"""
Évaluation protocolée de la fusion multimodale (logs+métriques+traces) sur
RCAEval RE2 — même discipline méthodologique que src/models/evaluate.py
(HDFS, logs seuls), étendue à la comparaison mono-modalité vs fusion.

Protocole (construit par src/data/sources/rcaeval.py au moment du parse()):
  - Chaque cas RCAEval ne fournit qu'UNE fenêtre normale et UNE fenêtre
    anormale (avant/après inject_time.txt) — pas de split pré-fourni comme
    pour LogHub. Le connecteur assigne chaque CAS (pas chaque fenêtre) à
    'train' (~70%, fenêtre normale uniquement) ou 'test' (~30%, les deux
    fenêtres) de façon déterministe, pour garantir qu'aucun cas du test
    n'a contribué à l'entraînement, même partiellement.
  - Entraînement UNIQUEMENT sur les fenêtres normales du split 'train'.
  - Évaluation sur test_normal + test_abnormal, jamais vus à l'entraînement.

Question empirique posée ici (pas supposée a priori): la fusion des trois
modalités fait-elle mieux que chaque modalité seule ? On compare donc
4 jeux de features (metrics/logs/traces/fused) en filtrant les colonnes de
features.parquet par préfixe (metric_*, event_*/2gram_*, trace_*) — ce
fichier unique est produit par src/data/sources/rcaeval.py::parse(), qui
agrège déjà chaque modalité par run_id via src/data/features.py.
"""

from pathlib import Path
import argparse
import logging
import sys
from datetime import datetime

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.detection_models import AnomalyDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_CONFIGS = [
    ("isolation_forest_auto", "isolation_forest", {"contamination": "auto"}),
    ("isolation_forest_0.1", "isolation_forest", {"contamination": 0.1}),
    ("one_class_svm_nu0.1", "one_class_svm", {"nu": 0.1}),
]

# (nom, préfixes de colonnes) — le nom 'fused' garde toutes les colonnes;
# les autres sont des ablations mono-modalité pour mesurer l'apport réel de
# la fusion plutôt que le supposer. cf. features.py: build_metrics_agg_matrix
# préfixe "metric_", build_sequence_features "event_"/"2gram_",
# build_traces_agg_matrix "trace_".
FEATURE_SETS = [
    ("metrics", ("metric_",)),
    ("logs", ("event_", "2gram_")),
    ("traces", ("trace_",)),
    ("fused", None),
]

# Modalités utilisées pour la fusion tardive (late_fusion_logreg, cf. plus
# bas) — mêmes préfixes que FEATURE_SETS, sans l'entrée 'fused'.
MODALITY_PREFIXES = [(name, prefixes) for name, prefixes in FEATURE_SETS if prefixes is not None]


def load_labeled_data(source_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    features_path = source_dir / "features.parquet"
    labels_path = source_dir / "labels.parquet"
    if not features_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"'{features_path}' ou '{labels_path}' introuvable. "
            "Lance d'abord: python -m src.data.acquire --source rcaeval --subset RE2 (fetch puis parse)"
        )
    return pd.read_parquet(features_path), pd.read_parquet(labels_path)


def build_features_for_set(features_df: pd.DataFrame, prefixes) -> pd.DataFrame:
    if prefixes is None:
        return features_df
    cols = [c for c in features_df.columns if c.startswith(prefixes)]
    if not cols:
        raise ValueError(f"Aucune colonne ne correspond aux préfixes {prefixes}")
    return features_df[cols]


def build_modality_blocks(features_df: pd.DataFrame):
    """
    Construit la matrice concaténée (métriques, logs, traces) dans un ordre
    FIXE, avec le nombre de colonnes par modalité — nécessaire à
    MultimodalAutoencoder (src/models/detection_models.py), qui tranche son
    tenseur d'entrée par position plutôt que par nom de colonne, contrairement
    à build_features_for_set ci-dessus qui ne garde qu'un seul jeu à la fois.

    Returns:
        (X, modality_dims): X = DataFrame concaténé, modality_dims = dict
        {nom_modalité: nb_colonnes} dans le même ordre que les blocs de X.
    """
    blocks, modality_dims = [], {}
    for name, prefixes in MODALITY_PREFIXES:
        cols = [c for c in features_df.columns if c.startswith(prefixes)]
        if not cols:
            continue
        blocks.append(features_df[cols])
        modality_dims[name] = len(cols)

    if not blocks:
        raise ValueError("Aucune modalité disponible pour construire la matrice jointe")

    return pd.concat(blocks, axis=1), modality_dims


def run_joint_fusion_autoencoder(features_df: pd.DataFrame, labels_indexed: pd.DataFrame, epochs: int = 50,
                                  variational_modalities: set = None) -> dict:
    """
    Fusion PRÉCOCE apprise conjointement: un seul MultimodalAutoencoder
    entraîné de bout en bout sur les 3 modalités concaténées (une branche
    encodeur/décodeur par modalité, goulot d'étranglement latent partagé) —
    par opposition à run_late_fusion ci-dessous (3 détecteurs indépendants +
    un combinateur logistique appris après coup sur leurs scores).

    Même protocole que evaluate_config: fit uniquement sur les runs normaux
    du split 'train', évaluation directe sur test_normal+test_abnormal.
    Contrairement à run_late_fusion, pas besoin de split val supplémentaire
    ici: l'autoencodeur est non supervisé (seuil de reconstruction recalculé
    par AnomalyDetector.predict), il n'y a pas de combinateur à ajuster
    séparément sur des labels.

    variational_modalities (optionnel): rend ces branches variationnelles
    (VAE) au lieu de déterministes — permet de comparer, dans le même run,
    la fusion précoce classique à une variante VAE modalité par modalité.
    """
    X, modality_dims = build_modality_blocks(features_df)

    split = labels_indexed.reindex(X.index)["fault_type"]
    y = pd.Series(labels_indexed.reindex(X.index)["label"].to_numpy(), index=X.index)

    train_mask = split == "train"
    test_mask = split.isin(["test_normal", "test_abnormal"])

    X_train, X_test = X[train_mask], X[test_mask]
    y_test = y[test_mask]

    logger.info(f"joint_fusion_autoencoder — modalités: {modality_dims} | train: {len(X_train)} | test: {len(X_test)}")

    label = "joint_fusion_vae_metrics" if variational_modalities else "joint_fusion_autoencoder"

    detector = AnomalyDetector(model_type="multimodal_autoencoder", variational_modalities=variational_modalities)
    detector.build_model(modality_dims=modality_dims)
    detector.train(X_train.values, epochs=epochs)

    y_pred = detector.predict(X_test.values)
    precision = precision_score(y_test, y_pred, pos_label=-1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=-1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=-1, zero_division=0)

    scores = detector.anomaly_score(X_test.values)
    y_true_binary = (y_test == -1).astype(int)
    try:
        auc = roc_auc_score(y_true_binary, scores)
    except ValueError as e:
        logger.warning(f"AUC non calculable pour {label}: {e}")
        auc = None

    logger.info(
        f"{label}: précision={precision:.4f}, rappel={recall:.4f}, "
        f"F1={f1:.4f}, AUC={auc if auc is None else round(auc, 4)}"
    )

    return {
        "feature_set": label,
        "config": "multimodal_autoencoder" if not variational_modalities else f"multimodal_autoencoder+vae({','.join(sorted(variational_modalities))})",
        "precision_anomalie": round(precision, 4),
        "rappel_anomalie": round(recall, 4),
        "f1_anomalie": round(f1, 4),
        "auc": round(auc, 4) if auc is not None else None,
        "n_predictions_anomalie": int((y_pred == -1).sum()),
        "n_vraies_anomalies": int((y_test == -1).sum()),
        "n_test": len(y_test),
    }


def evaluate_config(name: str, model_type: str, kwargs: dict, X_train: pd.DataFrame, X_test: pd.DataFrame, y_test) -> dict:
    detector = AnomalyDetector(model_type=model_type, **kwargs)
    detector.build_model()
    detector.train(X_train.values)

    y_pred = detector.predict(X_test.values)
    precision = precision_score(y_test, y_pred, pos_label=-1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=-1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=-1, zero_division=0)

    try:
        scores = -detector.model.decision_function(X_test.values)
        y_true_binary = (y_test == -1).astype(int)
        auc = roc_auc_score(y_true_binary, scores)
    except (AttributeError, ValueError) as e:
        logger.warning(f"AUC non calculable pour '{name}': {e}")
        auc = None

    return {
        "config": name,
        "precision_anomalie": round(precision, 4),
        "rappel_anomalie": round(recall, 4),
        "f1_anomalie": round(f1, 4),
        "auc": round(auc, 4) if auc is not None else None,
        "n_predictions_anomalie": int((y_pred == -1).sum()),
        "n_vraies_anomalies": int((y_test == -1).sum()),
        "n_test": len(y_test),
    }


def _evaluate_modality_ablation(feature_set_name: str, features_df: pd.DataFrame, prefixes: tuple, labels_indexed: pd.DataFrame) -> list:
    """Factorise la boucle MODEL_CONFIGS déjà utilisée pour FEATURE_SETS
    ci-dessus, réutilisée pour les ablations traces_gat/logs_lstm (mêmes
    étapes: filtrer par préfixe, split train/test, évaluer chaque config)."""
    X = build_features_for_set(features_df, prefixes)
    aligned = labels_indexed.reindex(X.index)
    split = aligned["fault_type"]
    y = pd.Series(aligned["label"].to_numpy(), index=X.index)
    train_mask = split == "train"
    test_mask = split.isin(["test_normal", "test_abnormal"])
    X_train, X_test = X[train_mask], X[test_mask]
    y_test = y[test_mask]

    rows = []
    for name, model_type, kwargs in MODEL_CONFIGS:
        logger.info(f"--- {feature_set_name} / {name} ---")
        result = evaluate_config(name, model_type, kwargs, X_train, X_test, y_test)
        result["feature_set"] = feature_set_name
        rows.append(result)
        logger.info(
            f"{name}: précision={result['precision_anomalie']}, rappel={result['rappel_anomalie']}, "
            f"F1={result['f1_anomalie']}, AUC={result['auc']}"
        )
    return rows


def _late_fusion_split(labels_indexed: pd.DataFrame, seed: int = 42):
    """Split train/val/test partagé par toutes les variantes de fusion
    tardive (même seed -> même partition, comparaison à périmètre égal)."""
    split = labels_indexed["fault_type"]
    y_all = pd.Series(labels_indexed["label"].to_numpy(), index=labels_indexed.index)
    train_ids = split[split == "train"].index
    test_ids = split[split.isin(["test_normal", "test_abnormal"])].index
    val_ids, final_test_ids = train_test_split(
        test_ids, test_size=0.5, random_state=seed, stratify=y_all.loc[test_ids]
    )
    return train_ids, val_ids, final_test_ids, y_all


def _run_late_fusion_from_blocks(blocks: list, train_ids, val_ids, final_test_ids, y_all: pd.Series) -> dict:
    """
    Cœur de la fusion tardive, factorisé pour accepter des blocs
    (nom, DataFrame de colonnes) venant potentiellement de SOURCES
    DIFFÉRENTES — ex: le score isolation_forest sur les stats agrégées
    'traces' ET sur l'embedding GAT 'traces_gat' EN PARALLÈLE (pas en
    remplacement l'un de l'autre), pour laisser le combinateur choisir plutôt
    que de supposer que le second remplace forcément le premier avec profit
    (mesuré empiriquement faux: F1 0,72 -> 0,62 en remplaçant traces par
    traces_gat/logs_lstm dans ce même combinateur).

    Principe: un détecteur isolation_forest par bloc (entraîné sur les runs
    'train' normaux uniquement), puis un combinateur appris (régression
    logistique) sur les scores d'anomalie continus. Pour rester sans fuite
    de données, le split 'test' existant est lui-même divisé en deux: une
    moitié 'val' sert à entraîner UNIQUEMENT les poids du combinateur,
    l'autre moitié 'test' (jamais vue ni par les détecteurs par bloc ni par
    le combinateur) sert au rapport final.
    """
    val_scores, test_scores = {}, {}
    for name, block_df in blocks:
        detector = AnomalyDetector(model_type="isolation_forest", contamination="auto")
        detector.build_model()
        detector.train(block_df.loc[train_ids].values)
        val_scores[name] = detector.anomaly_score(block_df.loc[val_ids].values)
        test_scores[name] = detector.anomaly_score(block_df.loc[final_test_ids].values)

    val_X = pd.DataFrame(val_scores, index=val_ids)
    test_X = pd.DataFrame(test_scores, index=final_test_ids)

    # Normalisation (z-score) selon les stats du split val UNIQUEMENT
    # (jamais le split test final) pour éviter toute fuite.
    mean, std = val_X.mean(), val_X.std().replace(0, 1)
    val_X_scaled = (val_X - mean) / std
    test_X_scaled = (test_X - mean) / std

    y_val_binary = (y_all.loc[val_ids] == -1).astype(int)
    y_test_binary = (y_all.loc[final_test_ids] == -1).astype(int)

    combiner = LogisticRegression()
    combiner.fit(val_X_scaled.values, y_val_binary.values)
    weights_info = dict(zip(val_X.columns, combiner.coef_[0]))

    test_pred = combiner.predict(test_X_scaled.values)
    test_proba = combiner.predict_proba(test_X_scaled.values)[:, 1]

    precision = precision_score(y_test_binary, test_pred, zero_division=0)
    recall = recall_score(y_test_binary, test_pred, zero_division=0)
    f1 = f1_score(y_test_binary, test_pred, zero_division=0)
    auc = roc_auc_score(y_test_binary, test_proba)

    logger.info(f"Fusion tardive (logreg sur scores/modalité) - poids appris: {weights_info}")
    logger.info(f"late_fusion_logreg: précision={precision:.4f}, rappel={recall:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

    return {
        "feature_set": "late_fusion_logreg",
        "config": "logreg_on_modality_scores",
        "precision_anomalie": round(precision, 4),
        "rappel_anomalie": round(recall, 4),
        "f1_anomalie": round(f1, 4),
        "auc": round(auc, 4),
        "n_predictions_anomalie": int(test_pred.sum()),
        "n_vraies_anomalies": int(y_test_binary.sum()),
        "n_test": len(final_test_ids),
    }


def run_late_fusion(features_df: pd.DataFrame, labels_indexed: pd.DataFrame, seed: int = 42) -> dict:
    """Fusion tardive (score-level) plutôt que la concaténation brute des
    features (jeu 'fused' plus haut) — motivée par le constat empirique que
    la concaténation dégrade la performance par rapport à la meilleure
    modalité seule (trop de colonnes peu informatives, notamment les
    bigrammes de logs, qui noient le signal métriques/traces). Un bloc par
    modalité de MODALITY_PREFIXES, cf. _run_late_fusion_from_blocks."""
    train_ids, val_ids, final_test_ids, y_all = _late_fusion_split(labels_indexed, seed=seed)
    blocks = [
        (name, features_df[[c for c in features_df.columns if c.startswith(prefixes)]])
        for name, prefixes in MODALITY_PREFIXES
        if any(c.startswith(prefixes) for c in features_df.columns)
    ]
    return _run_late_fusion_from_blocks(blocks, train_ids, val_ids, final_test_ids, y_all)


def merge_gat_trace_features(features_df: pd.DataFrame, source_dir: Path) -> pd.DataFrame:
    """
    Remplace les colonnes trace_* (build_traces_agg_matrix, 6 statistiques
    agrégées) par les embeddings appris par graph_encoder.py (cache
    trace_gat_features.parquet, même préfixe trace_* donc
    build_features_for_set/build_modality_blocks n'ont besoin d'aucune
    modification). fillna(0) pour les runs sans embedding (cas sans traces,
    déjà à 0 aujourd'hui pour ces runs). Dupliqué depuis train_rcaeval.py —
    même convention que MODALITY_PREFIXES ci-dessus (scripts indépendants).
    """
    gat_path = source_dir / "trace_gat_features.parquet"
    if not gat_path.exists():
        raise FileNotFoundError(
            f"'{gat_path}' introuvable. Lance d'abord: python -m src.models.graph_encoder --subset {source_dir.name}"
        )
    gat_df = pd.read_parquet(gat_path).reindex(features_df.index, fill_value=0.0)
    features_df = features_df.drop(columns=[c for c in features_df.columns if c.startswith("trace_")])
    return pd.concat([features_df, gat_df], axis=1)


def merge_lstm_log_features(features_df: pd.DataFrame, source_dir: Path) -> pd.DataFrame:
    """
    Remplace les colonnes event_*/2gram_* par les embeddings appris par
    log_sequence_encoder.py (cache event_lstm_features.parquet, préfixe
    event_* déjà reconnu par MODALITY_PREFIXES/FEATURE_SETS ci-dessus).
    Dupliqué depuis train_rcaeval.py — même convention que
    merge_gat_trace_features (scripts indépendants).
    """
    lstm_path = source_dir / "event_lstm_features.parquet"
    if not lstm_path.exists():
        raise FileNotFoundError(
            f"'{lstm_path}' introuvable. Lance d'abord: python -m src.models.log_sequence_encoder --subset {source_dir.name}"
        )
    lstm_df = pd.read_parquet(lstm_path).reindex(features_df.index, fill_value=0.0)
    features_df = features_df.drop(columns=[c for c in features_df.columns if c.startswith(("event_", "2gram_"))])
    return pd.concat([features_df, lstm_df], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluation protocolée de la fusion multimodale sur RCAEval")
    parser.add_argument("--source-dir", type=str, default="data/interim/rcaeval/RE2", help="Dossier contenant features.parquet + labels.parquet")
    parser.add_argument("--use-gat-traces", action="store_true",
                         help="Ajoute une ablation 'traces_gat' et une fusion jointe utilisant les embeddings GAT en cache (python -m src.models.graph_encoder), en plus des résultats trace_* existants.")
    parser.add_argument("--use-lstm-logs", action="store_true",
                         help="Ajoute une ablation 'logs_lstm' et une fusion jointe utilisant les embeddings LSTM en cache (python -m src.models.log_sequence_encoder), en plus des résultats event_*/2gram_* existants.")
    args = parser.parse_args()

    source_dir = REPO_ROOT / args.source_dir
    features_df, labels_df = load_labeled_data(source_dir)

    labels_indexed = labels_df.set_index("run_id")
    labels_indexed = labels_indexed.loc[~labels_indexed.index.duplicated(keep="first")]

    results = []
    for feature_set, prefixes in FEATURE_SETS:
        logger.info(f"=== Jeu de features: {feature_set} ===")
        try:
            X = build_features_for_set(features_df, prefixes)
        except ValueError as e:
            logger.warning(f"Jeu '{feature_set}' ignoré: {e}")
            continue

        aligned = labels_indexed.reindex(X.index)
        split = aligned["fault_type"]  # cf. rcaeval.py: fault_type réutilisé pour stocker le split (train/test_normal/test_abnormal)
        # rcaeval.py écrit déjà label au format -1/1 (contrairement à loghub.py qui écrit du 0/1
        # brut converti ensuite via schema.to_sklearn_labels) - pas de reconversion ici.
        y = pd.Series(aligned["label"].to_numpy(), index=X.index)

        train_mask = split == "train"
        test_mask = split.isin(["test_normal", "test_abnormal"])

        X_train, X_test = X[train_mask], X[test_mask]
        y_test = y[test_mask]

        logger.info(f"Train (normal uniquement): {len(X_train)} runs | Test: {len(X_test)} runs ({(y_test == -1).sum()} anomalies)")

        for name, model_type, kwargs in MODEL_CONFIGS:
            logger.info(f"--- {feature_set} / {name} ---")
            result = evaluate_config(name, model_type, kwargs, X_train, X_test, y_test)
            result["feature_set"] = feature_set
            results.append(result)
            logger.info(
                f"{name}: précision={result['precision_anomalie']}, rappel={result['rappel_anomalie']}, "
                f"F1={result['f1_anomalie']}, AUC={result['auc']}"
            )

    features_df_gat = None
    if args.use_gat_traces:
        features_df_gat = merge_gat_trace_features(features_df, source_dir)
        logger.info("=== Jeu de features: traces_gat ===")
        results.extend(_evaluate_modality_ablation("traces_gat", features_df_gat, ("trace_",), labels_indexed))

    features_df_lstm = None
    if args.use_lstm_logs:
        features_df_lstm = merge_lstm_log_features(features_df, source_dir)
        logger.info("=== Jeu de features: logs_lstm ===")
        results.extend(_evaluate_modality_ablation("logs_lstm", features_df_lstm, ("event_",), labels_indexed))

    # features_df "amélioré": GAT traces / LSTM logs par-dessus les colonnes
    # brutes quand demandé — les deux merge_* ne touchent que leur propre
    # préfixe, donc composables. Calculé une seule fois, réutilisé plus bas
    # à la fois pour la fusion tardive et la fusion jointe cumulées.
    features_df_upgraded = features_df_gat if features_df_gat is not None else features_df
    if args.use_lstm_logs:
        features_df_upgraded = merge_lstm_log_features(features_df_upgraded, source_dir)
    has_upgrade = args.use_gat_traces or args.use_lstm_logs

    logger.info("=== Fusion tardive (late_fusion_logreg) ===")
    results.append(run_late_fusion(features_df, labels_indexed))

    if has_upgrade:
        # Remplace traces/logs bruts par traces_gat/logs_lstm dans le même
        # combinateur — mesuré: DÉGRADE le résultat (F1 0,7179 -> 0,6222).
        # traces_gat a une meilleure F1 en isolation mais un AUC légèrement
        # inférieur à traces brut (0,7378 vs 0,7784): un score continu moins
        # bien calibré pour le combinateur, même si son seuil isolation_forest
        # propre sépare mieux. Gardé pour la comparaison, cf. all_signals
        # ci-dessous pour la variante qui a effectivement aidé.
        logger.info("=== Fusion tardive, features remplacées (GAT traces / LSTM logs) ===")
        result = run_late_fusion(features_df_upgraded, labels_indexed)
        result["feature_set"] = "late_fusion_logreg_upgraded"
        results.append(result)

        # Variante: AJOUTE traces_gat/logs_lstm comme blocs supplémentaires
        # au lieu de remplacer traces/logs bruts — laisse le combinateur
        # décider plutôt que de supposer que le nouveau signal est
        # strictement meilleur (contredit juste au-dessus).
        logger.info("=== Fusion tardive, tous les signaux (bruts + GAT/LSTM en plus) ===")
        train_ids, val_ids, final_test_ids, y_all = _late_fusion_split(labels_indexed)
        blocks = [
            (name, features_df[[c for c in features_df.columns if c.startswith(prefixes)]])
            for name, prefixes in MODALITY_PREFIXES
            if any(c.startswith(prefixes) for c in features_df.columns)
        ]
        if features_df_gat is not None:
            cols = [c for c in features_df_gat.columns if c.startswith("trace_")]
            blocks.append(("traces_gat", features_df_gat[cols]))
        if features_df_lstm is not None:
            cols = [c for c in features_df_lstm.columns if c.startswith("event_")]
            blocks.append(("logs_lstm", features_df_lstm[cols]))
        result = _run_late_fusion_from_blocks(blocks, train_ids, val_ids, final_test_ids, y_all)
        result["feature_set"] = "late_fusion_logreg_all_signals"
        results.append(result)

    logger.info("=== Fusion jointe (joint_fusion_autoencoder) ===")
    results.append(run_joint_fusion_autoencoder(features_df, labels_indexed))

    logger.info("=== Fusion jointe, branche métriques en VAE (joint_fusion_vae_metrics) ===")
    results.append(run_joint_fusion_autoencoder(features_df, labels_indexed, variational_modalities={"metrics"}))

    if features_df_gat is not None:
        logger.info("=== Fusion jointe, VAE métriques + GAT traces ===")
        result = run_joint_fusion_autoencoder(features_df_gat, labels_indexed, variational_modalities={"metrics"})
        result["feature_set"] = "joint_fusion_vae_metrics+gat_traces"
        result["config"] = "multimodal_autoencoder+vae(metrics)+gat_traces"
        results.append(result)

    if features_df_lstm is not None:
        # features_df_upgraded déjà calculé plus haut (GAT + LSTM cumulés si
        # les deux flags sont passés, sinon LSTM seul par-dessus les brutes).
        label = "joint_fusion_vae_metrics+gat_traces+lstm_logs" if features_df_gat is not None else "joint_fusion_vae_metrics+lstm_logs"
        logger.info(f"=== Fusion jointe, {label} ===")
        result = run_joint_fusion_autoencoder(features_df_upgraded, labels_indexed, variational_modalities={"metrics"})
        result["feature_set"] = label
        result["config"] = f"multimodal_autoencoder+vae(metrics)+{'gat_traces+' if features_df_gat is not None else ''}lstm_logs"
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df = results_df[["feature_set"] + [c for c in results_df.columns if c != "feature_set"]]
    print("\n" + results_df.to_string(index=False))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = REPO_ROOT / "experiments" / f"evaluation_rcaeval_{timestamp}.csv"
    out_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(out_path, index=False)
    logger.info(f"Résultats sauvegardés: {out_path}")


if __name__ == "__main__":
    main()
