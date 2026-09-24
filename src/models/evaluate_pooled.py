"""
Fusion tardive évaluée sur le POOL des runs de test RE2+RE3 plutôt que sur un
seul sous-ensemble à la fois.

Contexte: evaluate_multimodal.py --use-gat-traces --use-lstm-logs donne des
verdicts OPPOSÉS selon le sous-ensemble — sur RE2 (43 runs de validation),
remplacer traces/logs bruts par les embeddings GAT/LSTM dégrade le F1
(0,7179 -> 0,6222) ; sur RE3 (23 runs de validation), ça l'améliore
(0,8333 -> 0,8571). Trop peu d'échantillons dans les deux cas pour trancher
individuellement. Chaque détecteur par modalité reste entraîné/évalué sur
SON propre dataset (pas de fuite train RE2 -> test RE3 ou l'inverse, chaque
isolation_forest ne voit jamais l'autre source) — seuls les SCORES continus
résultants sont regroupés pour ajuster/évaluer le combinateur final sur un
ensemble de validation bien plus grand (~109 au lieu de 23-43).

Importe (plutôt que dupliquer, à l'inverse de la convention habituelle
d'indépendance entre scripts de ce module) MODALITY_PREFIXES/
load_labeled_data/merge_gat_trace_features/merge_lstm_log_features depuis
evaluate_multimodal.py: ce script étend directement sa logique de blocs par
modalité à plusieurs sources plutôt que de constituer une évaluation
indépendante — dupliquer cette logique une 3e fois (déjà présente dans
train_rcaeval.py) aurait rendu un bug déjà corrigé une fois plus probable
qu'utile ici.
"""

from datetime import datetime
from pathlib import Path
import argparse
import logging
import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.models.detection_models import AnomalyDetector
from src.models.evaluate_multimodal import (
    MODALITY_PREFIXES,
    load_labeled_data,
    merge_gat_trace_features,
    merge_lstm_log_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _blocks_for(features_df: pd.DataFrame, source_dir: Path, upgraded: bool) -> list:
    """Blocs (nom, DataFrame) par modalité — traces/logs bruts, ou
    traces_gat/logs_lstm à la place si upgraded=True (jamais les deux en
    même temps ici, contrairement à la variante 'all_signals' de
    evaluate_multimodal.py: on veut comparer raw vs GAT/LSTM à effectif de
    validation égal, pas les combiner)."""
    blocks = [
        (name, features_df[[c for c in features_df.columns if c.startswith(prefixes)]])
        for name, prefixes in MODALITY_PREFIXES
        if any(c.startswith(prefixes) for c in features_df.columns)
    ]
    if upgraded:
        blocks = [(n, b) for n, b in blocks if n not in ("traces", "logs")]
        gat_df = merge_gat_trace_features(features_df, source_dir)
        gat_cols = [c for c in gat_df.columns if c.startswith("trace_")]
        if gat_cols:
            blocks.append(("traces_gat", gat_df[gat_cols]))
        lstm_df = merge_lstm_log_features(features_df, source_dir)
        lstm_cols = [c for c in lstm_df.columns if c.startswith("event_")]
        if lstm_cols:
            blocks.append(("logs_lstm", lstm_df[lstm_cols]))
    return blocks


def compute_modality_scores(features_df: pd.DataFrame, labels_indexed: pd.DataFrame, blocks: list) -> tuple:
    """Un isolation_forest par bloc, entraîné sur le train (normal) de CE
    dataset, score calculé sur son propre test — jamais mélangé à
    l'entraînement d'une autre source. Retourne (scores_df, y), indexés par
    run_id (déjà uniques entre RE2/RE3: préfixés 'RE2-'/'RE3-' à la source,
    cf. rcaeval.py::parse)."""
    split = labels_indexed["fault_type"]
    y_all = pd.Series(labels_indexed["label"].to_numpy(), index=labels_indexed.index)
    train_ids = split[split == "train"].index
    test_ids = split[split.isin(["test_normal", "test_abnormal"])].index

    scores = {}
    for name, block_df in blocks:
        detector = AnomalyDetector(model_type="isolation_forest", contamination="auto")
        detector.build_model()
        detector.train(block_df.loc[train_ids].values)
        scores[name] = detector.anomaly_score(block_df.loc[test_ids].values)

    return pd.DataFrame(scores, index=test_ids), y_all.loc[test_ids]


def run_pooled_late_fusion(sources: list, upgraded: bool, seed: int = 42) -> dict:
    all_scores, all_y, n_per_source = [], [], []
    for source_dir in sources:
        features_df, labels_df = load_labeled_data(source_dir)
        labels_indexed = labels_df.set_index("run_id")
        labels_indexed = labels_indexed.loc[~labels_indexed.index.duplicated(keep="first")]

        blocks = _blocks_for(features_df, source_dir, upgraded)
        scores_df, y = compute_modality_scores(features_df, labels_indexed, blocks)
        all_scores.append(scores_df)
        all_y.append(y)
        n_per_source.append(len(scores_df))

    # Ne garde que les modalités présentes dans TOUTES les sources: RE3 n'a
    # aucune colonne 'logs' brute (0/90 cas avec cluster_id, contre une
    # minorité pour RE2 — cf. rcaeval.py) alors que logs_lstm existe pour les
    # deux (le mining Drain3 ne dépend pas de cluster_id). Sans ce filtre,
    # la concaténation introduit des NaN (colonne absente d'une source) que
    # LogisticRegression refuse. Loggé explicitement: c'est un résultat en
    # soi (couverture inégale des features brutes selon les sources).
    common_cols = set.intersection(*(set(df.columns) for df in all_scores))
    dropped = set.union(*(set(df.columns) for df in all_scores)) - common_cols
    if dropped:
        logger.warning(f"Modalités absentes d'au moins une source, exclues du pool: {sorted(dropped)}")
    all_scores = [df[sorted(common_cols)] for df in all_scores]

    pooled_scores = pd.concat(all_scores)
    pooled_y = pd.concat(all_y)
    logger.info(f"Pool: {len(pooled_scores)} runs de test au total ({dict(zip([s.name for s in sources], n_per_source))}), modalités: {sorted(common_cols)}")

    val_ids, test_ids = train_test_split(
        pooled_scores.index, test_size=0.5, random_state=seed, stratify=(pooled_y == -1)
    )

    mean, std = pooled_scores.loc[val_ids].mean(), pooled_scores.loc[val_ids].std().replace(0, 1)
    val_X = (pooled_scores.loc[val_ids] - mean) / std
    test_X = (pooled_scores.loc[test_ids] - mean) / std

    y_val_binary = (pooled_y.loc[val_ids] == -1).astype(int)
    y_test_binary = (pooled_y.loc[test_ids] == -1).astype(int)

    combiner = LogisticRegression()
    combiner.fit(val_X.values, y_val_binary.values)
    weights_info = dict(zip(val_X.columns, combiner.coef_[0]))

    test_pred = combiner.predict(test_X.values)
    test_proba = combiner.predict_proba(test_X.values)[:, 1]

    precision = precision_score(y_test_binary, test_pred, zero_division=0)
    recall = recall_score(y_test_binary, test_pred, zero_division=0)
    f1 = f1_score(y_test_binary, test_pred, zero_division=0)
    auc = roc_auc_score(y_test_binary, test_proba)

    label = "pooled_late_fusion_upgraded" if upgraded else "pooled_late_fusion_raw"
    logger.info(f"{label} — poids: {weights_info}")
    logger.info(
        f"{label}: précision={precision:.4f}, rappel={recall:.4f}, F1={f1:.4f}, AUC={auc:.4f}, "
        f"n_val={len(val_ids)}, n_test={len(test_ids)}"
    )

    return {
        "feature_set": label,
        "precision_anomalie": round(precision, 4),
        "rappel_anomalie": round(recall, 4),
        "f1_anomalie": round(f1, 4),
        "auc": round(auc, 4),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "n_vraies_anomalies_test": int(y_test_binary.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fusion tardive évaluée sur le pool des runs de test de plusieurs sous-ensembles RCAEval")
    parser.add_argument("--source-dirs", type=str, nargs="+", default=["data/interim/rcaeval/RE2", "data/interim/rcaeval/RE3"])
    args = parser.parse_args()
    sources = [REPO_ROOT / d for d in args.source_dirs]

    results = [
        run_pooled_late_fusion(sources, upgraded=False),
        run_pooled_late_fusion(sources, upgraded=True),
    ]
    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = REPO_ROOT / "experiments" / f"evaluation_pooled_{timestamp}.csv"
    out_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(out_path, index=False)
    logger.info(f"Résultats sauvegardés: {out_path}")


if __name__ == "__main__":
    main()
