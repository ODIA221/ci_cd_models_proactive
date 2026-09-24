"""
Entraîne et sauvegarde un modèle PERSISTANT (triplet .joblib/
_preprocessor.joblib/_meta.json, même contrat que src/models/train.py) à
partir des features RCAEval déjà fusionnées (metric_*/event_*/trace_*
concaténés par rcaeval.py::parse()) — contrairement à evaluate_multimodal.py,
qui n'entraîne que des détecteurs éphémères pour mesurer une métrique,
jamais sauvegardés dans models/, donc jamais servables par l'API.

Nécessaire pour que le module de corrélation causale (src/causal/) ait
quelque chose à expliquer en conditions réelles: sans modèle RCAEval dans le
registre models/*_meta.json, l'API/dashboard ne peuvent servir que le modèle
démo (data/raw/metrics/dataset_metrics.csv), qui n'a ni run_id ni traces.

Le run_id est conservé comme INDEX du DataFrame de features (déjà le cas
dans features.parquet, cf. rcaeval.py) tout au long de
CICDPreprocessor.fit/transform, qui ignore les colonnes 'id'/'timestamp' par
motif mais ne touche jamais à l'index — même mécanisme que 'timestamp' pour
le pipeline démo (load_data.py::load_metrics, qui met 'timestamp' en index
plutôt qu'en colonne). Aucune modification de train.py/preprocess.py n'est
donc nécessaire pour faire transiter run_id.
"""

from pathlib import Path
import argparse
import json
import logging
import sys
from datetime import datetime

import joblib
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data.preprocess import CICDPreprocessor
from src.data.sources.rcaeval import RCAEvalConnector
from src.models.detection_models import AnomalyDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Mêmes préfixes que evaluate_multimodal.py::MODALITY_PREFIXES — dupliqué
# volontairement (scripts indépendants, même convention que evaluate.py/
# evaluate_multimodal.py qui ne partagent pas non plus de code entre eux).
MODALITY_PREFIXES = [
    ("metrics", ("metric_",)),
    ("logs", ("event_", "2gram_")),
    ("traces", ("trace_",)),
]


def group_by_modality(features_df: pd.DataFrame):
    """
    Réordonne les colonnes de features_df par modalité (metrics, logs,
    traces) et retourne le nombre de colonnes par bloc. Nécessaire pour
    MultimodalAutoencoder (src/models/detection_models.py), qui tranche son
    tenseur d'entrée par position: CICDPreprocessor ne réordonne jamais les
    colonnes (pas de catégorielles dans les features RCAEval, donc pas de
    get_dummies), donc l'ordre fixé ici survit intact jusqu'à X_train/X_test.
    """
    blocks, modality_dims = [], {}
    for name, prefixes in MODALITY_PREFIXES:
        cols = [c for c in features_df.columns if c.startswith(prefixes)]
        if not cols:
            continue
        blocks.append(features_df[cols])
        modality_dims[name] = len(cols)
    return pd.concat(blocks, axis=1), modality_dims


def merge_gat_trace_features(features_df: pd.DataFrame, source_dir: Path) -> pd.DataFrame:
    """
    Remplace les colonnes trace_* (build_traces_agg_matrix, 6 statistiques
    agrégées) par les embeddings appris par graph_encoder.py (cache
    trace_gat_features.parquet, même préfixe trace_* donc group_by_modality
    ci-dessus n'a besoin d'aucune modification). fillna(0) pour les runs
    sans embedding (cas sans traces, déjà à 0 aujourd'hui pour ces runs).
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
    Remplace les colonnes event_*/2gram_* (sac d'événements + bigrammes) par
    les embeddings appris par log_sequence_encoder.py (cache
    event_lstm_features.parquet, préfixe event_* déjà reconnu par
    group_by_modality ci-dessus). fillna(0) pour les runs sans embedding.
    """
    lstm_path = source_dir / "event_lstm_features.parquet"
    if not lstm_path.exists():
        raise FileNotFoundError(
            f"'{lstm_path}' introuvable. Lance d'abord: python -m src.models.log_sequence_encoder --subset {source_dir.name}"
        )
    lstm_df = pd.read_parquet(lstm_path).reindex(features_df.index, fill_value=0.0)
    features_df = features_df.drop(columns=[c for c in features_df.columns if c.startswith(("event_", "2gram_"))])
    return pd.concat([features_df, lstm_df], axis=1)


def train_rcaeval_model(source_dir: Path, model_type: str = "isolation_forest", horizon_seconds: int = None,
                         variational_modalities: set = None, use_gat_traces: bool = False,
                         use_lstm_logs: bool = False, **model_kwargs) -> str:
    if (use_gat_traces or use_lstm_logs) and horizon_seconds is not None:
        # Les caches trace_gat_features.parquet / event_lstm_features.parquet
        # sont construits sur la fenêtre complète (720s) uniquement — les
        # combiner avec des features metrics tronquées à horizon_seconds
        # mélangerait deux horizons différents pour la même prédiction,
        # silencieusement.
        raise ValueError("--use-gat-traces/--use-lstm-logs ne sont pas supportés avec --horizon-seconds (caches non tronqués)")

    if horizon_seconds is not None:
        # Chapitre 10 (proactivité): entraîne sur des fenêtres tronquées à
        # `horizon_seconds` de part et d'autre de inject_time plutôt que sur
        # la fenêtre complète (720s) — cf. RCAEvalConnector.build_horizon_features
        # et evaluate_proactive.py, qui a mesuré qu'un horizon de 15s détecte
        # déjà aussi bien que la fenêtre complète sur la modalité traces.
        # run_id garde le MÊME nom que la fenêtre complète (pas de suffixe
        # _h{horizon}) pour que GET /explain/{run_id} retrouve toujours les
        # spans bruts déjà persistés par parse().
        connector = RCAEvalConnector()
        features_df, labels_df = connector.build_horizon_features(horizon=horizon_seconds, subset=source_dir.name)
        logger.info(f"Fenêtres tronquées à {horizon_seconds}s de part et d'autre de inject_time")
    else:
        features_df = pd.read_parquet(source_dir / "features.parquet")
        labels_df = pd.read_parquet(source_dir / "labels.parquet")

    if use_gat_traces:
        features_df = merge_gat_trace_features(features_df, source_dir)
    if use_lstm_logs:
        features_df = merge_lstm_log_features(features_df, source_dir)

    features_df, modality_dims = group_by_modality(features_df)
    labels_df = labels_df.set_index("run_id")
    labels_df = labels_df.loc[~labels_df.index.duplicated(keep="first")]

    split = labels_df.reindex(features_df.index)["fault_type"]
    X_train_raw = features_df[split == "train"]
    X_test_raw = features_df[split.isin(["test_normal", "test_abnormal"])]
    logger.info(f"Train (normal uniquement): {len(X_train_raw)} runs | Test: {len(X_test_raw)} runs")

    preprocessor = CICDPreprocessor(missing_strategy="knn")
    preprocessor.fit(X_train_raw)
    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    if model_type == "multimodal_autoencoder" and variational_modalities:
        model_kwargs["variational_modalities"] = variational_modalities

    detector = AnomalyDetector(model_type=model_type, **model_kwargs)
    if model_type == "autoencoder":
        detector.build_model(input_dim=X_train.shape[1])
    elif model_type == "multimodal_autoencoder":
        detector.build_model(modality_dims=modality_dims)
    else:
        detector.build_model()
    detector.train(X_train.values)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"{model_type}_rcaeval_{timestamp}"
    models_dir = REPO_ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    detector.save(models_dir / f"{model_id}.joblib")
    joblib.dump(preprocessor, models_dir / f"{model_id}_preprocessor.joblib")

    meta = {
        "model_type": model_type,
        "created_at": datetime.now().isoformat(),
        "feature_columns": X_train.columns.tolist(),
        "source_metrics_filename": None,
        "source_logs_filename": None,
        "target_col": None,
        "source": "rcaeval",
        # Chemin relatif au repo, utilisé par GET /explain/{run_id} pour
        # localiser traces/ et labels.parquet du même dossier RCAEval.
        "source_dir": str(source_dir.resolve().relative_to(REPO_ROOT)),
        # Nécessaire à model_registry.load_triplet() pour reconstruire
        # l'architecture (build_model) avant de charger le state_dict.
        "modality_dims": modality_dims if model_type == "multimodal_autoencoder" else None,
        # Nécessaire à model_registry.load_triplet() pour reconstruire les
        # têtes VAE (mu/logvar) avant load_state_dict — même raison que
        # modality_dims ci-dessus. Liste triée (JSON n'a pas de set).
        "variational_modalities": sorted(variational_modalities) if (model_type == "multimodal_autoencoder" and variational_modalities) else [],
        # Seuil de reconstruction appris sur X_train (cf. AnomalyDetector.train)
        # — save()/load() ne portent que les poids, donc sans ce champ un
        # modèle rechargé perd son seuil et /predict casse sur les requêtes
        # à une seule ligne (cf. model_registry.load_triplet).
        "threshold_": float(detector.threshold_) if detector.threshold_ is not None else None,
        # Chapitre 10 (proactivité): None = entraîné sur la fenêtre complète
        # (720s), sinon nombre de secondes de part et d'autre de inject_time
        # utilisées pour construire les features d'entraînement/test.
        "horizon_seconds": horizon_seconds,
    }
    with open(models_dir / f"{model_id}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info(f"Modèle sauvegardé: {model_id}")

    predictions = detector.predict(X_test.values)
    scores = detector.anomaly_score(X_test.values)

    # Contrairement à train.py (données démo, peu de colonnes), les features
    # RCAEval fusionnées comptent ~3000+ colonnes: on ne réexporte que
    # l'identifiant + le résultat, pas les features brutes.
    results_df = pd.DataFrame({
        "run_id": X_test_raw.index,
        "anomaly_score": scores,
        "prediction": predictions,
        "anomalie": predictions == -1,
    }).sort_values("anomaly_score", ascending=False)

    results_path = REPO_ROOT / "experiments" / f"results_{model_id}.csv"
    results_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(results_path, index=False)
    logger.info(f"Résultats sauvegardés: {results_path}")

    return model_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraîne un modèle RCAEval persistant, servable par l'API")
    parser.add_argument("--source-dir", type=str, default="data/interim/rcaeval/RE2")
    parser.add_argument("--model-type", type=str, default="isolation_forest", choices=["isolation_forest", "one_class_svm", "autoencoder", "multimodal_autoencoder"])
    parser.add_argument("--horizon-seconds", type=int, default=None, help="Si fourni: entraîne sur des fenêtres tronquées à N secondes de part et d'autre de inject_time (proactivité) au lieu de la fenêtre complète")
    parser.add_argument("--variational-modalities", type=str, default=None,
                         help="Liste séparée par virgules des modalités à rendre variationnelles (VAE), ex: 'metrics'. Ignoré hors multimodal_autoencoder.")
    parser.add_argument("--use-gat-traces", action="store_true",
                         help="Remplace les stats agrégées trace_* par les embeddings GAT mis en cache (python -m src.models.graph_encoder). Incompatible avec --horizon-seconds.")
    parser.add_argument("--use-lstm-logs", action="store_true",
                         help="Remplace le sac d'événements/bigrammes event_*/2gram_* par les embeddings LSTM mis en cache (python -m src.models.log_sequence_encoder). Incompatible avec --horizon-seconds.")
    args = parser.parse_args()
    variational_modalities = set(args.variational_modalities.split(",")) if args.variational_modalities else None
    train_rcaeval_model(REPO_ROOT / args.source_dir, model_type=args.model_type, horizon_seconds=args.horizon_seconds,
                         variational_modalities=variational_modalities, use_gat_traces=args.use_gat_traces,
                         use_lstm_logs=args.use_lstm_logs)


if __name__ == "__main__":
    main()
