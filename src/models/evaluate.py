"""
Évaluation protocolée de la détection d'anomalies sur des données labellisées

Contrairement à src/models/train.py (entraînement non supervisé sans vérité
terrain, où le "taux d'anomalies détectées" ne prouve rien), ce script mesure
précision/rappel/F1/AUC réels contre des labels connus.

Protocole (standard pour l'évaluation d'anomalies sur séquences, ex. HDFS):
  - Entraînement UNIQUEMENT sur les séquences normales du split 'train'
    (aucune anomalie vue à l'entraînement — c'est la définition même de la
    détection d'anomalies non supervisée).
  - Évaluation sur test_normal + test_abnormal, jamais vus à l'entraînement.
Entraîner et évaluer sur les mêmes données (ou piocher un contamination/nu
"oracle" à partir des labels de test) fausserait les scores par fuite de
données (data leakage) — à éviter absolument, y compris en explorant des
variantes de ce script.

Feature engineering: compare deux jeux de features issus de
src/data/features.py — le sac d'événements (bag-of-events) par run_id, qui
ignore l'ordre des événements, et sa combinaison avec des comptes de
bigrammes de transitions (build_sequence_features), qui capture une partie
de cet ordre. Les deux jeux sont évalués sur les mêmes modèles/splits pour
mesurer si les bigrammes apportent un gain réel (pas juste plausible).
"""

from pathlib import Path
import argparse
import logging
import sys
from datetime import datetime

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data import schema
from data.features import build_sequence_features
from models.detection_models import AnomalyDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# (model_type, kwargs) — contamination='auto' est le choix méthodologiquement
# correct ici car le train set est composé UNIQUEMENT de séquences normales
# (cf. docstring du module): fixer contamination à une valeur non nulle sur
# des données d'entraînement sans anomalie fausserait le seuil de décision.
MODEL_CONFIGS = [
    ("isolation_forest_auto", "isolation_forest", {"contamination": "auto"}),
    ("isolation_forest_0.1", "isolation_forest", {"contamination": 0.1}),
    ("one_class_svm_nu0.1", "one_class_svm", {"nu": 0.1}),
]

# (nom, include_bigrams) — cf. build_sequence_features. Comparés sur les
# mêmes MODEL_CONFIGS/splits pour isoler l'effet du feature engineering.
FEATURE_SETS = [
    ("bag_of_events", False),
    ("bag_of_events+bigrams", True),
]


def load_labeled_data(source_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    logs_path = source_dir / "logs.parquet"
    labels_path = source_dir / "labels.parquet"
    if not logs_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"'{logs_path}' ou '{labels_path}' introuvable. "
            "Lance d'abord: python -m src.data.acquire --source loghub --dataset hdfs"
        )
    return pd.read_parquet(logs_path), pd.read_parquet(labels_path)


def evaluate_config(name: str, model_type: str, kwargs: dict, X_train: pd.DataFrame, X_test: pd.DataFrame, y_test) -> dict:
    detector = AnomalyDetector(model_type=model_type, **kwargs)
    detector.build_model()
    detector.train(X_train.values)

    y_pred = detector.predict(X_test.values)
    precision = precision_score(y_test, y_pred, pos_label=-1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=-1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=-1, zero_division=0)

    try:
        scores = -detector.model.decision_function(X_test.values)  # plus grand = plus anormal
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluation protocolée sur données labellisées")
    parser.add_argument("--source-dir", type=str, default="data/interim/loghub/hdfs", help="Dossier contenant logs.parquet + labels.parquet")
    args = parser.parse_args()

    source_dir = REPO_ROOT / args.source_dir
    logs_df, labels_df = load_labeled_data(source_dir)

    labels_indexed = labels_df.set_index("run_id")
    labels_indexed = labels_indexed.loc[~labels_indexed.index.duplicated(keep="first")]

    results = []
    for feature_name, include_bigrams in FEATURE_SETS:
        logger.info(f"=== Jeu de features: {feature_name} ===")
        X = build_sequence_features(logs_df, include_bigrams=include_bigrams)

        aligned = labels_indexed.reindex(X.index)
        split = aligned["fault_type"]  # cf. loghub.py: fault_type réutilisé pour stocker le split d'origine
        y = pd.Series(schema.to_sklearn_labels(aligned["label"]), index=X.index)

        train_mask = split == "train"
        test_mask = split.isin(["test_normal", "test_abnormal"])

        X_train, X_test = X[train_mask], X[test_mask]
        y_test = y[test_mask]

        logger.info(f"Train (normal uniquement): {len(X_train)} runs | Test: {len(X_test)} runs ({(y_test == -1).sum()} anomalies)")

        if feature_name == FEATURE_SETS[0][0]:
            # Baseline naïve pour contexte (toujours prédire "normal") — identique quel que soit le feature set
            baseline_accuracy = (y_test == 1).mean()
            logger.info(
                f"Baseline 'toujours normal': précision=0.0, rappel=0.0, "
                f"accuracy={baseline_accuracy:.4f} (accuracy trompeuse ici — classe très déséquilibrée)"
            )

        for name, model_type, kwargs in MODEL_CONFIGS:
            logger.info(f"--- {feature_name} / {name} ---")
            result = evaluate_config(name, model_type, kwargs, X_train, X_test, y_test)
            result["feature_set"] = feature_name
            results.append(result)
            logger.info(
                f"{name}: précision={result['precision_anomalie']}, rappel={result['rappel_anomalie']}, "
                f"F1={result['f1_anomalie']}, AUC={result['auc']}"
            )

    results_df = pd.DataFrame(results)
    results_df = results_df[["feature_set"] + [c for c in results_df.columns if c != "feature_set"]]
    print("\n" + results_df.to_string(index=False))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = REPO_ROOT / "experiments" / f"evaluation_{timestamp}.csv"
    out_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(out_path, index=False)
    logger.info(f"Résultats sauvegardés: {out_path}")


if __name__ == "__main__":
    main()
