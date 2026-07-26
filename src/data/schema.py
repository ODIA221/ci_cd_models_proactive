"""
Schéma unifié logs / métriques / traces / labels
Contrat commun que doivent respecter tous les connecteurs de src/data/sources/
avant d'écrire leurs sorties dans data/interim/<source>/

Format: une table "tidy" (longue) par modalité, reliées entre elles par
les colonnes de jointure (timestamp, source, run_id, service).
Cela permet de fusionner les modalités plus tard (chapitre 5 de la thèse)
sans avoir à re-normaliser chaque source individuellement.
"""

from typing import List
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Colonnes obligatoires par modalité (les colonnes absentes doivent être
# explicitement mises à None/NaN par le connecteur, jamais omises)
METRICS_COLUMNS: List[str] = ["timestamp", "source", "run_id", "service", "metric_name", "value", "unit"]
LOGS_COLUMNS: List[str] = ["timestamp", "source", "run_id", "service", "level", "template_id", "message", "label"]
TRACES_COLUMNS: List[str] = [
    "timestamp", "source", "run_id", "span_id", "parent_span_id",
    "service", "operation", "duration_ms", "status",
]
LABELS_COLUMNS: List[str] = ["run_id", "source", "label", "fault_type"]

# Colonnes optionnelles de labels.parquet: présentes uniquement pour les
# sources qui exposent un vrai ground truth de cause racine (ex. RCAEval,
# où le nom de dossier <service>_<fault_type> encode le service injecté).
# Ne sont PAS ajoutées à LABELS_COLUMNS pour ne pas casser les connecteurs
# (loghub, otel_demo) qui ne les fournissent pas — validate_labels_df ne les
# exige donc jamais. root_cause_fault_type diffère de la colonne fault_type
# ci-dessus (qui encode train/test_normal/test_abnormal, pas le type de faute
# injectée).
OPTIONAL_LABELS_COLUMNS: List[str] = ["root_cause_service", "root_cause_fault_type"]


def _validate(df: pd.DataFrame, required_columns: List[str], modality: str) -> None:
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes pour la modalité '{modality}': {missing}")
    if df.empty:
        logger.warning(f"DataFrame '{modality}' vide")
    logger.info(f"Validation '{modality}' réussie: {len(df)} lignes, {len(df.columns)} colonnes")


def validate_metrics_df(df: pd.DataFrame) -> None:
    """Vérifie qu'un DataFrame de métriques respecte le schéma unifié (tidy/long)"""
    _validate(df, METRICS_COLUMNS, "metrics")


def validate_logs_df(df: pd.DataFrame) -> None:
    """Vérifie qu'un DataFrame de logs respecte le schéma unifié (tidy/long)"""
    _validate(df, LOGS_COLUMNS, "logs")


def validate_traces_df(df: pd.DataFrame) -> None:
    """Vérifie qu'un DataFrame de traces respecte le schéma unifié (tidy/long)"""
    _validate(df, TRACES_COLUMNS, "traces")


def validate_labels_df(df: pd.DataFrame) -> None:
    """Vérifie qu'un DataFrame de labels respecte le schéma unifié"""
    _validate(df, LABELS_COLUMNS, "labels")


def pivot_metrics_to_wide(df: pd.DataFrame, index: str = "timestamp") -> pd.DataFrame:
    """
    Reformate des métriques tidy (une ligne par (timestamp, metric_name))
    vers le format large (une colonne par métrique) attendu par
    CICDDataLoader/CICDPreprocessor.

    Args:
        df: DataFrame au schéma METRICS_COLUMNS
        index: colonne à utiliser comme index (typiquement 'timestamp')

    Returns:
        DataFrame large, une colonne par metric_name
    """
    validate_metrics_df(df)
    wide = df.pivot_table(index=index, columns="metric_name", values="value", aggfunc="mean")
    wide.columns.name = None
    return wide


def to_sklearn_labels(labels: pd.Series) -> np.ndarray:
    """
    Convertit une série de labels 0/1 (0=normal, 1=anomalie) ou
    'normal'/'anomaly' vers la convention scikit-learn utilisée par
    AnomalyDetector: -1 = anomalie, 1 = normal.
    """
    if labels.dtype == object:
        normalized = labels.astype(str).str.lower().map(
            {"normal": 1, "anomaly": -1, "anomalie": -1, "0": 1, "1": -1}
        )
    else:
        normalized = labels.map({0: 1, 1: -1})

    if normalized.isna().any():
        raise ValueError("Valeurs de label non reconnues, attendu 0/1 ou normal/anomaly")

    return normalized.to_numpy()
