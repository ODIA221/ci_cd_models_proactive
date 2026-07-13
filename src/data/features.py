"""
Extraction de features à partir des tables tidy du schéma unifié (schema.py)
Première brique de feature engineering: sac d'événements (bag-of-events) par run_id.

Limite connue: un sac d'événements ignore l'ordre des événements dans la
séquence (contrairement à des approches séquentielles type LSTM/DeepLog).
C'est une baseline volontairement simple, pas l'état de l'art — suffisante
pour obtenir une première mesure de performance chiffrée avant d'investir
dans un modèle plus complexe.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def build_event_count_matrix(logs_df: pd.DataFrame, id_col: str = "run_id", event_col: str = "template_id") -> pd.DataFrame:
    """
    Construit une matrice large (une ligne par run_id, une colonne par
    valeur distincte de `event_col`) comptant les occurrences de chaque
    événement dans la séquence de chaque run.

    Args:
        logs_df: DataFrame au schéma LOGS_COLUMNS (ou proche)
        id_col: colonne identifiant la séquence/le run (typiquement 'run_id')
        event_col: colonne identifiant le type d'événement (typiquement 'template_id')

    Returns:
        DataFrame indexé par `id_col`, une colonne par événement distinct, valeurs = compte
    """
    counts = (
        logs_df.groupby([id_col, event_col]).size().rename("count").reset_index()
    )
    matrix = counts.pivot(index=id_col, columns=event_col, values="count").fillna(0)
    matrix.columns = [f"event_{c}" for c in matrix.columns]
    logger.info(f"Matrice d'événements construite: {matrix.shape[0]} runs, {matrix.shape[1]} types d'événements")
    return matrix
