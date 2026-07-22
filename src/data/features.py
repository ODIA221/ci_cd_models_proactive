"""
Extraction de features à partir des tables tidy du schéma unifié (schema.py)

- build_event_count_matrix: sac d'événements (bag-of-events) par run_id —
  baseline volontairement simple, mais qui ignore l'ordre des événements
  dans la séquence (contrairement à des approches séquentielles type
  LSTM/DeepLog).
- build_event_ngram_matrix / build_sequence_features: complètent le sac
  d'événements avec des comptes de bigrammes de transitions, pour capturer
  une partie de l'ordre sans passer par un modèle séquentiel complet. Encore
  une approximation (pas de mémoire au-delà de 2 événements), mais un premier
  pas mesurable au-delà du bag-of-events pur — cf. evaluate.py qui compare
  les deux jeux de features sur les mêmes modèles.
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


def build_event_ngram_matrix(logs_df: pd.DataFrame, id_col: str = "run_id", event_col: str = "template_id", n: int = 2) -> pd.DataFrame:
    """
    Complète build_event_count_matrix en comptant les n-grammes consécutifs
    d'événements (transitions) par run_id, au lieu d'un sac d'événements qui
    ignore l'ordre. Ne remplace pas le sac d'événements: à combiner avec lui
    via build_sequence_features.

    L'ordre des événements est déduit de l'ordre des lignes de `logs_df` au
    sein de chaque run_id (ordre d'insertion), pas d'une colonne timestamp
    (absente dans les échantillons LogHub pré-calculés — cf. loghub.py).
    """
    if n < 2:
        raise ValueError("n doit être >= 2 (utiliser build_event_count_matrix pour n=1)")

    grouped = logs_df.groupby(id_col, sort=False)[event_col]
    shifted = [grouped.shift(-i) for i in range(n)]
    valid = shifted[-1].notna()

    ngram = shifted[0][valid].astype(str)
    for s in shifted[1:]:
        ngram = ngram.str.cat(s[valid].astype(str), sep="->")

    ngram_df = pd.DataFrame({id_col: logs_df.loc[valid, id_col].to_numpy(), "ngram": ngram.to_numpy()})
    counts = ngram_df.groupby([id_col, "ngram"]).size().rename("count").reset_index()
    matrix = counts.pivot(index=id_col, columns="ngram", values="count").fillna(0)
    matrix.columns = [f"{n}gram_{c}" for c in matrix.columns]
    logger.info(f"Matrice de {n}-grammes construite: {matrix.shape[0]} runs, {matrix.shape[1]} transitions distinctes")
    return matrix


def build_sequence_features(logs_df: pd.DataFrame, id_col: str = "run_id", event_col: str = "template_id", include_bigrams: bool = True) -> pd.DataFrame:
    """
    Matrice de features par run_id combinant le sac d'événements (ordre
    ignoré) et, si include_bigrams=True, les comptes de bigrammes de
    transitions (ordre partiellement restauré) — dépasse la limite du sac
    d'événements pur documentée en tête de ce module.

    Runs à un seul événement: aucun bigramme possible, colonnes bigrammes à 0
    (reindex sur l'index du sac d'événements, qui couvre tous les run_id).
    """
    bag = build_event_count_matrix(logs_df, id_col, event_col)
    if not include_bigrams:
        return bag

    bigrams = build_event_ngram_matrix(logs_df, id_col, event_col, n=2)
    bigrams = bigrams.reindex(bag.index, fill_value=0)
    return pd.concat([bag, bigrams], axis=1)


def build_metrics_agg_matrix(metrics_df: pd.DataFrame, id_col: str = "run_id", metric_col: str = "metric_name", value_col: str = "value") -> pd.DataFrame:
    """
    Agrège les métriques tidy (une ligne par timestamp x metric_name) en une
    matrice large, une ligne par run_id, quatre colonnes (mean/std/min/max)
    par métrique — résume la fenêtre temporelle d'un run sans dépendre de sa
    longueur exacte (nombre de points variable selon les cas).
    """
    grouped = metrics_df.groupby([id_col, metric_col])[value_col].agg(["mean", "std", "min", "max"])
    wide = grouped.unstack(metric_col)
    wide.columns = [f"metric_{metric}_{stat}" for stat, metric in wide.columns]
    wide = wide.fillna(0)
    logger.info(f"Matrice de métriques agrégées construite: {wide.shape[0]} runs, {wide.shape[1]} colonnes")
    return wide


def build_traces_agg_matrix(traces_df: pd.DataFrame, id_col: str = "run_id") -> pd.DataFrame:
    """
    Agrège les traces (une ligne par span) en une matrice large, une ligne
    par run_id: nombre de spans, durée moyenne/écart-type/max, nombre de
    services distincts impliqués, taux de spans en erreur (status != 0).
    Substitut léger à un encodeur de graphe (GAT/HGTN, cf. littérature) —
    capture un signal de dépendance de service sans entraîner de modèle.
    """
    g = traces_df.groupby(id_col)
    agg = pd.DataFrame({
        "trace_span_count": g.size(),
        "trace_duration_mean": g["duration_ms"].mean(),
        "trace_duration_std": g["duration_ms"].std(),
        "trace_duration_max": g["duration_ms"].max(),
        "trace_n_services": g["service"].nunique(),
        "trace_error_rate": g["status"].apply(lambda s: (pd.to_numeric(s, errors="coerce").fillna(0) != 0).mean()),
    })
    agg = agg.fillna(0)
    logger.info(f"Matrice de traces agrégées construite: {agg.shape[0]} runs, {agg.shape[1]} colonnes")
    return agg


def build_multimodal_features(
    metrics_df: pd.DataFrame = None,
    logs_df: pd.DataFrame = None,
    traces_df: pd.DataFrame = None,
    id_col: str = "run_id",
    include_bigrams: bool = True,
) -> pd.DataFrame:
    """
    Fusionne les features par modalité (métriques agrégées, logs
    bag-of-events+bigrammes, traces agrégées) en une seule matrice par
    run_id, alignée sur l'union des run_id couverts par les modalités
    fournies (0 pour un run absent d'une modalité — ex. traces manquantes
    pour Sock Shop). Fusion par concaténation simple, pas d'attention
    croisée: baseline volontairement classique avant d'investir dans un
    modèle de fusion appris.
    """
    parts = []
    if metrics_df is not None and not metrics_df.empty:
        parts.append(build_metrics_agg_matrix(metrics_df, id_col=id_col))
    if logs_df is not None and not logs_df.empty:
        parts.append(build_sequence_features(logs_df, id_col=id_col, include_bigrams=include_bigrams))
    if traces_df is not None and not traces_df.empty:
        parts.append(build_traces_agg_matrix(traces_df, id_col=id_col))

    if not parts:
        raise ValueError("Aucune modalité fournie (metrics_df/logs_df/traces_df tous vides ou None)")

    index = parts[0].index
    for p in parts[1:]:
        index = index.union(p.index)

    aligned = [p.reindex(index, fill_value=0) for p in parts]
    fused = pd.concat(aligned, axis=1)
    logger.info(f"Features multimodales fusionnées: {fused.shape[0]} runs, {fused.shape[1]} colonnes ({len(parts)} modalité(s))")
    return fused
