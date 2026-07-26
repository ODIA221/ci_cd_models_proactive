"""
Corrélation causale: relie une alerte (run_id anormal) à une chaîne de spans
interprétable, via une heuristique de propagation sur l'arbre de spans
(src/causal/span_tree.py) — pas de découverte causale statistique (PC/FCI/
causal-learn), décision prise pour rester cohérent avec l'aversion déjà
documentée du projet pour cette dépendance (cf. rcaeval.py, env 2023 figé,
conflits avec mlflow).

Intuition du modèle de propagation (fautes RCAEval: cpu/mem/disk/delay/loss/
socket injectées sur UN service): le service fautif dégrade son propre span
(status='error' ou durée anormalement élevée), ce qui ralentit en cascade ses
appelants (spans ANCÊTRES également suspects) mais pas nécessairement ce
qu'il appelle lui-même (spans enfants). La cause racine est donc, parmi les
spans suspects, celui situé le PLUS EN PROFONDEUR d'une chaîne suspecte
continue (aucun enfant suspect) — cf. rank_suspect_services.
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.causal.span_tree import build_span_tree

# Un span en erreur est un signal plus fiable qu'un simple écart de latence
# (le seuil z-score peut être franchi par du bruit) — score constant élevé,
# indépendant de la magnitude, pour toujours passer devant un pic de latence
# modéré lors du tri par score.
_ERROR_SCORE = 5.0


def compute_baseline_stats(normal_traces_df: pd.DataFrame) -> pd.DataFrame:
    """
    Moyenne/écart-type de duration_ms par (service, operation), à partir de
    spans connus normaux — jamais les fenêtres anormales (mêmes principes
    train/test que evaluate.py: on ne fitte jamais sur des données déjà en
    anomalie).
    """
    grouped = normal_traces_df.groupby(["service", "operation"])["duration_ms"]
    stats = grouped.agg(["mean", "std"]).rename(
        columns={"mean": "mean_duration_ms", "std": "std_duration_ms"}
    )
    stats["std_duration_ms"] = stats["std_duration_ms"].fillna(0.0)
    return stats


def load_run_traces(source_dir: Path, run_id: str) -> pd.DataFrame:
    """Charge les spans bruts d'un run_id depuis data/interim/<source>/<subset>/traces/."""
    path = source_dir / "traces" / f"{run_id.replace('/', '__')}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Pas de spans bruts pour run_id='{run_id}' sous {path}")
    return pd.read_parquet(path)


def load_baseline_stats(source_dir: Path, force_recompute: bool = False) -> pd.DataFrame:
    """
    Charge (ou calcule et met en cache dans trace_baseline.parquet) les stats
    de latence normale par (service, operation), à partir des runs
    'test_normal' de labels.parquet — seule fenêtre normale pour laquelle des
    spans bruts sont persistés (les fenêtres 'train' ne le sont pas, cf.
    rcaeval.py: coût de parsing).
    """
    cache_path = source_dir / "trace_baseline.parquet"
    if cache_path.exists() and not force_recompute:
        return pd.read_parquet(cache_path)

    labels_df = pd.read_parquet(source_dir / "labels.parquet")
    normal_run_ids = labels_df.loc[labels_df["fault_type"] == "test_normal", "run_id"]

    frames = []
    for run_id in normal_run_ids:
        try:
            frames.append(load_run_traces(source_dir, run_id))
        except FileNotFoundError:
            continue

    if not frames:
        raise FileNotFoundError(f"Aucun span 'test_normal' trouvé sous {source_dir / 'traces'}")

    stats = compute_baseline_stats(pd.concat(frames, ignore_index=True))
    stats.to_parquet(cache_path)
    return stats


def _compute_suspects(
    run_traces: pd.DataFrame, baseline_stats: pd.DataFrame, z_threshold: float
) -> Dict[str, tuple]:
    """
    Détermine les spans suspects d'un run, vectorisé sur tout le DataFrame
    plutôt que span par span: sur un run à ~2*10^5 spans (cas RCAEval
    réels), un .loc[] par span dans une boucle Python coûtait ~70s (mesuré),
    contre <1s avec un merge pandas. Retourne {span_id: (reason, score)},
    limité aux spans effectivement suspects (généralement <1% du run).
    """
    df = run_traces.merge(baseline_stats, how="left", left_on=["service", "operation"], right_index=True)

    status_numeric = pd.to_numeric(df["status"], errors="coerce")
    is_error = status_numeric.fillna(0.0) != 0.0
    non_numeric = status_numeric.isna() & df["status"].notna()
    if non_numeric.any():
        str_status = df.loc[non_numeric, "status"].astype(str).str.strip().str.lower()
        is_error.loc[non_numeric] = ~str_status.isin(["ok", "0", "success", ""])

    std = df["std_duration_ms"].fillna(0.0).clip(lower=1e-6)
    z = (df["duration_ms"] - df["mean_duration_ms"]) / std
    is_latency_suspect = (z > z_threshold).fillna(False)

    suspect_mask = (is_error | is_latency_suspect).to_numpy()
    if not suspect_mask.any():
        return {}

    latency_only = (is_latency_suspect & ~is_error).to_numpy()[suspect_mask]
    span_ids = df["span_id"].astype(str).to_numpy()[suspect_mask]
    z_values = z.to_numpy()[suspect_mask]

    suspects: Dict[str, tuple] = {}
    for span_id, is_latency, z_val in zip(span_ids, latency_only, z_values):
        if is_latency:
            suspects[span_id] = (f"latency_zscore={z_val:.2f}", float(z_val))
        else:
            suspects[span_id] = ("status_error", _ERROR_SCORE)
    return suspects


def rank_suspect_services(
    traces_df: pd.DataFrame,
    run_id: str,
    baseline_stats: pd.DataFrame,
    z_threshold: float = 3.0,
) -> List[dict]:
    """
    Classe les services suspects derrière l'anomalie observée sur `run_id`.

    Chaque span est marqué "suspect" (erreur ou latence hors baseline), puis
    seuls les spans suspects sans ENFANT suspect ("feuilles" de la chaîne
    suspecte, cf. docstring de module) sont retenus comme candidats de cause
    racine — un span suspect avec un enfant suspect n'est qu'un relais de la
    cascade, pas l'origine. Les votes sont agrégés par service (un service
    peut apparaître sur plusieurs feuilles suspectes) et triés par score
    décroissant.

    Retourne une liste ordonnée de dicts:
    [{"service", "span_id", "operation", "reason", "score", "n_suspect_spans"}]
    — liste vide si aucun signal suspect n'a été détecté sur ce run.
    """
    run_traces = traces_df[traces_df["run_id"] == run_id]
    if run_traces.empty:
        return []

    suspects = _compute_suspects(run_traces, baseline_stats, z_threshold)
    if not suspects:
        return []

    tree = build_span_tree(traces_df, run_id)

    leaves = [
        span_id for span_id in suspects
        if not any(child in suspects for child in tree[span_id]["children"])
    ]

    votes: Dict[str, dict] = {}
    for leaf_id in leaves:
        node = tree[leaf_id]
        reason, score = suspects[leaf_id]
        service = node["service"]
        entry = votes.setdefault(service, {
            "service": service,
            "span_id": leaf_id,
            "operation": node["operation"],
            "reason": reason,
            "score": 0.0,
            "n_suspect_spans": 0,
        })
        entry["score"] += score
        entry["n_suspect_spans"] += 1
        if score >= entry.get("_best_score", -1):
            entry["_best_score"] = score
            entry["span_id"] = leaf_id
            entry["operation"] = node["operation"]
            entry["reason"] = reason

    ranked = sorted(votes.values(), key=lambda e: e["score"], reverse=True)
    for entry in ranked:
        entry.pop("_best_score", None)
    return ranked
