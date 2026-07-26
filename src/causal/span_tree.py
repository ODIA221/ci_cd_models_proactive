"""
Reconstruction de l'arbre parent/enfant des spans d'un run à partir d'un
DataFrame au schéma TRACES_COLUMNS (schema.py) — brique de base du module de
corrélation causale (src/causal/correlation.py).

Implémentation en dict pur (pas de networkx/graphviz): un run CI/CD compte au
plus quelques milliers de spans, largement dans le budget d'un parcours
Python natif, et ça évite d'ajouter une dépendance de graphe à un projet qui
a déjà écarté causal-learn pour des raisons de conflits (cf. rcaeval.py).
"""

from typing import Dict, List, Optional

import pandas as pd

# Valeurs observées chez différents exporteurs de traces pour "pas de parent"
# (span racine): NaN/None (pandas), chaîne vide, ou l'ID sentinelle "0" utilisé
# par certains exporteurs Jaeger/OTel.
_NO_PARENT_SENTINELS = {None, "", "0", "0000000000000000"}


def build_span_tree(traces_df: pd.DataFrame, run_id: str) -> Dict[str, dict]:
    """
    Construit l'arbre des spans d'un run_id: dict span_id -> noeud, chaque
    noeud portant ses attributs (service, operation, duration_ms, status,
    timestamp, parent_span_id) et la liste de ses enfants (children).

    Un span dont le parent_span_id ne correspond à aucun span présent dans ce
    run (parent hors fenêtre temporelle, ou véritable racine) est traité comme
    une racine locale — cf. root_spans().
    """
    run_spans = traces_df[traces_df["run_id"] == run_id]

    tree: Dict[str, dict] = {}
    for row in run_spans.itertuples(index=False):
        span_id = str(row.span_id)
        tree[span_id] = {
            "span_id": span_id,
            "parent_span_id": None if pd.isna(row.parent_span_id) else str(row.parent_span_id),
            "service": row.service,
            "operation": getattr(row, "operation", None),
            "duration_ms": row.duration_ms,
            "status": row.status,
            "timestamp": getattr(row, "timestamp", None),
            "children": [],
        }

    for span_id, node in tree.items():
        parent_id = node["parent_span_id"]
        if parent_id in tree and parent_id != span_id:
            tree[parent_id]["children"].append(span_id)

    return tree


def root_spans(tree: Dict[str, dict]) -> List[str]:
    """Spans sans parent connu dans cet arbre (racine réelle ou racine locale)."""
    return [
        span_id for span_id, node in tree.items()
        if node["parent_span_id"] in _NO_PARENT_SENTINELS or node["parent_span_id"] not in tree
    ]


def ancestors(tree: Dict[str, dict], span_id: str) -> List[str]:
    """
    Chaîne des ancêtres de `span_id`, du parent immédiat vers la racine
    (span_id lui-même exclu). S'arrête dès qu'un parent_span_id sort de
    l'arbre (racine locale) ou en cas de cycle (garde-fou défensif: les
    traces réelles ne devraient jamais en avoir).
    """
    chain: List[str] = []
    seen = {span_id}
    current = tree.get(span_id, {}).get("parent_span_id")
    while current in tree and current not in seen:
        chain.append(current)
        seen.add(current)
        current = tree[current]["parent_span_id"]
    return chain
