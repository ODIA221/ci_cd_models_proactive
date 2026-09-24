"""
Vues d'exploration causale (LogPipeGuard-Multimodal v2) — fonctions pures
JSON (réponse de GET /causal/{run_id}) -> figure Plotly. Aucune dépendance au
modèle: le dashboard reste un client HTTP pur de l'API.

Écarts assumés par rapport à l'article (section 3.2) :
- couleur du score a_i : rampe SÉQUENTIELLE à une teinte (gris clair ->
  rouge critique), pas rouge -> vert — l'opposition rouge/vert est illisible
  pour ~8 % des hommes (deutéranopie) et un score est une magnitude, pas une
  polarité ;
- les glyphes représentent des services, pas des étapes CI/CD (cf.
  src/causal/signals.py) ;
- la vue de corrélation modale montre les contributions par modalité,
  pas des "poids d'attention intermodaux": le détecteur retenu (fusion
  tardive) n'a pas d'attention croisée (cf. docs/06).
"""

from collections import deque
import math
from typing import List, Optional, Set

import pandas as pd
import plotly.graph_objects as go

SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
AXIS = "#898781"
TEXT_SECONDARY = "#52514e"
HIGHLIGHT = "#2a78d6"  # sélection / chemin causal (bleu, distinct de la rampe rouge)
# Rampe séquentielle score d'anomalie: neutre -> critique.
SCORE_SCALE = [[0.0, "#f0efec"], [0.5, "#ec835a"], [1.0, "#d03b3b"]]


def _score_color(score: float) -> str:
    stops = [(0.0, (0xF0, 0xEF, 0xEC)), (0.5, (0xEC, 0x83, 0x5A)), (1.0, (0xD0, 0x3B, 0x3B))]
    score = min(max(score, 0.0), 1.0)
    for (x0, c0), (x1, c1) in zip(stops, stops[1:]):
        if score <= x1:
            t = (score - x0) / (x1 - x0)
            return "#%02x%02x%02x" % tuple(round(a + (b - a) * t) for a, b in zip(c0, c1))
    return "#d03b3b"


def _layout(fig: go.Figure, title: str, height: int) -> go.Figure:
    fig.update_layout(
        title=title, height=height, plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
        margin=dict(l=10, r=10, t=50, b=10), showlegend=False,
        font=dict(color=TEXT_SECONDARY),
    )
    return fig


def filtered_nodes(signals: dict, threshold: float) -> List[dict]:
    return [n for n in signals["nodes"] if n["anomaly_score"] >= threshold]


# --- Vue 1: chronologie ---------------------------------------------------------

def timeline_figure(signals: dict, threshold: float = 0.0, highlight: Optional[Set[str]] = None) -> go.Figure:
    """
    Une ligne par service, glyphe placé à l'instant de première déviation
    (onset, s après le début de la fenêtre sondée). Encodage (Table 1):
    couleur = a_i ; hauteur = entropie des logs H(L_i) ; largeur = rapport de
    durée médiane des spans (sondée / référence) ; bordure pointillée =
    déviation métrique notable. Services sans onset: colonne "aucune
    déviation" à droite.
    """
    highlight = highlight or set()
    nodes = filtered_nodes(signals, threshold)
    probe_s = signals["window"]["probe_seconds"]
    no_onset_x = probe_s * 1.08
    nodes = sorted(nodes, key=lambda n: (n["onset_s"] is None, n["onset_s"] or 0, -n["anomaly_score"]))
    max_entropy = max([(n["logs"] or {}).get("entropy_bits", 0) for n in nodes] + [1e-6])

    fig = go.Figure()
    hover_x, hover_y, hover_text = [], [], []
    for row, n in enumerate(nodes):
        x = n["onset_s"] if n["onset_s"] is not None else no_onset_x
        ratio = (n["traces"] or {}).get("duration_ratio") or 1.0
        half_w = probe_s * 0.012 * min(max(ratio, 0.5), 4.0)
        entropy = (n["logs"] or {}).get("entropy_bits", 0.0)
        half_h = 0.15 + 0.25 * (entropy / max_entropy)
        metric_dev = n["modality_contributions"].get("metrics", 0) >= 0.5
        selected = n["service"] in highlight
        fig.add_shape(
            type="rect", x0=x - half_w, x1=x + half_w, y0=row - half_h, y1=row + half_h,
            fillcolor=_score_color(n["anomaly_score"]),
            line=dict(color=HIGHLIGHT if selected else TEXT_SECONDARY, width=3 if selected else 1.5,
                      dash="dot" if metric_dev else "solid"),
        )
        onset_label = "—" if n["onset_s"] is None else f"{n['onset_s']:.0f} s"
        hover_x.append(x)
        hover_y.append(row)
        hover_text.append(
            f"<b>{n['display_name']}</b><br>a_i={n['anomaly_score']:.2f} "
            f"({n['n_modalities_deviating']} modalité(s) en déviation)<br>"
            f"début: {onset_label}<br>"
            f"entropie logs: {entropy:.2f} bits · durée ×{ratio:.2f}"
        )
    fig.add_trace(go.Scatter(x=hover_x, y=hover_y, mode="markers", marker=dict(size=18, opacity=0),
                             hovertext=hover_text, hoverinfo="text"))
    fig.add_vline(x=0, line=dict(color=AXIS, dash="dash"))
    fig.add_vrect(x0=probe_s * 1.02, x1=probe_s * 1.14, fillcolor=GRID, opacity=0.4, line_width=0,
                  annotation_text="aucune déviation", annotation_position="top")
    fig.update_yaxes(tickvals=list(range(len(nodes))), ticktext=[n["display_name"] for n in nodes],
                     autorange="reversed", gridcolor=GRID)
    fig.update_xaxes(title="secondes après le début de la fenêtre analysée (injection de la faute pour un run anormal)",
                     gridcolor=GRID, linecolor=AXIS, range=[-probe_s * 0.03, probe_s * 1.15])
    return _layout(fig, "Chronologie — première déviation par service", max(300, 36 * len(nodes) + 100))


# --- Vue 2: propagation ---------------------------------------------------------

def _depths(signals: dict) -> dict:
    """Profondeur d'appel depuis les racines (services jamais appelés)."""
    edges = signals["edges"]
    callees = {e["callee"] for e in edges}
    services = {e["caller"] for e in edges} | callees
    children = {}
    for e in edges:
        children.setdefault(e["caller"], set()).add(e["callee"])
    depth = {s: 0 for s in services - callees} or {min(services): 0} if services else {}
    queue = deque(depth)
    while queue:
        s = queue.popleft()
        for c in children.get(s, ()):
            if c not in depth:
                depth[c] = depth[s] + 1
                queue.append(c)
    for s in services:
        depth.setdefault(s, 0)
    return depth


def propagation_figure(signals: dict, threshold: float = 0.0, highlight: Optional[Set[str]] = None,
                       path: Optional[List[str]] = None) -> Optional[go.Figure]:
    """
    Graphe d'appels observé; flèches dans le sens de PROPAGATION D'UN
    SYMPTÔME (appelé -> appelant). Épaisseur = alpha GAT (attention de
    l'appelant vers l'appelé) si disponible, sinon volume d'appels (log).
    Taille/couleur des nœuds = a_i. Disposition en couches par profondeur
    d'appel (pas de simulation physique: lisible et stable d'un rendu à
    l'autre). None si le run n'a pas de traces.
    """
    edges = signals["edges"]
    if not edges:
        return None
    highlight = highlight or set()
    path_edges = set(zip(path, path[1:])) if path else set()
    scores = {n["service"]: n for n in signals["nodes"]}
    keep = {s for s, n in scores.items() if n["anomaly_score"] >= threshold} | highlight | set(path or [])
    depth = _depths(signals)
    layers = {}
    for s in sorted(depth, key=lambda s: -scores.get(s, {}).get("anomaly_score", 0)):
        if s in keep:
            layers.setdefault(depth[s], []).append(s)
    pos = {s: (d, -(i - (len(members) - 1) / 2)) for d, members in layers.items() for i, s in enumerate(members)}

    has_alpha = signals.get("attention") is not None
    fig = go.Figure()
    for e in edges:
        src, dst = e["callee"], e["caller"]  # sens de propagation
        if src not in pos or dst not in pos:
            continue
        alpha = e.get("alpha_caller_to_callee")
        width = 1 + 8 * alpha if has_alpha and alpha is not None else 1 + math.log10(1 + e["n_calls"])
        on_path = (e["callee"], e["caller"]) in path_edges
        fig.add_annotation(
            x=pos[dst][0], y=pos[dst][1], ax=pos[src][0], ay=pos[src][1],
            xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=0.8,
            arrowwidth=width if not on_path else width + 2, standoff=14, startstandoff=14,
            arrowcolor=HIGHLIGHT if on_path else AXIS, opacity=1 if on_path or not path_edges else 0.35,
        )
    xs, ys, sizes, colors, lines, texts, hovers = [], [], [], [], [], [], []
    for s, (x, y) in pos.items():
        n = scores.get(s)
        a = n["anomaly_score"] if n else 0.0
        xs.append(x); ys.append(y)
        sizes.append(14 + 30 * a)
        colors.append(_score_color(a))
        lines.append(HIGHLIGHT if s in highlight or s in (path or []) else TEXT_SECONDARY)
        texts.append(n["display_name"] if n else s)
        incoming = [e for e in edges if e["caller"] == s and e.get("alpha_caller_to_callee") is not None]
        att = "<br>".join(f"α→{e['callee']}: {e['alpha_caller_to_callee']:.2f}" for e in sorted(incoming, key=lambda e: -e["alpha_caller_to_callee"])[:4])
        hovers.append(f"<b>{texts[-1]}</b><br>a_i={a:.2f}" + (f"<br>{att}" if att else ""))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text", text=texts, textposition="bottom center",
                             marker=dict(size=sizes, color=colors, line=dict(color=lines, width=2)),
                             hovertext=hovers, hoverinfo="text"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    title = "Propagation — épaisseur = attention GAT α" if has_alpha else "Propagation — épaisseur = volume d'appels (attention indisponible)"
    return _layout(fig, title, 520)


# --- Vue 3: corrélation modale ---------------------------------------------------

def modal_figure(signals: dict, threshold: float = 0.0) -> go.Figure:
    """Coordonnées parallèles: un trait par service, axes = contribution de
    chaque modalité (0 si modalité absente pour ce service) puis a_i."""
    nodes = filtered_nodes(signals, threshold)
    df = pd.DataFrame([{
        "logs": n["modality_contributions"].get("logs", 0.0),
        "métriques": n["modality_contributions"].get("metrics", 0.0),
        "traces": n["modality_contributions"].get("traces", 0.0),
        "a_i": n["anomaly_score"],
    } for n in nodes])
    if df.empty:
        df = pd.DataFrame(columns=["logs", "métriques", "traces", "a_i"])
    fig = go.Figure(go.Parcoords(
        line=dict(color=df["a_i"], colorscale=SCORE_SCALE, cmin=0, cmax=1, showscale=True,
                  colorbar=dict(title="a_i")),
        dimensions=[dict(label=c, values=df[c], range=[0, 1]) for c in df.columns],
    ))
    return _layout(fig, "Corrélation modale — contribution de chaque modalité par service", 380)


def static_attribution_table(signals: dict) -> pd.DataFrame:
    """Condition B (attribution statique): mêmes signaux, en tableau brut."""
    rows = []
    for n in signals["nodes"]:
        top_log = (n["logs"] or {}).get("top_templates") or [{}]
        rows.append({
            "service": n["display_name"],
            "a_i": round(n["anomaly_score"], 3),
            "c_metriques": round(n["modality_contributions"].get("metrics", 0), 3),
            "c_logs": round(n["modality_contributions"].get("logs", 0), 3),
            "c_traces": round(n["modality_contributions"].get("traces", 0), 3),
            "metrique_principale": (n["metrics"] or {}).get("top_metric"),
            "template_log_principal": (top_log[0].get("template") or "")[:60],
            "beta_log": top_log[0].get("beta"),
            "debut_s": n["onset_s"],
        })
    return pd.DataFrame(rows)


def attention_table(signals: dict) -> pd.DataFrame:
    return pd.DataFrame([
        {"appelant": e["caller"], "appelé": e["callee"], "n_appels": e["n_calls"],
         "α appelant→appelé": e.get("alpha_caller_to_callee"), "α appelé→appelant": e.get("alpha_callee_to_caller")}
        for e in signals["edges"]
    ])
