"""
Couche d'extraction des signaux causaux (LogPipeGuard-Multimodal v2) — calcule,
pour UN run RCAEval, les signaux par service consommés par les vues
d'exploration causale (API GET /causal/{run_id}, dashboard).

Correspondance avec la formalisation de l'article (section 3.1) et écarts
ASSUMÉS — chaque signal porte un champ "provenance" pour que l'interface ne
présente jamais une heuristique comme une sortie de modèle :

- Nœuds v_i : des SERVICES (Online Boutique, Train Ticket, Sock Shop), pas des
  étapes de pipeline CI/CD — RCAEval ne contient aucune étape build/test/deploy.
- a_i ∈ [0,1] : score d'anomalie PAR SERVICE, heuristique (écart robuste
  fenêtre sondée vs fenêtre de référence, par modalité, écrasé dans [0,1]).
  Les détecteurs entraînés (src/models/) ne produisent qu'un score PAR RUN.
- α_ij : poids d'attention réels du GAT de src/models/graph_encoder.py —
  mais ce GAT est un auto-encodeur entraîné à reconstruire des graphes
  NORMAUX, pas un modèle de cause racine : ses poids sont associatifs
  (échelon 1 de Pearl), leur utilité est mesurée par
  src/models/evaluate_causal.py, pas supposée.
- β_i,k : PAS d'attention sur les logs (l'encodeur LSTM de
  log_sequence_encoder.py n'en a pas). Substitut : log2 du rapport des taux
  d'occurrence de chaque template entre fenêtre sondée et référence.
- H(L_i) : entropie de Shannon (bits) de la distribution des templates.

Fenêtres : pour un run '__abnormal', référence = avant inject_time, fenêtre
sondée = après (ce qu'un ingénieur aurait sous la main au moment du
diagnostic). Pour un run '__normal', la fenêtre normale est coupée en deux
(référence = 1re moitié, sondée = 2nde) : permet de voir à quoi ressemblent
les vues quand il n'y a RIEN à expliquer (taux de fausses pistes).
"""

from pathlib import Path
from typing import Dict, List, Optional
import json
import logging
import math
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

SIGNALS_VERSION = 2
BIN_SECONDS = 10
# Plafond des écarts robustes: une série constante en référence (MAD=0,
# fréquent pour memory-failures-total etc.) donnerait sinon un écart infini
# au moindre changement.
SCORE_CAP = 50.0
# "Seuil notable" par modalité (écart pour lequel la contribution écrasée
# vaut 0,5) — choix heuristique, documenté dans docs/07-*.md.
NOTABLE = {"metrics": 3.0, "traces": 3.0, "logs": math.log2(3)}
ONSET_Z = 3.0
_EXCLUDED_METRIC_PREFIXES = ("gke-", "loadgenerator")
_ERROR_PATTERN = re.compile(r"error|exception|fail|timeout|refused|unavailable", re.IGNORECASE)
_SPRING_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}[^\[]*\[[^\]]*\]\s+(\S+)\s+:\s(.*)$")
# Tout jeton contenant un chiffre (UUID, id hexadécimal, nombre, adresse)
# -> <*>, comme le pré-masquage de Drain. La v1 ne masquait que les hex de
# 8+ caractères: les UUID d'Online Boutique ("3c9f-…") laissaient un template
# distinct PAR requête, ce qui gonflait artificiellement β (vu dans la vue
# "templates les plus modifiés", corrigé en SIGNALS_VERSION 2).
_MASK_PATTERN = re.compile(r"[\w.:-]*\d[\w.:-]*")


def canonical_service(name) -> Optional[str]:
    """Clé de jointure inter-modalités: 'frontendservice' (traces) et
    'frontend' (logs/métriques) -> 'frontend' ; 'ts-travel-service' ->
    'ts-travel'."""
    if name is None or (isinstance(name, float) and math.isnan(name)):
        return None
    return re.sub(r"[-_]?service$", "", str(name).strip().lower()) or None


def _canonical_series(series: pd.Series) -> pd.Series:
    """canonical_service vectorisé sur les valeurs UNIQUES (quelques dizaines
    de services pour ~10^6 spans/lignes de logs: un .map(fonction) ligne à
    ligne coûtait ~10-20 s par run, mesuré sur Train Ticket)."""
    return series.map({v: canonical_service(v) for v in series.dropna().unique()})


def case_dir_for_run(run_id: str, subset: str = "RE2") -> Path:
    case_id = run_id.rsplit("__", 1)[0]
    return REPO_ROOT / "data" / "raw" / "rcaeval" / subset / case_id


def _windows(case_dir: Path, run_id: str, t_start: float, t_end: float) -> dict:
    inject = float((case_dir / "inject_time.txt").read_text().strip())
    if run_id.endswith("__abnormal"):
        return {"inject": inject, "base": (t_start, inject), "probe": (inject, t_end), "origin": inject}
    mid = (t_start + inject) / 2
    return {"inject": inject, "base": (t_start, mid), "probe": (mid, inject), "origin": mid}


def _robust_scale(base: np.ndarray) -> tuple:
    med = float(np.median(base)) if base.size else 0.0
    mad = float(np.median(np.abs(base - med))) * 1.4826 if base.size else 0.0
    return med, max(mad, 0.05 * abs(med), 1e-6)


def _deviation(base: np.ndarray, probe: np.ndarray) -> float:
    """Écart robuste |médiane(sondée) - médiane(réf)| / MAD(réf), plafonné."""
    if base.size < 3 or probe.size < 1:
        return 0.0
    med, scale = _robust_scale(base)
    return float(min(abs(np.median(probe) - med) / scale, SCORE_CAP))


def _onset(base: np.ndarray, probe: np.ndarray, probe_bins: np.ndarray, origin: float) -> Optional[float]:
    """Premier bin de la fenêtre sondée dont l'écart robuste dépasse ONSET_Z
    sur 2 bins consécutifs (1 seul bin = trop sensible au bruit). Retourne des
    secondes relatives à l'origine de la fenêtre sondée, ou None."""
    if base.size < 3 or probe.size < 2:
        return None
    med, scale = _robust_scale(base)
    over = np.abs(probe - med) / scale > ONSET_Z
    hits = np.flatnonzero(over[:-1] & over[1:])
    if hits.size == 0:
        return None
    return float(probe_bins[hits[0]] * BIN_SECONDS - origin)


def _binned(ts: pd.Series, values: pd.Series, agg: str) -> pd.Series:
    # Horodatages manquants: présents dans au moins un cas RCAEval
    # (checkoutservice_mem/2) — ignorés plutôt que de faire échouer le run.
    valid = ts.notna() & np.isfinite(ts)
    bins = (ts[valid] // BIN_SECONDS).astype("int64")
    return values[valid].groupby(bins).agg(agg)


def _split_series(series: pd.Series, windows: dict) -> tuple:
    """Série indexée par bin -> (base, probe, probe_bins), bins manquants à 0
    pour les comptages (un template absent d'un bin a bien 0 occurrence)."""
    lo_b, hi_b = (int(windows["base"][0] // BIN_SECONDS), int(windows["base"][1] // BIN_SECONDS))
    # ceil: le bin contenant inject_time mélange avant/après injection — il
    # est exclu de la fenêtre sondée (sinon onsets négatifs artificiels).
    lo_p, hi_p = (int(math.ceil(windows["probe"][0] / BIN_SECONDS)), int(windows["probe"][1] // BIN_SECONDS))
    full = series.reindex(range(lo_b, hi_p + 1))
    base = full.loc[lo_b:hi_b - 1].dropna().to_numpy(dtype=float)
    probe_s = full.loc[lo_p:hi_p].dropna()
    return base, probe_s.to_numpy(dtype=float), probe_s.index.to_numpy()


# --- Métriques ---------------------------------------------------------------

def metric_signals(case_dir: Path, windows: dict) -> Dict[str, dict]:
    path = case_dir / "metrics.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "time" not in df.columns:
        return {}
    ts = df["time"].astype(float)
    out: Dict[str, dict] = {}
    for col in df.columns:
        if col == "time" or "_" not in col or col.startswith(_EXCLUDED_METRIC_PREFIXES):
            continue
        raw_service, kind = col.split("_", 1)
        service = canonical_service(raw_service)
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() < 10:
            continue
        # Compteurs cumulés Prometheus (*-total): seul leur taux a un sens.
        if kind.endswith("-total"):
            values = values.diff()
        base, probe, probe_bins = _split_series(_binned(ts, values, "mean"), windows)
        dev = _deviation(base, probe)
        entry = out.setdefault(service, {"score": 0.0, "top_metric": None, "onset_s": None,
                                         "display_name": raw_service, "n_metrics": 0})
        entry["n_metrics"] += 1
        if dev > entry["score"]:
            entry.update(score=dev, top_metric=kind, onset_s=_onset(base, probe, probe_bins, windows["origin"]))
    return out


# --- Logs --------------------------------------------------------------------

def _entropy_bits(counts: pd.Series) -> float:
    p = counts[counts > 0] / counts.sum() if counts.sum() else counts[:0]
    return float(-(p * np.log2(p)).sum()) if len(p) else 0.0


def log_signals(case_dir: Path, windows: dict, top_k: int = 5) -> Dict[str, dict]:
    path = case_dir / "logs.csv"
    if not path.exists():
        return {}
    cols = pd.read_csv(path, nrows=0).columns
    usecols = [c for c in ("timestamp", "container_name", "message", "level", "log_template") if c in cols]
    df = pd.read_csv(path, usecols=usecols)
    if df.empty or "timestamp" not in df.columns:
        return {}
    df["ts"] = df["timestamp"] / 1e9
    df = df[(df["ts"] >= windows["base"][0]) & (df["ts"] < windows["probe"][1])]
    df["service"] = _canonical_series(df["container_name"])
    # Masquage/regex sur les messages UNIQUES seulement: un logs.csv Train
    # Ticket fait ~160 Mo (~10^6 lignes), mais bien moins de messages
    # distincts — l'appliquer ligne à ligne coûtait ~110 s (mesuré).
    df["message"] = df["message"].astype(str)
    # Troncature AVANT regex: les messages Train Ticket sont quasi tous
    # uniques (~600 octets, traces de pile Java) — la regex sur le texte
    # entier coûtait encore ~110 s; le template n'en garde que 100 caractères.
    unique_msgs = pd.Series(df["message"].unique())
    heads = unique_msgs.str.slice(0, 300).str.lower()
    # Préfixe Spring Boot (Train Ticket: "date heure  niveau pid --- [thread]
    # logger : message") retiré: sinon il consomme à lui seul les 100
    # caractères du template et tous les messages d'un service se confondent.
    spring = heads.str.extract(_SPRING_PREFIX)
    body = (spring[0] + " : " + spring[1]).where(spring[0].notna(), heads)
    masked = dict(zip(unique_msgs, body.str.slice(0, 160).str.replace(_MASK_PATTERN, "<*>", regex=True).str.slice(0, 100)))
    error_like = dict(zip(unique_msgs, heads.str.contains(_ERROR_PATTERN)))
    df["template"] = df["message"].map(masked)
    if "log_template" in df.columns:
        df["template"] = df["log_template"].where(df["log_template"].notna(), df["template"]).astype(str)
    level = df["level"].astype(str).str.lower() if "level" in df.columns else pd.Series("", index=df.index)
    df["is_error"] = level.isin(["error", "warn", "warning", "severe", "fatal"]) | df["message"].map(error_like)
    df["in_probe"] = df["ts"] >= windows["probe"][0]

    t_base = windows["base"][1] - windows["base"][0]
    t_probe = windows["probe"][1] - windows["probe"][0]
    out: Dict[str, dict] = {}
    for service, g in df.groupby("service"):
        counts = g.groupby(["template", "in_probe"]).size().unstack(fill_value=0).reindex(columns=[False, True], fill_value=0)
        counts.columns = ["n_base", "n_probe"]
        counts = counts[(counts["n_base"] + counts["n_probe"]) >= 5]
        if counts.empty:
            continue
        # β substitut: log2 du rapport des taux lissés (+1) — >0 template plus
        # fréquent après, <0 template qui disparaît (service qui se tait).
        counts["beta"] = np.log2(((counts["n_probe"] + 1) / t_probe) / ((counts["n_base"] + 1) / t_base))
        counts = counts.reindex(counts["beta"].abs().sort_values(ascending=False).index)
        top = counts.head(top_k)

        top_template = top.index[0]
        series = _binned(g["ts"], (g["template"] == top_template).astype(float), "sum")
        base, probe, probe_bins = _split_series(series.reindex(range(series.index.min(), series.index.max() + 1), fill_value=0.0), windows)

        probe_g = g[g["in_probe"]]
        out[service] = {
            "display_name": str(g["container_name"].iloc[0]),
            "score": float(min(abs(top["beta"].iloc[0]), SCORE_CAP)),
            "entropy_bits": _entropy_bits(probe_g["template"].value_counts()),
            "error_rate_probe": float(probe_g["is_error"].mean()) if len(probe_g) else 0.0,
            "error_rate_base": float(g.loc[~g["in_probe"], "is_error"].mean()) if (~g["in_probe"]).any() else 0.0,
            "onset_s": _onset(base, probe, probe_bins, windows["origin"]),
            "top_templates": [
                {"template": str(t), "beta": float(r.beta), "n_base": int(r.n_base), "n_probe": int(r.n_probe)}
                for t, r in top.iterrows()
            ],
        }
    return out


# --- Traces + graphe + attention GAT ------------------------------------------

def load_raw_spans(case_dir: Path, windows: dict) -> pd.DataFrame:
    path = case_dir / "traces.csv"
    if not path.exists():
        return pd.DataFrame()
    raw = pd.read_csv(path, usecols=lambda c: c in {"startTime", "serviceName", "operationName", "spanID", "parentSpanID", "duration", "statusCode"})
    if "startTime" not in raw.columns:
        return pd.DataFrame()
    # Mêmes colonnes/unités que graph_encoder.build_and_cache_gat_features
    # (duration brute RCAEval en µs, nommée duration_ms dans tout le
    # pipeline): l'attention n'a de sens que sur des features construites
    # exactement comme à l'entraînement.
    spans = pd.DataFrame({
        "ts": raw["startTime"] / 1e6,
        "service": raw["serviceName"],
        "operation": raw.get("operationName"),
        "span_id": raw["spanID"],
        "parent_span_id": raw["parentSpanID"],
        "duration_ms": raw["duration"],
        "status": raw["statusCode"],
    })
    return spans[(spans["ts"] >= windows["base"][0]) & (spans["ts"] < windows["probe"][1])]


def trace_signals(spans: pd.DataFrame, windows: dict) -> Dict[str, dict]:
    if spans.empty:
        return {}
    spans = spans.assign(canon=_canonical_series(spans["service"]),
                         is_error=(pd.to_numeric(spans["status"], errors="coerce").fillna(0) != 0).astype(float),
                         in_probe=spans["ts"] >= windows["probe"][0])
    out: Dict[str, dict] = {}
    for service, g in spans.groupby("canon"):
        dur_series = _binned(g["ts"], g["duration_ms"].astype(float), "median")
        err_series = _binned(g["ts"], g["is_error"], "sum")
        base_d, probe_d, bins_d = _split_series(dur_series, windows)
        base_e, probe_e, bins_e = _split_series(err_series, windows)
        dev_d, dev_e = _deviation(base_d, probe_d), _deviation(base_e, probe_e)
        probe_g, base_g = g[g["in_probe"]], g[~g["in_probe"]]
        base_median = float(base_g["duration_ms"].median()) if len(base_g) else float("nan")
        out[service] = {
            "display_name": str(g["service"].iloc[0]),
            "score": max(dev_d, dev_e),
            "reason": "latence" if dev_d >= dev_e else "erreurs",
            "n_spans_probe": int(len(probe_g)),
            "duration_ratio": float(probe_g["duration_ms"].median() / base_median) if len(probe_g) and base_median > 0 else None,
            "onset_s": _onset(base_d, probe_d, bins_d, windows["origin"]) if dev_d >= dev_e else _onset(base_e, probe_e, bins_e, windows["origin"]),
            "first_seen_s": float(probe_g["ts"].min() - windows["origin"]) if len(probe_g) else None,
            "last_seen_s": float(probe_g["ts"].max() - windows["origin"]) if len(probe_g) else None,
        }
    return out


def call_graph(spans_probe: pd.DataFrame) -> List[dict]:
    """Arêtes dirigées appelant -> appelé (parent_span -> span), agrégées par
    paire de services canoniques, auto-appels exclus."""
    if spans_probe.empty:
        return []
    span_to_service = spans_probe.drop_duplicates("span_id").set_index("span_id")["service"]
    caller = spans_probe["parent_span_id"].map(span_to_service)
    pairs = pd.DataFrame({"caller": _canonical_series(caller), "callee": _canonical_series(spans_probe["service"])}).dropna()
    pairs = pairs[pairs["caller"] != pairs["callee"]]
    counts = pairs.groupby(["caller", "callee"]).size().reset_index(name="n_calls")
    return [{"caller": r.caller, "callee": r.callee, "n_calls": int(r.n_calls)} for r in counts.itertuples(index=False)]


def gat_attention(spans_probe: pd.DataFrame, subset: str = "RE2") -> Optional[dict]:
    """Returns {"nodes": [canon...], "layer1": [[...]], "layer2": [[...]]} ou
    None si le modèle GAT n'a pas été sauvegardé / pas de spans."""
    if spans_probe.empty:
        return None
    from src.models.graph_encoder import build_service_graph, load_gat_model
    try:
        model, vocab = load_gat_model(subset)
    except FileNotFoundError as e:
        logger.warning(str(e))
        return None
    graph = build_service_graph(spans_probe[["service", "span_id", "parent_span_id", "duration_ms", "status"]], vocab)
    if graph is None:
        return None
    x, adjacency, node_names = graph
    weights = model.attention_weights(x, adjacency)
    return {
        "nodes": [canonical_service(n) for n in node_names],
        "layer1": weights["layer1"].round(4).tolist(),
        "layer2": weights["layer2"].round(4).tolist(),
    }


# --- Assemblage ---------------------------------------------------------------

def _squash(score: float, modality: str) -> float:
    return float(score / (score + NOTABLE[modality])) if score > 0 else 0.0


def compute_run_signals(run_id: str, subset: str = "RE2") -> dict:
    case_dir = case_dir_for_run(run_id, subset)
    if not (case_dir / "inject_time.txt").exists():
        raise FileNotFoundError(f"Cas brut introuvable pour run_id='{run_id}' ({case_dir})")

    times = pd.read_csv(case_dir / "metrics.csv", usecols=["time"])["time"] if (case_dir / "metrics.csv").exists() else pd.Series(dtype=float)
    inject = float((case_dir / "inject_time.txt").read_text().strip())
    t_start = float(times.min()) if len(times) else inject - 600
    t_end = float(times.max()) if len(times) else inject + 600
    windows = _windows(case_dir, run_id, t_start, t_end)

    metrics = metric_signals(case_dir, windows)
    logs = log_signals(case_dir, windows)
    spans = load_raw_spans(case_dir, windows)
    traces = trace_signals(spans, windows)
    spans_probe = spans[spans["ts"] >= windows["probe"][0]] if not spans.empty else spans

    services = sorted(set(metrics) | set(logs) | set(traces))
    nodes = []
    for s in services:
        per_modality = {
            "metrics": metrics.get(s), "logs": logs.get(s), "traces": traces.get(s),
        }
        contributions = {m: _squash(v["score"], m) for m, v in per_modality.items() if v is not None}
        onsets = [v["onset_s"] for v in per_modality.values() if v is not None and v.get("onset_s") is not None]
        display = next((v["display_name"] for v in (traces.get(s), logs.get(s), metrics.get(s)) if v), s)
        nodes.append({
            "service": s,
            "display_name": display,
            "anomaly_score": max(contributions.values()) if contributions else 0.0,
            "modality_contributions": contributions,
            "n_modalities_deviating": sum(c >= 0.5 for c in contributions.values()),
            "onset_s": min(onsets) if onsets else None,
            "metrics": metrics.get(s),
            "logs": logs.get(s),
            "traces": traces.get(s),
        })

    attention = gat_attention(spans_probe, subset)
    edges = call_graph(spans_probe)
    if attention is not None:
        idx = {n: i for i, n in enumerate(attention["nodes"])}
        a1 = attention["layer1"]
        for e in edges:
            i, j = idx.get(e["caller"]), idx.get(e["callee"])
            # alpha[i][j] = attention que l'appelant i accorde à l'appelé j:
            # c'est dans ce sens que se propage un symptôme (appelé dégradé ->
            # appelant ralenti), cf. docstring de src/causal/correlation.py.
            e["alpha_caller_to_callee"] = a1[i][j] if i is not None and j is not None else None
            e["alpha_callee_to_caller"] = a1[j][i] if i is not None and j is not None else None

    return {
        "version": SIGNALS_VERSION,
        "run_id": run_id,
        "subset": subset,
        "window": {"kind": "post_injection" if run_id.endswith("__abnormal") else "normal_second_half",
                   "probe_seconds": windows["probe"][1] - windows["probe"][0],
                   "base_seconds": windows["base"][1] - windows["base"][0]},
        "modalities_available": {"metrics": bool(metrics), "logs": bool(logs), "traces": bool(traces)},
        "provenance": {
            "anomaly_score": "heuristique: max par modalité de s/(s+seuil), s = écart robuste médiane/MAD fenêtre sondée vs référence",
            "alpha": "poids d'attention GAT couche 1 (moyenne des têtes), auto-encodeur entraîné sur graphes normaux — associatif, non causal" if attention else "indisponible (pas de traces ou modèle GAT non sauvegardé)",
            "beta": "substitut: log2 du rapport des taux d'occurrence des templates (pas d'attention sur les logs dans le modèle)",
            "onset": f"premier couple de bins de {BIN_SECONDS}s consécutifs avec écart robuste > {ONSET_Z}",
        },
        "nodes": sorted(nodes, key=node_sort_key, reverse=True),
        "edges": edges,
        "attention": attention,
    }


def signals_cache_path(run_id: str, subset: str = "RE2") -> Path:
    return REPO_ROOT / "data" / "interim" / "rcaeval" / subset / "causal_signals" / f"{run_id.replace('/', '__')}.json"


def load_or_compute_run_signals(run_id: str, subset: str = "RE2", force: bool = False) -> dict:
    """Calcul coûteux (relit logs/métriques/traces bruts, ~5-20 s): mis en
    cache JSON par run, invalidé si SIGNALS_VERSION change."""
    path = signals_cache_path(run_id, subset)
    if path.exists() and not force:
        cached = json.loads(path.read_text())
        if cached.get("version") == SIGNALS_VERSION:
            cached["nodes"].sort(key=node_sort_key, reverse=True)
            return cached
    signals = compute_run_signals(run_id, subset)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(signals, ensure_ascii=False))
    return signals


# --- Classements de cause racine (comparés dans evaluate_causal.py) ------------

def node_sort_key(node: dict) -> tuple:
    """a_i sature au plafond (SCORE_CAP) sur la majorité des runs anormaux:
    mesuré, 48/86 runs RE2 ont plusieurs services ex aequo au sommet. Sans
    départage explicite, l'ordre (alphabétique) des services décidait du
    rang 1. Départage: nombre de modalités en déviation (cohérence
    intermodale), puis somme des contributions."""
    return (node["anomaly_score"], node["n_modalities_deviating"], sum(node["modality_contributions"].values()))


def rank_by_anomaly_score(signals: dict) -> List[str]:
    return [n["service"] for n in sorted(signals["nodes"], key=node_sort_key, reverse=True)]


def rank_by_onset(signals: dict, min_score: float = 0.5) -> List[str]:
    """Précédence temporelle: parmi les services notablement anormaux, le
    premier à dévier est supposé être la cause (hypothèse classique en RCA,
    fausse dès que la fenêtre de 10 s masque l'ordre réel)."""
    anomalous = [n for n in signals["nodes"] if n["anomaly_score"] >= min_score and n["onset_s"] is not None]
    first = [n["service"] for n in sorted(anomalous, key=lambda n: (n["onset_s"], tuple(-v for v in node_sort_key(n))))]
    return first + [s for s in rank_by_anomaly_score(signals) if s not in first]


def rank_by_propagation(signals: dict, use_attention: bool = True, damping: float = 0.85, n_iter: int = 100,
                        degree_normalized: bool = False) -> List[str]:
    """
    PageRank personnalisé inspiré de MicroRCA sur le graphe d'appels: le
    marcheur part des services anormaux (personnalisation = a_i) et se
    déplace d'un appelant vers un appelé proportionnellement à l'anomalie de
    l'appelé — il "descend" vers la dépendance la plus dégradée, où la masse
    s'accumule. Boucle propre pondérée par a_i (un service anormal sans
    appelé anormal retient le marcheur).

    Poids de base d'une arête = alpha GAT (use_attention) ou 1 (ablation):
    la comparaison des deux, à ranker identique, est ce qui mesure l'apport
    réel de l'attention. degree_normalized (sans attention): poids
    1/(1+nb voisins de l'appelant) — exactement ce que vaudrait une
    attention UNIFORME du GAT (softmax sur voisins + boucle propre); sert à
    distinguer un apport appris d'un simple effet de normalisation. Services absents du graphe (pas de spans): ajoutés
    ensuite par a_i décroissant.
    """
    edges = signals["edges"]
    scores = {n["service"]: n["anomaly_score"] for n in signals["nodes"]}
    graph_nodes = sorted({e["caller"] for e in edges} | {e["callee"] for e in edges})
    if not graph_nodes or (use_attention and signals.get("attention") is None):
        return rank_by_anomaly_score(signals)

    idx = {n: i for i, n in enumerate(graph_nodes)}
    n = len(graph_nodes)
    a = np.array([scores.get(s, 0.0) for s in graph_nodes]) + 1e-6
    w = np.diag(a)
    neighbors: Dict[str, set] = {}
    for e in edges:
        neighbors.setdefault(e["caller"], set()).add(e["callee"])
        neighbors.setdefault(e["callee"], set()).add(e["caller"])
    for e in edges:
        if use_attention:
            base = e.get("alpha_caller_to_callee")
        elif degree_normalized:
            base = 1.0 / (1 + len(neighbors[e["caller"]]))
        else:
            base = 1.0
        w[idx[e["caller"]], idx[e["callee"]]] += (base or 0.0) * a[idx[e["callee"]]]
    transition = w / w.sum(axis=1, keepdims=True)

    personalization = a / a.sum()
    rank = np.full(n, 1.0 / n)
    for _ in range(n_iter):
        rank = damping * rank @ transition + (1 - damping) * personalization
    final = {s: rank[idx[s]] for s in graph_nodes}
    ordered = sorted(final, key=final.get, reverse=True)
    return ordered + [s for s in rank_by_anomaly_score(signals) if s not in final]


# --- Exploration "et si" -------------------------------------------------------

def what_if(signals: dict, hypothesis: str, min_score: float = 0.5) -> dict:
    """
    Test d'hypothèse STRUCTUREL (pas contrefactuel): si `hypothesis` est la
    cause racine, les symptômes qu'elle peut expliquer sont ceux de ses
    appelants transitifs (propagation appelé -> appelant). Retourne les
    services anormaux expliqués / non expliqués par cette hypothèse, et la
    part des services en déviation couverte. Une hypothèse qui laisse des
    services anormaux inexpliqués est affaiblie — pas réfutée (panne
    multiple, dépendance non tracée, bruit).
    """
    hypothesis = canonical_service(hypothesis)
    callers_of: Dict[str, set] = {}
    for e in signals["edges"]:
        callers_of.setdefault(e["callee"], set()).add(e["caller"])

    explained, frontier = {hypothesis}, [hypothesis]
    while frontier:
        for caller in callers_of.get(frontier.pop(), ()):
            if caller not in explained:
                explained.add(caller)
                frontier.append(caller)

    anomalous = {n["service"] for n in signals["nodes"] if n["anomaly_score"] >= min_score}
    hyp_node = next((n for n in signals["nodes"] if n["service"] == hypothesis), None)
    return {
        "hypothesis": hypothesis,
        "hypothesis_anomaly_score": hyp_node["anomaly_score"] if hyp_node else None,
        "propagation_path": sorted(explained - {hypothesis}),
        "explained_anomalous": sorted(anomalous & explained),
        "unexplained_anomalous": sorted(anomalous - explained),
        "coverage": len(anomalous & explained) / len(anomalous) if anomalous else None,
        "min_score": min_score,
        "caveat": "Propagation déduite du graphe d'appels observé (traces). Sans traces, seule l'hypothèse elle-même est 'expliquée'.",
    }


def causal_path(signals: dict, service: str) -> List[List[str]]:
    """Chemins de propagation depuis `service` (supposé cause) vers les
    racines du graphe d'appels (services jamais appelés), via ses appelants."""
    service = canonical_service(service)
    callers_of: Dict[str, set] = {}
    for e in signals["edges"]:
        callers_of.setdefault(e["callee"], set()).add(e["caller"])
    paths, stack = [], [[service]]
    while stack and len(paths) < 20:
        path = stack.pop()
        nxt = [c for c in callers_of.get(path[-1], ()) if c not in path]
        if not nxt:
            if len(path) > 1:
                paths.append(path)
            continue
        stack.extend(path + [c] for c in nxt)
    return paths


def markdown_report(signals: dict, hypothesis: Optional[str] = None, top_n: int = 5) -> str:
    """Rapport exportable (étape 4 du flux d'usage, section 4.3)."""
    lines = [
        f"# Rapport de diagnostic — `{signals['run_id']}`",
        "",
        f"- Fenêtre analysée : {signals['window']['kind']} ({signals['window']['probe_seconds']:.0f} s, référence {signals['window']['base_seconds']:.0f} s)",
        f"- Modalités disponibles : " + ", ".join(m for m, ok in signals["modalities_available"].items() if ok),
        "",
        "## Services les plus anormaux",
        "",
        "| Rang | Service | a_i | Modalités en déviation | Début (s) | Indice principal |",
        "|---|---|---|---|---|---|",
    ]
    for rank, n in enumerate(signals["nodes"][:top_n], start=1):
        hints = []
        if n["metrics"]:
            hints.append(f"métrique {n['metrics']['top_metric']}")
        if n["traces"]:
            hints.append(f"traces ({n['traces']['reason']})")
        if n["logs"] and n["logs"]["top_templates"]:
            hints.append(f"log « {n['logs']['top_templates'][0]['template'][:50]} »")
        onset = f"{n['onset_s']:.0f}" if n["onset_s"] is not None else "—"
        lines.append(f"| {rank} | {n['display_name']} | {n['anomaly_score']:.2f} | {n['n_modalities_deviating']} | {onset} | {'; '.join(hints)} |")

    if hypothesis:
        wi = what_if(signals, hypothesis)
        lines += ["", f"## Hypothèse retenue : {wi['hypothesis']}", "",
                  f"- Services anormaux expliqués par propagation : {', '.join(wi['explained_anomalous']) or '—'}",
                  f"- Services anormaux NON expliqués : {', '.join(wi['unexplained_anomalous']) or '—'}",
                  f"- Couverture : {wi['coverage']:.0%}" if wi["coverage"] is not None else "- Couverture : n/a"]

    lines += ["", "## Provenance des signaux", ""] + [f"- **{k}** : {v}" for k, v in signals["provenance"].items()]
    lines += ["", "_Les poids d'attention et scores ci-dessus sont des indices associatifs, pas une preuve causale._"]
    return "\n".join(lines)


# --- Export JSON-LD --------------------------------------------------------------

# Vocabulaire propre au projet, en URN (non résolvable) plutôt qu'une URL
# inventée: il n'existe pas d'ontologie publique établie pour ces signaux.
# service.name suit la convention sémantique OpenTelemetry.
JSONLD_CONTEXT = {
    "@vocab": "urn:logpipeguard:vocab#",
    "schema": "https://schema.org/",
    "name": "schema:name",
    "serviceName": "https://opentelemetry.io/docs/specs/semconv/resource/#service",
    "caller": {"@type": "@id"},
    "callee": {"@type": "@id"},
}


def to_jsonld(signals: dict) -> dict:
    """
    Sérialisation JSON-LD des signaux d'un run (section 4.2 de l'article).
    Interopérabilité SÉMANTIQUE seulement: ni Grafana ni Jaeger ne consomment
    nativement du JSON-LD — une intégration Grafana passerait par une source
    de données JSON générique lisant l'endpoint REST, non testée ici.
    """
    run_iri = f"urn:logpipeguard:run:{signals['subset']}:{signals['run_id']}"

    def service_iri(s: str) -> str:
        return f"{run_iri}#service={s}"

    return {
        "@context": JSONLD_CONTEXT,
        "@id": run_iri,
        "@type": "DiagnosticRun",
        "name": signals["run_id"],
        "window": signals["window"],
        "modalitiesAvailable": signals["modalities_available"],
        "provenance": signals["provenance"],
        "services": [
            {
                "@id": service_iri(n["service"]),
                "@type": "Service",
                "serviceName": n["display_name"],
                "anomalyScore": round(n["anomaly_score"], 4),
                "modalityContributions": {k: round(v, 4) for k, v in n["modality_contributions"].items()},
                "onsetSeconds": n["onset_s"],
                "topMetric": (n["metrics"] or {}).get("top_metric"),
                "logEntropyBits": (n["logs"] or {}).get("entropy_bits"),
            }
            for n in signals["nodes"]
        ],
        "calls": [
            {
                "@type": "Call",
                "caller": service_iri(e["caller"]),
                "callee": service_iri(e["callee"]),
                "callCount": e["n_calls"],
                "gatAttentionCallerToCallee": e.get("alpha_caller_to_callee"),
            }
            for e in signals["edges"]
        ],
    }
