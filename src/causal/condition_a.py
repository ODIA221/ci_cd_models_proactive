"""
Environnement de la condition A de l'étude utilisateur (docs/08, section 7) :
charge les 60 exécutions du pool d'étude, ANONYMISÉES, dans une pile
d'observabilité standard — VictoriaMetrics (stockage compatible Prometheus,
requêtes PromQL), Loki (journaux), Jaeger (traces) — consultée via Grafana.

Anonymisation : chaque exécution n'est identifiée que par son alias
`alerte-<8 hex>` (le même que celui affiché par le mode étude du dashboard).
Aucun nom de dossier RCAEval, aucun run_id, aucun type de panne n'est
envoyé à la pile, et les dates d'origine sont retirées du texte des journaux
(scrub_dates). Les horodatages sont décalés : chaque exécution occupe son
propre créneau d'une heure à partir de BASE_EPOCH, dans un ordre fixé par
l'alias (donc sans lien avec le service fautif), ce qui isole les
exécutions dans le temps (indispensable pour Jaeger, qui filtre par service
et par période) sans révéler leur date réelle.

Écarts assumés par rapport aux conditions B et C, documentés dans docs/08 :
- métriques ré-échantillonnées toutes les 5 s (les valeurs RCAEval se
  répètent sur plusieurs secondes : interpolation d'un relevé plus lâche) ;
- traces ÉCHANTILLONNÉES : toutes les traces contenant une erreur, plus une
  fraction des autres (10 % par défaut, tirage déterministe par traceID) —
  35 millions de spans pour 60 exécutions ne tiennent pas dans un Jaeger
  local ; c'est aussi la pratique courante en production.
- les colonnes `*-total` de RCAEval sont déjà des taux par seconde (elles
  décroissent : vérifié), exportées en `*_per_second` pour qu'un ingénieur
  ne leur applique pas rate() par réflexe.

Usage:
    python -m src.causal.condition_a --list                 # alias et créneaux, sans rien charger
    python -m src.causal.condition_a --limit 2              # charge 2 exécutions (test)
    python -m src.causal.condition_a                        # charge les 60
"""

from pathlib import Path
import argparse
import functools
import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Iterable, List

import pandas as pd
import requests

from src.causal import signals

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_EPOCH = int(datetime(2026, 1, 5, tzinfo=timezone.utc).timestamp())
SLOT_SECONDS = 3600
METRIC_STEP_SECONDS = 5
VM_URL = "http://localhost:8428"
LOKI_URL = "http://localhost:3100"
OTLP_URL = "http://localhost:4318"
GRAFANA_URL = f"http://localhost:{os.environ.get('CONDITION_A_GRAFANA_PORT', '3300')}"
JAEGER_UI_URL = "http://localhost:16686"


def alias_for(run_id: str) -> str:
    """Même alias que le mode étude du dashboard (causal_page.py)."""
    return "alerte-" + hashlib.sha1(run_id.encode()).hexdigest()[:8]


def study_pool(subset: str = "RE2") -> List[str]:
    """Les exécutions anormales de test avec traces — le pool de /study/tasks."""
    labels = pd.read_parquet(REPO_ROOT / "data" / "interim" / "rcaeval" / subset / "labels.parquet")
    abnormal = labels.loc[labels["fault_type"] == "test_abnormal", "run_id"]
    traces_dir = REPO_ROOT / "data" / "interim" / "rcaeval" / subset / "traces"
    return sorted(r for r in abnormal if (traces_dir / f"{r.replace('/', '__')}.parquet").exists())


def _time_bounds(path: Path) -> tuple:
    """Premier et dernier horodatage de metrics.csv (colonne 'time', en tête)
    en ne lisant que le début et la fin du fichier : relire 60 fichiers de
    jusqu'à 27 Mo prenait 75 s, trop pour un participant à chaque tâche.
    Les lignes sans horodatage (constatées dans un cas) sont ignorées."""
    def first_time(lines):
        for line in lines:
            value = line.split(",", 1)[0].strip()
            if value and value.lower() != "time":
                try:
                    return int(float(value))
                except ValueError:
                    continue
        raise ValueError(f"aucun horodatage dans {path}")

    with open(path, "rb") as f:
        head = f.read(2_097_152).decode(errors="ignore").splitlines()
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 2_097_152))
        tail = f.read().decode(errors="ignore").splitlines()[1:]
    return first_time(head[1:]), first_time(reversed(tail))


@functools.lru_cache(maxsize=4)
def slots(subset: str = "RE2") -> pd.DataFrame:
    """Un créneau d'une heure par exécution, ordonné par ALIAS (et non par
    run_id, dont l'ordre alphabétique regroupe les cas par service fautif)."""
    pool = study_pool(subset)
    rows = sorted(({"run_id": r, "alias": alias_for(r)} for r in pool), key=lambda d: d["alias"])
    for k, row in enumerate(rows):
        case_dir = signals.case_dir_for_run(row["run_id"], subset)
        t_min, t_max = _time_bounds(case_dir / "metrics.csv")
        inject = int((case_dir / "inject_time.txt").read_text().strip())
        row["slot_start"] = BASE_EPOCH + k * SLOT_SECONDS
        row["shift"] = row["slot_start"] - t_min
        row["alert_time"] = inject + row["shift"]
        row["window_end"] = t_max + row["shift"]
    return pd.DataFrame(rows)


def window_for(run_id: str, subset: str = "RE2") -> dict:
    """Créneau d'une exécution, recalculé de façon déterministe (pas de fichier
    de correspondance alias -> run_id à protéger)."""
    table = slots(subset)
    row = table[table["run_id"] == run_id]
    if row.empty:
        raise KeyError(f"{run_id} n'appartient pas au pool d'étude")
    return row.iloc[0].to_dict()


def grafana_link(run_id: str, subset: str = "RE2") -> dict:
    w = window_for(run_id, subset)
    return {
        "alias": w["alias"],
        "alert_time_utc": datetime.fromtimestamp(w["alert_time"], timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "grafana_url": (f"{GRAFANA_URL}/d/condition-a/condition-a?var-run={w['alias']}"
                        f"&from={w['slot_start'] * 1000}&to={w['window_end'] * 1000}"),
        "jaeger_url": (f"{JAEGER_UI_URL}/search?start={w['slot_start'] * 1_000_000}&end={w['window_end'] * 1_000_000}"
                       f"&tags={requests.utils.quote(json.dumps({'run': w['alias']}))}&limit=200"),
    }


def _post(url: str, attempts: int = 5, **kwargs) -> requests.Response:
    """POST avec nouvelles tentatives sur erreur serveur ou réseau : la VM
    Docker (quelques Go de RAM partagés avec d'autres conteneurs) fait
    parfois expirer Loki sous charge (constaté : DeadlineExceeded à la 21e
    exécution). Pour les métriques et les traces, renvoyer un lot est sans
    effet (VictoriaMetrics dédoublonne, Jaeger écrase par ID). Pour Loki, un
    lot renvoyé alors qu'il avait en fait été reçu crée des DOUBLONS
    (constaté : 2 à 7 087 lignes en trop sur 5 exécutions) : --check les
    détecte, ./run.sh study-env-reload-logs les élimine."""
    for attempt in range(1, attempts + 1):
        try:
            r = requests.post(url, timeout=300, **kwargs)
            if r.status_code < 500:
                return r
            reason = f"HTTP {r.status_code}: {r.text[:200]}"
        except requests.exceptions.RequestException as e:
            reason = str(e)
        if attempt == attempts:
            raise RuntimeError(f"Échec après {attempts} tentatives sur {url} : {reason}")
        wait = 2 ** attempt
        logger.warning(f"{url} : {reason} — nouvelle tentative dans {wait} s ({attempt}/{attempts})")
        time.sleep(wait)


# --- Métriques -> VictoriaMetrics -------------------------------------------------

def _metric_name(kind: str) -> str:
    kind = re.sub(r"-total$", "-per-second", kind)
    return "rcaeval_" + re.sub(r"[^a-zA-Z0-9_]", "_", kind)


def load_metrics(case_dir: Path, alias: str, shift: int) -> int:
    df = pd.read_csv(case_dir / "metrics.csv")
    # Lignes sans horodatage (258 dans un cas Online Boutique) : impossibles
    # à placer dans le temps, donc écartées.
    df = df[df["time"].notna()]
    df = df[(df["time"].astype(int) - int(df["time"].min())) % METRIC_STEP_SECONDS == 0]
    ts_ms = ((df["time"].astype(int) + shift) * 1000).astype(str).tolist()
    lines = []
    for col in df.columns:
        if col == "time" or "_" not in col:
            continue
        service, kind = col.split("_", 1)
        name = _metric_name(kind)
        node = service.startswith("gke-")
        labels = f'node="{service}"' if node else f'service="{service}"'
        values = pd.to_numeric(df[col], errors="coerce").tolist()
        for t, v in zip(ts_ms, values):
            if v == v:  # NaN exclus
                lines.append(f'{name}{{{labels},run="{alias}"}} {v} {t}')
    for start in range(0, len(lines), 200_000):
        r = _post(f"{VM_URL}/api/v1/import/prometheus", data="\n".join(lines[start:start + 200_000]))
        r.raise_for_status()
    return len(lines)


# --- Journaux -> Loki ---------------------------------------------------------------

# Dates d'origine présentes DANS les messages : préfixe de toutes les lignes
# Train Ticket (« 2024-01-23 15:40:35.271 info ... ») et dates isolées
# (« 2024/01/15 18:32:02 traces export ... », constaté sur Online Boutique).
# Elles ne désignent pas le service fautif, mais trahissent la date réelle
# de l'enregistrement, que le décalage temporel est censé masquer.
_LEADING_DATETIME = re.compile(r"^\s*\d{4}[-/]\d{2}[-/]\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?\s*")
_INLINE_DATE = re.compile(r"\b\d{4}[-/]\d{2}[-/]\d{2}(?:[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)?\b")


def scrub_dates(message: str) -> str:
    return _INLINE_DATE.sub("<date>", _LEADING_DATETIME.sub("", message))


def _byte_batches(df: pd.DataFrame, max_bytes: int = 2_000_000) -> Iterable[pd.DataFrame]:
    """Lots bornés en OCTETS, pas en lignes : 20 000 lignes Train Ticket
    (traces de pile Java) faisaient 9,7 Mo, au-delà de la limite de Loki."""
    sizes = df["message"].str.len().to_numpy() + 64
    start, acc = 0, 0
    for i, size in enumerate(sizes):
        if acc + size > max_bytes and i > start:
            yield df.iloc[start:i]
            start, acc = i, 0
        acc += size
    if start < len(df):
        yield df.iloc[start:]


def load_logs(case_dir: Path, alias: str, shift: int) -> int:
    cols = [c for c in ("timestamp", "container_name", "message", "level") if c in pd.read_csv(case_dir / "logs.csv", nrows=0).columns]
    df = pd.read_csv(case_dir / "logs.csv", usecols=cols).dropna(subset=["timestamp", "message"])
    df["ts"] = (df["timestamp"].astype("int64") + shift * 1_000_000_000).astype(str)
    unique = pd.Series(df["message"].astype(str).unique())
    df["message"] = df["message"].astype(str).map(dict(zip(unique, unique.map(scrub_dates))))
    df["level"] = df["level"].fillna("inconnu").astype(str).str.lower() if "level" in df else "inconnu"
    df = df.sort_values("ts")
    sent = 0
    for chunk in _byte_batches(df):
        streams = [
            {"stream": {"run": alias, "service": str(svc), "level": str(lvl)},
             "values": [[t, str(m)] for t, m in zip(g["ts"], g["message"])]}
            for (svc, lvl), g in chunk.groupby(["container_name", "level"])
        ]
        r = _post(f"{LOKI_URL}/loki/api/v1/push", json={"streams": streams})
        if r.status_code >= 300:
            raise RuntimeError(f"Loki a refusé le lot ({r.status_code}): {r.text[:300]}")
        sent += len(chunk)
    return sent


# --- Traces -> Jaeger (OTLP/HTTP) -----------------------------------------------------

def _keep_trace(trace_id: str, rate: float) -> bool:
    return int(hashlib.sha1(trace_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < rate


def load_traces(case_dir: Path, alias: str, shift: int, sample_rate: float, batch: int = 5_000) -> tuple:
    df = pd.read_csv(case_dir / "traces.csv",
                     usecols=["traceID", "spanID", "parentSpanID", "serviceName", "operationName", "startTime", "duration", "statusCode"])
    df["is_error"] = pd.to_numeric(df["statusCode"], errors="coerce").fillna(0) != 0
    error_traces = set(df.loc[df["is_error"], "traceID"])
    traces = df["traceID"].unique()
    kept = {t for t in traces if t in error_traces or _keep_trace(str(t), sample_rate)}
    df = df[df["traceID"].isin(kept)]

    def spans(rows: pd.DataFrame) -> Iterable[dict]:
        for r in rows.itertuples(index=False):
            start_ns = (int(r.startTime) + shift * 1_000_000) * 1000
            span = {
                "traceId": str(r.traceID).zfill(32), "spanId": str(r.spanID).zfill(16),
                "name": str(r.operationName), "kind": 2,
                "startTimeUnixNano": str(start_ns), "endTimeUnixNano": str(start_ns + int(r.duration) * 1000),
                "attributes": [{"key": "run", "value": {"stringValue": alias}}],
                "status": {"code": 2 if r.is_error else 0},
            }
            if isinstance(r.parentSpanID, str) and r.parentSpanID:
                span["parentSpanId"] = r.parentSpanID.zfill(16)
            yield span

    for start in range(0, len(df), batch):
        chunk = df.iloc[start:start + batch]
        payload = {"resourceSpans": [
            {"resource": {"attributes": [{"key": "service.name", "value": {"stringValue": str(svc)}}]},
             "scopeSpans": [{"scope": {"name": "rcaeval-replay"}, "spans": list(spans(g))}]}
            for svc, g in chunk.groupby("serviceName")
        ]}
        r = _post(f"{OTLP_URL}/v1/traces", json=payload)
        r.raise_for_status()
    return len(df), len(kept), len(traces)


def check_loaded(table: pd.DataFrame, subset: str = "RE2") -> int:
    """Compare, pour chaque alias, les lignes de journal INTERROGEABLES dans
    Loki aux lignes du fichier source. Loki accepte (HTTP 204) des lignes
    qu'il ne restitue qu'après écriture sur disque et publication de l'index :
    une perte ou un retard seraient sinon invisibles. Retourne le nombre
    d'alias incomplets."""
    incomplete = 0
    for row in table.itertuples(index=False):
        case_dir = signals.case_dir_for_run(row.run_id, subset)
        expected = int(pd.read_csv(case_dir / "logs.csv", usecols=["timestamp", "message"]).dropna().shape[0])
        r = requests.get(f"{LOKI_URL}/loki/api/v1/query", timeout=120, params={
            "query": f'sum(count_over_time({{run="{row.alias}"}}[2h]))',
            "time": (int(row.window_end) + 60) * 1_000_000_000})
        result = r.json()["data"]["result"]
        got = int(float(result[0]["value"][1])) if result else 0
        mark = "ok" if got == expected else ("DOUBLONS" if got > expected else "INCOMPLET")
        incomplete += got != expected
        print(f"{row.alias} : {got}/{expected} lignes de journal interrogeables — {mark}")
    return incomplete


def check_stack() -> None:
    for name, url in (("VictoriaMetrics", f"{VM_URL}/health"), ("Loki", f"{LOKI_URL}/ready"),
                      ("Jaeger", f"{JAEGER_UI_URL}/"), ("Grafana", f"{GRAFANA_URL}/api/health")):
        try:
            ok = requests.get(url, timeout=5).status_code < 500
        except requests.exceptions.RequestException:
            ok = False
        if not ok:
            raise SystemExit(f"{name} ne répond pas ({url}). Lance d'abord : ./run.sh study-env-up")


def main() -> None:
    parser = argparse.ArgumentParser(description="Charge les exécutions anonymisées de la condition A")
    parser.add_argument("--list", action="store_true", help="Affiche alias et créneaux sans rien charger")
    parser.add_argument("--limit", type=int, default=None, help="Ne charge que les N premières exécutions (test)")
    parser.add_argument("--trace-sample", type=float, default=0.10, help="Fraction des traces sans erreur conservées")
    parser.add_argument("--check", action="store_true",
                        help="Vérifie que toutes les lignes de journal chargées sont interrogeables (après quelques minutes)")
    parser.add_argument("--skip", type=int, default=0,
                        help="Saute les N premières exécutions (reprise après interruption ; recharger une exécution ne crée pas de doublon)")
    parser.add_argument("--only", choices=["metrics", "logs", "traces"], default=None,
                        help="Ne (re)charge qu'une source (ex. après réinitialisation d'un seul service)")
    parser.add_argument("--subset", default="RE2")
    args = parser.parse_args()

    table = slots(args.subset)
    if args.list:
        # Volontairement sans run_id : cette liste peut être montrée à l'équipe d'étude.
        for row in table.itertuples(index=False):
            print(f"{row.alias}  alerte à {datetime.fromtimestamp(row.alert_time, timezone.utc):%Y-%m-%d %H:%M:%S} UTC")
        return

    check_stack()
    rows = table.iloc[args.skip:]
    rows = rows.head(args.limit) if args.limit else rows
    if args.check:
        missing = check_loaded(rows, args.subset)
        if missing:
            raise SystemExit(f"{missing} exécution(s) en écart. Lignes manquantes : patienter quelques minutes "
                             "(écriture sur disque de Loki) puis relancer --check. Lignes en trop (doublons) : "
                             "./run.sh study-env-reload-logs")
        print("Toutes les exécutions vérifiées sont complètes.")
        return
    t0 = time.time()
    for i, row in enumerate(rows.itertuples(index=False), start=args.skip + 1):
        case_dir = signals.case_dir_for_run(row.run_id, args.subset)
        n_m = load_metrics(case_dir, row.alias, row.shift) if args.only in (None, "metrics") else 0
        n_l = load_logs(case_dir, row.alias, row.shift) if args.only in (None, "logs") else 0
        n_s, n_t, n_tot = (load_traces(case_dir, row.alias, row.shift, args.trace_sample)
                           if args.only in (None, "traces") else (0, 0, 0))
        logger.info(f"[{i}/{len(table)}] {row.alias} : {n_m} points de métriques, {n_l} lignes de journal, "
                    f"{n_s} spans ({n_t}/{n_tot} traces conservées) — {time.time() - t0:.0f} s écoulées")
    logger.info(f"Terminé. Grafana : {GRAFANA_URL}/d/condition-a/condition-a")
    logger.info("Les journaux deviennent interrogeables quelques minutes après le chargement : "
                "vérifier ensuite avec  ./run.sh study-env-load --check")


if __name__ == "__main__":
    main()
