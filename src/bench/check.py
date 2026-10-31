"""
Contrôle des exécutions PipeFault terminées (article 5, section 6.4), à
lancer après chaque lot : complétude de la télémétrie et vérification
automatique de l'effet de chaque faute. Écrit data/raw/pipefault/check.csv.
Les règles et seuils ci-dessous sont ceux du préenregistrement : ne pas les
modifier après le début de la campagne de test.

Règles d'exclusion :
- effet de la faute non vérifié (critères ci-dessous) ;
- plus de 10 % des pas de 15 s manquants dans une modalité ;
- échec du pipeline étranger à la faute (exécution normale ou d'exécution
  dont le pipeline n'est pas vert).

Usage: python -m src.bench.check
"""

from pathlib import Path
import json
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bench.campaign import LABELS, OUT

STEP = 15
MAX_MISSING = 0.10
LATENCY_RATIO = 1.5      # p90 après / p90 avant
ERROR_DELTA = 0.05       # taux d'erreur après - avant
SERVICE_ALIASES = {"product-catalog": ["product-catalog", "productcatalog"], "image-provider": ["image-provider"]}


def failed_stage(run_dir: Path):
    path = run_dir / "jenkins_stages.json"
    if not path.exists():
        return None
    for stage in json.loads(path.read_text()).get("stages", []):
        if stage.get("status") not in ("SUCCESS", "NOT_EXECUTED"):
            return stage.get("name")
    return None


def missing_fraction(df: pd.DataFrame, col: str, t0: float, t1: float, scale: float) -> float:
    if df.empty:
        return 1.0
    bins = ((df[col] / scale - t0) // STEP).astype(int).unique()
    expected = max(1, int((t1 - t0) // STEP))
    return 1 - min(1.0, len([b for b in bins if 0 <= b < expected]) / expected)


def span_effect(traces: pd.DataFrame, services: list, t_split: float) -> dict:
    """Effet d'exécution sur les spans des services donnés (et de leurs
    appelants directs pour les erreurs) : rapport des p90 et écart des taux
    d'erreur entre avant et après t_split."""
    if traces.empty or t_split is None:
        return {"latency_ratio": None, "error_delta": None}
    t = traces.assign(after=traces["start_time_us"] / 1e6 >= t_split)
    ids = set(t.loc[t["service"].isin(services), "span_id"])
    callers = t[t["span_id"].isin(set(t.loc[t["service"].isin(services), "parent_span_id"].dropna()))]
    scope = pd.concat([t[t["span_id"].isin(ids)], callers])
    scope = scope.assign(err=scope["status"].astype(str).str.upper().isin(["ERROR", "TRUE", "2"]))
    before, after = scope[~scope["after"]], scope[scope["after"]]
    if before.empty or after.empty:
        return {"latency_ratio": None, "error_delta": None}
    p90 = lambda d: d.loc[d["service"].isin(services), "duration_us"].quantile(0.9)
    ratio = p90(after) / p90(before) if p90(before) else None
    return {"latency_ratio": ratio, "error_delta": after["err"].mean() - before["err"].mean()}


def check_run(label: dict) -> dict:
    run_dir = OUT / "runs" / label["run_id"]
    row = {"run_id": label["run_id"], "split": label["split"], "family": label["family"], "fault": label["fault"]}
    t0, t1 = label["t_ref_start"], label["t_end"]
    try:
        metrics = pd.read_csv(run_dir / "metrics.csv")
        traces = pd.read_csv(run_dir / "traces.csv")
        logs = pd.read_csv(run_dir / "logs.csv")
        docker = json.loads((run_dir / "docker_state.json").read_text())
    except FileNotFoundError as e:
        return {**row, "inclus": False, "raison": f"fichier manquant : {Path(e.filename).name}"}

    row["manque_metriques"] = missing_fraction(metrics, "timestamp", t0, t1, 1)
    row["manque_traces"] = missing_fraction(traces, "start_time_us", t0, t1, 1e6)
    row["manque_journaux"] = missing_fraction(logs, "timestamp_ns", t0, t1, 1e9)
    row["etape_en_echec"] = failed_stage(run_dir)
    target = label["fault_service"]
    names = SERVICE_ALIASES.get(target, [target]) if target else []
    effect = span_effect(traces, names, label.get("t_inject"))
    restarts = sum((v.get("restart_count") or 0) + bool(v.get("oom_killed")) for k, v in docker.items()
                   if target and target in k)
    row.update(effect, redemarrages=restarts)

    fam, code = label["family"], label["fault"]
    origin = label.get("origin_stage")
    if fam == "ci":
        if code == "C2":
            ok = any(a > 0 for a in (label.get("test_attempts") or []))
        else:
            ok = row["etape_en_echec"] == origin
    elif fam in ("inter_etapes", "execution"):
        pipeline_green = label.get("jenkins_result") == "SUCCESS"
        visible = ((effect["latency_ratio"] or 0) >= LATENCY_RATIO or (effect["error_delta"] or 0) >= ERROR_DELTA
                   or restarts > 0)
        ok = pipeline_green and visible
    else:
        ok = label.get("jenkins_result") == "SUCCESS"

    worst = max(row["manque_metriques"], row["manque_traces"], row["manque_journaux"])
    reasons = []
    if not ok:
        reasons.append("effet non vérifié" if fam != "normale" else "pipeline non vert")
    if worst > MAX_MISSING:
        reasons.append(f"télémétrie incomplète ({worst:.0%} de pas manquants)")
    return {**row, "effet_verifie": ok, "inclus": not reasons, "raison": "; ".join(reasons)}


def main() -> None:
    labels = [json.loads(l) for l in LABELS.read_text().splitlines() if l.strip()] if LABELS.exists() else []
    if not labels:
        raise SystemExit("Aucune exécution terminée (labels.jsonl vide).")
    df = pd.DataFrame([check_run(l) for l in labels])
    df.to_csv(OUT / "check.csv", index=False)
    print(df.groupby(["split", "fault"]).agg(n=("run_id", "size"), inclus=("inclus", "sum")).to_string())
    bad = df[~df["inclus"]]
    if len(bad):
        print("\nExclues :\n" + bad[["run_id", "fault", "raison"]].to_string(index=False))


if __name__ == "__main__":
    main()
