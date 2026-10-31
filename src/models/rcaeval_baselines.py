"""
Méthodes de référence de RCAEval exécutées sur les MÊMES exécutions anormales
de test que ./run.sh evaluate-causal --compare-rankers (86 cas RE2), pour que
les P@1/P@3 de LogPipeGuard aient un point de comparaison publié.

Choix de protocole (à rapporter tel quel) :
- chaque méthode est appelée comme dans RCAEval/main.py : simple_metrics.csv,
  fenêtre de 5 min avant / 5 min après l'instant d'injection RÉEL, mêmes
  prétraitements (latence-90 seulement, inf -> NaN, ffill, 0), même SLI ;
- multi-sources (mmBARO, mmNSigma) : logts/tracets fenêtrés comme dans la
  branche multi-sources de main.py ;
- traces (TraceRCA, MicroRank) : traces.csv sur la même fenêtre de 10 min
  (instant d'injection converti en microsecondes, unité de startTime) ;
- le classement de métriques/opérations est ramené aux services (premier
  rang de chaque service), puis canonical_service() comme nos méthodes.

Le décorateur `rca` de RCAEval renvoie en silence un classement factice
(ordre des colonnes) quand une méthode plante : il est remplacé ici par une
version qui laisse remonter l'erreur, pour que l'échec soit compté comme tel.
RCD et MM-RCD exigent Python 3.8 dans RCAEval, PDiagnose n'existe que sous
forme de script aux chemins codés en dur : non exécutées, et signalées.

Usage: python -m src.models.rcaeval_baselines [--methods baro,nsigma,...]
       [--workers 2] [--timeout 900] [--limit N]
"""

from pathlib import Path
from datetime import datetime
import argparse
import ast
import importlib
import logging
import multiprocessing as mp
from queue import Empty
import sys
import time
import types
import warnings

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.causal.signals import canonical_service, case_dir_for_run

RCAEVAL_ROOT = REPO_ROOT / "RCAEval"
EXPERIMENTS = REPO_ROOT / "experiments"
WINDOW_ROWS = 10 * 60 // 2      # main.py : length=10 min -> 300 lignes de chaque côté
SIDE_ROWS_TS = 10 * 4 // 2      # main.py : logts/tracets -> 20 lignes de chaque côté
TRACE_WINDOW_S = 5 * 60

# nom -> (module, fonction, famille de données)
METHODS = {
    "baro": ("baro", "baro", "metriques"),
    "nsigma": ("__init__", "nsigma", "metriques"),
    "circa": ("circa", "circa", "metriques"),
    "pc_pagerank": ("pc_pagerank", "pc_pagerank", "metriques"),
    "microcause": ("microcause", "microcause", "metriques"),
    "causalrca": ("causalrca", "causalrca", "metriques"),
    "mmbaro": ("baro", "mmbaro", "multi_sources"),
    "mmnsigma": ("baro", "mmnsigma", "multi_sources"),
    "tracerca": ("tracerca", "tracerca", "traces"),
    "microrank": ("microrank", "microrank", "traces"),
}
NOT_RUN = {
    "rcd": "RCAEval n'importe RCD que sous Python 3.8 (is_py38) ; ce dépôt est en 3.12",
    "mmrcd": "idem RCD (Python 3.8 uniquement dans RCAEval)",
    "pdiagnose": "script main() aux chemins codés en dur, pas de fonction appelable",
    "microcause": "essai sur 2 cas RE2-OB : > 300 s par cas (4 cœurs, 8 Go) ; lançable avec --methods microcause",
    "causalrca": "essai sur 2 cas RE2-OB : > 300 s par cas (4 cœurs, 8 Go) ; lançable avec --methods causalrca",
}
DEFAULT_METHODS = ["baro", "nsigma", "mmbaro", "mmnsigma", "circa", "pc_pagerank", "tracerca", "microrank"]


def _strict_rca(func):
    return func


def _patch_sknetwork() -> None:
    """RCAEval appelle PageRank().fit_transform(), retiré de scikit-network
    (0.33 n'expose plus que fit/fit_predict, qui renvoie les mêmes scores).
    Adaptateur d'API seulement, l'algorithme n'est pas modifié."""
    from sknetwork.ranking import PageRank
    if not hasattr(PageRank, "fit_transform"):
        PageRank.fit_transform = lambda self, *a, **k: self.fit_predict(*a, **k)


def _load_method(module_name: str, func_name: str):
    """Charge une méthode RCAEval SANS exécuter RCAEval/e2e/__init__.py (qui
    importe toutes les méthodes et avale les erreurs)."""
    if str(RCAEVAL_ROOT) not in sys.path:
        sys.path.insert(0, str(RCAEVAL_ROOT))
    _patch_sknetwork()
    e2e_dir = RCAEVAL_ROOT / "RCAEval" / "e2e"
    if "RCAEval.e2e" not in sys.modules:
        importlib.import_module("RCAEval")
        stub = types.ModuleType("RCAEval.e2e")
        stub.__path__ = [str(e2e_dir)]
        stub.rca = _strict_rca
        sys.modules["RCAEval.e2e"] = stub
    if module_name == "__init__":
        # nsigma est défini dans __init__.py : on n'exécute que sa définition.
        tree = ast.parse((e2e_dir / "__init__.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == func_name)
        namespace = {"StandardScaler": importlib.import_module("sklearn.preprocessing").StandardScaler,
                     "preprocess": importlib.import_module("RCAEval.io.time_series").preprocess}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(e2e_dir / "__init__.py"), "exec"), namespace)
        return namespace[func_name]
    return getattr(importlib.import_module(f"RCAEval.e2e.{module_name}"), func_name)


def _sli(system: str, service: str, data: pd.DataFrame) -> str:
    # Mêmes règles que main.py pour RE2 (RE2-SS passe par la branche OB).
    if system == "RE2-TT":
        return f"{service}_latency" if f"{service}_latency" in data else "ts-ui-dashboard_latency"
    if f"{service}_latency" in data:
        return f"{service}_latency"
    return "frontend_1" if "frontend_1" in data and "frontend_latency" not in data else "frontend_latency"


def _window(df: pd.DataFrame, inject_time: float, rows: int) -> pd.DataFrame:
    return pd.concat([df[df["time"] < inject_time].tail(rows), df[df["time"] >= inject_time].head(rows)],
                     ignore_index=True)


def build_input(case_dir: Path, system: str, service: str, family: str):
    inject_time = int((case_dir / "inject_time.txt").read_text().strip())
    if family == "traces":
        if not (case_dir / "traces.csv").exists():
            return None, inject_time, None
        spans = pd.read_csv(case_dir / "traces.csv")
        lo, hi = (inject_time - TRACE_WINDOW_S) * 1_000_000, (inject_time + TRACE_WINDOW_S) * 1_000_000
        spans = spans[(spans["startTime"] >= lo) & (spans["startTime"] < hi)].reset_index(drop=True)
        return spans, inject_time * 1_000_000, None

    data = pd.read_csv(case_dir / "simple_metrics.csv")
    data = data.loc[:, ~data.columns.str.endswith("_latency-50")]
    data = data.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    data = _window(data, inject_time, WINDOW_ROWS)
    data = data.rename(columns={c: c.replace("_latency-90", "_latency") for c in data.columns if c.endswith("_latency-90")})
    sli = _sli(system, service, data)
    if family == "metriques":
        return data, inject_time, sli

    def ts(name: str) -> pd.DataFrame:
        path = case_dir / name
        return _window(pd.read_csv(path), inject_time, SIDE_ROWS_TS) if path.exists() else pd.DataFrame()

    # logs/traces/cluster_info sont lus mais inutilisés par mmnsigma : None.
    return {"metric": data, "logs": None, "logts": ts("logts.csv"), "traces": None,
            "tracets_err": ts("tracets_err.csv"), "tracets_lat": ts("tracets_lat.csv"),
            "cluster_info": None}, inject_time, sli


def services_from_ranks(ranks) -> list:
    """'checkoutservice_cpu', 'ts-order-service_latency', 'frontendservice_GetProduct'
    -> services canoniques dans l'ordre du premier rang atteint."""
    out = []
    for r in ranks or []:
        service = canonical_service(str(r).split("_")[0])
        if service and service not in out:
            out.append(service)
    return out


def _worker(method: str, case_dir: str, system: str, service: str, queue) -> None:
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    module_name, func_name, family = METHODS[method]
    try:
        func = _load_method(module_name, func_name)
        data, inject_time, sli = build_input(Path(case_dir), system, service, family)
        if data is None:
            queue.put({"statut": "non_applicable", "erreur": "pas de traces pour ce cas"})
            return
        num_node = (len(data.columns) - 1) if isinstance(data, pd.DataFrame) else None
        out = func(data, inject_time, dataset=system.lower(), anomalies=None, dk_select_useful=False,
                   sli=sli, verbose=False, n_iter=num_node, args=None)
        queue.put({"statut": "ok", "ranks": [str(r) for r in (out.get("ranks") or [])][:200]})
    except Exception as e:  # noqa: BLE001 — toute erreur de la méthode est un résultat à rapporter
        queue.put({"statut": "echec", "erreur": f"{type(e).__name__}: {str(e)[:200]}"})


def run_one(method: str, case_dir: Path, system: str, service: str, timeout: float) -> dict:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(method, str(case_dir), system, service, queue))
    start = time.time()
    proc.start()
    # Lire le résultat AVANT join : un processus fils qui a mis un gros objet
    # dans la file ne se termine pas tant que la file n'est pas vidée. (La
    # version précédente faisait join puis get, et a déclaré « timeout » un
    # cas après 5 s sur la campagne du 2026-10-31.)
    try:
        result = queue.get(timeout=timeout)
    except Empty:
        alive = proc.is_alive()
        proc.terminate()
        proc.join()
        if alive:
            return {"statut": "timeout", "erreur": f"> {timeout:.0f} s", "duree_s": time.time() - start}
        return {"statut": "echec", "erreur": f"processus terminé sans résultat (code {proc.exitcode})",
                "duree_s": time.time() - start}
    proc.join(60)
    if proc.is_alive():
        proc.terminate()
    result["duree_s"] = time.time() - start
    return result


def test_cases() -> pd.DataFrame:
    latest = sorted(EXPERIMENTS.glob("causal_rankers_2*.csv"))[-1]
    cases = pd.read_csv(latest).drop_duplicates("run_id")[["run_id", "systeme", "type_faute", "root_cause_service"]]
    return cases.reset_index(drop=True), latest.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Méthodes de référence RCAEval sur les cas de test RE2")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS),
                        help=f"Parmi : {', '.join(METHODS)}")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=900, help="Secondes par méthode et par cas")
    parser.add_argument("--limit", type=int, default=None, help="N premiers cas seulement (essai)")
    parser.add_argument("--rerun", default=None,
                        help="Fichier causal_baselines_*.csv : ne relance que ses lignes en échec ou timeout, "
                             "et écrit un nouveau fichier complet (lignes abouties reprises telles quelles)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise SystemExit(f"Méthodes inconnues : {unknown}")
    cases, source = test_cases()
    if args.limit:
        cases = cases.head(args.limit)
    log.info(f"{len(cases)} cas (repris de {source}), méthodes : {', '.join(methods)}")

    jobs = [(m, row) for m in methods for row in cases.itertuples(index=False)]
    rows = []
    previous = None
    if args.rerun:
        previous = pd.read_csv(args.rerun)
        retry = previous[previous["statut"].isin(["echec", "timeout"])]
        keys = set(zip(retry["methode"], retry["run_id"]))
        methods = sorted(set(retry["methode"]))
        jobs = [(m, row) for m in methods for row in cases.itertuples(index=False) if (m, row.run_id) in keys]
        rows = previous[~previous.set_index(["methode", "run_id"]).index.isin(list(keys))
                        & previous["run_id"].notna()].to_dict("records")
        log.info(f"Relance de {len(jobs)} cas en échec/timeout de {args.rerun}")

    def record(method, row, res):
        services = services_from_ranks(res.get("ranks"))
        rank = services.index(row.root_cause_service) + 1 if row.root_cause_service in services else None
        rows.append({"run_id": row.run_id, "systeme": row.systeme, "type_faute": row.type_faute,
                     "root_cause_service": row.root_cause_service, "methode": method,
                     "famille": METHODS[method][2], "statut": res["statut"], "erreur": res.get("erreur"),
                     "duree_s": round(res["duree_s"], 1), "rang": rank,
                     "hit_at_1": rank == 1 if res["statut"] == "ok" else None,
                     "hit_at_3": (rank is not None and rank <= 3) if res["statut"] == "ok" else None,
                     "top5": " > ".join(services[:5])})
        log.info(f"[{len(rows)}/{len(jobs)}] {method} {row.run_id}: {res['statut']} rang={rank} ({res['duree_s']:.0f} s)")

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, m, case_dir_for_run(r.run_id), r.systeme,
                               r.run_id.split("/")[1].rsplit("_", 1)[0], args.timeout): (m, r) for m, r in jobs}
        for fut in futures:
            m, r = futures[fut]
            record(m, r, fut.result())

    for name, reason in NOT_RUN.items():
        if name in methods:
            continue
        rows.append({"run_id": None, "methode": name, "statut": "non_executee", "erreur": reason})

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = EXPERIMENTS / f"causal_baselines_{stamp}.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    done = df[df["run_id"].notna()]
    summary = done.groupby("methode").agg(
        n=("run_id", "nunique"), ok=("statut", lambda s: int((s == "ok").sum())),
        p1=("hit_at_1", lambda s: s.dropna().astype(bool).mean()), p3=("hit_at_3", lambda s: s.dropna().astype(bool).mean()),
        duree_mediane_s=("duree_s", "median")).round(2)
    print(f"\nRésultats : {out} (source des cas : {source})\n" + summary.to_string())
    print("\nP@1/P@3 calculés sur les cas où la méthode a abouti ('ok') ; voir causal_summary pour la comparaison appariée.")


if __name__ == "__main__":
    main()
