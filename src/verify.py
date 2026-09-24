"""
Vérification de bout en bout du projet LogPipeGuard (`./run.sh verify`).

Contrôle, dans l'ordre, tout ce qui doit marcher pour démontrer le projet :
environnement, données, modèle GAT, couche de signaux causaux, chaque
endpoint de l'API (sur une instance DÉDIÉE, démarrée sur un port libre —
jamais celle que l'utilisateur a éventuellement lancée sur :8000), build et
typage du frontend, rendu réel de l'interface dans Chrome headless, parcours
complet du dashboard Streamlit (AppTest) et scripts annexes.

Avec --full: relance aussi l'évaluation des classements de cause racine et
compare les chiffres obtenus à ceux publiés dans docs/07.

Ne modifie aucune donnée: les signaux déjà en cache sont réutilisés, et
aucune session d'étude n'est écrite (seul le refus d'un run_id invalide est
testé sur POST /study/sessions).

Usage: python -m src.verify [--full]
"""

from pathlib import Path
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
from urllib.parse import quote

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SUBSET_DIR = REPO_ROOT / "data" / "interim" / "rcaeval" / "RE2"
EXAMPLE_RUN = "RE2-OB/checkoutservice_disk/2__abnormal"
EXAMPLE_RUN_NO_TRACES = "RE2-SS/carts_cpu/3__abnormal"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
# Chiffres publiés dans docs/07-couche-exploration-causale-v2.md (86 runs).
PUBLISHED_P_AT_1 = {"score_multimodal": 0.709302, "precedence_temporelle": 0.662791,
                    "propagation_attention": 0.453488, "propagation_degre": 0.453488}

results = []


class Skip(Exception):
    """Vérification non applicable ici (prérequis optionnel absent)."""


def check(section: str, name: str):
    def decorator(fn):
        def run(*args, **kwargs):
            start = time.time()
            try:
                detail = fn(*args, **kwargs) or ""
                status = "OK"
            except Skip as e:
                status, detail = "IGNORÉ", str(e)
            except AssertionError as e:
                status, detail = "ÉCHEC", str(e) or "assertion fausse"
            except Exception as e:  # noqa: BLE001 — on veut tout rapporter
                status, detail = "ÉCHEC", f"{type(e).__name__}: {e}"
                if os.environ.get("VERIFY_DEBUG"):
                    traceback.print_exc()
            results.append((section, name, status, detail))
            icon = {"OK": "✅", "ÉCHEC": "❌", "IGNORÉ": "⚪"}[status]
            print(f"  {icon} {name}" + (f" — {detail}" if detail else "") + f" ({time.time() - start:.1f}s)", flush=True)
            return status == "OK"
        return run
    return decorator


# --- 1. Environnement ----------------------------------------------------------

@check("Environnement", "Dépendances Python importables")
def check_python_deps():
    import fastapi, streamlit, torch, pandas, plotly, scipy, sklearn  # noqa: F401,E401
    return f"Python {sys.version.split()[0]}, torch {torch.__version__}, streamlit {streamlit.__version__}"


@check("Environnement", "Syntaxe de run.sh")
def check_run_sh():
    out = subprocess.run(["bash", "-n", str(REPO_ROOT / "run.sh")], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr.strip()


@check("Environnement", "Node.js / npm (pour le frontend)")
def check_node():
    npm = shutil.which("npm")
    if not npm:
        raise Skip("npm introuvable (installer Node >= 18) — le frontend ne peut pas être reconstruit")
    version = subprocess.run(["node", "--version"], capture_output=True, text=True).stdout.strip()
    return f"node {version}"


# --- 2. Données et modèles -----------------------------------------------------

@check("Données", "RCAEval RE2 (brut + normalisé)")
def check_rcaeval():
    import pandas as pd
    assert (REPO_ROOT / "data" / "raw" / "rcaeval" / "RE2").exists(), "données brutes absentes: ./run.sh acquire --source rcaeval --subset RE2"
    labels = pd.read_parquet(SUBSET_DIR / "labels.parquet")
    counts = labels["fault_type"].value_counts().to_dict()
    assert counts.get("test_abnormal", 0) > 0, "aucun run test_abnormal"
    return f"{counts}"


@check("Données", "Modèles de détection RCAEval entraînés")
def check_models():
    from src.api import model_registry
    rcaeval = [m for m in model_registry.list_model_triplets() if m.get("source_dir")]
    assert rcaeval, "aucun modèle RCAEval: ./run.sh showcase-rcaeval"
    return f"{len(rcaeval)} modèles RCAEval servables"


@check("Données", "Modèle GAT sauvegardé + attention normalisée")
def check_gat():
    import numpy as np
    from src.models.graph_encoder import load_gat_model
    from src.causal import signals
    model, vocab = load_gat_model("RE2")
    sig = signals.load_or_compute_run_signals(EXAMPLE_RUN)
    assert sig["attention"] is not None, "attention absente sur un run avec traces"
    rows = np.array(sig["attention"]["layer1"]).sum(axis=1)
    assert np.allclose(rows, 1, atol=1e-3), f"lignes d'attention ne somment pas à 1: {rows}"
    return f"{len(vocab)} services dans le vocabulaire"


@check("Données", "Cache des signaux causaux (172 runs de test)")
def check_signals_cache():
    from src.causal import signals
    files = list((SUBSET_DIR / "causal_signals").glob("*.json"))
    current = sum(json.loads(f.read_text()).get("version") == signals.SIGNALS_VERSION for f in files)
    if current < 172:
        return f"{current}/172 à jour — les autres seront calculés au 1er affichage (~1 min chacun)"
    return f"{current}/172 à jour (version {signals.SIGNALS_VERSION})"


# --- 3. Couche de signaux causaux ------------------------------------------------

@check("Signaux causaux", "Signaux d'un run avec traces")
def check_signals_run():
    from src.causal import signals
    sig = signals.load_or_compute_run_signals(EXAMPLE_RUN)
    assert sig["nodes"] and sig["edges"], "nœuds ou arêtes vides"
    for key in ("anomaly_score", "alpha", "beta", "onset"):
        assert key in sig["provenance"], f"provenance '{key}' manquante"
    top = signals.rank_by_anomaly_score(sig)[0]
    return f"{len(sig['nodes'])} services, {len(sig['edges'])} arêtes, rang 1 = {top} (vérité: checkout)"


@check("Signaux causaux", "Run sans traces (Sock Shop) : rien d'extrapolé")
def check_signals_no_traces():
    from src.causal import signals
    sig = signals.load_or_compute_run_signals(EXAMPLE_RUN_NO_TRACES)
    assert not sig["modalities_available"]["traces"], "traces annoncées disponibles"
    assert sig["edges"] == [] and sig["attention"] is None, "graphe/attention inventés sans traces"


@check("Signaux causaux", "« Et si », chemin causal, rapport, JSON-LD")
def check_signals_tools():
    from src.causal import signals
    sig = signals.load_or_compute_run_signals(EXAMPLE_RUN)
    wi = signals.what_if(sig, "checkoutservice")
    assert "checkout" in wi["explained_anomalous"], wi
    assert signals.causal_path(sig, "checkout"), "aucun chemin depuis checkout"
    report = signals.markdown_report(sig, "checkout")
    assert report.startswith("# Rapport") and "Provenance" in report
    ld = signals.to_jsonld(sig)
    assert "@context" in ld and ld["services"], "JSON-LD incomplet"
    return f"couverture de l'hypothèse checkout = {wi['coverage']:.0%}"


@check("Signaux causaux", "Classements de cause racine (dont ablation attention vs 1/degré)")
def check_rankers():
    from src.causal import signals
    sig = signals.load_or_compute_run_signals(EXAMPLE_RUN)
    att = signals.rank_by_propagation(sig, use_attention=True)
    deg = signals.rank_by_propagation(sig, use_attention=False, degree_normalized=True)
    assert set(att) == set(deg), "classements sur des ensembles différents"
    return f"précédence: {signals.rank_by_onset(sig)[0]}, propagation(α): {att[0]}, propagation(1/degré): {deg[0]}"


# --- 4. API (instance dédiée) ------------------------------------------------------

def start_test_api() -> str:
    import uvicorn
    from src.api.main import app
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(60):
        try:
            if requests.get(f"{base}/health", timeout=1).status_code == 200:
                return base
        except requests.exceptions.RequestException:
            time.sleep(0.5)
    raise RuntimeError("l'API de test ne démarre pas")


def api_checks(base: str) -> None:
    import pandas as pd

    @check("API", "GET /health, /models, /sources")
    def basics():
        assert requests.get(f"{base}/health").json()["status"] == "ok"
        models = requests.get(f"{base}/models").json()["models"]
        sources = requests.get(f"{base}/sources").json()["sources"]
        return f"{len(models)} modèles, {len(sources)} sources"

    @check("API", "POST /predict (modèle RCAEval, vrais runs de test)")
    def predict():
        from src.api import model_registry
        meta = next(m for m in model_registry.list_model_triplets()
                    if m.get("source_dir") and not m.get("horizon_seconds") and m["model_type"] == "isolation_forest")
        features = pd.read_parquet(SUBSET_DIR / "features.parquet")
        labels = pd.read_parquet(SUBSET_DIR / "labels.parquet").set_index("run_id")
        test = features[labels.reindex(features.index)["fault_type"].isin(["test_normal", "test_abnormal"])].head(20)
        records = json.loads(test[meta["feature_columns"]].reset_index().to_json(orient="records"))
        r = requests.post(f"{base}/predict", json={"records": records, "model_id": meta["model_id"]}, timeout=120)
        assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:200]}"
        preds = r.json()["predictions"]
        assert len(preds) == 20 and all(p["run_id"] for p in preds)
        return f"{sum(p['anomalie'] for p in preds)}/20 anomalies ({meta['model_id']})"

    @check("API", "POST /predict rejette des colonnes invalides (422 structuré)")
    def predict_422():
        r = requests.post(f"{base}/predict", json={"records": {"colonne_bidon": 1.0}}, timeout=60)
        assert r.status_code == 422 and "missing_columns" in r.json()["detail"], r.text[:200]

    @check("API", "GET /explain/{run_id} (heuristique v1)")
    def explain():
        from src.api import model_registry
        model_id = next(m["model_id"] for m in model_registry.list_model_triplets() if m.get("source_dir"))
        r = requests.get(f"{base}/explain/{EXAMPLE_RUN}", params={"model_id": model_id}, timeout=120)
        assert r.status_code == 200, r.text[:200]
        chain = r.json()["causal_chain"]
        return f"rang 1 = {chain[0]['service'] if chain else '—'}"

    @check("API", "GET /runs")
    def runs():
        data = requests.get(f"{base}/runs").json()
        assert len(data) == 172, f"{len(data)} runs au lieu de 172"
        return f"{sum(r['anormal'] for r in data)} anormaux, {sum(r['signaux_en_cache'] for r in data)} en cache"

    @check("API", "GET /causal/{run_id} + /whatif + /report + /jsonld")
    def causal():
        sig = requests.get(f"{base}/causal/{EXAMPLE_RUN}", timeout=300).json()
        assert sig["nodes"]
        wi = requests.get(f"{base}/causal/{EXAMPLE_RUN}/whatif", params={"hypothesis": "checkout"}).json()
        assert "paths" in wi and "coverage" in wi
        report = requests.get(f"{base}/causal/{EXAMPLE_RUN}/report", params={"hypothesis": "checkout"})
        assert report.status_code == 200 and report.text.startswith("# Rapport")
        ld = requests.get(f"{base}/causal/{EXAMPLE_RUN}/jsonld")
        assert ld.headers["content-type"].startswith("application/ld+json"), ld.headers["content-type"]
        assert requests.get(f"{base}/causal/RE2-OB/inexistant/1__abnormal").status_code == 404

    @check("API", "Étude utilisateur: tâches contrebalancées, aucune écriture")
    def study():
        tasks = requests.get(f"{base}/study/tasks", params={"participant_id": "VERIFY"}).json()
        assert len(tasks) == 15
        per_condition = {c: sum(t["condition"] == c for t in tasks) for c in {t["condition"] for t in tasks}}
        assert sorted(per_condition.values()) == [5, 5, 5], per_condition
        sessions = REPO_ROOT / "experiments" / "user_study" / "sessions.jsonl"
        before = sessions.read_text() if sessions.exists() else None
        r = requests.post(f"{base}/study/sessions", json={
            "participant_id": "VERIFY", "task_index": 0, "run_id": "RE2-OB/inexistant/1__abnormal",
            "condition": "A_manuel", "answer_service": "x", "diagnosis_seconds": 1, "confidence_likert": 3})
        assert r.status_code == 404, f"run_id invalide accepté (HTTP {r.status_code})"
        after = sessions.read_text() if sessions.exists() else None
        assert before == after, "une session a été écrite pendant la vérification"
        return f"5 tâches par condition: {per_condition}"

    @check("API", "Interface web servie sous /ui/")
    def ui_served():
        if not (REPO_ROOT / "frontend" / "dist" / "index.html").exists():
            raise Skip("frontend/dist absent (./run.sh ui-build)")
        html = requests.get(f"{base}/ui/").text
        asset = html.split('src="')[1].split('"')[0]
        assert requests.get(f"{base}{asset}").status_code == 200, f"{asset} introuvable"
        return asset

    for fn in (basics, predict, predict_422, explain, runs, causal, study, ui_served):
        fn()


# --- 5. Frontend -----------------------------------------------------------------

@check("Frontend", "Typage TypeScript strict (tsc)")
def check_typecheck():
    if not shutil.which("npm"):
        raise Skip("npm introuvable")
    if not (REPO_ROOT / "frontend" / "node_modules").exists():
        raise Skip("dépendances absentes (./run.sh ui-build)")
    out = subprocess.run(["npm", "run", "typecheck"], cwd=REPO_ROOT / "frontend", capture_output=True, text=True)
    assert out.returncode == 0, (out.stdout + out.stderr)[-400:]


@check("Frontend", "Build à jour avec les sources")
def check_build_fresh():
    dist = REPO_ROOT / "frontend" / "dist" / "index.html"
    assert dist.exists(), "frontend/dist absent: ./run.sh ui-build"
    sources = [p for p in (REPO_ROOT / "frontend" / "src").rglob("*") if p.is_file()]
    newer = [p.name for p in sources if p.stat().st_mtime > dist.stat().st_mtime]
    assert not newer, f"sources plus récentes que le build: {newer} — ./run.sh ui-build"


def check_browser_render(base: str) -> None:
    @check("Frontend", "Rendu réel dans Chrome headless (React exécuté, données chargées)")
    def render():
        if not Path(CHROME).exists():
            raise Skip("Google Chrome introuvable")
        if not (REPO_ROOT / "frontend" / "dist" / "index.html").exists():
            raise Skip("frontend/dist absent")
        url = f"{base}/ui/?run={quote(EXAMPLE_RUN)}&service=checkout"
        profile = REPO_ROOT / ".run" / "chrome-verify"
        profile.mkdir(parents=True, exist_ok=True)
        # --dump-dom seul capture le DOM dès l'événement "load", AVANT la
        # 2e requête (/whatif) — mesuré: panneau « Et si » absent alors que
        # l'API l'a bien servi. Le temps virtuel laisse les requêtes aboutir,
        # mais Chrome ne se termine plus ensuite (animations infinies: ondes
        # de propagation, simulation de forces): arrêt forcé, DOM déjà écrit.
        proc = subprocess.Popen(
            [CHROME, "--headless=new", "--disable-gpu", f"--user-data-dir={profile}",
             "--virtual-time-budget=10000", "--dump-dom", url],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        try:
            dom, _ = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            dom, _ = proc.communicate()
        for expected in ("exploration causale", "Chronologie", "Propagation", "Corrélation modale", "Et si la cause racine"):
            assert expected in dom, f"'{expected}' absent du DOM rendu ({len(dom)} caractères)"
        assert "<canvas" in dom and "<circle" in dom, "canevas de chronologie ou nœuds du graphe non rendus"
        return f"{dom.count('<circle')} nœuds SVG, chronologie sur canevas"
    render()


# --- 6. Dashboard Streamlit ----------------------------------------------------------

def check_dashboard(base: str) -> None:
    from streamlit.testing.v1 import AppTest

    def app():
        at = AppTest.from_file(str(REPO_ROOT / "src" / "dashboard" / "app.py"), default_timeout=300)
        at.run()
        at.sidebar.text_input[0].set_value(base).run()
        return at

    @check("Dashboard", "Détection RCAEval + chaîne causale + exploration v2")
    def detection():
        at = app()
        box = at.sidebar.selectbox[0]
        choice = next(o for o in box.options if o.startswith("isolation_forest_rcaeval"))
        box.set_value(choice).run()
        at.radio[0].set_value("RCAEval (test, avec chaîne causale)").run()
        next(b for b in at.button if "détection" in b.label).click().run()
        assert not at.exception, [e.value for e in at.exception]
        headers = [h.value for h in at.header]
        assert "Exploration causale (v2)" in headers, headers
        explore = next(s for s in at.selectbox if s.label == "Run à explorer")
        explore.set_value(EXAMPLE_RUN).run()
        next(s for s in at.selectbox if s.label.startswith("Service à examiner")).set_value("checkout").run()
        assert not at.exception, [e.value for e in at.exception]
        assert any("Et si" in s.value for s in at.subheader), "panneau « et si » absent"
        return f"modèle {choice.split(' ')[0]}"

    @check("Dashboard", "Mode étude: tâche affichée, run_id jamais montré")
    def study_mode():
        at = app()
        at.sidebar.toggle[0].set_value(True).run()
        at.text_input(key="study_participant").set_value("VERIFY").run()
        at.button[0].click().run()
        assert not at.exception, [e.value for e in at.exception]
        alert = " ".join(e.value for e in at.error)
        assert "alerte-" in alert, alert
        assert "RE2-" not in alert and "__abnormal" not in alert, f"run_id révélé au participant: {alert}"

    detection()
    study_mode()


# --- 7. Scripts ---------------------------------------------------------------------

@check("Scripts", "Analyse d'étude: refuse de produire des chiffres sans sessions")
def check_study_analysis():
    sessions = REPO_ROOT / "experiments" / "user_study" / "sessions.jsonl"
    out = subprocess.run([sys.executable, "-m", "src.causal.study_analysis"], cwd=REPO_ROOT, capture_output=True, text=True)
    if sessions.exists():
        assert out.returncode == 0, out.stderr[-300:]
        return "sessions réelles présentes: analyse exécutée"
    assert out.returncode != 0 and "Aucune session" in (out.stdout + out.stderr), "chiffres produits sans données"


@check("Scripts", "Réévaluation complète vs chiffres publiés (docs/07)")
def check_full_evaluation():
    import pandas as pd
    before = set((REPO_ROOT / "experiments").glob("causal_rankers_summary_*.csv"))
    out = subprocess.run([sys.executable, "src/models/evaluate_causal.py", "--compare-rankers", "--workers", "2"],
                         cwd=REPO_ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-400:]
    new = sorted(set((REPO_ROOT / "experiments").glob("causal_rankers_summary_*.csv")) - before)
    assert new, "aucun résumé produit"
    summary = pd.read_csv(new[-1])
    summary = summary[summary["sous_ensemble"] == "tous_runs_anormaux"].set_index("ranker")["precision_at_1"]
    gaps = {k: round(summary[k] - v, 3) for k, v in PUBLISHED_P_AT_1.items() if abs(summary[k] - v) > 1e-3}
    assert not gaps, f"écarts avec docs/07: {gaps}"
    return f"P@1 identiques à la documentation ({new[-1].name})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Vérification de bout en bout de LogPipeGuard")
    parser.add_argument("--full", action="store_true", help="Relance aussi l'évaluation causale complète (~10 min, même avec le cache)")
    args = parser.parse_args()

    print("\n[1/7] Environnement")
    check_python_deps(); check_run_sh(); check_node()
    print("\n[2/7] Données et modèles")
    check_rcaeval(); check_models(); check_gat(); check_signals_cache()
    print("\n[3/7] Couche de signaux causaux")
    check_signals_run(); check_signals_no_traces(); check_signals_tools(); check_rankers()
    print("\n[4/7] API (instance de test dédiée)")
    base = start_test_api()
    print(f"  (API de test sur {base})")
    api_checks(base)
    print("\n[5/7] Frontend")
    check_typecheck(); check_build_fresh(); check_browser_render(base)
    print("\n[6/7] Dashboard Streamlit")
    check_dashboard(base)
    print("\n[7/7] Scripts")
    check_study_analysis()
    if args.full:
        check_full_evaluation()
    else:
        results.append(("Scripts", "Réévaluation complète", "IGNORÉ", "ajouter --full"))
        print("  ⚪ Réévaluation complète — ajouter --full")

    n_ok = sum(r[2] == "OK" for r in results)
    n_fail = sum(r[2] == "ÉCHEC" for r in results)
    n_skip = sum(r[2] == "IGNORÉ" for r in results)
    print(f"\n==> {n_ok} OK, {n_fail} échec(s), {n_skip} ignoré(s)")
    for section, name, status, detail in results:
        if status == "ÉCHEC":
            print(f"    ❌ [{section}] {name}: {detail}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
