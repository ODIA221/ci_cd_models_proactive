"""
API FastAPI de service pour LogPipeGuard

Prototype de recherche: aucune authentification, usage local uniquement.
Ne remplace pas l'entraînement (qui reste une commande CLI: `./run.sh demo`,
plusieurs minutes) — cette API ne fait que du chargement/inférence sur un
modèle déjà entraîné.
"""

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from src.api import model_registry
from src.api.schemas import (
    ExplainResponse,
    HealthResponse,
    StudySessionRecord,
    StudyTask,
    ModelsListResponse,
    PredictRequest,
    PredictResponse,
    SourcesResponse,
)
from src.causal import correlation, signals
from src.data.sources.registry import list_sources

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

app = FastAPI(
    title="LogPipeGuard API",
    description="API de détection d'anomalies CI/CD (prototype de recherche — sans authentification, usage local)",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/models", response_model=ModelsListResponse)
def models() -> ModelsListResponse:
    return ModelsListResponse(models=model_registry.list_model_triplets())


@app.get("/sources", response_model=SourcesResponse)
def sources() -> SourcesResponse:
    return SourcesResponse(sources=list_sources())


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        model_id = model_registry.resolve_model_id(request.model_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    detector, preprocessor, meta = model_registry.load_triplet(model_id)

    records = request.records_as_list()
    df = pd.DataFrame(records)
    run_ids = df["run_id"].tolist() if "run_id" in df.columns else [None] * len(df)
    df = df.drop(columns=["timestamp", "run_id"], errors="ignore")

    expected = set(meta["feature_columns"])
    got = set(df.columns)
    missing = sorted(expected - got)
    unexpected = sorted(got - expected)
    if missing or unexpected:
        raise HTTPException(
            status_code=422,
            detail={"missing_columns": missing, "unexpected_columns": unexpected},
        )

    df = df[meta["feature_columns"]]

    try:
        X_processed = preprocessor.transform(df)
        predictions = detector.predict(X_processed.values)
        scores = detector.anomaly_score(X_processed.values)
    except Exception as e:
        logger.exception("Échec de la prédiction")
        raise HTTPException(status_code=422, detail=str(e))

    return PredictResponse(
        model_id=model_id,
        predictions=[
            {"prediction": int(p), "anomaly_score": float(s), "anomalie": bool(p == -1), "run_id": r}
            for p, s, r in zip(predictions, scores, run_ids)
        ],
    )


@app.get("/explain/{run_id:path}", response_model=ExplainResponse)
def explain(run_id: str, model_id: Optional[str] = None) -> ExplainResponse:
    """
    Relie un run_id anormal à une chaîne causale de spans suspects (cf.
    src/causal/correlation.py). Seuls les modèles entraînés via
    src/models/train_rcaeval.py exposent un 'source_dir' en méta — le modèle
    démo (train.py) n'a ni run_id ni traces, donc rien à expliquer.
    """
    try:
        resolved_model_id = model_registry.resolve_model_id(model_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    meta = model_registry.load_meta(resolved_model_id)
    source_dir_rel = meta.get("source_dir")
    if not source_dir_rel:
        raise HTTPException(
            status_code=400,
            detail=f"Corrélation causale non disponible pour le modèle '{resolved_model_id}' (aucune trace associée)",
        )

    source_dir = REPO_ROOT / source_dir_rel
    try:
        traces = correlation.load_run_traces(source_dir, run_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Pas de spans de trace pour run_id='{run_id}'")

    baseline_stats = correlation.load_baseline_stats(source_dir)
    causal_chain = correlation.rank_suspect_services(traces, run_id, baseline_stats)

    return ExplainResponse(run_id=run_id, model_id=resolved_model_id, causal_chain=causal_chain)


# --- Exploration causale (v2) -------------------------------------------------
# Signaux par service pour les vues chronologie / propagation / corrélation
# modale du dashboard. Indépendants du détecteur (qui ne score que le run
# entier): calculés depuis les données brutes RCAEval du run, cf.
# src/causal/signals.py pour la provenance exacte de chaque signal.

def _load_signals(run_id: str, subset: str) -> dict:
    try:
        return signals.load_or_compute_run_signals(run_id, subset)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/runs")
def runs(subset: str = "RE2") -> list:
    """Runs de test explorables (normaux ET anormaux: un run normal montre à
    quoi ressemblent les vues quand il n'y a rien à expliquer). Le service
    fautif n'est PAS renvoyé — mais il figure dans le run_id RCAEval."""
    labels = pd.read_parquet(REPO_ROOT / "data" / "interim" / "rcaeval" / subset / "labels.parquet")
    test = labels[labels["fault_type"].isin(["test_normal", "test_abnormal"])]
    cache_dir = signals.signals_cache_path("x", subset).parent
    return [
        {"run_id": r.run_id, "anormal": r.fault_type == "test_abnormal",
         "signaux_en_cache": (cache_dir / f"{r.run_id.replace('/', '__')}.json").exists()}
        for r in test.itertuples(index=False)
    ]


@app.get("/causal/{run_id:path}/jsonld")
def causal_jsonld(run_id: str, subset: str = "RE2") -> JSONResponse:
    return JSONResponse(signals.to_jsonld(_load_signals(run_id, subset)), media_type="application/ld+json")


@app.get("/causal/{run_id:path}/whatif")
def causal_what_if(run_id: str, hypothesis: str, subset: str = "RE2", min_score: float = 0.5) -> dict:
    signals_data = _load_signals(run_id, subset)
    result = signals.what_if(signals_data, hypothesis, min_score=min_score)
    result["paths"] = signals.causal_path(signals_data, hypothesis)
    return result


@app.get("/causal/{run_id:path}/report", response_class=PlainTextResponse)
def causal_report(run_id: str, subset: str = "RE2", hypothesis: Optional[str] = None) -> str:
    return signals.markdown_report(_load_signals(run_id, subset), hypothesis=hypothesis)


@app.get("/causal/{run_id:path}")
def causal(run_id: str, subset: str = "RE2") -> dict:
    """Premier appel sur un run: 5-30 s (relecture des fichiers bruts), puis cache JSON."""
    return _load_signals(run_id, subset)


# --- Instrumentation de l'étude utilisateur -------------------------------------
# N'IMPLÉMENTE PAS l'étude (recrutement, consentement, comité d'éthique): se
# contente d'enregistrer fidèlement ce que font de vrais participants. La
# vérité terrain ne quitte jamais le serveur (correction faite ici).

STUDY_DIR = REPO_ROOT / "experiments" / "user_study"


def _study_labels(subset: str) -> pd.DataFrame:
    labels = pd.read_parquet(REPO_ROOT / "data" / "interim" / "rcaeval" / subset / "labels.parquet")
    return labels[labels["fault_type"] == "test_abnormal"].set_index("run_id")


@app.get("/study/tasks", response_model=list[StudyTask])
def study_tasks(participant_id: str, subset: str = "RE2", n: int = 15) -> list:
    """Tirage déterministe par participant (graine = participant_id) de runs
    anormaux AVEC traces (sinon la condition C n'a pas de graphe à montrer)."""
    labels = _study_labels(subset)
    traces_dir = REPO_ROOT / "data" / "interim" / "rcaeval" / subset / "traces"
    candidates = sorted(r for r in labels.index if (traces_dir / f"{r.replace('/', '__')}.parquet").exists())
    rng = random.Random(participant_id)
    chosen = rng.sample(candidates, min(n, len(candidates)))
    # Ordre des conditions contrebalancé par rotation (carré latin 3x3).
    conditions = ["A_manuel", "B_attribution_statique", "C_exploration_causale"]
    offset = int.from_bytes(participant_id.encode(), "little") % 3
    return [
        StudyTask(task_index=i, run_id=r, condition=conditions[(i + offset) % 3])
        for i, r in enumerate(chosen)
    ]


@app.post("/study/sessions")
def study_record(record: StudySessionRecord) -> dict:
    labels = _study_labels(record.subset)
    if record.run_id not in labels.index:
        raise HTTPException(status_code=404, detail=f"run_id inconnu ou non anormal: '{record.run_id}'")
    truth = labels.loc[record.run_id, "root_cause_service"]
    row = record.model_dump()
    row["correct"] = signals.canonical_service(record.answer_service) == signals.canonical_service(truth)
    row["recorded_at"] = datetime.now().isoformat()
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    with open(STUDY_DIR / "sessions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    # Le participant n'apprend PAS s'il a juste (biais d'apprentissage sur les
    # tâches suivantes): seul l'accusé d'enregistrement est renvoyé.
    return {"status": "enregistré"}


# --- Interface web v2 (React + D3, frontend/) ------------------------------------
# Servie seulement si le build existe (cd frontend && npm install && npm run
# build, ou ./run.sh ui-build): l'API reste utilisable sans Node installé.
FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/ui", StaticFiles(directory=FRONTEND_DIST, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def root_redirect() -> RedirectResponse:
        return RedirectResponse("/ui/")
