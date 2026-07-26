"""
API FastAPI de service pour LogPipeGuard

Prototype de recherche: aucune authentification, usage local uniquement.
Ne remplace pas l'entraînement (qui reste une commande CLI: `./run.sh demo`,
plusieurs minutes) — cette API ne fait que du chargement/inférence sur un
modèle déjà entraîné.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api import model_registry
from src.api.schemas import (
    ExplainResponse,
    HealthResponse,
    ModelsListResponse,
    PredictRequest,
    PredictResponse,
    SourcesResponse,
)
from src.causal import correlation
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
