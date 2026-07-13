"""
API FastAPI de service pour LogPipeGuard

Prototype de recherche: aucune authentification, usage local uniquement.
Ne remplace pas l'entraînement (qui reste une commande CLI: `./run.sh demo`,
plusieurs minutes) — cette API ne fait que du chargement/inférence sur un
modèle déjà entraîné.
"""

import logging

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api import model_registry
from src.api.schemas import (
    HealthResponse,
    ModelsListResponse,
    PredictRequest,
    PredictResponse,
    SourcesResponse,
)
from src.data.sources.registry import list_sources

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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
    df = pd.DataFrame(records).drop(columns=["timestamp"], errors="ignore")

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
            {"prediction": int(p), "anomaly_score": float(s), "anomalie": bool(p == -1)}
            for p, s in zip(predictions, scores)
        ],
    )
