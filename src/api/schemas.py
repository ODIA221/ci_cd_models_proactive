"""
Schémas Pydantic pour l'API FastAPI (requêtes/réponses)
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class ModelInfo(BaseModel):
    model_id: str
    model_type: str
    created_at: str
    feature_columns: List[str]
    source_metrics_filename: Optional[str] = None
    source_logs_filename: Optional[str] = None
    target_col: Optional[str] = None


class ModelsListResponse(BaseModel):
    models: List[ModelInfo]


# Any plutôt que float: un enregistrement peut inclure une clé 'timestamp'
# (chaîne), ignorée par l'endpoint /predict avant conversion des features en
# float — valider Dict[str, float] ici rejetterait le payload avant même
# d'avoir pu retirer 'timestamp'.
PredictRecord = Dict[str, Any]


class PredictRequest(BaseModel):
    records: Union[PredictRecord, List[PredictRecord]] = Field(
        ..., description="Un enregistrement unique ou une liste d'enregistrements (nom de métrique -> valeur)"
    )
    model_id: Optional[str] = Field(None, description="Identifiant du modèle à utiliser (par défaut: le plus récent)")

    def records_as_list(self) -> List[PredictRecord]:
        return self.records if isinstance(self.records, list) else [self.records]


class PredictionItem(BaseModel):
    prediction: int
    anomaly_score: float
    anomalie: bool


class PredictResponse(BaseModel):
    model_id: str
    predictions: List[PredictionItem]


class SourceInfo(BaseModel):
    name: str
    display_name: str
    modalities: List[str]
    status: str
    estimated_size: str
    requires_token: bool
    requires_manual_download: bool
    requires_docker: bool


class SourcesResponse(BaseModel):
    sources: List[SourceInfo]
