"""
Registre des modèles entraînés disponibles pour l'API
Scanne models/*_meta.json (produit par src/models/train.py) pour résoudre et
charger un triplet (detector, preprocessor, meta) sans jamais réajuster quoi
que ce soit — le prétraitement et le modèle sont figés à l'entraînement.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging

import joblib

from src.models.detection_models import AnomalyDetector
from src.data.preprocess import CICDPreprocessor

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"


def list_model_triplets() -> List[Dict]:
    """
    Liste les modèles entraînés après l'introduction de cette fonctionnalité
    (avec un fichier _meta.json). Les modèles plus anciens (bare .joblib sans
    métadonnées) sont ignorés — un retrain via `./run.sh demo` est nécessaire
    pour qu'ils apparaissent ici.
    """
    triplets = []
    for meta_path in MODELS_DIR.glob("*_meta.json"):
        model_id = meta_path.name[: -len("_meta.json")]
        model_path = MODELS_DIR / f"{model_id}.joblib"
        preprocessor_path = MODELS_DIR / f"{model_id}_preprocessor.joblib"

        if not model_path.exists() or not preprocessor_path.exists():
            logger.warning(f"Triplet incomplet pour '{model_id}', ignoré (fichier manquant)")
            continue

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["model_id"] = model_id
        triplets.append(meta)

    triplets.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return triplets


def resolve_model_id(model_id: Optional[str] = None) -> str:
    """Retourne le model_id demandé (validé) ou le plus récent si non précisé."""
    triplets = list_model_triplets()
    if not triplets:
        raise ValueError("Aucun modèle entraîné disponible. Lance d'abord: ./run.sh demo")

    if model_id is None:
        return triplets[0]["model_id"]

    if not any(t["model_id"] == model_id for t in triplets):
        raise ValueError(f"model_id inconnu: '{model_id}'")

    return model_id


def load_triplet(model_id: str) -> Tuple[AnomalyDetector, CICDPreprocessor, Dict]:
    """Charge (detector, preprocessor, meta) pour un model_id déjà résolu."""
    meta_path = MODELS_DIR / f"{model_id}_meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["model_id"] = model_id

    detector = AnomalyDetector(model_type=meta["model_type"])
    if meta["model_type"] == "autoencoder":
        detector.build_model(input_dim=len(meta["feature_columns"]))
    detector.load(MODELS_DIR / f"{model_id}.joblib")

    preprocessor = joblib.load(MODELS_DIR / f"{model_id}_preprocessor.joblib")

    return detector, preprocessor, meta
