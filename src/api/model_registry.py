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


def load_meta(model_id: str) -> Dict:
    """Charge uniquement les métadonnées d'un model_id, sans charger le
    detector/preprocessor (évite de désérialiser un autoencoder PyTorch
    juste pour lire feature_columns/source_dir, ex: GET /explain)."""
    meta_path = MODELS_DIR / f"{model_id}_meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["model_id"] = model_id
    return meta


def load_triplet(model_id: str) -> Tuple[AnomalyDetector, CICDPreprocessor, Dict]:
    """Charge (detector, preprocessor, meta) pour un model_id déjà résolu."""
    meta = load_meta(model_id)

    if meta["model_type"] == "multimodal_autoencoder":
        # variational_modalities change la forme des poids (têtes mu/logvar):
        # doit être lu depuis meta pour que build_model reconstruise la même
        # architecture que celle sauvegardée, sinon load_state_dict échoue.
        detector = AnomalyDetector(model_type=meta["model_type"],
                                    variational_modalities=set(meta.get("variational_modalities") or []))
    else:
        detector = AnomalyDetector(model_type=meta["model_type"])

    if meta["model_type"] == "autoencoder":
        detector.build_model(input_dim=len(meta["feature_columns"]))
    elif meta["model_type"] == "multimodal_autoencoder":
        detector.build_model(modality_dims=meta["modality_dims"])
    detector.load(MODELS_DIR / f"{model_id}.joblib")

    if meta["model_type"] in ("autoencoder", "multimodal_autoencoder"):
        # save()/load() ne portent que les poids (state_dict) — sans ceci,
        # un detector rechargé a threshold_=None et predict() retombe sur le
        # 95e percentile du batch REÇU: sur une requête à une seule ligne,
        # x > percentile([x], 95) == x > x == False toujours -> /predict
        # renverrait "normal" pour absolument tout, silencieusement.
        detector.threshold_ = meta.get("threshold_")

    preprocessor = joblib.load(MODELS_DIR / f"{model_id}_preprocessor.joblib")

    return detector, preprocessor, meta
