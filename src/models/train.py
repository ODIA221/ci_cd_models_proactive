"""
Script d'entraînement principal
Pipeline complet: chargement -> prétraitement -> entraînement -> évaluation
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime

# Ajout de la racine du repo pour les imports. Important: on importe via
# `src.xxx` (pas `data.xxx`) pour que le nom de module qualifié soit
# identique à celui utilisé par src/api/ — sinon joblib.dump(preprocessor)
# sérialise la classe sous le nom 'data.preprocess.CICDPreprocessor', que
# l'API (qui importe 'src.data.preprocess.CICDPreprocessor') ne peut pas
# désérialiser (ModuleNotFoundError: No module named 'data.preprocess').
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data.load_data import CICDDataLoader
from src.data.preprocess import CICDPreprocessor
from src.models.detection_models import AnomalyDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_pipeline(data_path: Path, 
                   metrics_filename: str = None,
                   logs_filename: str = None,
                   model_type: str = 'isolation_forest',
                   target_col: str = None):
    """
    Pipeline complet d'entraînement pour la détection d'anomalies CI/CD
    
    Args:
        data_path: Chemin vers le dossier data
        metrics_filename: Nom du fichier de métriques
        logs_filename: Nom du fichier de logs
        model_type: Type de modèle ('isolation_forest', 'one_class_svm', 'autoencoder')
        target_col: Colonne cible pour la validation (optionnel)
    """
    
    # 1. Initialisation
    loader = CICDDataLoader(data_path)
    preprocessor = CICDPreprocessor(missing_strategy='knn')
    
    # 2. Chargement des données
    logger.info("=" * 50)
    logger.info("DÉBUT DU PIPELINE D'ENTRAÎNEMENT")
    logger.info("=" * 50)
    
    if metrics_filename:
        logger.info(f"Chargement des métriques: {metrics_filename}")
        df = loader.load_metrics(metrics_filename)
    elif logs_filename:
        logger.info(f"Chargement des logs: {logs_filename}")
        df = loader.load_logs(logs_filename)
    else:
        raise ValueError("Spécifiez metrics_filename ou logs_filename")
    
    logger.info(f"Dimensions des données: {df.shape}")
    
    # 3. Séparation des features et de la cible (si disponible)
    if target_col and target_col in df.columns:
        y = df[target_col].values
        X = df.drop(columns=[target_col])
        logger.info(f"Colonne cible trouvée: {target_col}")
    else:
        X = df
        y = None
        logger.info("Pas de colonne cible - détection non supervisée")
    
    # 4. Prétraitement
    logger.info("\n--- PRÉTRAITEMENT ---")
    preprocessor.fit(X)
    X_processed = preprocessor.transform(X)
    logger.info(f"Données après prétraitement: {X_processed.shape}")
    
    # 5. Construction et entraînement du modèle
    logger.info(f"\n--- ENTRAÎNEMENT ({model_type}) ---")
    
    detector = AnomalyDetector(model_type=model_type)
    
    if model_type == 'autoencoder':
        detector.build_model(input_dim=X_processed.shape[1])
    else:
        detector.build_model()
    
    detector.train(X_processed.values if hasattr(X_processed, 'values') else X_processed)
    
    # 6. Évaluation
    logger.info("\n--- ÉVALUATION ---")
    results = detector.evaluate(
        X_processed.values if hasattr(X_processed, 'values') else X_processed,
        y_true=y
    )
    
    logger.info(f"Anomalies détectées: {results['anomalies_detectees']} ({results['taux_anomalies']:.2%})")
    
    if 'auc' in results:
        logger.info(f"AUC: {results['auc']:.4f}")
    
    # 7. Sauvegarde du modèle
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = Path("models") / f"{model_type}_{timestamp}.joblib"
    model_path.parent.mkdir(exist_ok=True)
    detector.save(model_path)
    logger.info(f"Modèle sauvegardé: {model_path}")

    # 7bis. Sauvegarde du préprocesseur ajusté + métadonnées, pour pouvoir
    # rejouer exactement le même prétraitement sur de nouvelles données
    # (ex: une requête API) sans jamais réajuster scaler/imputer/encoders.
    preprocessor_path = Path("models") / f"{model_type}_{timestamp}_preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)

    meta = {
        "model_type": model_type,
        "created_at": datetime.now().isoformat(),
        "feature_columns": X_processed.columns.tolist(),
        "source_metrics_filename": metrics_filename,
        "source_logs_filename": logs_filename,
        "target_col": target_col,
        # Seuil de reconstruction appris sur X_train (AnomalyDetector.train,
        # model_type='autoencoder' uniquement) — save()/load() ne portent que
        # les poids; sans ce champ, un modèle rechargé via l'API perdrait son
        # seuil et /predict casserait sur les requêtes à une seule ligne
        # (cf. src/api/model_registry.py::load_triplet).
        "threshold_": float(detector.threshold_) if detector.threshold_ is not None else None,
    }
    meta_path = Path("models") / f"{model_type}_{timestamp}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info(f"Préprocesseur et métadonnées sauvegardés: {preprocessor_path.name}, {meta_path.name}")

    # 8. Sauvegarde des résultats
    # Score continu (plus élevé = plus anormal), utile pour trier/prioriser les alertes.
    anomaly_score = detector.anomaly_score(X_processed.values if hasattr(X_processed, 'values') else X_processed)

    # Repose sur X.reindex(...) plutôt que sur un alignement positionnel: robuste même
    # si le prétraitement a supprimé des lignes (missing_strategy='drop').
    results_df = X.reindex(X_processed.index).reset_index()
    results_df['anomaly_score'] = anomaly_score
    results_df['prediction'] = results['predictions']
    results_df['anomalie'] = results['predictions'] == -1
    results_df = results_df.sort_values('anomaly_score', ascending=False)

    results_path = Path("experiments") / f"results_{timestamp}.csv"
    results_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(results_path, index=False)
    logger.info(f"Résultats sauvegardés (triés par score d'anomalie décroissant): {results_path}")
    
    return detector, results


if __name__ == "__main__":
    # Chemin robuste au répertoire courant: <repo_root>/data
    DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data"

    detector, results = train_pipeline(
        data_path=DATA_PATH,
        metrics_filename="metrics/dataset_metrics.csv",
        model_type="isolation_forest",
        target_col=None  # Pas de colonne cible connue - détection non supervisée
    )