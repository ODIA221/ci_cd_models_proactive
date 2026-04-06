"""
Script d'entraînement principal
Pipeline complet: chargement -> prétraitement -> entraînement -> évaluation
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Ajout du chemin src pour les imports
sys.path.append(str(Path(__file__).parent.parent))

from data.load_data import CICDDataLoader
from data.preprocess import CICDPreprocessor
from models.detection_models import AnomalyDetector

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
    X_processed = preprocessor.fit_transform(X)
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
    
    # 8. Sauvegarde des résultats
    results_df = pd.DataFrame({
        'prediction': results['predictions'],
        'anomalie': results['predictions'] == -1
    })
    results_path = Path("experiments") / f"results_{timestamp}.csv"
    results_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(results_path, index=False)
    logger.info(f"Résultats sauvegardés: {results_path}")
    
    return detector, results


if __name__ == "__main__":
    # Configuration - À ADAPTER AVEC VOS FICHIERS
    DATA_PATH = Path("../data")  # Chemin relatif depuis src/models/
    
    # Exemple d'utilisation avec vos données
    # Remplacez 'compute_dataset.csv' par le nom de votre fichier
    detector, results = train_pipeline(
        data_path=DATA_PATH,
        metrics_filename="compute_dataset.csv",  # À modifier
        model_type="isolation_forest",  # Testez les différents modèles
        target_col=None  # Mettez le nom si vous avez une colonne d'étiquettes
    )