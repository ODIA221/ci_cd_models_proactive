"""
Module de chargement des données pour la détection d'anomalies CI/CD
Gère les différents formats (CSV, Parquet, JSON) et types de données (logs, métriques, traces)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Union, Optional, Tuple
import json
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CICDDataLoader:
    """
    Chargeur de données pour les pipelines CI/CD
    Supporte les logs, métriques et traces
    """
    
    def __init__(self, data_path: Path):
        """
        Args:
            data_path: Chemin vers le dossier contenant les données brutes
        """
        self.data_path = Path(data_path)
        self.raw_path = self.data_path / 'raw'
        self.processed_path = self.data_path / 'processed'
        
        # Création des dossiers s'ils n'existent pas
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
    
    def load_metrics(self, filename: str) -> pd.DataFrame:
        """
        Charge les métriques (CPU, mémoire, réseau, etc.)
        
        Args:
            filename: Nom du fichier (CSV ou Parquet)
            
        Returns:
            DataFrame avec les métriques
        """
        filepath = self.raw_path / filename
        
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filename.endswith('.parquet'):
            df = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Format non supporté: {filename}")
        
        logger.info(f"Métriques chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
        
        # Conversion des timestamps si présents
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        return df
    
    def load_logs(self, filename: str) -> pd.DataFrame:
        """
        Charge les logs bruts
        
        Args:
            filename: Nom du fichier (CSV ou JSON)
            
        Returns:
            DataFrame avec les logs
        """
        filepath = self.raw_path / filename
        
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filename.endswith('.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Format non supporté: {filename}")
        
        logger.info(f"Logs chargés: {df.shape[0]} lignes")
        
        # Vérification de la colonne de message
        if 'message' not in df.columns and 'log' not in df.columns:
            logger.warning("Aucune colonne 'message' ou 'log' trouvée")
        
        return df
    
    def load_traces(self, filename: str) -> Dict:
        """
        Charge les traces (format JSON généralement)
        
        Args:
            filename: Nom du fichier JSON
            
        Returns:
            Dictionnaire contenant les traces
        """
        filepath = self.raw_path / filename
        
        with open(filepath, 'r') as f:
            traces = json.load(f)
        
        logger.info(f"Traces chargées: {len(traces) if isinstance(traces, list) else 'structure dict'}")
        
        return traces
    
    def save_processed_data(self, df: pd.DataFrame, filename: str, format: str = 'parquet'):
        """
        Sauvegarde les données nettoyées
        
        Args:
            df: DataFrame à sauvegarder
            filename: Nom du fichier
            format: 'csv' ou 'parquet' (recommandé pour gros volumes)
        """
        filepath = self.processed_path / filename
        
        if format == 'csv':
            df.to_csv(filepath, index=False)
        elif format == 'parquet':
            df.to_parquet(filepath, index=False)
        else:
            raise ValueError(f"Format non supporté: {format}")
        
        logger.info(f"Données sauvegardées: {filepath}")

# Exemple d'utilisation
if __name__ == "__main__":
    # Initialisation
    loader = CICDDataLoader(Path("../../data"))
    
    # Chargement des métriques (à adapter avec vos noms de fichiers)
    # metrics_df = loader.load_metrics("compute_dataset.csv")
    # print(metrics_df.head())