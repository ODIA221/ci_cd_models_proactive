"""
Prétraitement des données pour la détection d'anomalies CI/CD
Normalisation, encodage, gestion des valeurs manquantes
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class CICDPreprocessor:
    """
    Prétraitement spécifique pour les données CI/CD
    """
    
    def __init__(self, categorical_strategy: str = 'label', 
                 numerical_strategy: str = 'standard',
                 missing_strategy: str = 'knn'):
        """
        Args:
            categorical_strategy: 'label' ou 'onehot'
            numerical_strategy: 'standard' ou 'minmax'
            missing_strategy: 'knn', 'mean', 'median', 'drop'
        """
        self.categorical_strategy = categorical_strategy
        self.numerical_strategy = numerical_strategy
        self.missing_strategy = missing_strategy
        
        self.scaler = None
        self.label_encoders = {}
        self.imputer = None
        
    def detect_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Détecte automatiquement les colonnes numériques et catégorielles
        
        Returns:
            (numerical_cols, categorical_cols)
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Exclure les colonnes d'identifiant ou de timestamp
        exclude_patterns = ['id', 'timestamp', 'date', 'time', 'pipeline_id', 'job_id']
        
        numerical_cols = [col for col in numerical_cols 
                         if not any(pattern in col.lower() for pattern in exclude_patterns)]
        categorical_cols = [col for col in categorical_cols 
                           if not any(pattern in col.lower() for pattern in exclude_patterns)]
        
        logger.info(f"Colonnes numériques: {len(numerical_cols)}, catégorielles: {len(categorical_cols)}")
        
        return numerical_cols, categorical_cols
    
    def handle_missing_values(self, df: pd.DataFrame, numerical_cols: List[str]) -> pd.DataFrame:
        """
        Gère les valeurs manquantes selon la stratégie choisie
        
        Pour votre dataset avec 99.9% de complétude, l'imputation KNN est idéale
        """
        df_clean = df.copy()
        
        if self.missing_strategy == 'drop':
            df_clean = df_clean.dropna()
            logger.info(f"Lignes supprimées: {len(df) - len(df_clean)}")
            
        elif self.missing_strategy in ['mean', 'median']:
            self.imputer = SimpleImputer(strategy=self.missing_strategy)
            df_clean[numerical_cols] = self.imputer.fit_transform(df_clean[numerical_cols])
            
        elif self.missing_strategy == 'knn':
            # KNN imputer pour préserver les relations entre variables
            self.imputer = KNNImputer(n_neighbors=5)
            df_clean[numerical_cols] = self.imputer.fit_transform(df_clean[numerical_cols])
        
        return df_clean
    
    def encode_categorical(self, df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
        """
        Encode les variables catégorielles (status, environnement, etc.)
        """
        df_encoded = df.copy()
        
        if self.categorical_strategy == 'label':
            for col in categorical_cols:
                if col in df_encoded.columns:
                    self.label_encoders[col] = LabelEncoder()
                    # Gestion des NaN
                    mask = df_encoded[col].notna()
                    df_encoded.loc[mask, col] = self.label_encoders[col].fit_transform(
                        df_encoded.loc[mask, col].astype(str)
                    )
                    # Remplacer les NaN restants par -1
                    df_encoded[col] = df_encoded[col].fillna(-1)
                    
        elif self.categorical_strategy == 'onehot':
            df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, dummy_na=True)
        
        logger.info(f"Encodage terminé: {df_encoded.shape[1]} colonnes")
        
        return df_encoded
    
    def normalize_features(self, df: pd.DataFrame, numerical_cols: List[str]) -> pd.DataFrame:
        """
        Normalise les caractéristiques numériques
        """
        df_normalized = df.copy()
        
        self.scaler = StandardScaler()
        df_normalized[numerical_cols] = self.scaler.fit_transform(df_normalized[numerical_cols])
        
        return df_normalized
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applique tout le pipeline de prétraitement
        """
        logger.info("Début du prétraitement...")
        
        # 1. Détection des types
        numerical_cols, categorical_cols = self.detect_column_types(df)
        
        # 2. Gestion des valeurs manquantes (sur les numériques)
        df_processed = self.handle_missing_values(df, numerical_cols)
        
        # 3. Encodage des variables catégorielles
        df_processed = self.encode_categorical(df_processed, categorical_cols)
        
        # 4. Normalisation (re-détecter les colonnes numériques après encodage)
        numerical_cols_updated = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        df_processed = self.normalize_features(df_processed, numerical_cols_updated)
        
        logger.info(f"Prétraitement terminé: {df_processed.shape}")
        
        return df_processed

# Exemple d'utilisation
if __name__ == "__main__":
    # Test avec des données simulées
    sample_data = pd.DataFrame({
        'cpu_usage': [45.2, 78.1, np.nan, 32.5, 95.3],
        'memory_mb': [1024, 2048, 1536, np.nan, 4096],
        'status': ['success', 'failed', 'success', 'running', 'failed'],
        'duration_ms': [1200, 4500, 2300, 890, 12000]
    })
    
    preprocessor = CICDPreprocessor(missing_strategy='knn')
    processed = preprocessor.fit_transform(sample_data)
    print(processed)