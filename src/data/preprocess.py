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

        # État figé à l'issue de fit(), réutilisé tel quel par transform()
        # pour appliquer un prétraitement identique sur de nouvelles données
        # (ex: une requête API) sans jamais réajuster scaler/imputer/encoders.
        self.numerical_cols = []
        self.categorical_cols = []
        self.numerical_cols_final = []
        self.fitted_columns = []
        self.is_fitted = False
        
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
    
    def fit(self, df: pd.DataFrame) -> "CICDPreprocessor":
        """
        Ajuste le pipeline de prétraitement (détection colonnes, imputation,
        encodage, normalisation) et conserve l'état pour transform().
        """
        logger.info("Début du prétraitement (fit)...")

        # 1. Détection des types
        self.numerical_cols, self.categorical_cols = self.detect_column_types(df)

        # 2. Gestion des valeurs manquantes (sur les numériques)
        df_processed = self.handle_missing_values(df, self.numerical_cols)

        # 3. Encodage des variables catégorielles
        df_processed = self.encode_categorical(df_processed, self.categorical_cols)

        # 4. Normalisation (re-détecter les colonnes numériques après encodage)
        self.numerical_cols_final = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        self.fitted_columns = df_processed.columns.tolist()
        df_processed = self.normalize_features(df_processed, self.numerical_cols_final)

        self.is_fitted = True
        logger.info(f"Prétraitement ajusté: {df_processed.shape}")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applique le prétraitement déjà ajusté par fit() à de nouvelles
        données, sans rien réajuster (scaler/imputer/encoders figés).
        Nécessaire pour prédire de façon cohérente sur des données jamais
        vues à l'entraînement (ex: une requête API).
        """
        if not self.is_fitted:
            raise RuntimeError("CICDPreprocessor.transform() appelé avant fit()")

        df_processed = df.copy()

        # 1. Imputation (mêmes colonnes numériques qu'à l'entraînement)
        if self.missing_strategy == "drop":
            df_processed = df_processed.dropna()
        elif self.imputer is not None:
            cols = [c for c in self.numerical_cols if c in df_processed.columns]
            df_processed[cols] = self.imputer.transform(df_processed[cols])

        # 2. Encodage catégoriel (valeurs jamais vues -> -1, comme les NaN)
        if self.categorical_strategy == "label":
            for col in self.categorical_cols:
                if col not in df_processed.columns:
                    continue
                encoder = self.label_encoders[col]
                known = set(encoder.classes_)
                values = df_processed[col].astype(str)
                unseen_mask = ~values.isin(known)
                if unseen_mask.any():
                    logger.warning(f"Valeurs catégorielles inconnues dans '{col}': {values[unseen_mask].unique().tolist()}")
                encoded = pd.Series(-1, index=df_processed.index, dtype="object")
                encoded[~unseen_mask] = encoder.transform(values[~unseen_mask])
                df_processed[col] = encoded
        elif self.categorical_strategy == "onehot":
            df_processed = pd.get_dummies(df_processed, columns=self.categorical_cols, dummy_na=True)

        # 3. Réindexation stricte sur les colonnes vues à l'entraînement
        df_processed = df_processed.reindex(columns=self.fitted_columns)

        # 4. Normalisation (scaler déjà ajusté)
        df_processed[self.numerical_cols_final] = self.scaler.transform(df_processed[self.numerical_cols_final])

        return df_processed

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajuste puis applique le pipeline de prétraitement en un seul appel
        (comportement historique, conservé pour train.py/evaluate.py).
        """
        return self.fit(df).transform(df)

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