"""
Modèles de détection d'anomalies pour pipelines CI/CD
Implémente Isolation Forest, One-Class SVM, et Autoencoder
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, Optional
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """
    Détecteur d'anomalies multi-modèle pour CI/CD
    """
    
    def __init__(self, model_type: str = 'isolation_forest', **kwargs):
        """
        Args:
            model_type: 'isolation_forest', 'one_class_svm', 'autoencoder'
            **kwargs: Paramètres spécifiques au modèle
        """
        self.model_type = model_type
        self.model = None
        self.kwargs = kwargs
        
    def build_model(self, input_dim: Optional[int] = None):
        """
        Construit le modèle selon le type choisi
        """
        if self.model_type == 'isolation_forest':
            self.model = IsolationForest(
                contamination=self.kwargs.get('contamination', 0.1),
                random_state=self.kwargs.get('random_state', 42),
                n_estimators=self.kwargs.get('n_estimators', 100)
            )
            
        elif self.model_type == 'one_class_svm':
            self.model = OneClassSVM(
                nu=self.kwargs.get('nu', 0.1),
                kernel=self.kwargs.get('kernel', 'rbf'),
                gamma=self.kwargs.get('gamma', 'auto')
            )
            
        elif self.model_type == 'autoencoder':
            if input_dim is None:
                raise ValueError("input_dim requis pour l'autoencodeur")
            self.model = Autoencoder(input_dim=input_dim, **self.kwargs)
            
        else:
            raise ValueError(f"Modèle non reconnu: {self.model_type}")
        
        return self.model
    
    def train(self, X_train: np.ndarray, X_val: Optional[np.ndarray] = None, 
              epochs: int = 50, batch_size: int = 32):
        """
        Entraîne le modèle
        
        Args:
            X_train: Données d'entraînement
            X_val: Données de validation (optionnel)
            epochs: Nombre d'époques (pour autoencoder)
            batch_size: Taille du batch (pour autoencoder)
        """
        if self.model_type in ['isolation_forest', 'one_class_svm']:
            self.model.fit(X_train)
            
        elif self.model_type == 'autoencoder':
            self.model.train_model(X_train, X_val, epochs, batch_size)
        
        logger.info(f"Modèle {self.model_type} entraîné avec succès")
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Prédit les anomalies
        
        Returns:
            Array avec -1 pour anomalie, 1 pour normal
        """
        if self.model_type in ['isolation_forest', 'one_class_svm']:
            predictions = self.model.predict(X_test)
            
        elif self.model_type == 'autoencoder':
            reconstructions = self.model.predict(X_test)
            mse = np.mean((X_test - reconstructions) ** 2, axis=1)
            threshold = np.percentile(mse, 95)  # Seuil à 95%
            predictions = np.where(mse > threshold, -1, 1)
        
        return predictions
    
    def evaluate(self, X_test: np.ndarray, y_true: Optional[np.ndarray] = None) -> Dict:
        """
        Évalue les performances du modèle
        """
        y_pred = self.predict(X_test)
        
        results = {
            'anomalies_detectees': np.sum(y_pred == -1),
            'taux_anomalies': np.mean(y_pred == -1),
            'predictions': y_pred
        }
        
        if y_true is not None:
            # Conversion: y_true attendu en -1 (anomalie) et 1 (normal)
            results['classification_report'] = classification_report(y_true, y_pred)
            results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
            
            # AUC si disponible
            if len(np.unique(y_true)) == 2:
                y_true_binary = (y_true == -1).astype(int)
                y_pred_binary = (y_pred == -1).astype(int)
                results['auc'] = roc_auc_score(y_true_binary, y_pred_binary)
        
        return results
    
    def save(self, path: Path):
        """Sauvegarde le modèle"""
        if self.model_type in ['isolation_forest', 'one_class_svm']:
            joblib.dump(self.model, path)
        elif self.model_type == 'autoencoder':
            torch.save(self.model.state_dict(), path)
        logger.info(f"Modèle sauvegardé: {path}")
    
    def load(self, path: Path):
        """Charge un modèle sauvegardé"""
        if self.model_type in ['isolation_forest', 'one_class_svm']:
            self.model = joblib.load(path)
        elif self.model_type == 'autoencoder':
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
        logger.info(f"Modèle chargé: {path}")


class Autoencoder(nn.Module):
    """
    Autoencodeur profond pour la détection d'anomalies
    Particulièrement adapté pour les séquences temporelles CI/CD
    """
    
    def __init__(self, input_dim: int, hidden_dims: list = [64, 32, 16], 
                 latent_dim: int = 8, dropout: float = 0.2):
        """
        Args:
            input_dim: Dimension d'entrée
            hidden_dims: Dimensions des couches cachées
            latent_dim: Dimension de l'espace latent
            dropout: Taux de dropout
        """
        super().__init__()
        
        # Encodeur
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Décodeur
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.criterion = nn.MSELoss()
        self.optimizer = None
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def train_model(self, X_train, X_val=None, epochs=50, batch_size=32, lr=1e-3):
        """Entraîne l'autoencodeur"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        
        # Conversion en tenseurs
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            
            for batch_x, _ in train_loader:
                self.optimizer.zero_grad()
                outputs = self(batch_x)
                loss = self.criterion(outputs, batch_x)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.6f}")
    
    def predict(self, X_test):
        """Reconstruit les données de test"""
        device = next(self.parameters()).device
        self.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(device)
            reconstructions = self(X_tensor).cpu().numpy()
        
        return reconstructions


# Exemple d'utilisation
if __name__ == "__main__":
    # Simulation de données
    np.random.seed(42)
    X_train = np.random.randn(1000, 10)
    X_test = np.random.randn(200, 10)
    
    # Isolation Forest
    print("=== Isolation Forest ===")
    detector_if = AnomalyDetector(model_type='isolation_forest', contamination=0.1)
    detector_if.build_model()
    detector_if.train(X_train)
    results_if = detector_if.evaluate(X_test)
    print(f"Anomalies détectées: {results_if['anomalies_detectees']}")
    
    # Autoencodeur
    print("\n=== Autoencoder ===")
    detector_ae = AnomalyDetector(model_type='autoencoder', input_dim=10)
    detector_ae.build_model(input_dim=10)
    detector_ae.train(X_train, epochs=20)
    results_ae = detector_ae.evaluate(X_test)
    print(f"Anomalies détectées: {results_ae['anomalies_detectees']}")