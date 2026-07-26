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
        # Seuil de reconstruction (autoencoder/multimodal_autoencoder),
        # appris une fois sur X_train (normal uniquement) dans train() —
        # jamais recalculé sur X_test dans predict(), cf. docstring de
        # predict() pour la raison.
        self.threshold_ = None
        
    def build_model(self, input_dim: Optional[int] = None, modality_dims: Optional[Dict[str, int]] = None):
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

        elif self.model_type == 'multimodal_autoencoder':
            if modality_dims is None:
                raise ValueError("modality_dims requis pour l'autoencodeur multimodal")
            self.model = MultimodalAutoencoder(modality_dims=modality_dims, **self.kwargs)

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

        elif self.model_type in ('autoencoder', 'multimodal_autoencoder'):
            self.model.train_model(X_train, X_val, epochs, batch_size)
            # Seuil appris sur X_train (normal uniquement, cf. protocole
            # evaluate.py/evaluate_multimodal.py), PAS sur X_test dans
            # predict(): un seuil recalculé au 95e percentile de X_test
            # suppose ~5% d'anomalies dans CE batch précis — faux dès que la
            # prévalence réelle diffère (ex: test RCAEval ~50/50), ce qui
            # plafonnait artificiellement le rappel autour de 5% quel que
            # soit le pouvoir discriminant réel du modèle (AUC, lui,
            # indépendant du seuil, n'était pas affecté).
            reconstructions_train = self.model.predict(X_train)
            train_mse = np.mean((X_train - reconstructions_train) ** 2, axis=1)
            self.threshold_ = np.percentile(train_mse, 95)

        logger.info(f"Modèle {self.model_type} entraîné avec succès")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Prédit les anomalies

        Returns:
            Array avec -1 pour anomalie, 1 pour normal
        """
        if self.model_type in ['isolation_forest', 'one_class_svm']:
            predictions = self.model.predict(X_test)

        elif self.model_type in ('autoencoder', 'multimodal_autoencoder'):
            reconstructions = self.model.predict(X_test)
            mse = np.mean((X_test - reconstructions) ** 2, axis=1)
            if self.threshold_ is None:
                logger.warning(
                    "Seuil non appris (train() jamais appelé sur cette instance, ex: après load() seul) — "
                    "repli sur le 95e percentile de CE batch de test, biaisé si sa prévalence d'anomalies "
                    "diffère de celle de l'entraînement."
                )
                threshold = np.percentile(mse, 95)
            else:
                threshold = self.threshold_
            predictions = np.where(mse > threshold, -1, 1)

        return predictions

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Score continu d'anomalie (plus élevé = plus anormal), utile pour
        trier/prioriser des alertes plutôt que de se limiter à -1/1.
        """
        try:
            if self.model_type in ("autoencoder", "multimodal_autoencoder"):
                reconstructions = self.model.predict(X)
                return np.mean((X - reconstructions) ** 2, axis=1)
            return -self.model.decision_function(X)
        except Exception as e:
            logger.warning(f"Score d'anomalie non calculable: {e}")
            return np.full(len(X), np.nan)

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
        elif self.model_type in ('autoencoder', 'multimodal_autoencoder'):
            torch.save(self.model.state_dict(), path)
        logger.info(f"Modèle sauvegardé: {path}")

    def load(self, path: Path):
        """Charge un modèle sauvegardé"""
        if self.model_type in ['isolation_forest', 'one_class_svm']:
            self.model = joblib.load(path)
        elif self.model_type in ('autoencoder', 'multimodal_autoencoder'):
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


class MultimodalAutoencoder(nn.Module):
    """
    Autoencodeur multimodal: une branche encodeur/décodeur par modalité
    (métriques/logs/traces), fusionnées dans un goulot d'étranglement latent
    partagé — contrairement à Autoencoder (mono-branche, une seule modalité)
    et à la fusion tardive de evaluate_multimodal.py::run_late_fusion (3
    détecteurs indépendants + un combinateur appris après coup): ici la
    représentation latente est apprise CONJOINTEMENT à partir des 3
    modalités, pas recombinée après coup.

    Prend un seul tenseur 2D en entrée (colonnes des modalités concaténées
    dans l'ordre de `modality_dims`, un dict {nom: nb_colonnes}) plutôt que
    des tenseurs séparés par modalité — reste compatible avec l'appel
    `detector.train(X.values)` déjà utilisé partout ailleurs dans le repo.
    """

    def __init__(self, modality_dims: Dict[str, int], hidden_dims: list = [64, 32, 16],
                 branch_latent_dim: int = 16, latent_dim: int = 8, dropout: float = 0.2):
        super().__init__()
        self.modality_dims = modality_dims
        self.branch_latent_dim = branch_latent_dim

        # Bornes de tranchage du tenseur d'entrée concaténé, précalculées une
        # fois (ordre = ordre d'insertion de modality_dims).
        self._slices = {}
        offset = 0
        for name, dim in modality_dims.items():
            self._slices[name] = slice(offset, offset + dim)
            offset += dim

        def make_branch(in_dim: int, out_dim: int, dims: list) -> nn.Sequential:
            layers = []
            prev_dim = in_dim
            for h_dim in dims:
                layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU(), nn.Dropout(dropout)])
                prev_dim = h_dim
            layers.append(nn.Linear(prev_dim, out_dim))
            return nn.Sequential(*layers)

        # Une branche encodeur par modalité: dim modalité -> branch_latent_dim
        self.encoders = nn.ModuleDict({
            name: make_branch(dim, branch_latent_dim, hidden_dims)
            for name, dim in modality_dims.items()
        })

        # Goulot d'étranglement PARTAGÉ: concat des latents par modalité -> latent_dim
        fused_dim = branch_latent_dim * len(modality_dims)
        self.bottleneck = nn.Linear(fused_dim, latent_dim)
        self.unbottleneck = nn.Linear(latent_dim, fused_dim)

        # Une branche décodeur par modalité: branch_latent_dim -> dim modalité
        self.decoders = nn.ModuleDict({
            name: make_branch(branch_latent_dim, dim, list(reversed(hidden_dims)))
            for name, dim in modality_dims.items()
        })

        self.optimizer = None

    def forward(self, x):
        branch_latents = [self.encoders[name](x[:, s]) for name, s in self._slices.items()]
        z = self.bottleneck(torch.cat(branch_latents, dim=1))
        unfused = self.unbottleneck(z)
        chunks = torch.split(unfused, self.branch_latent_dim, dim=1)
        reconstructions = [self.decoders[name](chunk) for name, chunk in zip(self._slices.keys(), chunks)]
        return torch.cat(reconstructions, dim=1)

    def _modality_loss(self, outputs, targets):
        """
        Moyenne (pas somme) du Huber loss calculé séparément par modalité:
        une simple MSE sur toutes les colonnes concaténées écraserait
        totalement le signal traces (6 colonnes) derrière la branche
        métriques (~3000 colonnes RCAEval) — chaque modalité pèse donc pour
        1/3 de la perte, indépendamment de son nombre de colonnes.

        Huber (smooth L1) plutôt que MSE pour l'ENTRAÎNEMENT spécifiquement:
        ~15% des colonnes metric_* RCAEval sont des métriques par service
        quasi constantes (0 sauf pour le service concerné, cf.
        rcaeval.py::parse::fillna(0.0)) — un std quasi nul rend leur z-score
        StandardScaler énorme sur la poignée de lignes non nulles, et le MSE
        (qui met au carré) explosait (perte ~1e14, non convergente). Huber
        pénalise ces résidus extrêmes linéairement plutôt que
        quadratiquement, ce qui stabilise l'entraînement sans changer le
        prétraitement partagé avec isolation_forest/one_class_svm. Le score
        d'anomalie à l'inférence (AnomalyDetector.anomaly_score) reste un MSE
        classique, inchangé — seule la perte d'ENTRAÎNEMENT change ici.
        """
        losses = [nn.functional.smooth_l1_loss(outputs[:, s], targets[:, s]) for s in self._slices.values()]
        return torch.stack(losses).mean()

    def train_model(self, X_train, X_val=None, epochs=50, batch_size=32, lr=1e-3):
        """Entraîne l'autoencodeur multimodal (même boucle que Autoencoder.train_model, perte par modalité)"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

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
                loss = self._modality_loss(outputs, batch_x)
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