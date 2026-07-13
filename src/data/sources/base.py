"""
Interface commune à tous les connecteurs de sources de données externes
Chaque connecteur (loghub, rcaeval, gaia, travistorrent, github_actions, otel_demo)
implémente ce contrat pour être piloté de façon uniforme par acquire.py
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

DATA_ROOT = Path(__file__).resolve().parents[3] / "data"


class BaseConnector(ABC):
    """
    Contrat commun: status() / fetch() / parse() / info()

    Conventions de répertoires:
      - data/raw/<name>/       artefacts bruts téléchargés (gitignoré)
      - data/interim/<name>/   sorties normalisées au schéma unifié (gitignoré)
    """

    name: str

    def __init__(self):
        self.raw_dir = DATA_ROOT / "raw" / self.name
        self.interim_dir = DATA_ROOT / "interim" / self.name

    def status(self) -> str:
        """
        Statut par défaut basé sur la présence de fichiers dans raw_dir/interim_dir.
        Les connecteurs à sémantique différente (ex: otel_demo, qui interroge un
        stack en cours d'exécution plutôt que de télécharger) surchargent cette méthode.
        """
        if self.interim_dir.exists() and any(self.interim_dir.iterdir()):
            return "parsed"
        if self.raw_dir.exists() and any(self.raw_dir.iterdir()):
            return "downloaded"
        return "not_downloaded"

    @abstractmethod
    def fetch(self, force: bool = False, **kwargs) -> None:
        """Télécharge/collecte les données brutes vers self.raw_dir. Idempotent."""
        raise NotImplementedError

    @abstractmethod
    def parse(self, **kwargs) -> None:
        """Normalise les données brutes vers le schéma unifié dans self.interim_dir."""
        raise NotImplementedError

    def info(self) -> Dict:
        """Métadonnées de la source, surchargé pour enrichir avec registry.yaml."""
        return {"name": self.name, "status": self.status()}
