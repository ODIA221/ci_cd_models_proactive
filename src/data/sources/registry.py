"""
Chargement du registre des sources (registry.yaml) et résolution des connecteurs
"""

from pathlib import Path
from importlib import import_module
from typing import Dict, List
import logging

import yaml

from src.data.sources.base import BaseConnector

logger = logging.getLogger(__name__)

REGISTRY_PATH = Path(__file__).parent / "registry.yaml"


def load_registry() -> Dict[str, dict]:
    """Charge registry.yaml et retourne le dict {nom_source: métadonnées}"""
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_sources() -> List[dict]:
    """
    Liste toutes les sources avec leur statut courant, pour l'affichage `--list`.
    N'instancie/n'importe le module du connecteur que pour calculer le statut
    (aucun téléchargement n'est déclenché par cette fonction).
    """
    registry = load_registry()
    rows = []
    for name, meta in registry.items():
        try:
            connector = get_connector(name)
            status = connector.status()
        except Exception as e:
            logger.warning(f"Impossible de déterminer le statut de '{name}': {e}")
            status = "inconnu"

        rows.append(
            {
                "name": name,
                "display_name": meta.get("display_name", name),
                "modalities": meta.get("modalities", []),
                "status": status,
                "estimated_size": meta.get("estimated_size", "?"),
                "requires_token": meta.get("requires_token", False),
                "requires_manual_download": meta.get("requires_manual_download", False),
                "requires_docker": meta.get("requires_docker", False),
            }
        )
    return rows


def get_connector(name: str) -> BaseConnector:
    """
    Instancie le connecteur associé à `name` d'après registry.yaml.

    Args:
        name: identifiant de la source (ex: 'loghub', 'rcaeval', ...)

    Returns:
        Instance de BaseConnector
    """
    registry = load_registry()
    if name not in registry:
        raise ValueError(f"Source inconnue: '{name}'. Sources disponibles: {list(registry.keys())}")

    module_path, class_name = registry[name]["module"].split(":")
    module = import_module(module_path)
    connector_cls = getattr(module, class_name)
    return connector_cls()
