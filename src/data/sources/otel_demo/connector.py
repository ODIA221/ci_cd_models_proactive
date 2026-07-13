"""
Connecteur OpenTelemetry Demo ("Astronomy Shop") — même démonstrateur que
l'article OF4CD. Contrairement aux autres connecteurs, il n'y a rien à
télécharger: la donnée est exportée depuis un stack Prometheus/Jaeger/Loki
déjà en cours d'exécution (lancé via `docker compose up` dans un clone frère
de github.com/open-telemetry/opentelemetry-demo, non vendored ici).

Voir README.md dans ce dossier pour les instructions de lancement, et
collector-config.yaml pour la configuration du Collector OTel.
"""

from pathlib import Path
import logging

import requests

from src.data.sources.base import BaseConnector

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINTS = {
    "prometheus": "http://localhost:9090",
    "jaeger": "http://localhost:16686",
    "loki": "http://localhost:3100",
}

SETUP_INSTRUCTIONS = """
Le stack OpenTelemetry Demo n'est pas accessible ({endpoint}).

Pour le démarrer:
  1. git clone https://github.com/open-telemetry/opentelemetry-demo ../opentelemetry-demo
  2. cd ../opentelemetry-demo && docker compose up --no-build
  3. Vérifie: Prometheus http://localhost:9090, Jaeger http://localhost:16686,
     Loki http://localhost:3100 (Grafana derrière http://localhost:8080/grafana)
  4. Pour injecter une défaillance (scénarios SC1/SC2 d'OF4CD), active un
     feature flag flagd décrit dans src/flagd/demo.flagd.json du dépôt cloné.

Relance ensuite: python -m src.data.acquire --source otel_demo --export all
"""


class OTelDemoConnector(BaseConnector):
    name = "otel_demo"

    def __init__(self, endpoints: dict = None):
        super().__init__()
        self.endpoints = endpoints or DEFAULT_ENDPOINTS

    def _is_reachable(self, url: str, timeout: float = 2.0) -> bool:
        try:
            requests.get(url, timeout=timeout)
            return True
        except requests.exceptions.RequestException:
            return False

    def status(self) -> str:
        reachable = {name: self._is_reachable(url) for name, url in self.endpoints.items()}
        if all(reachable.values()):
            return "stack actif (prometheus, jaeger, loki accessibles)"
        if any(reachable.values()):
            missing = [n for n, ok in reachable.items() if not ok]
            return f"stack partiel (indisponible: {missing})"
        return "not_downloaded (stack non démarré)"

    def fetch(self, force: bool = False, **kwargs) -> None:
        """Rien à télécharger — affiche les instructions de lancement du stack."""
        for name, url in self.endpoints.items():
            if not self._is_reachable(url):
                raise RuntimeError(SETUP_INSTRUCTIONS.format(endpoint=f"{name} ({url})"))
        logger.info("Stack OpenTelemetry Demo déjà accessible, rien à faire.")

    def parse(self, export: str = "all", **kwargs) -> None:
        """
        Exporte les données du stack en cours d'exécution vers data/interim/otel_demo/.

        Args:
            export: 'prometheus', 'jaeger', 'loki' ou 'all'
        """
        from src.data.sources.otel_demo import export_jaeger, export_loki, export_prometheus

        exporters = {
            "prometheus": (export_prometheus.export, self.endpoints["prometheus"]),
            "jaeger": (export_jaeger.export, self.endpoints["jaeger"]),
            "loki": (export_loki.export, self.endpoints["loki"]),
        }

        targets = exporters.keys() if export == "all" else [export]
        self.interim_dir.mkdir(parents=True, exist_ok=True)

        for target in targets:
            if target not in exporters:
                raise ValueError(f"Export inconnu: '{target}' (attendu: {list(exporters)} ou 'all')")
            export_fn, endpoint = exporters[target]
            if not self._is_reachable(endpoint):
                raise RuntimeError(SETUP_INSTRUCTIONS.format(endpoint=f"{target} ({endpoint})"))
            export_fn(endpoint, self.interim_dir)
