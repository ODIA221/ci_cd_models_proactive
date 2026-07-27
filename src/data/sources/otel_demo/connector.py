"""
Connecteur OpenTelemetry Demo ("Astronomy Shop") — même démonstrateur que
l'article OF4CD. Contrairement aux autres connecteurs, il n'y a rien à
télécharger: la donnée est exportée depuis un stack Prometheus/Jaeger déjà en
cours d'exécution (lancé via `./run.sh otel-up` dans un clone frère de
github.com/open-telemetry/opentelemetry-demo, non vendored ici).

Note (upstream a évolué depuis l'écriture initiale de ce connecteur):
- Jaeger tourne désormais en v2 (architecture OTel Collector) et sert son API
  sous le préfixe /jaeger/ui (plus à la racine) — d'où le endpoint par défaut
  ci-dessous qui inclut déjà ce préfixe.
- Loki a été retiré du stack upstream, remplacé par OpenSearch. L'export logs
  n'est donc plus disponible via ce connecteur (voir README.md).

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
    "jaeger": "http://localhost:16686/jaeger/ui",
}

SETUP_INSTRUCTIONS = """
Le stack OpenTelemetry Demo n'est pas accessible ({endpoint}).

Pour le démarrer: ./run.sh otel-up
  (clone github.com/open-telemetry/opentelemetry-demo dans ../opentelemetry-demo
   si besoin, puis lance compose.yaml + compose.observability.yaml)

Vérifie ensuite: Prometheus http://localhost:9090, Jaeger http://localhost:16686/jaeger/ui
Pour injecter une défaillance (scénarios SC1/SC2 d'OF4CD), active un feature
flag flagd décrit dans src/flagd/demo.flagd.json du dépôt cloné.

Relance ensuite: python -m src.data.acquire --source otel_demo --export all
"""


class OTelDemoConnector(BaseConnector):
    name = "otel_demo"

    def __init__(self, endpoints: dict = None):
        super().__init__()
        self.endpoints = endpoints or DEFAULT_ENDPOINTS

    def _is_reachable(self, url: str, timeout: float = 5.0) -> bool:
        try:
            requests.get(url, timeout=timeout)
            return True
        except requests.exceptions.RequestException:
            return False

    def status(self) -> str:
        reachable = {name: self._is_reachable(url) for name, url in self.endpoints.items()}
        if all(reachable.values()):
            return "stack actif (prometheus, jaeger accessibles)"
        if any(reachable.values()):
            missing = [n for n, ok in reachable.items() if not ok]
            return f"stack partiel (indisponible: {missing})"
        return "not_downloaded (stack non démarré)"

    def _targets(self, export: str) -> list:
        if export == "all":
            return list(self.endpoints.keys())
        if export not in self.endpoints:
            raise ValueError(f"Export inconnu: '{export}' (attendu: {list(self.endpoints)} ou 'all')")
        return [export]

    def fetch(self, force: bool = False, **kwargs) -> None:
        """Rien à télécharger — vérifie juste que l'endpoint demandé est accessible."""
        export = kwargs.get("export", "all")
        for name in self._targets(export):
            url = self.endpoints[name]
            if not self._is_reachable(url):
                raise RuntimeError(SETUP_INSTRUCTIONS.format(endpoint=f"{name} ({url})"))
        logger.info("Stack OpenTelemetry Demo déjà accessible, rien à faire.")

    def parse(self, export: str = "all", **kwargs) -> None:
        """
        Exporte les données du stack en cours d'exécution vers data/interim/otel_demo/.

        Args:
            export: 'prometheus', 'jaeger' ou 'all'
        """
        from src.data.sources.otel_demo import export_jaeger, export_prometheus

        exporters = {
            "prometheus": (export_prometheus.export, self.endpoints["prometheus"]),
            "jaeger": (export_jaeger.export, self.endpoints["jaeger"]),
        }

        self.interim_dir.mkdir(parents=True, exist_ok=True)

        for target in self._targets(export):
            export_fn, endpoint = exporters[target]
            if not self._is_reachable(endpoint):
                raise RuntimeError(SETUP_INSTRUCTIONS.format(endpoint=f"{target} ({endpoint})"))
            export_fn(endpoint, self.interim_dir)
