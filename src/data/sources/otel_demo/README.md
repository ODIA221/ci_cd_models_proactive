# OpenTelemetry Demo ("Astronomy Shop") — source `otel_demo`

Même démonstrateur que celui validé dans l'article OF4CD. Contrairement aux
autres sources, il n'y a rien à télécharger : ce connecteur exporte les
données d'un stack Prometheus/Jaeger/Loki déjà en cours d'exécution.

## Prérequis

Docker et Docker Compose (non installés sur cette machine au moment de
l'écriture — à installer avant d'utiliser cette source).

## Lancer le stack

```bash
git clone https://github.com/open-telemetry/opentelemetry-demo ../opentelemetry-demo
cd ../opentelemetry-demo
docker compose up --no-build
```

Interfaces exposées:
- Boutique (trafic applicatif) : http://localhost:8080
- Grafana : http://localhost:8080/grafana
- Jaeger : http://localhost:8080/jaeger/ui (API interne : http://localhost:16686)
- Prometheus : http://localhost:9090
- Loki : http://localhost:3100 (pas d'UI dédiée, interrogé via son API)

Le Load Generator inclus dans le compose peuple automatiquement les données.

## Injecter des scénarios de défaillance (SC1/SC2, cf. OF4CD)

Les feature flags `flagd` du dépôt cloné (`src/flagd/demo.flagd.json`)
permettent de simuler des pannes : latence artificielle, erreurs de paiement,
etc. Active le flag correspondant avant de lancer un export pour capturer une
fenêtre de données incluant une anomalie labellisable.

## Attributs CI/CD (corrélation trace_id)

Pour ajouter les attributs `ci.pipeline.id`, `ci.job.id`, `git.commit.hash`
prescrits par OF4CD, fusionne `collector-config.yaml` (dans ce dossier) dans
la configuration du Collector du dépôt cloné
(`src/otel-collector/otelcol-config.yml`).

## Exporter les données

Une fois le stack démarré et un scénario éventuellement injecté :

```bash
python -m src.data.acquire --source otel_demo --export all
# ou individuellement: --export prometheus | jaeger | loki
```

Sorties : `data/interim/otel_demo/{metrics,traces,logs}.parquet` (schéma
unifié, voir `src/data/schema.py`).
