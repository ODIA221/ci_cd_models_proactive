# OpenTelemetry Demo ("Astronomy Shop") — source `otel_demo`

Même démonstrateur que celui validé dans l'article OF4CD. Contrairement aux
autres sources, il n'y a rien à télécharger : ce connecteur exporte les
données d'un stack Prometheus/Jaeger déjà en cours d'exécution.

## Prérequis

Docker et Docker Compose Desktop, démarré.

## Lancer le stack

```bash
./run.sh otel-up
```

Fait (idempotent) : clone `../opentelemetry-demo` si besoin, puis
`docker compose -f compose.yaml -f compose.observability.yaml up --no-build -d`
(la couche `compose.observability.yaml` ajoute Jaeger/Prometheus/Grafana/OpenSearch
au stack applicatif de base — sans elle, seule la boutique tourne).

Interfaces exposées:
- Boutique (trafic applicatif) : http://localhost:8080
- Prometheus : http://localhost:9090
- Jaeger : http://localhost:16686/jaeger/ui (l'API REST est servie sous ce
  même préfixe `/jaeger/ui` depuis le passage à Jaeger v2 — pas à la racine)

Le Load Generator inclus dans le compose peuple automatiquement les données.

**Note upstream (le dépôt `opentelemetry-demo` a évolué depuis l'écriture
initiale de ce connecteur)** : Loki a été retiré du stack et remplacé par
OpenSearch pour les logs. L'export logs n'est **pas implémenté** dans ce
connecteur (seuls `prometheus` et `jaeger` le sont) — le brancher sur
OpenSearch nécessiterait un nouvel exporter (`_search` sur l'index
`otel-logs-*`, cf. `otelcol-config-observability.yml` du dépôt cloné pour le
mapping exact). Par ailleurs OpenSearch peut rester `unhealthy` /ne jamais
démarrer sur Docker Desktop Mac tant que `vm.max_map_count` n'est pas relevé
à 262144 dans la VM Docker (`./run.sh otel-up` tente ce réglage
automatiquement sur macOS, best-effort) — `otel-collector` ne dépend
volontairement que du démarrage (pas de la santé) d'OpenSearch pour ne pas
bloquer tout le pipeline traces/métriques si les logs sont indisponibles.

## Injecter des scénarios de défaillance (SC1/SC2, cf. OF4CD)

Les feature flags `flagd` du dépôt cloné (`src/flagd/demo.flagd.json`)
permettent de simuler des pannes : latence artificielle, erreurs de paiement,
etc. Active le flag correspondant avant de lancer un export pour capturer une
fenêtre de données incluant une anomalie labellisable.

## Attributs CI/CD (corrélation trace_id)

Pour ajouter les attributs `ci.pipeline.id`, `ci.job.id`, `git.commit.hash`
prescrits par OF4CD, fusionne `collector-config.yaml` (dans ce dossier) dans
la configuration du Collector du dépôt cloné
(`src/otel-collector/otelcol-config-observability.yml`).

## Exporter les données

Une fois le stack démarré et un scénario éventuellement injecté :

```bash
python -m src.data.acquire --source otel_demo --export all
# ou individuellement: --export prometheus | jaeger
```

Sorties : `data/interim/otel_demo/{metrics,traces}.parquet` (schéma unifié,
voir `src/data/schema.py`). Pas de `logs.parquet` (voir note ci-dessus).
