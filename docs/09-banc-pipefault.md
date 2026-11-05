# 9. Banc PipeFault : chaîne CI/CD instrumentée avec injection de fautes

Ce chapitre décrit le banc construit pour l'article 5. Il répond à la
faiblesse principale des chapitres précédents : toutes les évaluations
portaient sur des pannes de microservices injectées à l'exécution (RCAEval),
où les nœuds sont des services, jamais des étapes de pipeline.

**État au 2026-10-31.** Le code du banc est écrit et vérifié sans Docker
(`./run.sh bench-selftest` : 57 vérifications sur 57, contre la démo réelle).
Il n'a **pas encore tourné de bout en bout** : aucune exécution du pilote n'a
eu lieu, et aucune donnée PipeFault n'existe. Les points marqués « à
confirmer au pilote » ci-dessous n'ont pas pu être vérifiés autrement.

## Ce que fait le banc

```
Jenkins (port 8090, JCasC, greffon OpenTelemetry)
   checkout -> build -> test-unit -> image-check -> deploy -> smoke
      |  construit 3 des 4 services livrables (checkout Go, payment Node.js,
      |  recommendation Python, ad Java) depuis une copie de la démo
      v
Démo OpenTelemetry 3.0.0 (../opentelemetry-demo) + générateur de charge
      v
Collector -> Jaeger (traces) · Prometheus (métriques) · Loki (journaux)
```

Une exécution dure environ 25 minutes : remise à l'état de référence, 5 min
de référence, pipeline, faute d'exécution éventuelle 60 s après la fin du
pipeline, 10 min d'observation, export. L'orchestrateur
(`src/bench/campaign.py`) tient la fenêtre d'observation, pas le pipeline.

**Trois services livrés par exécution, dont au plus un fautif.** Avec un seul
service livré, la référence naïve « dernier service déployé » serait toujours
juste sur les fautes inter-étapes.

## Catalogue de fautes (`src/bench/faults.py`)

| Code | Famille | Mécanisme | Services | Étape d'origine |
|---|---|---|---|---|
| C1 | CI | assertion d'un test synthétique inversée | les 4 | test-unit |
| C2 | CI | test synthétique instable (p = 0,3, 3 tentatives) | les 4 | test-unit |
| C3 | CI | `HTTP(S)_PROXY` injoignable pendant `docker build --no-cache` | les 4 | build |
| C4 | CI | erreur de compilation insérée | checkout, ad (langages compilés) | build |
| X1 | inter-étapes | `PAYMENT_ADDR` / `PRODUCT_CATALOG_ADDR` vers un port fermé | checkout, recommendation | deploy |
| X2 | inter-étapes | limite de tas trop basse (`GOMEMLIMIT`, `--max-old-space-size`, `-Xmx`) | checkout, payment, ad | deploy |
| X3 | inter-étapes | `cpus: 0.05` dans le manifeste | les 4 | deploy |
| X4 | inter-étapes | pause de 300 ms insérée dans la route principale | les 4 | build |
| X5 | inter-étapes | port d'écoute décalé, appelants inchangés | les 4 | deploy |
| X6 | inter-étapes | limite mémoire du conteneur trop basse | les 4 | deploy |
| R1 | exécution | drapeau `intlShippingSlowdown` ou `imageSlowLoad` | shipping, image-provider | — |
| R2 | exécution | drapeau `adHighCpu`, `emailMemoryLeak` ou `adManualGc` | ad, email | — |
| R3 | exécution | drapeau `paymentFailure`, `cartFailure`, `productCatalogFailure`, `adFailure` ou `recommendationCacheFailure` | 5 services | — |

Par rapport au premier jet de l'article 5, le catalogue a été ajusté au code
réel de la démo 3.0.0 : « délai client réduit » et « pool réduit » n'ont pas
de variable de configuration dans ces services (remplacés par X2 et X3),
« construction sans optimisation » n'a de sens que pour Go (remplacé par X5,
port décalé), et l'erreur de compilation n'existe que pour Go et Java.
**L'article doit reprendre ce tableau.**

**Les tests de l'étape test-unit sont synthétiques** (`src/bench/synthetic_tests.py`) :
ils vérifient des propriétés statiques du code livré, pas son comportement.
Ils ne voient donc ni les fautes de configuration ni la régression X4, comme
beaucoup de suites réelles. C'est une hypothèse de conception à rapporter
comme menace à la validité.

## Calendrier (`./run.sh bench-plan`)

Tiré une seule fois, graine par défaut 20261031, écrit dans
`data/raw/pipefault/schedule.csv` ; la commande refuse d'écraser un
calendrier existant.

| Bloc | Exécutions | Détail |
|---|---|---|
| dev (pilote) | 50 | 18 avec faute (toutes les familles) + 32 normales |
| test | 360 | 180 inter-étapes (6 × 30), 40 CI (4 × 10), 60 exécution (3 × 20), 80 normales |

`bench-run` exécute toujours le bloc dev avant le bloc test.

## Contrôle (`./run.sh bench-check`)

Règles préenregistrées (`src/bench/check.py`) : une exécution est exclue si
l'effet de sa faute n'est pas vérifié (étape d'origine en échec pour C1, C3,
C4 ; au moins une tentative en échec pour C2 ; pipeline vert ET, sur le
service fautif, p90 de latence ×1,5 ou taux d'erreur +5 points ou un
redémarrage, pour X et R), ou s'il manque plus de 10 % des pas de 15 s dans
une modalité. Si plus de 15 % des exécutions d'un type sont exclues, elles sont
remplacées en fin de campagne par un calendrier complémentaire tiré avec la
graine **20261032** (déclarée dans l'article 5, section 6.4).

## Puissance

`./run.sh bench-power` (`src/bench/power.py`, graine 20261031) recalcule le tableau 5
de l'article 5 ; résultat dans `experiments/pipefault_power_grid.csv`.

## Lancer

```bash
./run.sh bench-selftest              # sans Docker : fautes, configs, drapeaux, calendrier
./run.sh bench-up                    # démo + Jaeger/Prometheus/Loki + Jenkins
./run.sh bench-plan                  # une seule fois
./run.sh bench-run --limit 1         # première exécution du pilote
./run.sh bench-check                 # complétude et effet de la faute
./run.sh bench-status
./run.sh bench-down                  # arrêt, volumes conservés
```

**Ressources.** Les limites mémoire de compose (démo 3,2 Go + banc 3,2 Go)
totalisent environ 6,4 Go, sans compter les constructions d'images (Gradle pour `ad`).
Sur la machine de développement (4 cœurs, 8 Go), c'est à la limite : le
pilote est possible en fermant le reste, la campagne de 410 exécutions
(environ 170 h) demande une VM dédiée de 16 Go.

## À confirmer au pilote

- clés de configuration du greffon OpenTelemetry de Jenkins dans
  `bench/jenkins/casc.yaml` (elles changent selon les versions) ;
- ingestion OTLP par Loki 3.3 et étiquette `service_name` attendue par
  `export.py` ;
- noms des métriques exportées par le Collector vers Prometheus (expression
  `PIPEFAULT_METRICS` de `export.py`) ;
- effet réel de chaque faute (c'est précisément le rôle de `bench-check`
  sur le bloc dev : une faute sans effet vérifié doit être corrigée AVANT le
  bloc test, et la correction documentée).
