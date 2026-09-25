# LogPipeGuard — détection et explication d'anomalies CI/CD

Prototype de recherche (thèse, UCAD) pour détecter des anomalies dans des
systèmes livrés en continu à partir de trois sources de données — journaux,
métriques et traces — puis aider un ingénieur à en trouver la cause.

Le dépôt contient :

- une chaîne de **détection** multimodale (Isolation Forest, auto-encodeurs,
  VAE, GAT, LSTM, fusion tardive), évaluée sur le jeu public RCAEval ;
- une **couche d'exploration causale** (v2) : signaux par service, trois vues
  interactives, test d'hypothèse « et si », rapport PDF ;
- une **API FastAPI**, une **interface web** (React + D3) et un **dashboard**
  Streamlit ;
- une commande qui **vérifie tout** de bout en bout.

## Résultats mesurés

Chiffres obtenus avec ce code, sur RCAEval ; détails et limites dans [`docs/`](docs/README.md).

| Question | Résultat |
|---|---|
| Meilleure détection d'anomalies | Fusion tardive : F1 = 0,86 sur 23 exécutions de test RE2 ; 0,59 sur RE2 + RE3 regroupés (109 exécutions) |
| Localisation du service fautif | Le bon service est classé 1er dans 71 % des 86 exécutions anormales de test, dans les 3 premiers dans 95 % |
| L'attention du GAT explique-t-elle la panne ? | Non : elle est uniforme et n'apporte rien au-delà du degré des nœuds |
| Gain de temps de diagnostic pour un ingénieur | **Non mesuré** : l'étude utilisateur reste à conduire (protocole et instrumentation fournis) |

## Prérequis

| Outil | Pour quoi | Obligatoire |
|---|---|---|
| Python 3.12 | Tout le projet | Oui |
| Node.js ≥ 18 | Interface web v2 (`frontend/`) | Non — l'API et le dashboard fonctionnent sans |
| Google Chrome | Contrôle de rendu de `./run.sh verify` | Non — contrôle ignoré s'il est absent |
| Docker | Démo OpenTelemetry, Jenkins local | Non |
| ~5 Go de disque | Jeu RCAEval RE2 | Pour les fonctions RCAEval |

`run.sh` crée l'environnement virtuel `.venv` et installe `requirements.txt`
à chaque lancement. Si Node est installé via nvm, `run.sh` le charge lui-même.

## Démarrage rapide

```bash
./run.sh start
```

Une seule commande : elle construit l'interface web si nécessaire, démarre
l'API en arrière-plan, puis le dashboard, et ouvre le navigateur.

| Adresse | Contenu |
|---|---|
| http://localhost:8000/ui/ | Interface d'exploration causale (React + D3) |
| http://localhost:8501 | Dashboard Streamlit (détection, chaîne causale, mode étude) |
| http://localhost:8000/docs | API et documentation interactive (Swagger) |

`Ctrl+C` arrête tout. En cas de processus resté actif : `./run.sh stop`.

## Lancer l'interface web (frontend)

Trois façons, selon l'usage :

| Usage | Commande | Adresse |
|---|---|---|
| Tout lancer d'un coup | `./run.sh start` | http://localhost:8000/ui/ |
| API seule, interface déjà construite | `./run.sh ui-build` (une fois), puis `./run.sh serve` | http://localhost:8000/ui/ |
| Développer l'interface (rechargement à chaud) | `./run.sh ui-dev` | http://localhost:5173/ui/ |

En mode développement, chaque modification de `frontend/src/` s'affiche sans
recharger la page, et les appels API sont relayés vers le port 8000.
`ui-dev` démarre l'API si elle ne tourne pas déjà, et l'arrête avec `Ctrl+C`.
Sans `run.sh`, les commandes équivalentes sont (API lancée à part avec
`./run.sh serve`) :

```bash
cd frontend
npm install
npm run dev        # développement, http://localhost:5173/ui/
npm run build      # production, dans frontend/dist/ (servi par l'API sous /ui/)
npm run typecheck  # vérification TypeScript
```

Un lien direct ouvre une exécution et un service déjà sélectionnés :
`http://localhost:8000/ui/?run=RE2-OB/checkoutservice_disk/2__abnormal&service=checkout`.

## Préparer les données (première installation)

Les données et modèles ne sont pas versionnés. Sur une nouvelle machine :

```bash
./run.sh acquire --source rcaeval --subset RE2              # ~4,2 Go, une seule fois
./run.sh showcase-rcaeval                                   # entraîne 3 modèles servables par l'API
python3 -m src.models.graph_encoder --subset RE2 --save-model-only   # modèle GAT (~20 min)
./run.sh evaluate-causal --compare-rankers                  # facultatif : précalcule les 172 exécutions (~1 h)
```

Sans la dernière commande, les signaux d'une exécution sont calculés à son
premier affichage (quelques secondes à ~1 min), puis mis en cache.

## Tout vérifier

```bash
./run.sh verify          # ~2 min
./run.sh verify --full   # + refait l'évaluation et compare aux chiffres publiés (~10 min)
```

Contrôle l'environnement, les données, le modèle GAT, les signaux causaux,
chaque route de l'API (sur une instance de test séparée), le typage et le build
du frontend, son rendu réel dans Chrome, le parcours du dashboard et les
scripts d'analyse. Ne modifie aucune donnée. Code de sortie non nul en cas
d'échec.

## Commandes `run.sh`

| Commande | Rôle |
|---|---|
| `start` / `stop` | Tout lancer / tout arrêter |
| `serve` | API FastAPI seule (port 8000) |
| `dashboard` | Dashboard Streamlit seul (port 8501, nécessite `serve`) |
| `ui-build` | Construit l'interface web v2 |
| `ui-dev` | Interface web v2 en mode développement (port 5173, démarre l'API si besoin) |
| `verify [--full]` | Vérification de bout en bout |
| `setup` | Installe seulement l'environnement Python |
| `demo` (défaut) | Démo de détection sur les données d'exemple |
| `sources` / `acquire <args>` | Liste / télécharge les sources de données externes |
| `showcase-rcaeval` | Acquiert RCAEval RE2 et entraîne les modèles de démonstration |
| `train-rcaeval <args>` | Entraîne un modèle RCAEval servable |
| `evaluate` | Évaluation sur LogHub HDFS |
| `evaluate-multimodal <args>` | Fusion tardive vs jointe vs mono-modalité |
| `evaluate-causal [--compare-rankers]` | Localisation de la cause racine (P@1, P@3) |
| `evaluate-proactive <args>` | Détection à horizon court (15 s à 720 s) |
| `study-analysis` | Analyse des sessions d'étude utilisateur enregistrées |
| `otel-up` / `otel-down` | Démo OpenTelemetry (Docker) |
| `jenkins-up` / `jenkins-down` | Jenkins local (Docker) |

## API

| Route | Rôle |
|---|---|
| `GET /health`, `/models`, `/sources` | État, modèles entraînés, sources de données |
| `POST /predict` | Score d'anomalie d'une ou plusieurs exécutions |
| `GET /explain/{run_id}` | Chaîne causale par l'heuristique d'arbre de spans (v1) |
| `GET /runs` | Exécutions de test explorables |
| `GET /causal/{run_id}` | Signaux par service (score, attention, journaux, instants de déviation) |
| `GET /causal/{run_id}/whatif?hypothesis=…` | Test d'hypothèse de cause racine |
| `GET /causal/{run_id}/report.pdf` | Rapport de diagnostic en PDF, avec graphiques |
| `GET /causal/{run_id}/report` | Même rapport en Markdown |
| `GET /causal/{run_id}/jsonld` | Signaux en JSON-LD |
| `GET /study/tasks`, `POST /study/sessions` | Instrumentation de l'étude utilisateur |

Prototype de recherche : pas d'authentification, usage local uniquement.

## Structure du dépôt

```
src/
  data/        chargement, prétraitement, connecteurs (RCAEval, LogHub, GitHub Actions, GitLab CI, Jenkins, OTel…)
  models/      détecteurs, GAT (graph_encoder), LSTM, scripts d'entraînement et d'évaluation
  causal/      signaux par service, corrélation causale, rapport PDF, analyse d'étude
  api/         API FastAPI
  dashboard/   dashboard Streamlit
  verify.py    vérification de bout en bout
frontend/      interface web v2 (React 18, TypeScript, D3 v7)
docs/          documentation détaillée, chapitre par chapitre
notebooks/     explorations (non synchronisées avec src/)
run.sh         point d'entrée unique
```

## Documentation

La [documentation](docs/README.md) se lit comme une suite de chapitres : cours
sur les notions utilisées, diagnostic initial, VAE, GAT, LSTM, comparaison des
fusions, conclusions, puis la [couche d'exploration causale](docs/07-couche-exploration-causale-v2.md).

## Limites

- Les données annotées sont des **microservices** (RCAEval), pas des pipelines
  CI/CD : les nœuds analysés sont des services, pas des étapes de build/test/déploiement.
- Les pannes évaluées sont injectées sur un seul service à la fois.
- Le score par service est une heuristique : sur des fenêtres sans panne, il
  signale encore 2 services en médiane.
- Aucune affirmation sur le temps de diagnostic humain avant l'étude utilisateur.
