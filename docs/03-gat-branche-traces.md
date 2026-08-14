# 3. GAT pour la branche traces

Deuxième étape : remplacer les 6 statistiques agrégées de
`build_traces_agg_matrix` par un embedding appris via un réseau à attention
sur graphe (GAT), fait main en PyTorch pur (décision prise en amont :
`torch_geometric` n'est pas une dépendance du projet et n'a pas été
ajoutée).

## La découverte qui a changé la conception

L'hypothèse de départ était un graphe **span par span** : nœuds = spans
d'une trace, arêtes = relations parent/enfant (`parent_span_id`). En
inspectant les données réellement mises en cache
(`data/interim/rcaeval/RE2/traces/*.parquet`), chaque run RCAEval contient
en réalité **200 000 à 400 000 spans**. Une matrice d'adjacence dense pour
un graphe de cette taille est numériquement impossible (N² pour N≈300 000),
et un GAT épars aurait demandé des primitives de scatter/segment absentes
de PyTorch pur — exactement ce que l'usage de `torch_geometric` évite, mais
que la décision de rester "PyTorch pur" écartait.

**Solution retenue : un graphe au niveau service**, pas au niveau span :

- nœuds = les services distincts impliqués dans un run (34 au total sur
  RCAEval RE2 — Online Boutique, Sock Shop, Train Ticket confondus) ;
- arêtes = appels `parent_span_id → span_id` agrégés par paire de
  services (un appel entre deux spans de services A et B devient une arête
  A–B) ;
- features de nœud = statistiques par service (nombre d'appels, durée
  moyenne, taux d'erreur, en `log1p` pour limiter l'écrasement des
  services à fort volume) + un one-hot d'identité de service.

Un graphe à au plus 34 nœuds rend l'attention **dense** parfaitement
praticable — et cette granularité correspond à celle déjà utilisée par le
module de corrélation causale existant (`GET /explain/{run_id}` raisonne
déjà en "service suspect", pas en span individuel).

## Ce qui a été construit

`src/models/graph_encoder.py` (nouveau fichier) :

- `GATLayer` : attention de graphe multi-têtes fait main (projection
  linéaire, coefficients d'attention sur paires de nœuds concaténées,
  softmax masqué par l'adjacence, agrégation pondérée) ;
- `TraceGraphAutoencoder` : deux `GATLayer` empilées (encodeur) + décodeur
  MLP par nœud, perte Huber (même choix que
  `MultimodalAutoencoder._modality_loss`, pour la même raison de stabilité
  numérique face à des colonnes d'échelles très différentes) ;
- `build_service_graph()` : construit le graphe d'une fenêtre à partir des
  spans bruts ;
- un script CLI qui entraîne le modèle sur les graphes du split **train**
  uniquement (normal), puis encode tous les run_id disponibles (moyenne des
  embeddings de nœuds → vecteur de taille fixe, 16 dimensions) et met le
  résultat en cache dans `trace_gat_features.parquet`.

Ce précalcul est **hors ligne** (décision prise en amont, cohérente avec le
reste du dépôt) : pas d'entraînement de bout en bout avec
`MultimodalAutoencoder`, ce qui évite de toucher au contrat `/predict` de
l'API (une requête reste un vecteur plat de features, pas des spans bruts).
Limite assumée : un modèle utilisant les embeddings GAT ne peut scorer que
des run_id déjà présents dans le cache précalculé.

## Intégration

Les colonnes du cache (`trace_gat_0`...`trace_gat_15`) gardent le préfixe
`trace_` déjà reconnu partout dans le pipeline (`MODALITY_PREFIXES` dans
`train_rcaeval.py`/`evaluate_multimodal.py`) — remplacer les 6 anciennes
colonnes par les 16 nouvelles ne demande donc **aucune modification** du
code de groupement par modalité. Un flag `--use-gat-traces` déclenche ce
remplacement (`merge_gat_trace_features()`), dans `train_rcaeval.py` comme
dans `evaluate_multimodal.py`.

## Résultat mesuré

Sur RCAEval RE2 (`experiments/evaluation_rcaeval_20260813_165228.csv`) :

| Config | F1 | AUC |
|---|---|---|
| traces (6 stats agrégées, ancien) | 0,6667 | 0,7784 |
| traces_gat (GAT, nouveau) | **0,7035** | 0,7378 |
| fusion jointe (baseline) | 0,2941 | 0,6381 |
| fusion jointe + VAE métriques | 0,3519 | 0,6560 |
| fusion jointe + VAE métriques + GAT traces | **0,5806** | 0,7139 |

Le GAT seul bat déjà les 6 statistiques agrégées (meilleur rappel : 0,81 vs
0,62, au prix d'un AUC légèrement inférieur — point qui revient de façon
importante au [chapitre 5](05-fusion-tardive-vs-jointe.md)). Combiné au VAE
de l'étape précédente, la fusion jointe passe de F1=0,29 à F1=0,58 — un
quasi-doublement.

## Reproduire

```bash
python3 -m src.models.graph_encoder --subset RE2
python3 -m src.models.train_rcaeval --model-type multimodal_autoencoder --variational-modalities metrics --use-gat-traces
python3 -m src.models.evaluate_multimodal --source-dir data/interim/rcaeval/RE2 --use-gat-traces
```

Temps d'exécution mesuré pour `graph_encoder.py --subset RE2` : environ
12 minutes (271 cas, dont la construction des 241 graphes et
l'entraînement sur les 121 graphes normaux du split train).
