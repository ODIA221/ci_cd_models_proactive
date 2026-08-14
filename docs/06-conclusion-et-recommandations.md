# 6. Conclusion et recommandations

## Ce qui est validé

**La fusion tardive (ensemble par modalité + régression logistique) est la
meilleure approche mesurée pour la détection d'anomalies multimodale sur
RCAEval**, et de loin la plus robuste : c'est la seule des deux stratégies
qui reste fonctionnelle sur un petit jeu de données (RE3, F1=0,83–0,86)
comme sur un plus grand (RE2, F1=0,72), et validée par un pool des deux
(F1=0,55–0,59 selon la variante, sur 109 runs de test).

**L'architecture décrite par le résumé de thèse (fusion jointe VAE+GAT+LSTM
dans un goulot d'étranglement partagé) progresse réellement grâce aux
composants ajoutés** (F1 0,29 → 0,58 sur RE2 en ajoutant VAE puis GAT), mais
**reste systématiquement derrière la fusion tardive**, et **s'effondre**
sur un jeu de données plus petit (RE3 : F1=0, pire que le hasard). C'est un
résultat négatif honnête, mais informatif : la valeur ajoutée du résumé de
thèse ne peut pas se soutenir sur le seul argument "architecture plus
sophistiquée = meilleurs résultats" — sur ce problème et ce volume de
données, c'est mesurablement faux.

**GAT (traces) et LSTM (logs) sont des améliorations réelles par rapport
aux anciennes features**, mesurées en ablation isolée (traces : F1 0,67→0,70 ;
logs : F1 0,12→0,38, AUC quasi-hasard→légèrement au-dessus). Mais leur effet
une fois intégrées dans un combinateur de fusion tardive n'est **pas
distinguable du bruit** avec les volumes de données actuels (RE2+RE3
combinés : ΔF1=0,04, dans les deux sens selon le sous-ensemble pris
isolément).

## Ce qui reste ouvert

- **L'attention croisée** (le 4e composant du résumé de thèse) n'a pas été
  implémentée : après le constat que la fusion apprise (jointe) perd
  systématiquement face à un simple ensemble, il n'y avait plus de
  justification empirique à investir dans un mécanisme de fusion appris
  plus sophistiqué encore, sans d'abord comprendre pourquoi la fusion
  jointe sous-performe (voir piste ci-dessous).
- **Pourquoi la fusion jointe s'effondre sur peu de données** n'a pas été
  diagnostiqué en profondeur (latent_dim/branch_latent_dim trop
  restrictifs ? seuil de reconstruction mal calibré avec peu d'exemples
  train ? autre chose ?). Une piste pour la suite.
- **RE1** (375 cas métriques-seules) reste inutilisable : bug de noms de
  fichiers non corrigé (`data.csv`/`simple_data.csv` non reconnus par
  `rcaeval.py::_load_case_metrics`). Correctif probable : ajouter ces noms
  à la liste déjà présente (`simple_metrics.csv`, `metrics.csv`).
- La comparaison GAT/LSTM vs brut sur le pool RE2+RE3 (218 runs) reste basée
  sur un seul split val/test (seed=42) — une validation croisée (k-fold)
  donnerait une estimation encore plus fiable, mais n'a pas été faite ici.

## Recommandation pratique

Pour un modèle **utilisable en pratique** aujourd'hui (servi par l'API,
`./run.sh serve`), la fusion tardive brute reste le choix le plus robuste
et le mieux validé (F1=0,72 sur RE2). Les embeddings GAT/LSTM sont
disponibles et utilisables (`--use-gat-traces`, `--use-lstm-logs`) mais
n'apportent pas, à ce stade, une amélioration démontrée suffisamment
fiable pour justifier de les préférer par défaut.

Pour la **suite du travail de thèse**, deux directions se dégagent des
résultats de cette session :

1. Documenter honnêtement, dans le manuscrit, que l'architecture à fusion
   jointe (la vision initiale) n'est pas compétitive sur ce jeu de données
   à ce volume — c'est en soi une contribution scientifique (pourquoi la
   fusion apprise échoue là où un ensemble simple réussit est une question
   ouverte intéressante), pas un échec à cacher.
2. Si l'objectif reste de faire fonctionner la fusion jointe, chercher
   d'abord plus de données d'entraînement (RE1 corrigé + RE2 + RE3
   combinés en train, pas seulement en test) avant d'ajouter de la
   complexité architecturale supplémentaire (attention croisée) — la
   fragilité observée sur RE3 pointe vers un problème de volume de
   données, pas nécessairement de capacité du modèle.

## Récapitulatif des fichiers créés ou modifiés

| Fichier | Nature |
|---|---|
| `src/models/detection_models.py` | Modifié — `MultimodalAutoencoder` accepte `variational_modalities` (VAE) |
| `src/models/graph_encoder.py` | Nouveau — GAT pour les traces |
| `src/models/log_sequence_encoder.py` | Nouveau — LSTM pour les logs |
| `src/models/evaluate_pooled.py` | Nouveau — fusion tardive évaluée sur plusieurs sous-ensembles regroupés |
| `src/models/train_rcaeval.py` | Modifié — flags `--variational-modalities`, `--use-gat-traces`, `--use-lstm-logs` |
| `src/models/evaluate_multimodal.py` | Modifié — ablations et variantes de fusion pour GAT/LSTM |
| `src/api/model_registry.py` | Modifié — reconstruction correcte des modèles VAE au chargement |
| `src/data/sources/rcaeval.py` | Modifié — commentaire erroné sur `cluster_id` corrigé |
| `data/interim/rcaeval/RE2/trace_gat_features.parquet` | Généré — embeddings GAT (241 runs) |
| `data/interim/rcaeval/RE2/event_lstm_features.parquet` | Généré — embeddings LSTM (356 runs) |
| `data/interim/rcaeval/RE3/*` | Généré — RE3 acquis, parsé, GAT+LSTM entraînés |
| `experiments/evaluation_rcaeval_*.csv`, `evaluation_pooled_*.csv` | Générés — résultats bruts de chaque expérience citée dans cette documentation |

Voir aussi la note de suivi sur le suivi Git de ces fichiers :
[`git-tracking.md`](git-tracking.md) — un problème de configuration
distinct de ce travail, découvert en le documentant.
