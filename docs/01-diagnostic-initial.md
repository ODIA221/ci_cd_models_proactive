# 1. Diagnostic initial : le résumé de thèse face au code

## Le résumé de thèse

Le point de départ de ce travail est un résumé destiné à un article/chapitre
de thèse, décrivant :

> un cadre novateur d'apprentissage profond multimodal qui fusionne [logs,
> métriques, traces] [...]. Notre architecture combine un modèle séquentiel
> basé sur LSTM pour les logs, un auto-encodeur variationnel pour les flux
> métriques, et un réseau de neurones à attention sur graphe pour la
> modélisation des dépendances de traces, le tout intégré par un mécanisme
> d'attention croisée entre modalités [...] un score F1 de 0,936 sur
> Astronomy Shop et de 0,958 sur AnoMod [...] une réduction de 91,2 % du
> temps moyen de diagnostic [...].

## Ce que le code contenait réellement

Une lecture attentive de `src/models/detection_models.py` et des scripts
d'évaluation existants (`src/models/evaluate_multimodal.py`,
`src/models/evaluate_proactive.py`, `src/models/evaluate_causal.py`) a
révélé un écart important entre ce texte et l'implémentation :

| Élément du résumé | Réalité du code (avant ce travail) |
|---|---|
| LSTM pour les logs | Sac d'événements + bigrammes (`build_sequence_features`, `src/data/features.py`) — aucune notion d'ordre séquentiel réel, aucun réseau récurrent. |
| Auto-encodeur variationnel pour les métriques | `MultimodalAutoencoder` (`detection_models.py`) : un MLP déterministe par branche, goulot d'étranglement par simple concaténation — aucune reparamétrisation, aucun terme KL. |
| GAT pour les traces | `build_traces_agg_matrix` : 6 statistiques agrégées globales (nombre de spans, durée moyenne, taux d'erreur...) — le docstring de cette fonction dit lui-même explicitement *"substitut léger à un encodeur de graphe (GAT/HGTN)"*. |
| Attention croisée entre modalités | Concaténation simple des embeddings de branche avant le goulot d'étranglement. Aucun mécanisme d'attention. |
| Jeu de données "AnoMod" | Aucune occurrence de ce nom dans tout le dépôt (recherché par `grep` récursif). |
| F1 = 0,936 / 0,958 | Mesuré réellement sur RCAEval RE2 (`experiments/evaluation_rcaeval_20260726_140759.csv`), fusion trimodale complète (`joint_fusion_autoencoder`) : **F1 = 0,3077**. |
| Ablation : la fusion trimodale bat toutes les combinaisons bimodales | Mesuré : c'est l'inverse — la modalité **traces seule** (F1 ≈ 0,67) bat systématiquement la fusion complète (F1 ≈ 0,18–0,40) dans `experiments/evaluation_proactive_20260726_153239.csv`. |
| Détection proactive dès 40-60 % de l'avancement du pipeline | L'évaluation existante (`evaluate_proactive.py`) mesure des horizons en **secondes** (15s à 720s autour de l'injection de faute), pas un pourcentage d'avancement — et le F1 ne s'améliore pas monotonement avec l'horizon. |
| Réduction de 91,2 % du temps de diagnostic | Aucun calcul de ce type dans le dépôt (`causal_eval_*.csv` mesure hit@1/hit@3 sur la localisation de cause racine, pas un temps). |

## Interprétation

Ce texte décrivait une **cible visée**, pas l'état du code au moment où ce
travail a commencé. Le présenter tel quel comme des résultats obtenus
aurait été problématique. La décision prise avec l'auteur de la thèse a été
de **se rapprocher réellement de cette vision, par étapes validées** :
implémenter chaque composant un par un (VAE, puis GAT, puis LSTM), mesurer
son effet sur des données réelles avant de passer au suivant, plutôt que
d'écrire une architecture complète d'un coup sans savoir laquelle de ses
parties apporte quoi.

Les trois chapitres suivants documentent ces trois implémentations. Le
chapitre 5 documente pourquoi, au final, la stratégie de fusion elle-même
(pas seulement la qualité de chaque encodeur) s'est révélée être la
question la plus déterminante.

## Repère chiffré : le point de départ

Avant tout changement de code, sur RCAEval RE2 (271 cas, 172 runs de test) :

| Modalité seule | F1 | AUC |
|---|---|---|
| métriques (isolation_forest) | 0,35 | 0,66 |
| logs (bag-of-events) | 0,12 | 0,50 (hasard) |
| traces (6 stats agrégées) | 0,67 | 0,78 |
| fusion trimodale jointe | 0,31–0,40 | 0,60–0,65 |
| fusion tardive (ensemble + régression logistique) | 0,72 | 0,83 |

C'est cette dernière ligne — déjà présente dans le code avant ce
travail — qui allait rester la référence la plus difficile à battre tout
au long des chapitres suivants.
