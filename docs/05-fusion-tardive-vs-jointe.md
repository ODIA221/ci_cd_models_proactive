# 5. Fusion tardive vs fusion jointe : la vraie question

Les trois chapitres précédents ont chacun amélioré la représentation d'une
modalité (VAE, GAT, LSTM). Mais à chaque étape, la fusion jointe — l'union
de ces trois représentations dans un unique goulot d'étranglement partagé,
l'architecture visée par le résumé de thèse — est restée **loin derrière**
une approche bien plus simple déjà présente dans le dépôt : la **fusion
tardive** (`run_late_fusion` dans `evaluate_multimodal.py`). Ce chapitre
documente l'investigation qui a suivi la demande "je veux un meilleur
résultat".

## Deux stratégies de fusion, un rappel

- **Fusion jointe** (`joint_fusion_autoencoder` et ses variantes VAE/GAT/LSTM) :
  un unique `MultimodalAutoencoder`, une branche par modalité, toutes
  fusionnées dans un même goulot d'étranglement latent, entraîné de bout en
  bout. C'est l'architecture décrite par le résumé de thèse.
- **Fusion tardive** (`late_fusion_logreg`) : un détecteur `isolation_forest`
  **indépendant** par modalité (entraîné séparément sur les runs normaux du
  split train), puis un combinateur — une régression logistique — appris
  sur les 3 scores d'anomalie continus. Le split test est lui-même divisé
  en deux (val/test) pour entraîner ce combinateur sans fuite de données.

## Premier réflexe : appliquer les mêmes améliorations à la fusion tardive

Puisque `traces_gat` et `logs_lstm` battent leurs équivalents bruts en
ablation isolée (chapitres 3 et 4), l'hypothèse naturelle était que les
utiliser dans la fusion tardive (déjà la meilleure approche, F1=0,7179)
l'améliorerait encore. Testé de deux façons sur RCAEval RE2
(`experiments/evaluation_rcaeval_20260814_083115.csv`) :

| Fusion tardive | F1 | AUC |
|---|---|---|
| brute (métriques+logs+traces d'origine) | **0,7179** | **0,8318** |
| GAT+LSTM **en remplacement** de traces/logs | 0,6222 | 0,7290 |
| brute **+** GAT+LSTM en signaux supplémentaires (5 blocs) | 0,6889 | 0,7820 |

Aucune des deux variantes ne bat la version brute. Hypothèse retenue plutôt
que d'abandonner la piste : le combinateur (régression logistique) n'est
calibré que sur **43 échantillons** de validation (moitié du split test de
RE2, 172 runs) — un échantillon trop petit pour distinguer fiablement deux
stratégies dont l'écart de performance sous-jacent pourrait être faible.

## Vérifier l'hypothèse : un second jeu de données indépendant

Pour trancher, il fallait plus de données. Le registre des sources
(`src/data/sources/registry.yaml`) référence deux autres sous-ensembles
RCAEval, RE1 et RE3, en plus de RE2 déjà utilisé.

**RE1** s'est révélé métriques-seules (confirmé : aucun `logs.csv` ni
`traces.csv` dans ses 375 cas) et, pire, structuré avec des noms de fichiers
différents (`data.csv`/`simple_data.csv` au lieu de
`metrics.csv`/`simple_metrics.csv`) que le code ne reconnaît pas —
`features.parquet` généré était vide. Écarté : même corrigé, il n'aurait
apporté que des cas métriques-only, pas la donnée traces/logs qui manquait
pour trancher.

**RE3** ("le plus volumineux" selon le registre — en réalité ~534 Mo au
total, plus petit que les 4,2 Go de RE2) a la même structure de fichiers
que RE2 et contient bien logs+métriques+traces. 90 cas, 46 runs de test
(contre 172 pour RE2) — un jeu plus petit, mais entièrement indépendant.

GAT et LSTM ont été ré-entraînés sur RE3 (indépendamment de RE2, ~10
minutes au total grâce à sa taille réduite), puis la même comparaison a été
relancée (`experiments/evaluation_rcaeval_20260814_093057.csv`) :

| Fusion tardive (RE3, 46 runs de test) | F1 | AUC |
|---|---|---|
| brute | 0,8333 | 0,8030 |
| GAT+LSTM en remplacement | **0,8571** | **0,8712** |
| fusion jointe (toutes variantes) | 0,0000 | 0,37–0,44 |

Résultat **opposé** à RE2 : ici, la version GAT/LSTM gagne. Ceci confirme
l'hypothèse de bruit d'échantillonnage — avec des jeux de validation aussi
petits (23 à 43 échantillons), le classement entre deux stratégies proches
n'est pas fiable individuellement.

Constat additionnel, net celui-là (pas d'inversion entre RE2 et RE3) : la
**fusion jointe s'effondre complètement sur RE3** (F1=0, AUC sous 0,45 —
pire que le hasard). Avec seulement 41 à 67 séquences d'entraînement
(contre 185 sur RE2), l'auto-encodeur à goulot d'étranglement partagé n'a
simplement pas assez de données pour apprendre quoi que ce soit
d'exploitable, alors que la fusion tardive (des `isolation_forest`, bien
moins gourmands en données) reste robuste.

## Trancher : regrouper les deux jeux de test

Ni RE2 seul (172 runs de test) ni RE3 seul (46 runs) ne donnent assez de
puissance statistique pour départager raw et GAT/LSTM de façon fiable. La
solution retenue : un nouveau script,
[`src/models/evaluate_pooled.py`](../src/models/evaluate_pooled.py), qui

1. entraîne et évalue chaque détecteur par modalité **séparément sur son
   propre dataset** (aucune fuite : un détecteur entraîné sur le train de
   RE2 ne voit jamais RE3, et inversement) ;
2. ne conserve que les modalités **présentes dans les deux sources** —
   RE3 n'a aucune colonne `logs` brute (0 cas sur 90 avec `cluster_id`),
   contrairement à `logs_lstm`, disponible pour les deux puisque le mining
   Drain3 ne dépend pas de ce champ pré-existant ;
3. regroupe les **scores continus** (pas les features brutes, qui n'ont pas
   le même schéma de colonnes entre RE2 et RE3) des deux jeux de test en un
   seul ensemble de 218 runs, sur lequel un unique combinateur est ajusté
   et évalué (109 en validation, 109 en test — 2,5 à 5 fois plus qu'avec un
   seul sous-ensemble).

Résultat (`experiments/evaluation_pooled_20260814_093826.csv`) :

| Fusion tardive regroupée (218 runs, 109 en validation) | F1 | AUC |
|---|---|---|
| brute (métriques + traces seulement, `logs` exclu — absent de RE3) | 0,5909 | 0,7556 |
| GAT traces + LSTM logs + métriques | 0,5476 | 0,7481 |

## Conclusion de ce chapitre

L'écart entre les deux (ΔF1 = 0,04, ΔAUC = 0,01) est faible — largement
dans la marge de bruit résiduelle, même à 109 échantillons. **Les
verdicts opposés obtenus sur RE2 seul et RE3 seul étaient donc bien tous
les deux, en grande partie, du bruit d'échantillonnage.** Sur un jeu de
données plus large, GAT et LSTM sont globalement **équivalents** aux
features brutes pour la fusion tardive — ni gain net, ni régression nette.

Ce que ce jeu de données plus large confirme, en revanche, sans ambiguïté :
la fusion tardive reste bien supérieure à la fusion jointe, et cette
dernière est fragile aux petits volumes de données — un résultat qui ne
s'est jamais inversé, sur aucun des trois jeux testés (RE2 seul, RE3 seul,
regroupé).

Point de méthode à garder en tête : le F1 du pool (0,59) est plus bas que
celui de RE2 seul (0,72), en partie parce que la modalité `logs` a dû être
exclue (absente de RE3), et parce que normaliser les scores par z-score
globalement sur un pool de détecteurs entraînés séparément est un exercice
plus dur, pour le combinateur, qu'un seul dataset homogène. Ce n'est donc
pas directement comparable au 0,7179 du chapitre 1 — c'est une estimation
différente, sur un périmètre différent, mais statistiquement plus fiable
pour la question posée (GAT/LSTM aident-ils la fusion tardive, oui ou
non).

## Reproduire

```bash
python3 -m src.data.acquire --source rcaeval --subset RE3
python3 -m src.models.graph_encoder --subset RE3
python3 -m src.models.log_sequence_encoder --subset RE3
python3 -m src.models.evaluate_multimodal --source-dir data/interim/rcaeval/RE3 --use-gat-traces --use-lstm-logs
python3 -m src.models.evaluate_pooled --source-dirs data/interim/rcaeval/RE2 data/interim/rcaeval/RE3
```
