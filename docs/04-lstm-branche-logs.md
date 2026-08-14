# 4. LSTM pour la branche logs

Troisième étape, et la plus mouvementée : remplacer le sac
d'événements/bigrammes par un auto-encodeur séquentiel (LSTM) sur la vraie
séquence ordonnée de templates de log. Cette étape a révélé un bug de
données préexistant, puis a nécessité trois corrections successives avant
d'aboutir — documentées ici en détail parce qu'elles sont instructives au
même titre que le résultat final.

## La découverte : la modalité logs était presque vide

Avant de commencer à coder, une vérification des données brutes a révélé
pourquoi la modalité `logs` obtenait un AUC ≈ 0,50 (quasi aléatoire) dans
toutes les évaluations précédentes : le commentaire de
`src/data/sources/rcaeval.py` affirmait que `cluster_id` (le template de
log déjà calculé) était *"déjà présent... pas besoin de Drain3 ici"*.

En lisant les en-têtes des 271 fichiers `logs.csv` bruts :

| Système | Cas avec `cluster_id` | Cas sans |
|---|---|---|
| Online Boutique (OB) | 25 | 66 |
| Sock Shop (SS) | 0 | 90 |
| Train Ticket (TT) | 0 | 90 |
| **Total** | **25 / 271 (9 %)** | **246 / 271 (91 %)** |

`parse()` ignore silencieusement les logs quand `cluster_id` est absent —
les colonnes `event_*`/`2gram_*` étaient donc à 0 pour 246 runs sur 271.
Le commentaire erroné a été corrigé dans le code ; la bonne nouvelle est
que `src/data/log_parsing.py::mine_templates()` (Drain3) existait déjà dans
le dépôt, écrit pour d'autres connecteurs (Jenkins, GitLab CI) — il ne
restait qu'à l'appliquer à RCAEval.

## Conception retenue

Une seule instance `TemplateMiner` (Drain3) partagée sur les 271 cas, dans
un ordre déterministe, pour un vocabulaire de templates cohérent d'un run à
l'autre (le `cluster_id` pré-existant est ignoré : le réutiliser aurait
mélangé deux numérotations incohérentes entre les 25 cas qui l'ont et les
246 qui ne l'ont pas).

`LogSequenceAutoencoder` (`src/models/log_sequence_encoder.py`) :
Embedding → LSTM encodeur (dernier état caché → vecteur latent) → LSTM
décodeur en **teacher forcing décalé** : le décodeur ne reçoit jamais en
entrée le token qu'il doit prédire, seulement ses prédécesseurs + le
vecteur latent comme état initial. Sans ce décalage, un décodeur qui voit
directement sa propre cible peut simplement la recopier et contourner
totalement le goulot d'étranglement — le décalage force l'information à
passer par le vecteur compressé.

## Incident 1 : le vocabulaire de templates explose

Le premier lancement a montré une dérive brutale à la transition entre
systèmes :

| Cas traités | Vocabulaire | Temps pour 20 cas |
|---|---|---|
| 20 → 80 (Online Boutique) | 113 → 144 templates | ~45 secondes |
| 80 → 100 (premier cas Sock Shop) | 144 → **26 339** templates | **29 minutes** |

Cause : les trois systèmes ont des formats de log radicalement différents
(logs structurés Go pour Sock Shop `ts=... caller=... method=...`, stack
traces Java/Spring, payloads JSON à texte libre type lorem ipsum pour les
descriptions de produits, IDs Mongo hexadécimaux). Drain3, dont l'arbre de
clustering discrimine notamment par nombre de tokens, ne parvient pas à
regrouper ces formats — chaque ligne un peu différente devient un nouveau
cluster, et la recherche dans l'arbre ralentit d'autant plus que le nombre
de clusters connus augmente.

**Correction** : `drain_max_clusters = 1000` dans `TemplateMinerConfig` —
un cache LRU qui borne le nombre de templates activement comparés pour le
matching. Résultat : le même intervalle (cas 80 → 100) est passé de 29
minutes à 2,4 minutes.

## Incident 2 : la borne LRU ne borne pas le vocabulaire

Après cette première correction, le mining s'est terminé, mais
l'entraînement a immédiatement crashé (`IndexError: index out of range`).
En lisant le code source de Drain3 (`drain3/drain.py`) : `max_clusters`
limite uniquement un cache LRU (`id_to_cluster`) utilisé pour le matching —
mais `clusters_counter`, le compteur qui **attribue** les identifiants de
cluster, continue de grimper sans limite à chaque nouveau cluster créé,
qu'il soit ensuite gardé en cache ou évincé. Le calcul de la taille du
vocabulaire (`len(miner.drain.clusters)`, qui ne reflète que le cache
borné) sous-estimait donc largement le plus grand identifiant réellement
utilisé dans les séquences.

**Correction** : calcul de la taille du vocabulaire directement à partir du
plus grand token effectivement présent dans les séquences collectées,
sans dépendre d'un état interne de Drain3. Vocabulaire réel trouvé :
**272 534 templates**.

## Incident 3 : un vocabulaire de sortie trop grand pour être entraîné

Avec ce vocabulaire correctement calculé, l'entraînement était numériquement
correct mais **computationnellement infaisable** : la couche de sortie du
décodeur (`Linear(32, 272534)`) produit, pour un batch de 32 séquences de
200 pas de temps, un tenseur de logits de `32 × 200 × 272534` éléments —
et un softmax de cette taille à calculer à chaque pas de temps, pour
chaque batch, à chaque époque. Le processus a tourné plus de 2 heures sans
qu'une seule époque ne se termine, avant d'être arrêté manuellement.

**Correction** : plafonnement du vocabulaire du **modèle** (indépendamment
de celui de Drain3) aux `vocab_cap` (500 par défaut) templates les plus
fréquents du split train, le reste étant regroupé sous un unique token
"inconnu" (UNK) — pratique standard pour les vocabulaires à longue traîne
en NLP. Avec ce plafond, l'entraînement complet (50 époques) prend
**20 secondes** au lieu de plus de 2 heures.

En prévention d'un futur échec similaire, le résultat coûteux du mining
(~40 minutes sur RE2) est désormais mis en cache sur disque
(`_log_sequences_cache.pkl`) — un nouvel échec dans l'étape d'entraînement
ne demande plus de refaire le mining.

## Résultat mesuré

Sur RCAEval RE2, après ces trois corrections
(`experiments/evaluation_rcaeval_20260813_221630.csv`) :

| Config | F1 | AUC |
|---|---|---|
| logs (bag-of-événements, ancien — vide pour 91 % des runs) | 0,1224 | 0,5001 (hasard) |
| logs_lstm (nouveau, séquences réelles, vocabulaire plafonné) | **0,3803** | 0,5174 |
| fusion jointe, VAE + GAT | 0,5760 | 0,7281 |
| fusion jointe, VAE + GAT + LSTM | 0,4538 | 0,6616 |

Le LSTM seul bat nettement l'ancien sac d'événements (F1 0,38 contre 0,12),
mais son signal reste faible en absolu (AUC 0,517, à peine au-dessus du
hasard) — et l'ajouter à la fusion jointe **dégrade** le résultat par
rapport à VAE+GAT seuls. Ce résultat individuel s'est révélé, avec plus de
données, largement dû au bruit d'échantillonnage — voir le
[chapitre 5](05-fusion-tardive-vs-jointe.md).

## Reproduire

```bash
python3 -m src.models.log_sequence_encoder --subset RE2
python3 -m src.models.train_rcaeval --model-type multimodal_autoencoder --variational-modalities metrics --use-gat-traces --use-lstm-logs
python3 -m src.models.evaluate_multimodal --source-dir data/interim/rcaeval/RE2 --use-gat-traces --use-lstm-logs
```

Temps d'exécution mesuré pour `log_sequence_encoder.py --subset RE2` :
environ 40 minutes, presque entièrement dominées par le mining Drain3 sur
44 millions de lignes de log (l'entraînement du LSTM lui-même, une fois le
vocabulaire plafonné, ne prend que quelques secondes).
