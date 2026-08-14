# 0. Cours : les concepts et algorithmes derrière la solution

Ce chapitre est différent des autres : ce n'est pas un compte-rendu de ce
qui a été fait, mais un **cours** qui explique, à partir de zéro, chaque
concept et chaque algorithme mobilisé dans les chapitres 1 à 6. L'objectif
est qu'après l'avoir lu, une phrase comme *"un VAE sur la branche métriques,
un GAT sur la branche traces, un LSTM sur la branche logs, fusionnés par un
combinateur logistique plutôt qu'un goulot d'étranglement partagé"*
devienne entièrement compréhensible, terme par terme.

Chaque section part d'une intuition simple avant d'introduire le
vocabulaire technique, et relie systématiquement la théorie au code réel du
dépôt (`src/models/...`).

## Table des matières

1. [Le domaine : observabilité et anomalies dans un pipeline CI/CD](#1-le-domaine-observabilité-et-anomalies-dans-un-pipeline-cicd)
2. [La détection d'anomalies non supervisée](#2-la-détection-danomalies-non-supervisée)
3. [Les trois détecteurs de base](#3-les-trois-détecteurs-de-base)
4. [Auto-encodeurs et auto-encodeurs variationnels (VAE)](#4-auto-encodeurs-et-auto-encodeurs-variationnels-vae)
5. [Graphes et réseaux à attention de graphe (GAT)](#5-graphes-et-réseaux-à-attention-de-graphe-gat)
6. [Séquences et réseaux récurrents (LSTM)](#6-séquences-et-réseaux-récurrents-lstm)
7. [Le parsing de logs et Drain3](#7-le-parsing-de-logs-et-drain3)
8. [Fusionner plusieurs modalités : fusion jointe vs fusion tardive](#8-fusionner-plusieurs-modalités-fusion-jointe-vs-fusion-tardive)
9. [Évaluer un modèle : métriques et pièges statistiques](#9-évaluer-un-modèle-métriques-et-pièges-statistiques)
10. [Glossaire des abréviations](#10-glossaire-des-abréviations)

---

## 1. Le domaine : observabilité et anomalies dans un pipeline CI/CD

**CI/CD** (*Continuous Integration / Continuous Deployment*, intégration
continue / déploiement continu) désigne les chaînes automatisées qui
compilent, testent et déploient un logiciel à chaque changement de code.
Quand un pipeline CI/CD échoue ou se comporte anormalement (lenteur,
erreurs en cascade), on veut le détecter vite — et si possible comprendre
**pourquoi**.

Pour observer ce qui se passe dans un système distribué (typiquement, une
application découpée en plusieurs **microservices** qui s'appellent entre
eux via le réseau), on dispose de trois grandes familles de données,
appelées les "trois piliers de l'observabilité" :

- **Métriques** (*metrics*) : des séries de nombres dans le temps — usage
  CPU, mémoire, latence moyenne, nombre de requêtes par seconde. Une valeur
  par instant, facile à agréger (moyenne, écart-type...).
- **Logs** : des messages texte horodatés émis par le code
  (`"paiement autorisé"`, `"connexion refusée"`...). Une ligne par
  événement, riche en contenu mais moins structurée.
- **Traces** : la trajectoire complète d'une requête à travers les
  différents microservices qu'elle traverse. Une trace est composée de
  **spans** (un span = un appel, avec un service, une durée, un statut, et
  un lien vers son span "parent" — l'appel qui l'a déclenché).

Une **anomalie**, ici, c'est un comportement du système qui s'écarte de son
fonctionnement normal — une panne, une latence anormale, une erreur en
cascade. **RCA** (*Root Cause Analysis*, analyse de cause racine) désigne le
travail qui suit la détection : une fois l'anomalie repérée, quel service,
au juste, en est la cause ?

Le jeu de données utilisé dans ce projet, **RCAEval**, fournit exactement
ça : des cas de panne réels injectés dans trois applications de démonstration
microservices (Online Boutique, Sock Shop, Train Ticket), avec pour chaque
cas les métriques, logs et traces autour de l'incident, plus l'étiquette de
la vraie cause racine.

---

## 2. La détection d'anomalies non supervisée

### Pourquoi "non supervisée" ?

En apprentissage **supervisé**, on entraîne un modèle avec des exemples
étiquetés ("ceci est normal", "ceci est une anomalie") et il apprend à
reconnaître le motif qui les distingue. Le problème : les vraies anomalies
sont rares et souvent inédites (une nouvelle panne ne ressemble à aucune
panne déjà vue) — on ne peut pas compter sur un grand jeu d'exemples
étiquetés d'anomalies.

L'approche **non supervisée** retenue ici contourne le problème : on
entraîne le modèle **uniquement sur des données normales** (le
comportement habituel du système), et on lui apprend à reconnaître *ce à
quoi ressemble le normal*. Au moment de l'évaluation, tout ce qui
s'éloigne trop de ce que le modèle a appris comme "normal" est signalé
comme anomalie — sans jamais avoir montré au modèle un seul exemple de
panne pendant son entraînement.

C'est pour cette raison que, dans tout le code (`train_rcaeval.py`,
`evaluate_multimodal.py`...), on voit systématiquement :

```python
X_train = features[split == "train"]      # uniquement des runs NORMAUX
X_test = features[split.isin(["test_normal", "test_abnormal"])]  # les deux, à l'évaluation
```

### Score d'anomalie et seuil

La plupart des modèles utilisés ici ne renvoient pas directement
"normal"/"anomalie", mais un **score continu d'anomalie** : plus il est
élevé, plus l'exemple ressemble peu à ce que le modèle connaît. On choisit
ensuite un **seuil** au-delà duquel on décide "anomalie" — dans ce projet,
le seuil est calculé comme le 95e percentile des scores obtenus sur les
données d'entraînement (normales) : c'est-à-dire "un score plus élevé que
95 % de ce que j'ai vu de normal, c'est suspect".

Un score continu est aussi ce qui permet de **classer/prioriser** des
alertes (la plus suspecte en premier) plutôt que de se limiter à un verdict
binaire — voir `AnomalyDetector.anomaly_score()` dans
`src/models/detection_models.py`.

---

## 3. Les trois détecteurs de base

Avant VAE/GAT/LSTM, le dépôt utilise trois algorithmes de détection
d'anomalies classiques (bibliothèque `scikit-learn`), qui servent de
briques de base partout, y compris dans la fusion tardive du chapitre 5.

### Isolation Forest (forêt d'isolement)

Idée : un point anormal est **facile à isoler** du reste des données. En
construisant des arbres de décision aléatoires qui découpent l'espace des
données au hasard, un point normal (entouré de beaucoup d'autres points
semblables) demande beaucoup de découpes avant d'être isolé seul dans sa
branche, alors qu'un point anormal (isolé, atypique) se retrouve isolé en
très peu de découpes. Le score d'anomalie est basé sur cette profondeur
moyenne d'isolement, sur une forêt de plusieurs arbres.

### One-Class SVM (SVM à une classe)

Une variante des machines à vecteurs de support (*Support Vector Machine*,
SVM) adaptée à n'avoir vu qu'une seule classe (le "normal") à
l'entraînement : le modèle apprend une frontière qui englobe le plus
étroitement possible les données normales dans l'espace des features ; tout
ce qui tombe en dehors de cette frontière est jugé anormal.

### Autoencoder (auto-encodeur) classique

Un réseau de neurones entraîné à **reconstruire son entrée** en la faisant
passer par un goulot d'étranglement (une couche de dimension réduite, qui
force le réseau à compresser l'information). Concrètement : un **encodeur**
compresse l'entrée en un vecteur latent de petite taille, un **décodeur**
tente de reconstruire l'entrée d'origine à partir de ce vecteur.

Entraîné uniquement sur des données normales, l'auto-encodeur devient bon
pour compresser/reconstruire ce type de données — mais **mauvais** pour
reconstruire quelque chose de différent (une anomalie), qu'il n'a jamais
vu. L'**erreur de reconstruction** (l'écart entre l'entrée et sa
reconstruction) devient donc, elle-même, le score d'anomalie : plus elle
est grande, plus l'exemple est anormal.

C'est la base de `Autoencoder` et `MultimodalAutoencoder` dans
`src/models/detection_models.py`, et donc de tout ce qui suit.

---

## 4. Auto-encodeurs et auto-encodeurs variationnels (VAE)

### De l'auto-encodeur classique au VAE

L'auto-encodeur classique (section 3) apprend un vecteur latent
**unique et déterministe** pour chaque entrée : la même entrée donne
toujours exactement le même vecteur. Un **VAE** (*Variational
AutoEncoder*, auto-encodeur variationnel) change ce principe : au lieu
d'apprendre un point unique, l'encodeur apprend une **distribution de
probabilité** (concrètement, une loi normale/gaussienne, décrite par sa
moyenne `mu` et sa variance, exprimée en log — `logvar`) pour chaque
entrée. Le vecteur latent effectivement utilisé est ensuite **échantillonné
au hasard** dans cette distribution.

Pourquoi faire ça ? Deux raisons principales :

1. **Régularisation** : forcer les distributions latentes à rester proches
   d'une loi normale standard (moyenne 0, variance 1) — via un terme de
   pénalité appelé **divergence KL** (voir plus bas) — empêche le modèle
   d'apprendre un espace latent "en miettes" (un point isolé par exemple
   d'entraînement, sans structure) ; ça encourage un espace latent plus
   lisse et continu, où des entrées similaires ont des représentations
   proches.
2. **Robustesse** : puisque le décodeur doit reconstruire correctement à
   partir de n'importe quel point tiré au hasard *autour* de `mu` (pas
   seulement `mu` exactement), il apprend une représentation plus
   généraliste, moins sur-ajustée à chaque exemple d'entraînement précis.

### Le terme technique clé : la reparamétrisation

On ne peut pas faire descendre un gradient (nécessaire pour entraîner un
réseau de neurones) à travers une opération d'échantillonnage aléatoire
directement. L'astuce, appelée **reparametrization trick** (astuce de
reparamétrisation), consiste à réécrire l'échantillonnage comme :

```
z = mu + epsilon * exp(0.5 * logvar)
```

où `epsilon` est un nombre aléatoire tiré d'une loi normale standard, **en
dehors** du réseau (donc pas besoin de faire passer un gradient à travers
lui). `mu` et `logvar`, eux, sont produits par le réseau et reçoivent
normalement leur gradient. C'est exactement ce qui est implémenté dans
`MultimodalAutoencoder.forward()` (`src/models/detection_models.py`) :

```python
if self.training:
    std = torch.exp(0.5 * logvar)
    z_mod = mu + torch.randn_like(std) * std
else:
    z_mod = mu   # à l'inférence : déterministe, pas d'échantillonnage
```

### La divergence KL (Kullback-Leibler)

La **divergence KL** mesure l'écart entre deux distributions de
probabilité — ici, entre la distribution apprise par l'encodeur
(`mu`, `logvar`) et une loi normale standard (moyenne 0, variance 1),
qu'on utilise comme référence "neutre". Sa formule, pour une gaussienne :

```
KL = -0.5 * (1 + logvar - mu² - exp(logvar))
```

Ce terme est ajouté à la perte d'entraînement (en plus de l'erreur de
reconstruction habituelle), pondéré par un facteur `kl_weight` : plus ce
poids est élevé, plus le modèle est poussé à garder son espace latent
proche de la référence neutre, au risque de sacrifier un peu de qualité de
reconstruction (compromis classique en pratique VAE, parfois appelé
**β-VAE** quand on ajuste ce poids).

Piège rencontré dans ce projet (documenté au
[chapitre 2](02-vae-branche-metriques.md)) : `exp(logvar)` peut exploser
numériquement si `logvar` n'est pas borné, provoquant une perte `NaN` —
d'où le `torch.clamp(logvar, -6, 6)` ajouté dans le code.

### ELBO, en une phrase

La quantité que le VAE cherche à maximiser au total (perte de
reconstruction + terme KL, avec un signe inversé) s'appelle en théorie
l'**ELBO** (*Evidence Lower BOund*, borne inférieure de la vraisemblance) —
un terme que vous croiserez dans toute la littérature VAE, même s'il n'est
pas explicitement nommé ainsi dans le code du dépôt (qui manipule
directement "perte de reconstruction + poids × KL").

---

## 5. Graphes et réseaux à attention de graphe (GAT)

### Qu'est-ce qu'un graphe, ici ?

Un **graphe** est une structure de données faite de **nœuds** (les
éléments) et d'**arêtes** (les relations entre eux). Dans ce projet
(`src/models/graph_encoder.py`), un graphe représente, pour un run donné,
les **services** impliqués (les nœuds) et **qui appelle qui** (les arêtes,
déduites des relations parent/enfant entre spans de trace).

Une matrice d'**adjacence** est une façon simple de représenter les arêtes :
un tableau N×N (N = nombre de nœuds) où la case (i, j) vaut "vrai" si le
nœud i est relié au nœud j. Elle est dite **dense** quand on la stocke
entièrement (taille N², adaptée à un petit nombre de nœuds — ici, au plus
34 services) par opposition à une représentation **éparse** (*sparse*,
qui ne stocke que les arêtes existantes — nécessaire dès que N devient très
grand, ce qui aurait été le cas avec un graphe au niveau des spans plutôt
que des services, cf. [chapitre 3](03-gat-branche-traces.md)).

### Réseaux de neurones sur graphes et passage de messages

Un **GNN** (*Graph Neural Network*, réseau de neurones sur graphe) apprend
une représentation de chaque nœud en tenant compte de ses **voisins** dans
le graphe — l'intuition est le **passage de messages** (*message passing*) :
à chaque étape, chaque nœud "regarde" ses voisins directs, agrège leur
information, et met à jour sa propre représentation. En empilant plusieurs
étapes/couches, l'information peut circuler plus loin dans le graphe
(voisins de voisins, etc.).

### L'attention : pondérer les voisins différemment

Dans un GNN "simple", chaque voisin compte de la même façon (une moyenne).
Un **GAT** (*Graph Attention Network*, réseau à attention de graphe) va
plus loin : il apprend, pour chaque paire de nœuds reliés, un **poids
d'attention** — combien ce voisin précis doit compter pour ce nœud précis,
appris et différent d'une paire à l'autre. Concrètement (papier original de
Veličković et al., 2018, ré-implémenté à la main ici) :

1. Chaque nœud est projeté linéairement dans un nouvel espace (`W · h`).
2. Pour chaque paire de nœuds voisins (i, j), un score d'attention brut
   `e_ij` est calculé à partir de leurs représentations projetées
   concaténées.
3. Ces scores sont normalisés par un **softmax** sur tous les voisins d'un
   même nœud — ce qui donne des poids qui somment à 1, comme des
   pourcentages d'importance.
4. La nouvelle représentation du nœud est la moyenne pondérée (par ces
   poids d'attention) des représentations projetées de ses voisins.

**Attention multi-têtes** (*multi-head*) : plutôt qu'un seul jeu de poids
d'attention, on en calcule plusieurs en parallèle ("têtes"), chacune
pouvant apprendre à se concentrer sur un aspect différent de la relation
entre nœuds, puis on concatène ou moyenne leurs résultats. C'est le rôle du
paramètre `n_heads` dans `GATLayer` (`src/models/graph_encoder.py`).

### Pourquoi "fait main" et sans bibliothèque dédiée

Des bibliothèques comme `torch_geometric` fournissent des implémentations
de GAT prêtes à l'emploi, optimisées pour de très grands graphes épars.
Ici, la décision a été de l'implémenter directement en PyTorch pur (sans
nouvelle dépendance), rendue possible par le choix de travailler sur un
graphe au niveau **service** (au plus 34 nœuds) plutôt qu'au niveau
**span** (jusqu'à 400 000 nœuds par run) — voir le
[chapitre 3](03-gat-branche-traces.md) pour l'histoire complète de cette
découverte.

### Encodeur de graphe et pooling

Une fois que chaque **nœud** a sa propre représentation (après une ou
plusieurs couches GAT), il faut une représentation pour le **graphe entier**
(un vecteur unique par run, pour rejoindre les autres modalités). La
méthode utilisée ici est le **mean pooling** : la simple moyenne des
représentations de tous les nœuds. C'est le rôle de
`TraceGraphAutoencoder.encode_graph()`.

---

## 6. Séquences et réseaux récurrents (LSTM)

### Le problème des séquences

Une séquence de logs (une liste ordonnée d'événements dans le temps) a une
longueur **variable** d'un run à l'autre, et l'**ordre** compte (un
événement A suivi de B n'a pas le même sens que B suivi de A). Les réseaux
de neurones "classiques" (comme les MLP utilisés dans les auto-encodeurs
des sections précédentes) attendent une entrée de taille fixe et ne
capturent pas naturellement cette notion d'ordre.

### RNN et le problème de la mémoire longue

Un **RNN** (*Recurrent Neural Network*, réseau de neurones récurrent) lit
une séquence pas à pas, en maintenant un **état caché** qui résume ce qu'il
a vu jusque-là, mis à jour à chaque nouveau pas. En théorie, ça permet de
traiter des séquences de longueur variable et de tenir compte de l'ordre.
En pratique, un RNN "simple" a du mal à se souvenir d'informations
lointaines dans une longue séquence (le signal du gradient s'atténue ou
explose au fil des pas — problème de **gradient qui s'évanouit**, *vanishing
gradient*).

### LSTM : une mémoire mieux contrôlée

Un **LSTM** (*Long Short-Term Memory*, mémoire à court terme longue —
Hochreiter & Schmidhuber, 1997) est une variante de RNN conçue pour mieux
gérer ce problème. Il maintient, en plus de l'état caché, une **cellule de
mémoire** séparée, et utilise des **portes** (*gates* — des mécanismes
appris qui décident quoi garder, quoi oublier, quoi laisser sortir à
chaque pas) pour contrôler précisément ce qui doit être mémorisé
longtemps ou oublié. Le détail des portes internes n'est pas ré-implémenté
dans ce projet — le dépôt utilise directement `torch.nn.LSTM`, l'implémentation
standard de PyTorch — mais comprendre l'intuition ("un état caché + une
mémoire, avec des vannes apprises qui filtrent l'information") suffit pour
la suite.

### Encodeur-décodeur de séquences (seq2seq) et auto-encodage

Pour obtenir, comme pour les autres modalités, un **vecteur unique de
taille fixe** résumant toute une séquence de logs, le dépôt utilise un
schéma **encodeur-décodeur** (*seq2seq*, pour *sequence to sequence*) :

- un LSTM **encodeur** lit la séquence de tokens (les identifiants de
  templates de log, cf. section 7) un par un ; son dernier état caché,
  une fois toute la séquence lue, sert de résumé — le **vecteur latent**.
- un LSTM **décodeur** tente de reconstruire la séquence d'origine à
  partir de ce vecteur latent (le même principe de "reconstruction comme
  pretexte d'apprentissage" que l'auto-encodeur classique, section 3).

### Le piège du teacher forcing "trop généreux"

Le **teacher forcing** est une technique d'entraînement courante pour les
décodeurs de séquence : à chaque pas, au lieu de laisser le décodeur
utiliser sa propre prédiction (potentiellement fausse) du pas précédent
pour prédire le pas suivant, on lui donne directement la **vraie** valeur
attendue du pas précédent — ça stabilise et accélère l'entraînement.

Le piège, identifié pendant ce travail (cf.
[chapitre 4](04-lstm-branche-logs.md)) : si on donne au décodeur, à
l'instant `t`, le token qu'il doit justement prédire à l'instant `t` (et
non celui d'avant), il peut apprendre à simplement le **recopier**, sans
jamais avoir besoin du vecteur latent — ce qui rendrait ce vecteur inutile
comme résumé. La solution, un **décalage** (*shift*) : le décodeur ne voit
à l'instant `t` que le vrai token de l'instant `t-1` (précédé d'un token
spécial `START` pour le tout premier pas), jamais celui qu'il doit
prédire. Voir `LogSequenceAutoencoder._shift_right()`.

### Vocabulaire, token, embedding

En traitement automatique du langage (**NLP**, *Natural Language
Processing*), un modèle qui travaille sur des séquences de symboles
discrets (mots, ou ici templates de log) a besoin d'un **vocabulaire** :
l'ensemble fini des symboles possibles, chacun associé à un entier (le
**token**). Une couche d'**embedding** (`nn.Embedding`) associe à chaque
token un vecteur de nombres réels, appris pendant l'entraînement — une
façon de transformer un identifiant arbitraire en une représentation
numérique exploitable par le réseau.

Deux tokens spéciaux réservés dans ce projet :

- **PAD** (*padding*, remplissage) : token neutre utilisé pour compléter
  les séquences plus courtes que la longueur maximale fixée, afin que
  toutes les séquences d'un batch aient la même taille.
- **UNK** (*unknown*, inconnu) : token "fourre-tout" utilisé pour tous les
  templates de log trop rares pour mériter leur propre identifiant dédié
  (cf. section suivante et chapitre 4) — une pratique standard en NLP face
  à un vocabulaire à "longue traîne" (beaucoup de mots/symboles rares).

---

## 7. Le parsing de logs et Drain3

### Le problème : des logs, pas des événements

Une ligne de log est un texte libre, par exemple :

```
2024-01-20 03:02:06.682  INFO [orders,...] Received payment response: PaymentResponse{authorised=true}
```

Deux lignes de log qui décrivent le "même type" d'événement (un paiement
reçu) ne sont presque jamais **identiques** caractère pour caractère (l'ID
de commande, l'horodatage, le montant changent à chaque fois). Pour
raisonner sur des séquences d'événements (que ce soit un simple comptage ou
un LSTM), il faut d'abord regrouper les lignes qui décrivent le même type
d'événement sous un identifiant commun — c'est le **parsing de logs**
(*log parsing*) ou **mining de templates** (*template mining*).

### Un template, un cluster

Un **template** est le "moule" commun à toutes les lignes d'un même type
d'événement, avec les parties variables remplacées par un symbole
générique — par exemple `Received payment response: PaymentResponse{authorised=<*>}`.
Chaque template reçoit un identifiant numérique, appelé ici **cluster_id**
(l'algorithme regroupe les lignes similaires en "clusters", un cluster =
un template).

### Drain3

**Drain** est un algorithme de mining de templates en flux (il traite les
lignes une par une, sans avoir besoin de tout le corpus en mémoire à
l'avance), qui organise les templates connus dans un arbre pour retrouver
rapidement, pour chaque nouvelle ligne, le template existant le plus
proche (ou en créer un nouveau si aucun ne correspond suffisamment).
**Drain3** est l'implémentation Python utilisée dans ce projet
(bibliothèque `drain3`, déjà une dépendance du projet). Voir
`src/data/log_parsing.py` et `src/models/log_sequence_encoder.py`.

### Le piège de l'hétérogénéité (rencontré au chapitre 4)

Drain3 discrimine notamment les lignes par leur **nombre de tokens** (mots)
dans son arbre. Sur des formats de logs très différents (logs structurés,
piles d'erreurs Java, JSON à texte libre...), ce critère échoue à
regrouper des lignes pourtant du même type — chaque légère variation
créant un nouveau cluster, faisant exploser le nombre de templates
distincts (jusqu'à 272 534 sur RCAEval RE2 sans garde-fou). D'où les deux
correctifs détaillés au chapitre 4 : une borne sur le cache de recherche
(`drain_max_clusters`), et un plafonnement du vocabulaire final gardé pour
le modèle (le reste regroupé sous **UNK**).

---

## 8. Fusionner plusieurs modalités : fusion jointe vs fusion tardive

C'est la question centrale du [chapitre 5](05-fusion-tardive-vs-jointe.md) —
ici, l'explication conceptuelle de ce qui les distingue.

### Fusion précoce / jointe (*early / joint fusion*)

Toutes les modalités sont combinées **avant** ou **au sein** d'un même
modèle, entraîné de bout en bout comme un seul système. Dans ce projet :
`MultimodalAutoencoder` — une branche encodeur par modalité, mais toutes
leurs représentations sont concaténées puis compressées dans un **unique**
goulot d'étranglement partagé, et le modèle entier apprend en une seule
fois à minimiser une perte commune. Avantage théorique : le modèle peut
apprendre des interactions entre modalités (par exemple, "une hausse de
latence ET une erreur réseau ensemble sont plus suspectes que l'une des
deux seule"). Inconvénient observé dans ce projet : demande davantage de
données pour bien s'entraîner (cf. l'effondrement sur RE3, plus petit), et
une modalité bruitée peut "polluer" le goulot d'étranglement partagé.

### Fusion tardive (*late fusion*)

Chaque modalité a son **propre** détecteur, entraîné **indépendamment**
(ici, un `isolation_forest` par modalité). On obtient un score d'anomalie
continu par modalité, et un modèle simple et léger — ici, une **régression
logistique** (`LogisticRegression`, un modèle qui apprend une combinaison
linéaire pondérée de ses entrées, suivie d'une fonction qui transforme le
résultat en probabilité entre 0 et 1) — apprend à **combiner** ces scores
en une décision finale. C'est un exemple de méthode d'**ensemble** (*ensemble
method*) : combiner plusieurs modèles plus simples plutôt que d'en
entraîner un seul plus complexe.

### Pourquoi la fusion tardive a gagné, ici

Mesuré empiriquement dans ce projet (chapitre 5), pas supposé a priori : la
fusion tardive est restée plus robuste sur les deux jeux de données testés,
et surtout beaucoup moins fragile quand les données d'entraînement se
raréfient. Une explication plausible : chaque `isolation_forest` est un
modèle simple, avec peu de paramètres à apprendre, alors que
`MultimodalAutoencoder` (un réseau de neurones profond) a besoin de
davantage d'exemples pour converger correctement — un compromis classique
en apprentissage automatique entre la capacité d'un modèle (plus il est
complexe, plus il peut en théorie apprendre des motifs fins) et la
quantité de données nécessaire pour exploiter cette capacité sans
sur-apprendre du bruit.

---

## 9. Évaluer un modèle : métriques et pièges statistiques

### Vrais/faux positifs et négatifs

Pour un problème de détection binaire (anomalie / normal), chaque
prédiction tombe dans une de quatre cases :

- **Vrai positif** (VP) : une vraie anomalie, détectée comme telle.
- **Faux positif** (FP) : un run normal, détecté à tort comme anomalie
  (fausse alerte).
- **Vrai négatif** (VN) : un run normal, correctement jugé normal.
- **Faux négatif** (FN) : une vraie anomalie, manquée (jugée normale).

### Précision, rappel, F1

- **Précision** (*precision*) : parmi tout ce que le modèle a signalé
  comme anomalie, quelle proportion l'était vraiment ? `VP / (VP + FP)`.
  Une précision élevée signifie peu de fausses alertes.
- **Rappel** (*recall*) : parmi toutes les vraies anomalies, quelle
  proportion le modèle a-t-il su détecter ? `VP / (VP + FN)`. Un rappel
  élevé signifie peu d'anomalies manquées.
- **F1** : la moyenne harmonique de la précision et du rappel — une façon
  de résumer les deux en un seul chiffre, qui pénalise fortement un modèle
  très déséquilibré (par exemple, un modèle qui alerte sur tout aurait un
  rappel de 100 % mais une précision très faible, et donc un F1 bas).

Il y a presque toujours un **compromis** entre précision et rappel : un
seuil de détection plus permissif augmente le rappel (moins d'anomalies
manquées) mais fait baisser la précision (plus de fausses alertes), et
inversement.

### AUC et ROC

La courbe **ROC** (*Receiver Operating Characteristic*) trace, pour tous
les seuils de décision possibles, le taux de vrais positifs contre le taux
de faux positifs. L'**AUC** (*Area Under the Curve*, aire sous la courbe)
résume cette courbe en un seul chiffre entre 0 et 1 : elle mesure la
qualité du **classement** produit par le score continu du modèle,
indépendamment du choix d'un seuil précis — "si je prends une vraie
anomalie et un vrai normal au hasard, quelle est la probabilité que le
modèle donne un score plus élevé à l'anomalie ?". Une AUC de 0,5 équivaut
au hasard pur (une pièce de monnaie) ; 1,0 est un classement parfait.

C'est pour cette raison qu'un modèle peut avoir un **F1 plus élevé** (à un
seuil donné) mais une **AUC plus basse** (moins bon en général, sur tous
les seuils) qu'un autre — situation rencontrée explicitement au
[chapitre 3](03-gat-branche-traces.md) entre `traces` et `traces_gat`.

### Entraînement, validation, test

- **Train** (entraînement) : les données sur lesquelles le modèle apprend
  ses paramètres.
- **Validation** (ou **val**) : des données jamais vues à l'entraînement,
  utilisées pour ajuster des choix qui ne sont pas appris directement par
  le modèle (ici, les poids du combinateur de fusion tardive), ou pour
  comparer plusieurs variantes entre elles.
- **Test** : des données jamais vues ni à l'entraînement, ni pendant
  l'ajustement sur la validation — utilisées uniquement à la toute fin,
  pour rapporter un score final honnête. Mélanger ces rôles (par exemple,
  ajuster un choix sur les données de test) s'appelle une **fuite de
  données** (*data leakage*) et donne des résultats trop optimistes,
  invalides.

### Le piège central de ce projet : le bruit d'échantillonnage

Un score mesuré sur un **petit** ensemble de test (23 à 43 exemples, dans
ce projet) peut varier significativement d'un tirage à l'autre, sans que
ça reflète une vraie différence de qualité entre deux modèles — c'est le
**bruit d'échantillonnage** (*sampling noise*). C'est exactement ce qui
s'est produit au [chapitre 5](05-fusion-tardive-vs-jointe.md) : deux
conclusions opposées ("GAT/LSTM aident" / "GAT/LSTM nuisent") obtenues sur
deux petits jeux de test différents, réconciliées seulement en regroupant
davantage de données de test (218 exemples) pour obtenir une estimation
plus stable.

La **validation croisée** (*cross-validation*, ou **k-fold** — découper
les données en k parts, entraîner/valider k fois en faisant tourner quelle
part sert de validation) est une technique standard pour réduire ce bruit
sans avoir besoin de plus de données — évoquée comme piste pour la suite au
[chapitre 6](06-conclusion-et-recommandations.md), mais pas mise en œuvre
dans cette session (le choix a été fait de chercher plus de données
plutôt, cf. chapitre 5).

---

## 10. Glossaire des abréviations

| Sigle | Signification (anglais) | Traduction / explication |
|---|---|---|
| **AUC** | Area Under the (ROC) Curve | Aire sous la courbe ROC — qualité globale du classement d'un score, indépendamment du seuil (section 9) |
| **CI/CD** | Continuous Integration / Continuous Deployment | Intégration continue / déploiement continu — chaînes automatisées de build/test/déploiement (section 1) |
| **CLI** | Command-Line Interface | Interface en ligne de commande |
| **CPU** | Central Processing Unit | Processeur |
| **CSV** | Comma-Separated Values | Format de fichier tabulaire texte, valeurs séparées par des virgules |
| **ELBO** | Evidence Lower BOund | Borne inférieure de la vraisemblance — quantité maximisée par un VAE (section 4) |
| **F1** | (score F1, aussi F-mesure) | Moyenne harmonique précision/rappel (section 9) |
| **FN** | False Negative | Faux négatif — anomalie manquée (section 9) |
| **FP** | False Positive | Faux positif — fausse alerte (section 9) |
| **GAT** | Graph Attention Network | Réseau à attention de graphe (section 5) |
| **GNN** | Graph Neural Network | Réseau de neurones sur graphe (section 5) |
| **GPU** | Graphics Processing Unit | Processeur graphique, utilisé pour accélérer l'entraînement de réseaux de neurones |
| **HTTP** | HyperText Transfer Protocol | Protocole de communication web, utilisé par l'API REST du projet |
| **JSON** | JavaScript Object Notation | Format de données texte structuré, utilisé par l'API et certains logs |
| **KL** | Kullback-Leibler (divergence) | Mesure d'écart entre deux distributions de probabilité (section 4) |
| **LRU** | Least Recently Used | Politique de cache qui évince l'élément le moins récemment utilisé (mécanisme de Drain3, chapitre 4) |
| **LSTM** | Long Short-Term Memory | Réseau récurrent à mémoire longue/courte contrôlée (section 6) |
| **MLP** | MultiLayer Perceptron | Perceptron multicouche — le réseau de neurones "simple" (couches entièrement connectées) utilisé dans les encodeurs/décodeurs de ce projet |
| **NLP** | Natural Language Processing | Traitement automatique du langage naturel (section 6) |
| **PAD** | Padding | Remplissage — token neutre pour égaliser la longueur des séquences (section 6) |
| **RCA** | Root Cause Analysis | Analyse de cause racine (section 1) |
| **RNN** | Recurrent Neural Network | Réseau de neurones récurrent (section 6) |
| **ROC** | Receiver Operating Characteristic | Courbe reliant taux de vrais/faux positifs selon le seuil (section 9) |
| **seq2seq** | Sequence to Sequence | Architecture encodeur-décodeur pour transformer une séquence en une autre (ou la reconstruire) (section 6) |
| **SVM** | Support Vector Machine | Machine à vecteurs de support (section 3) |
| **UNK** | Unknown | Inconnu — token "fourre-tout" pour les éléments rares (sections 6-7) |
| **VAE** | Variational AutoEncoder | Auto-encodeur variationnel (section 4) |
| **VN** | Vrai Négatif | Run normal correctement jugé normal (section 9) |
| **VP** | Vrai Positif | Vraie anomalie correctement détectée (section 9) |

---

*Pour voir comment ces concepts s'articulent concrètement dans le code de
ce projet, revenir aux chapitres [2](02-vae-branche-metriques.md) (VAE),
[3](03-gat-branche-traces.md) (GAT), [4](04-lstm-branche-logs.md) (LSTM) et
[5](05-fusion-tardive-vs-jointe.md) (fusion et évaluation).*
