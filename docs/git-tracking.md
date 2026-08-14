# Note : `src/models/` n'est pas suivi par Git

Découvert en vérifiant l'état du dépôt à la fin de cette session de travail
(pas un problème lié aux changements documentés ci-avant, mais assez
important pour être signalé séparément).

## Le problème

`.gitignore` (ligne 24) contient :

```
models/
```

Cette règle est destinée à ignorer le dossier `models/` à la racine du
dépôt (les artefacts `.joblib`/`.json` produits par l'entraînement — cf.
le commentaire au-dessus, "Sorties d'entraînement"). Mais un motif
`.gitignore` sans `/` en tête correspond à **n'importe quel dossier de ce
nom, à n'importe quelle profondeur** — pas seulement à la racine. Résultat
vérifié (`git check-ignore -v`) :

```
.gitignore:24:models/	src/models/train_rcaeval.py
.gitignore:24:models/	src/models/evaluate_multimodal.py
.gitignore:24:models/	src/models/graph_encoder.py
```

`src/models/` tout entier est ignoré. Seuls deux fichiers de ce dossier
sont suivis par Git aujourd'hui — `detection_models.py` et `train.py` — et
uniquement parce qu'ils ont été ajoutés à un commit **avant** que cette
règle n'existe (Git ne cesse jamais de suivre un fichier déjà suivi juste
parce qu'une règle `.gitignore` apparaît après coup ; en revanche, un
fichier jamais ajouté reste invisible pour `git status`/`git add -A`).

Conséquence concrète observée : `src/models/train_rcaeval.py`,
`evaluate.py`, `evaluate_causal.py`, `evaluate_multimodal.py`,
`evaluate_proactive.py` — les scripts qui implémentent une bonne partie du
pipeline RCAEval documenté dans `CLAUDE.md` — n'ont **jamais** été commités,
alors qu'ils existaient déjà avant cette session. Le commit `d773f38b`
("renforcer models pour meilleur score"), fait en parallèle de ce travail
par l'utilisateur, n'a d'ailleurs capturé que 3 des fichiers modifiés
pendant cette session (`detection_models.py`, `model_registry.py`,
`rcaeval.py`) — `train_rcaeval.py` et `evaluate_multimodal.py`, pourtant
modifiés aussi, sont restés invisibles pour ce commit, silencieusement,
pour la même raison.

## Fichiers actuellement invisibles pour Git à cause de cette règle

```
src/models/evaluate.py
src/models/evaluate_causal.py
src/models/evaluate_multimodal.py
src/models/evaluate_pooled.py       (nouveau, cf. chapitre 5)
src/models/evaluate_proactive.py
src/models/graph_encoder.py         (nouveau, cf. chapitre 3)
src/models/log_sequence_encoder.py  (nouveau, cf. chapitre 4)
src/models/train_rcaeval.py
```

## Correctif suggéré (non appliqué)

Ancrer la règle à la racine du dépôt avec un `/` en tête :

```diff
 # Sorties d'entraînement
-models/
+/models/
 experiments/*.csv
```

Cette modification n'a **pas été appliquée** dans cette session : c'est un
changement qui affecte l'historique Git à venir (quels fichiers deviennent
suivables) et le choix de committer ou non l'ensemble de ces fichiers
appartient à l'auteur du dépôt. Après correction de `.gitignore`, les
fichiers listés ci-dessus apparaîtront comme non suivis (`??`) dans
`git status` et pourront être ajoutés/commités normalement.
