# 2. VAE pour la branche métriques

Première étape, la plus simple des trois : transformer la branche
**métriques** de `MultimodalAutoencoder` en auto-encodeur variationnel
(VAE), en laissant les branches logs/traces inchangées pour l'instant.

## Pourquoi commencer par les métriques

`MultimodalAutoencoder` (`src/models/detection_models.py`) a une branche
encodeur/décodeur par modalité, fusionnées dans un goulot d'étranglement
partagé. Rendre une branche variationnelle (encodeur → distribution latente
plutôt que vecteur unique, échantillonnage, terme de régularisation KL) est
une modification locale à cette seule branche — c'était la façon la plus
sûre de valider le principe avant de toucher aux deux autres modalités,
dont la structure des données est bien plus complexe (graphe pour les
traces, séquence pour les logs — cf. chapitres 3 et 4).

## Ce qui a été construit

`MultimodalAutoencoder` accepte désormais un paramètre
`variational_modalities: set[str]` (vide par défaut, donc rétrocompatible
avec tous les modèles déjà entraînés). Pour chaque modalité listée :

- l'encodeur produit deux vecteurs, `mu` et `logvar`, au lieu d'un seul
  vecteur latent déterministe ;
- à l'entraînement, le vecteur latent est **échantillonné** par
  reparamétrisation : `z = mu + eps * exp(0.5 * logvar)` ;
- à l'inférence (`predict()`), `z = mu` directement — un score de
  reconstruction déterministe et reproductible, nécessaire pour un seuil
  d'anomalie stable ;
- la perte d'entraînement gagne un terme KL
  (`-0.5 * mean(1 + logvar - mu² - exp(logvar))`), pondéré par
  `kl_weight` (1e-3 par défaut).

Les branches non listées dans `variational_modalities` gardent exactement
le comportement d'avant (MLP déterministe) — c'est ce qui permet d'ajouter
le VAE sur les métriques sans rien changer aux branches logs/traces.

## Un bug de stabilité numérique en cours de route

Le premier essai a produit une perte `NaN` dès la 10e époque. Cause : dans
le terme KL, `exp(logvar)` apparaît sans transformation — sur des colonnes
`metric_*` RCAEval quasi constantes (0 sauf pour un service précis), le
z-score du `StandardScaler` est énorme sur les quelques lignes non nulles,
et quelques pas de gradient suffisent à faire diverger `logvar` vers
l'infini. Corrigé en bornant `logvar` (`torch.clamp(logvar, -6, 6)`) juste
après son calcul — un garde-fou standard en pratique VAE, particulièrement
nécessaire ici vu l'échelle très inégale des colonnes métriques.

## Intégration dans le pipeline existant

- `src/models/train_rcaeval.py` : flag `--variational-modalities metrics`,
  persisté dans `meta.json` du modèle entraîné.
- `src/api/model_registry.py` : `load_triplet()` relit ce champ pour
  reconstruire la bonne architecture (têtes `mu`/`logvar`) avant de charger
  les poids — sans ça, un modèle VAE servi par l'API échouerait au
  chargement (`load_state_dict` avec des formes de poids incompatibles).
- `src/models/evaluate_multimodal.py` : la fusion jointe est évaluée à la
  fois sans VAE (`joint_fusion_autoencoder`, comportement inchangé) et avec
  (`joint_fusion_vae_metrics`), dans le même run, pour une comparaison
  directe.

## Résultat mesuré

Sur RCAEval RE2 (`experiments/evaluation_rcaeval_20260813_162337.csv`) :

| Config | F1 | AUC |
|---|---|---|
| Fusion jointe, métriques déterministes | 0,3619 | 0,6646 |
| Fusion jointe, métriques en VAE | **0,4425** | 0,6648 |

Amélioration réelle mais modeste (+0,08 F1, AUC quasi stable). Le VAE aide
la fusion jointe à mieux exploiter le signal métriques, mais celle-ci reste
loin derrière la modalité traces seule (F1 ≈ 0,67) ou la fusion tardive
(F1 = 0,72) — un thème qui revient à chaque étape, creusé au
[chapitre 5](05-fusion-tardive-vs-jointe.md).

Note de méthode : l'entraînement de l'auto-encodeur n'est pas seedé (comme
le reste du dépôt), donc ce chiffre varie légèrement d'un run à l'autre —
les runs ultérieurs (chapitres 3 et 4) montrent cette même comparaison avec
des valeurs proches mais pas identiques (ex. 0,42 vs 0,45 selon le run).
L'ordre de grandeur et le sens de l'effet restent constants.

## Reproduire

```bash
python3 -m src.models.train_rcaeval --model-type multimodal_autoencoder --variational-modalities metrics
python3 -m src.models.evaluate_multimodal --source-dir data/interim/rcaeval/RE2
```
