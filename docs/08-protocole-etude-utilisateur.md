# 8. Protocole préenregistré de l'étude utilisateur

**Statut : protocole, étude non conduite.** Ce document fixe, avant toute
collecte, les questions de recherche, les hypothèses, le plan, l'effectif et
l'analyse de l'étude utilisateur annoncée au [chapitre 7](07-couche-exploration-causale-v2.md).
Il est rédigé pour une soumission en deux étapes (*Registered Report*) : à
l'étape 2, les relecteurs vérifient que l'étude a suivi exactement ce plan.
Le code qui l'exécute est déjà en place et testé :

| Élément | Code |
|---|---|
| Tirage des tâches, plan par blocs, carré latin | `GET /study/tasks` (`src/api/main.py`) |
| Enregistrement des réponses, correction côté serveur | `POST /study/sessions`, mode « étude » du dashboard |
| Calcul de puissance par simulation | `python -m src.causal.study_power` |
| Analyse préenregistrée | `./run.sh study-analysis` (`src/causal/study_analysis.py`) |

## 1. Questions de recherche

- **QR1** — L'exploration interactive réduit-elle le temps nécessaire pour
  désigner le service à l'origine d'une panne ?
- **QR2** — Améliore-t-elle l'exactitude de ce diagnostic, au-delà de la simple
  mise à disposition des mêmes signaux sous forme de tableaux ?
- **QR3** — Modifie-t-elle la confiance déclarée et la charge cognitive ?

## 2. Hypothèses

Conditions : **A**, corrélation manuelle avec les outils habituels ;
**B**, mêmes signaux que C, en tableaux statiques ; **C**, interface
d'exploration interactive.

| Hypothèse | Énoncé | Statut | Test |
|---|---|---|---|
| **H1** | Le temps de diagnostic est plus court en C qu'en A | **Principale** | Modèle mixte sur log(temps), contraste C−A, α = 0,05 bilatéral |
| H2 | L'exactitude est plus élevée en C qu'en B | Secondaire | GEE logistique, contraste C−B |
| H3 | Le temps de diagnostic est plus court en C qu'en B | Secondaire | Modèle mixte sur log(temps), contraste C−B |
| H4 | La confiance déclarée est plus élevée en C qu'en A | Secondaire | Modèle mixte sur la note de Likert, contraste C−A |

Les hypothèses secondaires sont corrigées ensemble par la méthode de Holm.
Une seule hypothèse principale est déclarée, ce qui dispense le test de H1 de
toute correction pour comparaisons multiples.

**Analyses exploratoires**, présentées comme telles : NASA-TLX et SUS par
condition, nombre d'interactions en C, résultats par type de panne et par
application, entretien qualitatif.

## 3. Plan

**Intra-sujets, par blocs.** Chaque participant traite 15 exécutions
anormales, réparties en 3 blocs de 5, un par condition. Faire 5 tâches de
suite dans la même condition évite de passer de Grafana à l'interface à
chaque exécution, ce qui ajouterait un coût de bascule et risquerait de
contaminer une condition par l'autre.

**Contrebalancement.** L'ordre des blocs suit un carré latin 3 × 3
(ABC, BCA, CAB), attribué **à tour de rôle** selon le numéro d'inscription :
P01 reçoit ABC, P02 BCA, P03 CAB, P04 ABC, et ainsi de suite. L'effectif sera
un multiple de 3 pour que chaque ordre soit également représenté.

**Tâches.** Pour chaque participant, 15 exécutions sont tirées sans remise
parmi les 60 exécutions anormales de test de RCAEval RE2 qui disposent de
traces. Le tirage est déterministe, avec pour graine l'identifiant du
participant. Les tâches sont donc différentes d'un participant à l'autre,
et l'effet propre à chaque exécution se répartit au hasard entre conditions.

## 4. Participants

**Inclusion** : au moins un an d'expérience professionnelle en exploitation,
DevOps ou SRE ; usage courant d'au moins un outil de métriques, de journaux
ou de traces.

**Exclusion** : avoir participé au développement de LogPipeGuard ou à la
préparation des tâches ; avoir participé à l'étude pilote.

**Recrutement** : communautés DevOps et SRE de Dakar, entreprises partenaires
de l'UCAD, réseau professionnel des auteurs. Participation volontaire,
séance d'environ 90 minutes.

## 5. Effectif

L'effectif est calculé par simulation du plan exact et du modèle d'analyse
préenregistré (`python -m src.causal.study_power`). L'effet visé est
**d = 0,5**, exprimé en écarts-types d'une observation (un temps de
diagnostic, en log). Si l'écart-type du log-temps vaut environ 0,6, valeur
fréquente pour des temps de tâche, d = 0,5 correspond à un temps réduit
d'environ 26 %. Les parts de variance dues au participant (ρ_p) et à la tâche
(ρ_t) sont inconnues avant le pilote : le calcul en couvre plusieurs valeurs.

**Puissance du test de H1** (contraste C contre A, d = 0,5, α = 0,05 ;
300 simulations par cellule ; fichier `experiments/study_power_grid.csv`) :

| Participants | ρ_p = 0,2 ; ρ_t = 0,1 | ρ_p = 0,2 ; ρ_t = 0,2 | ρ_p = 0,4 ; ρ_t = 0,1 | ρ_p = 0,4 ; ρ_t = 0,2 |
|---|---|---|---|---|
| 6 | 0,54 | 0,58 | 0,73 | 0,71 |
| 9 | 0,71 | 0,74 | 0,86 | 0,87 |
| 12 | 0,87 | 0,85 | 0,96 | 0,93 |
| 15 | 0,92 | 0,94 | 0,97 | 0,98 |
| **18** | **0,97** | **0,98** | **0,99** | **0,99** |
| **24** | **0,98** | **1,00** | **1,00** | **1,00** |

**Effet plus faible** (d = 0,3), dans le scénario de variance le plus
défavorable (ρ_p = 0,2 ; ρ_t = 0,2) : puissance de 0,47 à 12 participants,
0,59 à 18, 0,74 à 24 et 0,82 à 30.

Le test est correctement calibré : sans effet simulé, il rejette à tort
l'hypothèse nulle dans 4,5 % des cas pour 5 % attendus (600 simulations,
12 participants).

**Décision préenregistrée** : effectif cible **24 participants** (8 par ordre
de blocs), minimum **18**.

- Avec 18 participants, la puissance dépasse 95 % dans tous les scénarios de
  la grille pour d = 0,5.
- 24 participants laissent une marge pour les abandons, et donnent 74 % de
  puissance si l'effet réel n'est que de d = 0,3.
- **Règle d'arrêt** : le recrutement s'arrête à 24 participants, ou à la fin
  d'une fenêtre de 8 semaines. En dessous de 18, l'étude est présentée comme
  **exploratoire**, et H1 n'est pas considérée comme testée.
- **Mise à jour après le pilote** : si les variances estimées sur le pilote
  sortent de la grille, l'effectif est recalculé avec le même script, avant
  le début de l'étude principale, et ce recalcul est documenté.

**Pourquoi ce chiffre diffère des 34 à 44 participants annoncés auparavant.**
Ce premier calcul supposait un test apparié sur la moyenne de chaque
participant, avec un effet d = 0,5 défini sur ces moyennes et une correction
de Bonferroni pour trois comparaisons. Le plan retenu ici définit l'effet sur
une seule tâche, exploite les 5 tâches par condition et par participant, et
ne déclare qu'une hypothèse principale. Les deux calculs ne portent donc pas
sur la même quantité.

## 6. Procédure

1. Consentement éclairé et questionnaire démographique (expérience, outils utilisés).
2. Tâche d'entraînement non comptée dans chaque condition, avant son bloc
   (environ 5 minutes par condition).
3. Trois blocs de 5 tâches. Pour chaque tâche : affichage de l'alerte, diagnostic,
   désignation du service fautif et confiance (Likert de 1 à 5). Durée
   maximale annoncée : **15 minutes**. Au-delà, l'expérimentateur demande la
   meilleure hypothèse du moment, et le temps est enregistré à 900 s.
4. **NASA-TLX à la fin de chaque bloc**, ce que permet le plan par blocs.
5. **SUS** à la fin des blocs B et C (les outils de la condition A ne sont pas les nôtres).
6. Entretien semi-directif d'environ 10 minutes.

Le participant ne voit jamais l'identifiant RCAEval de l'exécution, qui
contient le nom du service fautif : l'interface affiche un alias opaque. Il
n'apprend pas non plus s'il a eu juste avant la fin de la séance.

## 7. Matériel à préparer avant le pilote

- **Condition A** : un environnement Grafana avec Loki (journaux), Prometheus
  (métriques) et Jaeger ou Tempo (traces), chargé avec des copies des
  exécutions **anonymisées**. Les dossiers RCAEval portent le nom du service
  fautif et doivent être renommés avec le même alias que dans l'interface.
  **Cet environnement n'existe pas encore** : c'est le principal chantier
  technique avant l'étude.
- **Condition B** : déjà en place (tableaux du mode étude).
- **Condition C** : déjà en place (interface d'exploration).

## 8. Mesures

| Mesure | Définition | Rôle |
|---|---|---|
| Temps de diagnostic | De l'affichage de l'alerte à la validation de la réponse, plafonné à 900 s | Principal (H1, H3) |
| Exactitude | Le service désigné est le service où la panne a été injectée | Secondaire (H2) |
| Confiance | Échelle de Likert de 1 à 5, après chaque tâche | Secondaire (H4) |
| Charge cognitive | NASA-TLX brut, par bloc | Exploratoire |
| Utilisabilité | SUS, conditions B et C | Exploratoire |
| Interactions | Nombre d'actions dans l'interface (condition C) | Exploratoire |

## 9. Analyse

Exécutée telle quelle par `./run.sh study-analysis` :

- **H1 et H3** : modèle linéaire mixte sur log(temps), ordonnée aléatoire par
  participant, condition et position du bloc en effets fixes. Résultat
  rapporté comme un rapport des temps (exp du coefficient), avec IC à 95 %.
- **H2** : régression logistique estimée par GEE, groupée par participant
  (corrélation échangeable), même covariable de position, erreur standard
  corrigée pour petits échantillons (Mancl-DeRouen).
- **H4** : modèle linéaire mixte sur la note de Likert, même structure que H1.
- **Correction de Holm** sur H2 à H4.
- **Sensibilité** : test de Friedman, puis tests de Wilcoxon appariés avec
  correction de Bonferroni, sur les moyennes par participant. Si les
  conclusions divergent de l'analyse principale, les deux sont rapportées.
- **Validation du script** : sur des données simulées avec un effet connu
  (temps C/A = 0,70), il estime 0,72 [0,60 ; 0,86]. Sans effet, son taux de
  faux positifs est de 4,5 % pour 5 % attendus (600 simulations, 12 participants).

**Données exclues** : uniquement les tâches compromises par un incident
technique, documenté pendant la séance (panne de l'outil, mauvaise exécution
chargée). Aucune exclusion fondée sur la valeur d'une mesure.

## 10. Étude pilote

3 à 5 participants, exclus de l'analyse principale. Objectifs : vérifier la
durée des tâches et de la séance, la clarté des consignes et le
fonctionnement de la condition A, et estimer ρ_p, ρ_t et l'écart-type du
log-temps pour confirmer l'effectif (section 5).

## 11. Écarts au protocole

Tout écart (changement de matériel, d'effectif, de règle d'exclusion) est
daté, justifié et rapporté dans l'article d'étape 2. Une analyse qui
n'est pas décrite ici est présentée comme exploratoire.

## 12. Éthique et données

Avis d'un comité d'éthique avant tout recrutement. Données pseudonymisées
(identifiants P01, P02…), stockées dans `experiments/user_study/sessions.jsonl`.
Publication des données de séance et des questionnaires après l'étude, sous
réserve du consentement des participants ; les entretiens ne sont publiés
que sous forme d'extraits anonymisés.

## 13. Calendrier indicatif

| Semaines | Étape |
|---|---|
| 1 à 3 | Avis éthique, environnement de la condition A |
| 4 | Étude pilote, ajustements, recalcul éventuel de l'effectif |
| 5 à 12 | Étude principale (24 participants au plus) |
| 13 et 14 | Analyse préenregistrée, rédaction de l'étape 2 |

---

## Annexe — texte pour le manuscrit d'étape 1 (anglais)

> **Research questions.** RQ1: Does interactive exploration reduce the time
> engineers need to identify the service at the origin of a failure? RQ2: Does
> it improve diagnostic accuracy beyond providing the same signals as static
> tables? RQ3: Does it change self-reported confidence and cognitive load?
>
> **Hypotheses.** H1 (primary): diagnosis time is shorter with interactive
> exploration (C) than with manual correlation (A). H2: accuracy is higher in C
> than with static attribution (B). H3: diagnosis time is shorter in C than in
> B. H4: confidence is higher in C than in A. H1 is tested at α = .05
> (two-sided) without correction, as the single primary hypothesis; H2–H4 are
> corrected together with the Holm procedure.
>
> **Design.** Within-subjects, blocked: each participant completes 15
> anomalous runs in three blocks of five, one block per condition. Block order
> follows a 3 × 3 Latin square assigned in rotation by enrolment number (ABC,
> BCA, CAB), and the sample size is a multiple of three. Runs are drawn without
> replacement, per participant, from the 60 anomalous test runs of RCAEval RE2
> that include traces.
>
> **Sample size.** We estimated power by simulating the exact design and the
> pre-registered model. Effect size d = 0.5 is expressed in standard deviations
> of a single log-transformed diagnosis time (about a 26% time reduction if
> that standard deviation is 0.6). Across participant- and task-level variance
> shares of 0.2–0.4 and 0.1–0.2, 18 participants yield power of 0.97–0.99.
> We target 24 participants (minimum 18), which retains 0.74 power for
> d = 0.3 in the least favourable variance scenario. Without a simulated
> effect, the test rejects the null hypothesis in 4.5% of cases (nominal 5%). Recruitment stops at 24 participants or after an eight-week window;
> with fewer than 18, the study is reported as exploratory.
>
> **Analysis.** H1 and H3: linear mixed model on log diagnosis time, with a
> random intercept per participant and condition and block position as fixed
> effects, reported as a time ratio with a 95% confidence interval. H2:
> logistic GEE clustered by participant (exchangeable correlation, same
> covariate, Mancl–DeRouen small-sample correction). H4: linear mixed model on
> the Likert rating. Sensitivity analysis: Friedman and Bonferroni-corrected
> Wilcoxon signed-rank tests on per-participant means. Diagnosis times are
> capped at 900 s, the announced maximum per task. Tasks are excluded only for
> technical incidents documented during the session.
