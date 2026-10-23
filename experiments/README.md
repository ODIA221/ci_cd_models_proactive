# Résultats bruts

Ce dossier contient les résultats bruts cités dans l'article et dans la
documentation (`docs/`). Seuls ces fichiers sont versionnés ; les autres
sorties d'entraînement et d'évaluation (`results_*.csv`, nouvelles
évaluations) restent locales.

## Localisation de la cause racine (article, tables 3 et 4 ; `docs/07`)

Produits par `./run.sh evaluate-causal --compare-rankers` sur les 86
exécutions anormales de test de RCAEval RE2.

| Fichier | Contenu |
|---|---|
| `causal_rankers_summary_20260924_123129.csv` | P@1 et P@3 par méthode (table 3), sur toutes les exécutions anormales (86) et sur celles qui ont des traces (60) |
| `causal_rankers_by_fault_20260924_123129.csv` | P@1 par méthode et par type de panne (table 4) |
| `causal_rankers_20260924_123129.csv` | Détail par exécution et par méthode : rang du service fautif, présence de traces, uniformité de l'attention GAT (entropie normalisée) |
| `causal_false_leads_20260924_123129.csv` | Par exécution (normale ou anormale) : nombre de services avec aᵢ ≥ 0,5 et score maximal (section 6.3, fausses pistes) |

`./run.sh verify --full` relance cette évaluation et vérifie que les P@1
obtenus correspondent à ceux de ce fichier de synthèse.

## Détection d'anomalies (`docs/02` à `docs/06`)

| Fichier | Contenu |
|---|---|
| `evaluation_rcaeval_20260814_093057.csv` | Comparaison finale sur RE2 : mono-modalités, fusion tardive (F1 = 0,86), fusion jointe |
| `evaluation_pooled_20260814_093826.csv` | Fusion tardive sur RE2 + RE3 regroupés (F1 = 0,59, 109 exécutions de test) |
| `evaluation_rcaeval_20260813_165228.csv` | Branche traces : statistiques agrégées contre embeddings GAT (chapitre 3) |
| `evaluation_rcaeval_20260726_140759.csv`, `_20260813_162337.csv`, `_20260813_221630.csv`, `_20260814_083115.csv` | Étapes intermédiaires citées dans les chapitres 2, 4 et 5 |
| `evaluation_proactive_20260726_153239.csv` | Détection à horizon court, de 15 s à 720 s après l'incident |

Colonnes communes aux fichiers de détection : `precision_anomalie`,
`rappel_anomalie`, `f1_anomalie`, `auc`, et les effectifs de test.

## Étude utilisateur (`docs/08`)

| Fichier | Contenu |
|---|---|
| `study_power_grid.csv` | Puissance du test de l'hypothèse principale par simulation du plan exact (`./run.sh study-power`) : grille d'effectifs et de variances, plus la sensibilité à un effet faible (d = 0,3) |

Aucune donnée de participant n'est présente : l'étude n'a pas été conduite.
