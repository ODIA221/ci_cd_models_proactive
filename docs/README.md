# Documentation — Vers une détection multimodale d'anomalies CI/CD

Cette documentation retrace une session de travail visant à rapprocher le
code du dépôt (`src/`) de l'architecture décrite dans le résumé de thèse
(LSTM pour les logs, auto-encodeur variationnel pour les métriques, réseau
à attention sur graphe pour les traces, fusion par attention croisée), et à
mesurer honnêtement, étape par étape, si chaque ajout améliore réellement la
détection d'anomalies sur RCAEval.

Les chapitres 1 à 6 portent sur la **détection**. Le chapitre 7 porte sur
la **couche d'exploration causale** (v2) : signaux par service, interface
interactive, rapport PDF, et ce que ces outils valent réellement pour
localiser une panne.

Elle est écrite pour être lue dans l'ordre, comme des chapitres, mais
chaque page se suffit à elle-même si vous cherchez un point précis. Pour
installer et lancer le projet, voir le [README à la racine](../README.md).

## Plan

0. **[Cours : les concepts et algorithmes](00-cours-concepts-et-algorithmes.md)** — un cours complet, pédagogique, qui explique de zéro chaque notion utilisée (VAE, GAT, LSTM, fusion, métriques d'évaluation...) et un glossaire de toutes les abréviations. À lire en premier si un terme technique des chapitres suivants n'est pas familier.
1. **[Diagnostic initial](01-diagnostic-initial.md)** — pourquoi ce travail a commencé : l'écart entre le résumé de thèse et le code réel, chiffré précisément.
2. **[VAE pour la branche métriques](02-vae-branche-metriques.md)** — premier composant, le plus simple : encodeur variationnel greffé sur `MultimodalAutoencoder`.
3. **[GAT pour la branche traces](03-gat-branche-traces.md)** — réseau à attention de graphe fait main, et la découverte (200k+ spans/run) qui a changé sa conception.
4. **[LSTM pour la branche logs](04-lstm-branche-logs.md)** — auto-encodeur séquentiel sur des templates minés par Drain3, et les trois incidents rencontrés en le construisant.
5. **[Fusion tardive vs fusion jointe](05-fusion-tardive-vs-jointe.md)** — pourquoi l'ensemble simple (isolation_forest par modalité + régression logistique) bat systématiquement l'architecture à goulot d'étranglement partagé, validé sur deux jeux de données indépendants (RE2, RE3) puis regroupés.
6. **[Conclusion et recommandations](06-conclusion-et-recommandations.md)** — ce qui est validé, ce qui reste ouvert, quoi utiliser en pratique.
7. **[Couche d'exploration causale (v2)](07-couche-exploration-causale-v2.md)** — adaptation de l'article « Explication des échecs CI/CD » : ce qui est implémenté, ce qui est mesuré (l'attention GAT n'apporte rien au-delà d'une normalisation par le degré) et ce que l'article doit corriger.

Note annexe, sans rapport avec le contenu scientifique : **[pourquoi
`src/models/` n'était pas suivi par Git](git-tracking.md)** (règle
`.gitignore` trop large) — **corrigé le 2026-09-24**.

## Résumé exécutif

| Question | Réponse courte |
|---|---|
| Le code correspond-il au résumé de thèse ? | Non, au départ : MLP déterministe, pas de LSTM/VAE/GAT/attention croisée, F1 mesuré ≈0,31 contre 0,936/0,958 annoncés. |
| Le VAE, le GAT et le LSTM ajoutés aident-ils ? | Le GAT oui, mesurablement. Le LSTM légèrement, mais son effet net dans la fusion reste incertain. Le VAE aide la fusion jointe mais celle-ci reste dominée par une approche plus simple. |
| Quelle est la meilleure approche mesurée ? | La fusion tardive (un détecteur par modalité + régression logistique sur les scores), pas la fusion jointe visée par le résumé de thèse. |
| Ce résultat est-il fiable ? | Validé sur deux sous-ensembles RCAEval indépendants (RE2, RE3) puis sur leurs runs de test regroupés (218 au total) — voir [chapitre 5](05-fusion-tardive-vs-jointe.md). |
| Peut-on localiser le service fautif ? | Oui, en partie : un score robuste par service le classe 1er dans 71 % des 86 exécutions anormales de test RE2, dans les 3 premiers dans 95 %, même sans traces — voir [chapitre 7](07-couche-exploration-causale-v2.md). |
| L'attention du GAT explique-t-elle la panne ? | Non : elle est uniforme (entropie normalisée médiane 1,000) et donne exactement les mêmes classements qu'une pondération par 1/degré. |
| L'interface fait-elle gagner du temps aux ingénieurs ? | Non mesuré : l'étude utilisateur n'a pas été conduite. Le protocole et l'instrumentation existent. |

## Comment reproduire

Toutes les commandes s'exécutent depuis la racine du dépôt, avec le
virtualenv actif (`.venv`). Prérequis : RCAEval RE2 acquis
(`./run.sh acquire --source rcaeval --subset RE2`).

Détection (chapitres 2 à 6) :

```bash
python3 -m src.models.graph_encoder --subset RE2
python3 -m src.models.log_sequence_encoder --subset RE2
python3 -m src.models.evaluate_multimodal --source-dir data/interim/rcaeval/RE2 --use-gat-traces --use-lstm-logs
python3 -m src.models.evaluate_pooled
```

Exploration causale (chapitre 7) :

```bash
python3 -m src.models.graph_encoder --subset RE2 --save-model-only   # sauvegarde le GAT (attention)
./run.sh evaluate-causal --compare-rankers                          # ~1 h au 1er lancement, puis cache
./run.sh verify --full                                              # recompare aux chiffres du chapitre 7
```

Les résultats bruts (CSV) sont dans `experiments/` :
`evaluation_rcaeval_*.csv` et `evaluation_pooled_*.csv` (détection),
`causal_rankers*_*.csv` et `causal_false_leads_*.csv` (chapitre 7).
