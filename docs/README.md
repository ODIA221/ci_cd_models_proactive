# Documentation — Vers une détection multimodale d'anomalies CI/CD

Cette documentation retrace une session de travail visant à rapprocher le
code du dépôt (`src/`) de l'architecture décrite dans le résumé de thèse
(LSTM pour les logs, auto-encodeur variationnel pour les métriques, réseau
à attention sur graphe pour les traces, fusion par attention croisée), et à
mesurer honnêtement, étape par étape, si chaque ajout améliore réellement la
détection d'anomalies sur RCAEval.

Elle est écrite pour être lue dans l'ordre, comme des chapitres, mais
chaque page se suffit à elle-même si vous cherchez un point précis.

## Plan

1. **[Diagnostic initial](01-diagnostic-initial.md)** — pourquoi ce travail a commencé : l'écart entre le résumé de thèse et le code réel, chiffré précisément.
2. **[VAE pour la branche métriques](02-vae-branche-metriques.md)** — premier composant, le plus simple : encodeur variationnel greffé sur `MultimodalAutoencoder`.
3. **[GAT pour la branche traces](03-gat-branche-traces.md)** — réseau à attention de graphe fait main, et la découverte (200k+ spans/run) qui a changé sa conception.
4. **[LSTM pour la branche logs](04-lstm-branche-logs.md)** — auto-encodeur séquentiel sur des templates minés par Drain3, et les trois incidents rencontrés en le construisant.
5. **[Fusion tardive vs fusion jointe](05-fusion-tardive-vs-jointe.md)** — pourquoi l'ensemble simple (isolation_forest par modalité + régression logistique) bat systématiquement l'architecture à goulot d'étranglement partagé, validé sur deux jeux de données indépendants (RE2, RE3) puis regroupés.
6. **[Conclusion et recommandations](06-conclusion-et-recommandations.md)** — ce qui est validé, ce qui reste ouvert, quoi utiliser en pratique.

Note annexe, sans rapport avec le contenu scientifique ci-dessus mais
importante : **[le dossier `src/models/` n'est pas suivi par Git](git-tracking.md)**
à cause d'une règle `.gitignore` trop large — découvert en vérifiant l'état
du dépôt à la fin de cette session.

## Résumé exécutif

| Question | Réponse courte |
|---|---|
| Le code correspond-il au résumé de thèse ? | Non, au départ : MLP déterministe, pas de LSTM/VAE/GAT/attention croisée, F1 mesuré ≈0,31 contre 0,936/0,958 annoncés. |
| Le VAE, le GAT et le LSTM ajoutés aident-ils ? | Le GAT oui, mesurablement. Le LSTM légèrement, mais son effet net dans la fusion reste incertain. Le VAE aide la fusion jointe mais celle-ci reste dominée par une approche plus simple. |
| Quelle est la meilleure approche mesurée ? | La fusion tardive (un détecteur par modalité + régression logistique sur les scores), pas la fusion jointe visée par le résumé de thèse. |
| Ce résultat est-il fiable ? | Validé sur deux sous-ensembles RCAEval indépendants (RE2, RE3) puis sur leurs runs de test regroupés (218 au total) — voir [chapitre 5](05-fusion-tardive-vs-jointe.md). |

## Comment reproduire

Toutes les commandes citées dans cette documentation s'exécutent depuis la
racine du dépôt, avec le virtualenv actif (`.venv`) :

```bash
python3 -m src.models.graph_encoder --subset RE2
python3 -m src.models.log_sequence_encoder --subset RE2
python3 -m src.models.evaluate_multimodal --source-dir data/interim/rcaeval/RE2 --use-gat-traces --use-lstm-logs
python3 -m src.models.evaluate_pooled
```

Les fichiers de résultats bruts (CSV) référencés dans cette documentation
sont dans `experiments/evaluation_rcaeval_*.csv` et
`experiments/evaluation_pooled_*.csv`.
