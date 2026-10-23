# 7. Couche d'exploration causale (LogPipeGuard-Multimodal v2)

Ce chapitre documente l'adaptation, dans le code, de l'article « Explication
des échecs CI/CD : un cadre de corrélation visuelle causale pour les
journaux, métriques et traces », en partant de ce qui existait déjà
(chapitres 1 à 6). Chaque affirmation de l'article est ramenée à ce qui est
**réellement implémenté** et **réellement mesuré**. Là où le code ne peut
pas soutenir l'affirmation, ce chapitre le dit.

## Ce qui a été construit

| Fichier | Rôle |
|---|---|
| `src/models/graph_encoder.py` | Modifié : le GAT expose ses poids d'attention (`attention_weights`), est entraîné avec une graine fixe et **sauvegardé** (`--save-model-only`) |
| `src/causal/signals.py` | Nouveau : couche d'extraction des signaux causaux (score par service, attention, logs, onsets, graphe d'appels, « et si », rapport, JSON-LD) |
| `src/causal/report_pdf.py` | Nouveau : rapport de diagnostic en PDF avec graphiques (fpdf2 + matplotlib) |
| `src/api/main.py`, `schemas.py` | Nouveaux endpoints : `GET /runs`, `GET /causal/{run_id}` (+ `/whatif`, `/report`, `/report.pdf`, `/jsonld`), `GET /study/tasks`, `POST /study/sessions`, interface servie sous `/ui/` |
| `frontend/` | Nouveau : interface React 18 + TypeScript + D3 v7 (chronologie sur canevas défilable, graphe à forces animé, coordonnées parallèles, « et si », exports) |
| `src/dashboard/causal_views.py`, `causal_page.py` | Nouveau : mêmes vues dans le dashboard Streamlit + mode étude utilisateur |
| `src/models/evaluate_causal.py` | Modifié : `--compare-rankers` compare les classements de cause racine de la v2 |
| `src/causal/study_analysis.py` | Nouveau : analyse statistique des sessions d'étude **réellement enregistrées** |
| `src/verify.py` | Nouveau : vérification de bout en bout (`./run.sh verify [--full]`) |
| `run.sh` | Nouvelles commandes : `ui-build`, `ui-dev`, `verify`, `study-analysis` ; `start` construit l'interface si besoin et l'ouvre |

Commandes :

```bash
python -m src.models.graph_encoder --subset RE2 --save-model-only   # ~20 min, une fois
./run.sh evaluate-causal --compare-rankers                          # ~1 h au 1er lancement, puis cache
./run.sh start                                                      # tout lancer ; interface sur http://localhost:8000/ui/
./run.sh ui-dev                                                     # interface en mode développement (http://localhost:5173/ui/, démarre l'API si besoin)
./run.sh verify --full                                              # tout vérifier + recomparer aux chiffres ci-dessous
./run.sh study-analysis                                             # seulement après une vraie étude
```

## Correspondance article → implémentation

### Section 3.1 : formalisation

| Article | Implémenté | Écart |
|---|---|---|
| Nœuds vᵢ = étapes de pipeline CI/CD | Services RCAEval (Online Boutique, Train Ticket, Sock Shop) | **Aucune étape build/test/deploy dans les données.** Les vues parlent de services. |
| aᵢ ∈ [0,1] fourni par le modèle multimodal | Heuristique : pour chaque modalité, écart robuste (médiane/MAD, bins de 10 s) entre la fenêtre après injection et la fenêtre avant, écrasé en s/(s+seuil), puis maximum sur les modalités | Les détecteurs entraînés ne donnent qu'un score **par run**, pas par service |
| αᵢⱼ : attention GAT = influence causale | Vrais poids d'attention (couche 1, moyenne des têtes) du GAT de `graph_encoder.py` | Ce GAT est un auto-encodeur entraîné à reconstruire des graphes **normaux**. Mesuré ci-dessous : son attention est uniforme |
| βᵢ,ₖ : attention sur les journaux | Substitut : log2 du rapport des taux d'occurrence de chaque template (après/avant) | L'encodeur LSTM des logs n'a **pas** d'attention |
| H(Lᵢ) : entropie des journaux | Entropie de Shannon des templates dans la fenêtre analysée | — |
| Durée normalisée dᵢ | Rapport de durée médiane des spans (après/avant) | Pas de durée d'étape |

Chaque réponse de `GET /causal/{run_id}` contient un champ `provenance` qui
répète ces écarts, affiché dans les deux interfaces avant toute
interprétation.

### Section 3.2 : les trois vues

- **Chronologie** : une ligne par service, glyphe placé à la première
  déviation. Encodage de la Table 1 respecté (hauteur = entropie, largeur =
  durée, bordure pointillée = déviation métrique), **sauf la couleur** : une
  rampe à une teinte (gris → rouge) remplace le rouge → vert, illisible pour
  les daltoniens (deutéranopie) et inadaptée à une magnitude.
- **Propagation** : graphe d'appels observé dans les traces, flèches dans le
  sens de propagation d'un symptôme (appelé → appelant), épaisseur = α, taille
  et couleur = aᵢ, ondes animées le long du chemin sélectionné. Sans traces
  (Sock Shop), la vue le dit au lieu d'extrapoler.
- **Corrélation modale** : coordonnées parallèles des contributions de chaque
  modalité. **Pas** de « poids d'attention intermodaux » : le meilleur
  détecteur mesuré (fusion tardive, chapitre 5) n'a pas d'attention croisée.

### Section 3.3 : interaction

| Principe | Implémentation |
|---|---|
| Liaison et surbrillance | Sélection d'un service partagée par les trois vues (clic sur un glyphe, un nœud ou un trait) |
| Filtrage par seuil | Curseur aᵢ appliqué aux trois vues |
| Traçage du chemin causal au clic | Chemins appelé → appelants depuis le service sélectionné, surlignés et animés |
| Exploration « et si » | Test **structurel** : quels services anormaux l'hypothèse explique-t-elle par propagation ? Couverture affichée. Ce n'est **pas** un contrefactuel (échelon 3 de Pearl) |

### Section 4.2 : implémentation

| Article | Implémenté |
|---|---|
| React 18 + TypeScript | Oui (`frontend/`, TypeScript `strict`) |
| D3.js v7 | Oui : simulation de forces, drag, coordonnées parallèles, rampes |
| Disposition dirigée par forces, animation physique | Oui (`d3-force` : liens, répulsion, collision) |
| Chronologie sur canevas défilable | Oui (`<canvas>` dans un conteneur à défilement horizontal, zoom) |
| **Three.js** pour les graphes à grande échelle | **Non implémenté, volontairement.** Les graphes font au plus ~35 services (graphe au niveau service, chapitre 3). SVG/D3 en affiche des milliers. Ajouter WebGL n'apporterait rien de mesurable. **À retirer de l'article**, ou à présenter comme perspective si l'on passe un jour au niveau span (~200 000 nœuds/run). |
| FastAPI servant des signaux pré-calculés | Oui : calcul au premier appel (5 s à ~1 min selon le système), puis cache JSON |
| JSON-LD | Oui (`GET /causal/{run_id}/jsonld`, `application/ld+json`), vocabulaire en URN du projet + `service.name` OpenTelemetry. **Limite** : ni Grafana ni Jaeger ne lisent nativement du JSON-LD. L'« intégration » à ces outils n'est donc ni faite ni testée. Une piste plausible : une source de données JSON générique de Grafana lisant l'endpoint REST. |

### Section 4.3 : flux d'usage

1. **Alerte proactive** : existait déjà (`/predict`, modèles à horizon court, chapitre 10 de la thèse).
2. **Ouverture avec le run pré-chargé** : lien direct `/ui/?run=<run_id>&service=<service>`, et bouton depuis le dashboard Streamlit.
3. **Exploration** : les trois vues.
4. **Export** : rapport PDF avec graphiques (`/report.pdf`, `src/causal/report_pdf.py`), rapport Markdown (`/report`) et JSON-LD (`/jsonld`), boutons dans les deux interfaces.

## Résultats mesurés

Protocole : `./run.sh evaluate-causal --compare-rankers`, RCAEval RE2, les
**86 runs anormaux de test** (28 Online Boutique, 32 Train Ticket, 26 Sock
Shop ; faute cpu 15, delay 10, disk 17, loss 12, mem 16, socket 16). La
vérité terrain est le service où la faute a été injectée. On mesure
precision@k : le service fautif est-il classé dans les k premiers ? Fichiers
bruts : `experiments/causal_rankers*_20261023_194519.csv` ; tous les chiffres
ci-dessous se recalculent avec `python -m src.models.causal_summary`.

> **Correction du 2026-10-23 (signaux version 3).** Les colonnes de métriques
> RCAEval en `*-total` sont déjà des taux par seconde, pas des compteurs
> cumulés (elles décroissent, vérifié). Les versions 1 et 2 des signaux les
> différenciaient, et mesuraient donc la variation du taux au lieu du taux.
> La correction **dégrade** le score multimodal : P@1 de 0,71 à 0,64, P@3 de
> 0,95 à 0,88, et davantage de fausses pistes. Les chiffres ci-dessous sont
> ceux de la version corrigée ; les anciens restent dans l'historique Git
> (`causal_rankers*_20260924_123129.csv`). Le pipeline de détection
> (chapitres 2 à 6) n'était pas concerné.

Classements comparés :

- `score_multimodal` : aᵢ décroissant (départage : nombre de modalités en
  déviation, puis somme des contributions) ;
- `precedence_temporelle` : parmi les services avec aᵢ ≥ 0,5, le premier à dévier ;
- `propagation_attention` / `propagation_uniforme` / `propagation_degre` :
  PageRank personnalisé de type MicroRCA sur le graphe d'appels, arêtes
  pondérées respectivement par α (GAT), par 1, et par 1/(1+degré) ;
- `arbre_spans_v1` : l'heuristique existante (`GET /explain`), qui ne
  fonctionne qu'avec des spans en cache (60 runs).

### Localisation de la cause racine

| Classement | P@1 (86 runs) | P@3 (86 runs) | P@1 (60 runs avec traces) | P@3 (60 runs) |
|---|---|---|---|---|
| arbre_spans_v1 (existant) | — | — | **0,75** | **0,95** |
| score_multimodal | **0,64** | **0,88** | 0,65 | 0,90 |
| precedence_temporelle | 0,58 | 0,81 | 0,60 | 0,78 |
| propagation_attention | 0,42 | 0,71 | 0,33 | 0,65 |
| propagation_degre | 0,42 | 0,72 | 0,33 | 0,67 |
| propagation_uniforme | 0,36 | 0,62 | 0,25 | 0,52 |

Tests de McNemar appariés (P@1) :

- arbre_spans_v1 contre score_multimodal : 13 contre 7 runs gagnés seul, p = 0,263 ;
- arbre_spans_v1 contre precedence_temporelle : 16 contre 7, p = 0,093 ;
- score_multimodal contre precedence_temporelle : 10 contre 5, p = 0,302 ;
- attention contre uniforme : 5 contre 0, p = 0,062.

**Aucune différence n'est significative entre les trois meilleurs
classements**, mais l'heuristique existante a désormais l'estimation la plus
haute (0,75 contre 0,65 sur les mêmes 60 runs).

Par système (P@1) : Online Boutique 0,68 / 0,68 (arbre de spans /
score_multimodal), Train Ticket 0,81 / 0,62, Sock Shop — / 0,62. Seul apport
net du score multimodal : **il fonctionne sans traces** (Sock Shop, 26 runs),
là où l'heuristique par arbre de spans ne peut rien classer.

Par type de faute (P@1, score_multimodal) : cpu 0,80, delay 0,80, disk 0,76,
socket 0,62, mem 0,44, loss 0,42. La mémoire et la perte de paquets sont les
plus difficiles.

### Ce que valent les poids d'attention GAT

**Rien de plus qu'une normalisation par le degré.** (Ce résultat porte sur
les traces, que la correction ne modifie pas.)

- L'entropie normalisée des lignes d'attention (1 = parfaitement uniforme)
  a une médiane de **1,000** (minimum 0,999) sur les 60 runs avec traces.
- `propagation_attention` et `propagation_degre` (1/(1+degré), sans aucun
  modèle) donnent des **P@1 identiques sur les 86 runs**, jusque dans chaque
  type de faute ; à P@3, ils ne diffèrent que d'un run (en faveur du degré).
  L'avantage apparent de l'attention sur `propagation_uniforme` (5 runs
  contre 0, p = 0,062) vient de cette normalisation.
- Tous les classements par propagation sont **nettement moins bons** que le
  simple score par service. Le marcheur « descend » vers les appelés, alors
  qu'une faute sur un appelant (par exemple checkoutservice) contamine ses
  appelés par effet de charge. Sur `delay`, P@1 = 0,10.

### Fausses pistes (runs sans faute)

Sur les 86 runs **normaux** de test (deuxième moitié contre première moitié
d'une fenêtre sans faute), les vues signalent en médiane **3 services
« notablement anormaux »** (aᵢ ≥ 0,5 ; écart interquartile 2 à 10,75 ;
jusqu'à 24), et le score maximal atteint le plafond (0,94) dans plus de la
moitié des runs normaux. Sur les runs anormaux : 8 services en médiane
(jusqu'à 50). Un service rouge ne suffit donc pas à conclure : les seuils
(écart robuste de 3) n'ont pas été calibrés, et avec la correction, les taux
bruts (trafic, CPU) varient assez d'une demi-fenêtre à l'autre pour saturer
le score.

### Limites de cette évaluation

- **Pas de jeu de test séparé.** Trois décisions ont été prises *après*
  avoir vu les résultats de ces mêmes 86 runs : le masquage des
  identifiants dans les logs, le départage des ex æquo (64 runs sur 86 ont
  plusieurs services au plafond de aᵢ ; sans départage, par ordre
  alphabétique, P@1 = 0,35) et l'ajout de l'ablation par le degré. Les
  chiffres de `score_multimodal` sont donc probablement un peu optimistes.
  RE3 (déjà acquis, chapitre 5) peut servir de confirmation indépendante.
- 86 runs, 10 à 17 par type de faute : les chiffres par faute ont de larges
  intervalles de confiance.
- Fautes injectées sur **un seul** service par cas. Rien n'est mesuré sur les
  pannes multiples.
- Le GAT utilisé pour l'attention a été **réentraîné** (graine 42) : le
  modèle qui a produit les embeddings du chapitre 3 n'avait pas été
  sauvegardé. Ses embeddings en cache ne sont pas modifiés.

## Section 5.3 et 6 : l'étude utilisateur

**Elle n'a pas été menée, et aucun chiffre de la Table 3 de l'article n'est
produit par ce code.** Ce qui existe est l'**instrumentation** pour la mener :

Le protocole complet (questions de recherche, hypothèses, effectif calculé
par simulation, analyse préenregistrée) est au
[chapitre 8](08-protocole-etude-utilisateur.md). L'instrumentation :

- `GET /study/tasks?participant_id=P01` : 15 tâches tirées de manière
  déterministe parmi les runs anormaux avec traces, en **3 blocs de 5** (un
  par condition), ordre des blocs en carré latin attribué à tour de rôle
  (P01 : ABC, P02 : BCA, P03 : CAB) ;
- mode « étude utilisateur » du dashboard : chronomètre, réponse, confiance
  (Likert 1-5), nombre d'interactions ;
- `POST /study/sessions` : correction **côté serveur**, jamais renvoyée au
  participant, écriture dans `experiments/user_study/sessions.jsonl` ;
- `./run.sh study-analysis` : analyse préenregistrée (modèle mixte pour
  l'hypothèse principale, Holm pour les secondaires, tests de rangs en
  sensibilité). Refuse de tourner sans sessions, et refuse les tests à moins
  de 5 participants ayant passé les 3 conditions ;
- `python -m src.causal.study_power` : calcul de puissance par simulation du plan exact.

Biais identifiés et traités :

- **Le `run_id` RCAEval contient le service fautif**
  (`RE2-TT/ts-travel-service_delay/2__abnormal`). En mode étude, il est
  remplacé par un alias opaque. Pour la condition A (outils externes),
  l'expérimentateur doit préparer des **copies anonymisées** des données :
  les noms de dossiers trahissent aussi la réponse.
- **Condition A** : le dashboard ne reproduit pas Grafana/Loki/Tempo. Il ne
  fait que chronométrer. Les participants doivent utiliser de vrais outils.
- NASA-TLX et SUS se collectent hors outil.

Reste à faire, hors code : avis d'un comité d'éthique, consentement éclairé,
recrutement, environnement Grafana/Loki/Jaeger de la condition A, étude
pilote (voir le [chapitre 8](08-protocole-etude-utilisateur.md)).

## Ce que l'article doit changer pour rester honnête

1. **Abstract, section 7.3** : ne pas présenter les poids d'attention GAT
   comme un signal causal exploitable. Mesuré ici : ils sont uniformes et
   n'apportent rien au-delà d'une normalisation par le degré.
2. **Section 3.1** : les nœuds sont des services (RCAEval). Les scénarios
   SC1 (échec de build), SC2 (test flaky) et SC3 (erreur de déploiement)
   **n'existent pas** dans les données. RCAEval injecte cpu/mem/disk/delay/
   loss/socket. SC4 (pic de latence) correspond approximativement à
   `delay`, SC5 (fuite de ressources) à `mem`.
3. **Section 5.1** : le jeu « Astronomy Shop étendu, 5 000 exécutions
   annotées » n'est pas dans ce dépôt. Les résultats ci-dessus portent sur
   RCAEval RE2 (86 runs anormaux de test).
4. **Sections 1 et 2.1** : le F1 = 0,936 / 0,958 attribué à la v1 n'est pas
   reproduit par ce code. Meilleur F1 mesuré : 0,86 (fusion tardive, 23 runs
   de test RE2), 0,59 sur RE2+RE3 regroupés (chapitres 5 et 6).
5. **Section 4.2** : retirer Three.js, préciser les limites de JSON-LD.
6. **Section 6** : Table 3, pourcentages, 73 % de « découverte en amont »,
   citations : à remplacer par les données réelles de l'étude, ou à
   supprimer. Une formulation au conditionnel ne suffit pas si les chiffres
   restent dans le texte.
