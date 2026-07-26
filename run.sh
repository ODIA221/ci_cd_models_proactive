#!/usr/bin/env bash
# Point d'entrée unique du projet LogPipeGuard.
#
# Usage:
#   ./run.sh                    # installe l'environnement puis lance la démo
#                                # de détection d'anomalies sur les données d'exemple
#   ./run.sh setup               # installe seulement l'environnement (.venv + requirements.txt)
#   ./run.sh demo                # relance la démo (identique au comportement par défaut)
#   ./run.sh sources              # liste les sources de données externes et leur statut
#   ./run.sh acquire <args...>    # transmet les arguments à `python -m src.data.acquire`
#                                  # ex: ./run.sh acquire --source loghub --dataset hdfs
#                                  # ex: ./run.sh acquire --source rcaeval --subset RE2
#                                  # ex: ./run.sh acquire --source jenkins --jenkins-url https://ci.example.com --job my-job
#                                  # ex: ./run.sh acquire --source gitlab_ci --project namespace/nom
#   ./run.sh evaluate             # évaluation protocolée (précision/rappel/F1/AUC) sur
#                                  # données labellisées (LogHub HDFS par défaut)
#   ./run.sh train-rcaeval <args...>     # entraîne un modèle RCAEval PERSISTANT, servable par
#                                  # l'API/dashboard (triplet .joblib/_preprocessor.joblib/_meta.json)
#                                  # ex: ./run.sh train-rcaeval --model-type multimodal_autoencoder
#                                  # ex: ./run.sh train-rcaeval --model-type isolation_forest --horizon-seconds 60
#                                  # nécessite d'avoir acquis RCAEval au préalable (./run.sh acquire --source rcaeval --subset RE2)
#   ./run.sh evaluate-multimodal <args...>  # compare fusion tardive vs fusion neuronale jointe vs
#                                  # mono-modalité sur RCAEval (précision/rappel/F1/AUC réels)
#   ./run.sh evaluate-causal <args...>      # évalue la corrélation causale (precision@1/@3 contre
#                                  # le service fautif réellement injecté, RCAEval)
#   ./run.sh evaluate-proactive <args...>   # mesure le délai de détection minimal (proactivité):
#                                  # précision/rappel/F1/AUC par horizon (15s à 720s post-incident)
#   ./run.sh showcase-rcaeval        # PRÉPARE le dashboard: acquiert RCAEval RE2 si besoin (~4.2GB,
#                                  # une seule fois) puis entraîne 3 modèles servables — isolation_forest
#                                  # (fenêtre complète, pour la corrélation causale), multimodal_autoencoder
#                                  # (architecture hybride), isolation_forest --horizon-seconds 60
#                                  # (proactivité). Prend plusieurs minutes. Ensuite: ./run.sh start,
#                                  # choisir un modèle + la source "RCAEval (test, avec chaîne causale)"
#                                  # dans le dashboard. La généricité (Jenkins/GitLab CI) n'est PAS
#                                  # démontrable dans le dashboard: ce sont des connecteurs de données
#                                  # (cf. ./run.sh acquire), pas des modèles à visualiser. Pour Jenkins,
#                                  # ./run.sh jenkins-up démarre une instance locale à des fins de test.
#   ./run.sh jenkins-up             # démarre un Jenkins local (Docker, jenkins/jenkins:lts) pour tester
#                                  # le connecteur Jenkins sans instance réelle. Affiche le mot de passe
#                                  # initial à la première installation. Nécessite Docker démarré.
#   ./run.sh jenkins-down            # arrête le conteneur Jenkins local (conserve son volume de données)
#   ./run.sh otel-up               # clone (si besoin) ../opentelemetry-demo et lance le stack
#                                  # Docker Compose (nécessite Docker installé et démarré)
#   ./run.sh otel-down             # arrête le stack OpenTelemetry Demo
#   ./run.sh serve                 # démarre l'API FastAPI (http://localhost:8000, docs: /docs)
#   ./run.sh dashboard              # démarre le dashboard Streamlit (http://localhost:8501)
#                                  # nécessite l'API lancée dans un autre terminal (./run.sh serve)
#   ./run.sh start                  # tout-en-un: API en arrière-plan + dashboard au premier plan,
#                                  # ouvre le navigateur, arrête l'API automatiquement à la sortie (Ctrl+C)
#   ./run.sh stop                   # arrête l'API laissée en arrière-plan par `start`
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

VENV_DIR=".venv"
PYTHON_BIN="$VENV_DIR/bin/python3"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "==> Création de l'environnement virtuel ($VENV_DIR)..."
    python3 -m venv "$VENV_DIR"
fi

echo "==> Installation des dépendances (requirements.txt)..."
"$PYTHON_BIN" -m pip install --quiet --upgrade pip
"$PYTHON_BIN" -m pip install --quiet -r requirements.txt

COMMAND="${1:-demo}"

case "$COMMAND" in
    setup)
        echo "==> Environnement prêt (.venv)."
        ;;

    demo)
        echo "==> Démo: détection d'anomalies (Isolation Forest) sur data/raw/metrics/dataset_metrics.csv"
        echo "    (premier lancement ~2-3 min, incluant l'imputation KNN sur 41k lignes)"
        "$PYTHON_BIN" src/models/train.py
        echo
        echo "==> Terminé. Modèle sauvegardé dans models/, résultats dans experiments/."
        ;;

    sources)
        "$PYTHON_BIN" -m src.data.acquire --list
        ;;

    acquire)
        shift
        "$PYTHON_BIN" -m src.data.acquire "$@"
        ;;

    evaluate)
        shift
        "$PYTHON_BIN" src/models/evaluate.py "$@"
        ;;

    train-rcaeval)
        shift
        echo "==> Entraînement d'un modèle RCAEval persistant (servable par l'API/dashboard)..."
        "$PYTHON_BIN" src/models/train_rcaeval.py "$@"
        ;;

    evaluate-multimodal)
        shift
        "$PYTHON_BIN" src/models/evaluate_multimodal.py "$@"
        ;;

    evaluate-causal)
        shift
        "$PYTHON_BIN" src/models/evaluate_causal.py "$@"
        ;;

    evaluate-proactive)
        shift
        "$PYTHON_BIN" src/models/evaluate_proactive.py "$@"
        ;;

    showcase-rcaeval)
        echo "==> Préparation de la démo dashboard (corrélation causale, architecture hybride, proactivité)."
        echo "    La généricité (Jenkins/GitLab CI) n'est pas incluse: ce sont des connecteurs de données,"
        echo "    pas des modèles à visualiser, et nécessitent un vrai serveur/jeton (cf. ./run.sh acquire)."
        echo

        if [ ! -f "data/interim/rcaeval/RE2/features.parquet" ]; then
            echo "==> RCAEval RE2 non trouvé, acquisition (fetch+parse, ~4.2GB, peut prendre du temps)..."
            "$PYTHON_BIN" -m src.data.acquire --source rcaeval --subset RE2
        else
            echo "==> RCAEval RE2 déjà acquis (data/interim/rcaeval/RE2/features.parquet), réutilisation."
        fi

        echo
        echo "==> [1/3] isolation_forest, fenêtre complète (corrélation causale)..."
        "$PYTHON_BIN" src/models/train_rcaeval.py --model-type isolation_forest

        echo
        echo "==> [2/3] multimodal_autoencoder, fenêtre complète (architecture hybride)..."
        "$PYTHON_BIN" src/models/train_rcaeval.py --model-type multimodal_autoencoder

        echo
        echo "==> [3/3] isolation_forest, horizon 60s (proactivité)..."
        "$PYTHON_BIN" src/models/train_rcaeval.py --model-type isolation_forest --horizon-seconds 60

        echo
        echo "==> Terminé. Lance ./run.sh start, puis dans le dashboard:"
        echo "    1. Choisis un modèle dans la barre latérale (isolation_forest_rcaeval_*,"
        echo "       multimodal_autoencoder_rcaeval_*, ou celui avec horizon_seconds=60)"
        echo "    2. Sélectionne la source 'RCAEval (test, avec chaîne causale)'"
        echo "    3. Clique 'Lancer la détection', puis choisis une anomalie sous"
        echo "       'Chaîne causale' pour voir le service fautif identifié"
        ;;

    jenkins-up)
        if ! command -v docker >/dev/null 2>&1; then
            echo "Docker n'est pas installé (ou pas dans le PATH). Installe Docker Desktop avant de relancer."
            exit 1
        fi
        if ! docker info >/dev/null 2>&1; then
            echo "Docker est installé mais le daemon ne répond pas. Démarre Docker Desktop puis relance."
            exit 1
        fi

        if docker ps -a --format '{{.Names}}' | grep -qx jenkins; then
            echo "==> Conteneur 'jenkins' déjà présent, démarrage (docker start)..."
            docker start jenkins >/dev/null
        else
            echo "==> Création du conteneur Jenkins (jenkins/jenkins:lts, premier lancement: téléchargement de l'image)..."
            docker run -d --name jenkins -p 8080:8080 -p 50000:50000 -v jenkins_home:/var/jenkins_home jenkins/jenkins:lts >/dev/null
        fi

        echo "==> Attente du démarrage de Jenkins..."
        READY=false
        for _ in $(seq 1 60); do
            CODE="$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8080/login 2>/dev/null || true)"
            if [ "$CODE" = "200" ] || [ "$CODE" = "403" ]; then
                READY=true
                break
            fi
            sleep 2
        done

        echo
        if [ "$READY" = "true" ]; then
            echo "==> Jenkins accessible sur http://localhost:8080"
        else
            echo "==> Jenkins ne répond pas encore après 2 min — laisse-lui un peu plus de temps, ou vérifie: docker logs jenkins"
        fi

        if docker exec jenkins test -f /var/jenkins_home/secrets/initialAdminPassword 2>/dev/null; then
            echo "==> Mot de passe initial (première installation uniquement, à coller sur http://localhost:8080):"
            docker exec jenkins cat /var/jenkins_home/secrets/initialAdminPassword
        fi

        echo
        echo "Ensuite:"
        echo "  1. Termine la configuration sur http://localhost:8080 si besoin (plugins suggérés, compte admin)"
        echo "  2. Crée un job de test (Freestyle project, une étape shell) et lance-le plusieurs fois"
        echo "  3. Génère un jeton: menu utilisateur -> Configure -> API Token"
        echo "  4. export JENKINS_USER=... JENKINS_TOKEN=..."
        echo "  5. ./run.sh acquire --source jenkins --jenkins-url http://localhost:8080 --job <nom-du-job>"
        echo "Pour arrêter: ./run.sh jenkins-down"
        ;;

    jenkins-down)
        if docker ps -a --format '{{.Names}}' | grep -qx jenkins; then
            docker stop jenkins >/dev/null
            echo "==> Conteneur Jenkins arrêté (volume conservé; ./run.sh jenkins-up pour le relancer)."
        else
            echo "Aucun conteneur 'jenkins' trouvé, rien à arrêter."
        fi
        ;;

    otel-up)
        OTEL_DEMO_DIR="../opentelemetry-demo"

        if ! command -v docker >/dev/null 2>&1; then
            echo "Docker n'est pas installé (ou pas dans le PATH). Installe Docker Desktop avant de relancer."
            exit 1
        fi
        if ! docker info >/dev/null 2>&1; then
            echo "Docker est installé mais le daemon ne répond pas. Démarre Docker Desktop puis relance."
            exit 1
        fi

        if [ ! -d "$OTEL_DEMO_DIR" ]; then
            echo "==> Clone d'opentelemetry-demo dans $OTEL_DEMO_DIR (dossier frère, non vendored)..."
            git clone https://github.com/open-telemetry/opentelemetry-demo "$OTEL_DEMO_DIR"
        fi

        echo "==> Lancement du stack (docker compose up -d)..."
        (cd "$OTEL_DEMO_DIR" && docker compose up --no-build -d)

        echo
        echo "==> Stack démarré. Interfaces:"
        echo "    Boutique   : http://localhost:8080"
        echo "    Grafana    : http://localhost:8080/grafana"
        echo "    Jaeger UI  : http://localhost:8080/jaeger/ui"
        echo "    Prometheus : http://localhost:9090"
        echo
        echo "Une fois prêt, exporte les données avec: ./run.sh acquire --source otel_demo --export all"
        echo "Pour arrêter: ./run.sh otel-down"
        ;;

    otel-down)
        OTEL_DEMO_DIR="../opentelemetry-demo"
        if [ ! -d "$OTEL_DEMO_DIR" ]; then
            echo "'$OTEL_DEMO_DIR' introuvable, rien à arrêter."
            exit 1
        fi
        (cd "$OTEL_DEMO_DIR" && docker compose down)
        ;;

    serve)
        shift
        echo "==> Démarrage de l'API FastAPI (http://localhost:8000, docs interactives: /docs)..."
        "$PYTHON_BIN" -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000 "$@"
        ;;

    dashboard)
        shift
        echo "==> Démarrage du dashboard Streamlit (http://localhost:8501)..."
        echo "    Assure-toi que l'API tourne dans un autre terminal: ./run.sh serve"
        "$PYTHON_BIN" -m streamlit run src/dashboard/app.py "$@"
        ;;

    start)
        shift
        RUN_DIR=".run"
        API_LOG="$RUN_DIR/api.log"
        API_PID_FILE="$RUN_DIR/api.pid"
        mkdir -p "$RUN_DIR"

        if curl -s -o /dev/null http://localhost:8000/health; then
            echo "==> API déjà accessible sur le port 8000, réutilisation."
            STARTED_API=false
        else
            echo "==> Démarrage de l'API en arrière-plan (logs: $API_LOG)..."
            nohup "$PYTHON_BIN" -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 > "$API_LOG" 2>&1 &
            echo $! > "$API_PID_FILE"
            STARTED_API=true

            printf "    Attente de l'API"
            READY=false
            for _ in $(seq 1 30); do
                if curl -s -o /dev/null http://localhost:8000/health; then
                    READY=true
                    break
                fi
                printf "."
                sleep 1
            done
            echo

            if [ "$READY" != "true" ]; then
                echo "L'API ne répond toujours pas après 30s. Vérifie $API_LOG."
                exit 1
            fi
            echo "==> API prête sur http://localhost:8000 (docs: /docs)."
        fi

        cleanup() {
            echo
            if [ -n "${DASHBOARD_PID:-}" ] && kill -0 "$DASHBOARD_PID" 2>/dev/null; then
                echo "==> Arrêt du dashboard (PID $DASHBOARD_PID)..."
                kill "$DASHBOARD_PID" 2>/dev/null || true
            fi
            if [ "$STARTED_API" = "true" ] && [ -f "$API_PID_FILE" ]; then
                echo "==> Arrêt de l'API (PID $(cat "$API_PID_FILE"))..."
                kill "$(cat "$API_PID_FILE")" 2>/dev/null || true
                rm -f "$API_PID_FILE"
            fi
        }
        # macOS embarque bash 3.2 (2007, gelé pour raisons de licence GPLv3), où
        # un `trap` sur INT/TERM ne se déclenche PAS immédiatement si bash est
        # bloqué dans un `wait` sur un job en arrière-plan (contrairement à bash
        # 4.4+) — le signal reste en attente jusqu'à la fin du `wait`. On
        # contourne ça avec une boucle de `sleep` courts: le trap positionne un
        # simple indicateur, vérifié entre deux `sleep`, plutôt que de dépendre
        # d'un `wait` interrompu.
        STOP=0
        trap 'STOP=1' INT TERM
        trap cleanup EXIT

        (
            sleep 2
            if command -v open >/dev/null 2>&1; then
                open http://localhost:8501
            elif command -v xdg-open >/dev/null 2>&1; then
                xdg-open http://localhost:8501
            else
                echo "Ouvre manuellement: http://localhost:8501"
            fi
        ) &

        echo "==> Démarrage du dashboard (http://localhost:8501)... (Ctrl+C pour tout arrêter)"
        "$PYTHON_BIN" -m streamlit run src/dashboard/app.py "$@" &
        DASHBOARD_PID=$!

        while kill -0 "$DASHBOARD_PID" 2>/dev/null; do
            if [ "$STOP" = "1" ]; then
                break
            fi
            sleep 0.5
        done
        ;;

    stop)
        # Filet de sécurité indépendant du bon déclenchement du trap INT/TERM de
        # `start` (peu fiable selon le contexte de lancement, cf. commentaire dans
        # `start`): nettoie toujours par port, que le nettoyage automatique ait
        # fonctionné ou non.
        API_PID_FILE=".run/api.pid"
        rm -f "$API_PID_FILE"

        if lsof -ti:8000 >/dev/null 2>&1; then
            lsof -ti:8000 | xargs kill -9 2>/dev/null || true
            echo "==> API (port 8000) arrêtée."
        else
            echo "Rien à arrêter sur le port 8000 (API)."
        fi

        if lsof -ti:8501 >/dev/null 2>&1; then
            lsof -ti:8501 | xargs kill -9 2>/dev/null || true
            echo "==> Dashboard (port 8501) arrêté."
        else
            echo "Rien à arrêter sur le port 8501 (dashboard)."
        fi
        ;;

    *)
        echo "Commande inconnue: '$COMMAND'"
        echo "Usage: ./run.sh [setup|demo|sources|acquire <args...>|evaluate|train-rcaeval <args...>|evaluate-multimodal <args...>|evaluate-causal <args...>|evaluate-proactive <args...>|showcase-rcaeval|jenkins-up|jenkins-down|otel-up|otel-down|serve|dashboard|start|stop]"
        exit 1
        ;;
esac
