"""
Connecteur Jenkins — builds + logs de console d'un job

Nécessite un utilisateur + jeton d'API Jenkins (JENKINS_USER/JENKINS_TOKEN).
fetch() échoue explicitement si l'un des deux est absent, même logique que
github_actions.py::_require_token() — aucun appel n'est fait sans
authentification.

Limite assumée: jobs Jenkins de premier niveau uniquement (pas de dossiers
imbriqués /job/x/job/y/) — Jenkins n'a pas d'API "globale" comme GitHub
(chaque instance est auto-hébergée), et la prise en charge des dossiers
imbriqués n'a pas pu être vérifiée faute d'instance Jenkins réelle
accessible depuis cet environnement; documenté plutôt que deviné.

Contrairement à github_actions.py (qui télécharge les logs de console mais
ne les unifie jamais vers le schéma logs), ce connecteur va plus loin:
parse() mine les templates des lignes de console via drain3
(src/data/log_parsing.py) et dérive un label -1/1 depuis le statut du build
— la même discipline "démontrer que le pipeline marche, pas seulement que
les données sont téléchargées" que le reste du projet applique déjà à
RCAEval (cf. évaluations protocolées).
"""

from pathlib import Path
import json
import logging
import os

import pandas as pd
import requests

from src.data.sources.base import BaseConnector
from src.data import schema
from src.data.log_parsing import mine_templates

logger = logging.getLogger(__name__)

# Résultats de build Jenkins terminaux uniquement (result=null => encore en
# cours, exclu). SUCCESS -> normal, tout le reste -> anomalie.
_FAILURE_RESULTS = {"FAILURE", "UNSTABLE", "ABORTED"}


def _require_credentials() -> tuple:
    user = os.environ.get("JENKINS_USER")
    token = os.environ.get("JENKINS_TOKEN")
    if not user or not token:
        raise EnvironmentError(
            "JENKINS_USER et/ou JENKINS_TOKEN non définis. Crée un jeton d'API "
            "Jenkins (menu utilisateur -> Configure -> API Token) puis exporte: "
            "export JENKINS_USER=... JENKINS_TOKEN=... avant de relancer cette commande."
        )
    return user, token


class JenkinsConnector(BaseConnector):
    name = "jenkins"

    def fetch(self, force: bool = False, jenkins_url: str = None, job: str = None, max_builds: int = 100, **kwargs) -> None:
        """
        Récupère la liste des builds d'un job Jenkins et leurs logs de console.

        Args:
            jenkins_url: URL de base de l'instance Jenkins (ex: 'https://ci.example.com')
            job: nom du job (premier niveau uniquement, cf. docstring de module)
            max_builds: nombre maximum de builds à récupérer
        """
        if not jenkins_url:
            raise ValueError("Le paramètre 'jenkins_url' est requis (ex: 'https://ci.example.com').")
        if not job:
            raise ValueError("Le paramètre 'job' est requis (nom du job Jenkins).")
        auth = _require_credentials()

        job_dir = self.raw_dir / job
        logs_dir = job_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        response = requests.get(
            f"{jenkins_url.rstrip('/')}/job/{job}/api/json",
            params={"tree": f"builds[number,timestamp,duration,result]{{0,{max_builds}}}"},
            auth=auth,
            timeout=30,
        )
        response.raise_for_status()
        builds = response.json().get("builds", [])

        (job_dir / "builds.json").write_text(json.dumps(builds), encoding="utf-8")
        logger.info(f"{len(builds)} builds récupérés pour le job '{job}'")

        for build in builds:
            number = build["number"]
            log_path = logs_dir / f"{number}.txt"
            if log_path.exists() and not force:
                continue
            log_response = requests.get(
                f"{jenkins_url.rstrip('/')}/job/{job}/{number}/consoleText", auth=auth, timeout=30
            )
            if log_response.status_code == 200:
                log_path.write_text(log_response.text, encoding="utf-8")
            else:
                logger.warning(f"Console log indisponible pour le build {number} (statut {log_response.status_code})")

        logger.info(f"Jenkins job '{job}': builds + logs téléchargés -> {job_dir}")

    def parse(self, job: str = None, **kwargs) -> None:
        """
        Normalise les métadonnées de build vers une table tabulaire et les
        logs de console (via drain3) vers le schéma logs unifié + labels.
        """
        if not job:
            raise ValueError("Le paramètre 'job' est requis (même valeur que pour fetch()).")

        job_dir = self.raw_dir / job
        builds_path = job_dir / "builds.json"
        if not builds_path.exists():
            raise FileNotFoundError(f"'{builds_path}' introuvable. Lance fetch(jenkins_url=..., job='{job}') d'abord.")

        builds = json.loads(builds_path.read_text(encoding="utf-8"))

        builds_df = pd.DataFrame(
            {
                "run_id": [str(b["number"]) for b in builds],
                "job": job,
                "result": [b.get("result") for b in builds],
                "duration_ms": [b.get("duration") for b in builds],
                "timestamp": pd.to_datetime([b.get("timestamp") for b in builds], unit="ms", errors="coerce"),
            }
        )

        # Collecte (run_id, ligne) pour TOUS les builds d'abord, puis un seul
        # appel à mine_templates() sur le corpus entier: chaque appel
        # instancie un nouveau TemplateMiner (cf. log_parsing.py), donc miner
        # build par build donnerait des cluster_id incomparables d'un build à
        # l'autre (le cluster 1 du build 42 n'aurait aucun rapport avec le
        # cluster 1 du build 100) — inutilisable par
        # features.py::build_event_count_matrix, qui suppose un template_id
        # cohérent sur tout le corpus (comme le cluster_id déjà cohérent de
        # LogHub/RCAEval, calculé en amont sur l'ensemble du dataset).
        run_ids_per_line, lines, labels_rows = [], [], []
        for build in builds:
            result = build.get("result")
            if result is None:
                continue  # build encore en cours, pas de verdict à labelliser

            number = build["number"]
            run_id = str(number)
            log_path = job_dir / "logs" / f"{number}.txt"
            if not log_path.exists():
                continue

            build_lines = [line for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
            if not build_lines:
                continue

            lines.extend(build_lines)
            run_ids_per_line.extend([run_id] * len(build_lines))
            labels_rows.append({
                "run_id": run_id,
                "source": "jenkins",
                "label": -1 if result in _FAILURE_RESULTS else 1,
                "fault_type": result,
            })

        template_ids = mine_templates(lines) if lines else []
        logs_rows = [
            {
                "timestamp": None,
                "source": "jenkins",
                "run_id": run_id,
                "service": job,
                "level": None,
                "template_id": template_id,
                "message": line,
                "label": None,
            }
            for run_id, line, template_id in zip(run_ids_per_line, lines, template_ids)
        ]

        logs_df = pd.DataFrame(logs_rows, columns=schema.LOGS_COLUMNS)
        labels_df = pd.DataFrame(labels_rows, columns=schema.LABELS_COLUMNS)
        schema.validate_logs_df(logs_df)
        schema.validate_labels_df(labels_df)

        out_dir = self.interim_dir / job
        out_dir.mkdir(parents=True, exist_ok=True)
        builds_df.to_parquet(out_dir / "builds.parquet", index=False)
        logs_df.to_parquet(out_dir / "logs.parquet", index=False)
        labels_df.to_parquet(out_dir / "labels.parquet", index=False)
        logger.info(
            f"Jenkins job '{job}' parsé: {len(builds_df)} builds, {len(logs_df)} lignes de log, "
            f"{len(labels_df)} builds labellisés -> {out_dir}"
        )
