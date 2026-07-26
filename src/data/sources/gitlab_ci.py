"""
Connecteur GitLab CI — pipelines + traces de jobs d'un projet

Nécessite un jeton d'accès personnel GitLab (GITLAB_TOKEN, scope 'read_api').
fetch() échoue explicitement si absent, même logique que
github_actions.py::_require_token() — aucun appel n'est fait sans
authentification.

Comme jenkins.py (chapitre 9, généricité) et contrairement à
github_actions.py (qui télécharge des logs mais ne les unifie jamais):
parse() mine les templates des traces de jobs via drain3
(src/data/log_parsing.py) et dérive un label -1/1 depuis le statut du
pipeline.
"""

from pathlib import Path
from urllib.parse import quote
import json
import logging
import os

import pandas as pd
import requests

from src.data.sources.base import BaseConnector
from src.data import schema
from src.data.log_parsing import mine_templates

logger = logging.getLogger(__name__)

# États terminaux GitLab CI (running/pending/created/skipped/manual exclus:
# pas de verdict définitif à labelliser).
_TERMINAL_STATUSES = {"success", "failed", "canceled"}
_FAILURE_STATUSES = {"failed", "canceled"}


def _require_token() -> str:
    token = os.environ.get("GITLAB_TOKEN")
    if not token:
        raise EnvironmentError(
            "GITLAB_TOKEN non défini. Crée un jeton d'accès personnel GitLab "
            "(Préférences -> Access Tokens, scope 'read_api') puis exporte: "
            "export GITLAB_TOKEN=... avant de relancer cette commande."
        )
    return token


class GitLabCIConnector(BaseConnector):
    name = "gitlab_ci"

    def fetch(self, force: bool = False, project: str = None, gitlab_url: str = "https://gitlab.com", max_pipelines: int = 100, **kwargs) -> None:
        """
        Récupère les pipelines d'un projet GitLab, leurs jobs, et les traces
        (logs bruts) de chaque job.

        Args:
            project: ID numérique ou chemin 'namespace/nom' du projet
            gitlab_url: URL de base de l'instance GitLab (défaut: gitlab.com)
            max_pipelines: nombre maximum de pipelines à récupérer
        """
        if not project:
            raise ValueError("Le paramètre 'project' est requis (ID numérique ou 'namespace/nom').")
        token = _require_token()

        headers = {"PRIVATE-TOKEN": token}
        project_encoded = quote(str(project), safe="")
        project_dir = self.raw_dir / str(project).replace("/", "@")
        pipelines_dir = project_dir / "pipelines"
        logs_dir = project_dir / "logs"
        pipelines_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        pipelines = []
        page = 1
        while len(pipelines) < max_pipelines:
            response = requests.get(
                f"{gitlab_url.rstrip('/')}/api/v4/projects/{project_encoded}/pipelines",
                headers=headers,
                params={"per_page": 100, "page": page},
                timeout=30,
            )
            response.raise_for_status()
            batch = response.json()
            if not batch:
                break
            pipelines.extend(batch)
            page += 1

        pipelines = pipelines[:max_pipelines]
        (pipelines_dir / "pipelines.json").write_text(json.dumps(pipelines), encoding="utf-8")
        logger.info(f"{len(pipelines)} pipelines récupérés pour '{project}'")

        for pipeline in pipelines:
            pipeline_id = pipeline["id"]
            jobs_path = logs_dir / f"{pipeline_id}_jobs.json"
            if jobs_path.exists() and not force:
                continue

            jobs_response = requests.get(
                f"{gitlab_url.rstrip('/')}/api/v4/projects/{project_encoded}/pipelines/{pipeline_id}/jobs",
                headers=headers,
                timeout=30,
            )
            if jobs_response.status_code != 200:
                logger.warning(f"Jobs indisponibles pour le pipeline {pipeline_id} (statut {jobs_response.status_code})")
                continue
            jobs = jobs_response.json()
            jobs_path.write_text(json.dumps(jobs), encoding="utf-8")

            for job in jobs:
                job_id = job["id"]
                trace_path = logs_dir / f"{pipeline_id}_{job_id}.txt"
                if trace_path.exists() and not force:
                    continue
                trace_response = requests.get(
                    f"{gitlab_url.rstrip('/')}/api/v4/projects/{project_encoded}/jobs/{job_id}/trace",
                    headers=headers,
                    timeout=30,
                )
                if trace_response.status_code == 200:
                    trace_path.write_text(trace_response.text, encoding="utf-8")
                else:
                    logger.warning(f"Trace indisponible pour le job {job_id} (statut {trace_response.status_code})")

        logger.info(f"GitLab CI '{project}': pipelines + jobs + traces téléchargés -> {project_dir}")

    def parse(self, project: str = None, **kwargs) -> None:
        """
        Normalise les métadonnées de pipeline vers une table tabulaire et les
        traces de jobs (via drain3) vers le schéma logs unifié + labels.
        """
        if not project:
            raise ValueError("Le paramètre 'project' est requis (même valeur que pour fetch()).")

        project_dir = self.raw_dir / str(project).replace("/", "@")
        pipelines_path = project_dir / "pipelines" / "pipelines.json"
        if not pipelines_path.exists():
            raise FileNotFoundError(f"'{pipelines_path}' introuvable. Lance fetch(project='{project}') d'abord.")

        pipelines = json.loads(pipelines_path.read_text(encoding="utf-8"))
        logs_dir = project_dir / "logs"

        pipelines_df = pd.DataFrame(
            {
                "run_id": [str(p["id"]) for p in pipelines],
                "project": str(project),
                "status": [p.get("status") for p in pipelines],
                "created_at": pd.to_datetime([p.get("created_at") for p in pipelines], errors="coerce"),
                "updated_at": pd.to_datetime([p.get("updated_at") for p in pipelines], errors="coerce"),
                "ref": [p.get("ref") for p in pipelines],
            }
        )
        pipelines_df["duration_s"] = (pipelines_df["updated_at"] - pipelines_df["created_at"]).dt.total_seconds()

        # Même précaution que jenkins.py: un seul mine_templates() sur le
        # corpus entier (tous pipelines confondus), pas un par pipeline —
        # sinon les template_id ne seraient pas comparables d'un pipeline à
        # l'autre (cf. commentaire détaillé dans jenkins.py::parse()).
        run_ids_per_line, lines, labels_rows = [], [], []
        for pipeline in pipelines:
            status = pipeline.get("status")
            if status not in _TERMINAL_STATUSES:
                continue

            pipeline_id = pipeline["id"]
            run_id = str(pipeline_id)
            jobs_path = logs_dir / f"{pipeline_id}_jobs.json"
            if not jobs_path.exists():
                continue
            jobs = json.loads(jobs_path.read_text(encoding="utf-8"))

            pipeline_lines = []
            for job in jobs:
                job_id = job["id"]
                trace_path = logs_dir / f"{pipeline_id}_{job_id}.txt"
                if not trace_path.exists():
                    continue
                pipeline_lines.extend(
                    line for line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()
                )

            if not pipeline_lines:
                continue

            lines.extend(pipeline_lines)
            run_ids_per_line.extend([run_id] * len(pipeline_lines))
            labels_rows.append({
                "run_id": run_id,
                "source": "gitlab_ci",
                "label": -1 if status in _FAILURE_STATUSES else 1,
                "fault_type": status,
            })

        template_ids = mine_templates(lines) if lines else []
        logs_rows = [
            {
                "timestamp": None,
                "source": "gitlab_ci",
                "run_id": run_id,
                "service": str(project),
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

        out_dir = self.interim_dir / str(project).replace("/", "@")
        out_dir.mkdir(parents=True, exist_ok=True)
        pipelines_df.to_parquet(out_dir / "pipelines.parquet", index=False)
        logs_df.to_parquet(out_dir / "logs.parquet", index=False)
        labels_df.to_parquet(out_dir / "labels.parquet", index=False)
        logger.info(
            f"GitLab CI '{project}' parsé: {len(pipelines_df)} pipelines, {len(logs_df)} lignes de log, "
            f"{len(labels_df)} pipelines labellisés -> {out_dir}"
        )
