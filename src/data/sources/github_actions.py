"""
Connecteur GitHub Actions — runs de workflow + logs d'exécution

Nécessite un token personnel GitHub (variable d'environnement GITHUB_TOKEN).
fetch() échoue explicitement si le token est absent: aucun appel API n'est
fait sans authentification (limite d'API anonyme trop basse pour un usage
réel, et on veut éviter tout appel accidentel).

Alternative pour l'historique des fichiers de workflow (sans télémétrie de run):
outil `gigawork` (Cardoen et al., MSR 2024), installable via pip, cf. registry.yaml.
GHALogs (Moriconi et al., MSR'25) est référencé mais son lien exact n'a pas pu
être vérifié - voir le TODO dans registry.yaml plutôt que d'inventer une URL.
"""

from pathlib import Path
import logging
import os
import time

import pandas as pd
import requests

from src.data.sources.base import BaseConnector
from src.data import schema

logger = logging.getLogger(__name__)

API_BASE = "https://api.github.com"


def _require_token() -> str:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise EnvironmentError(
            "GITHUB_TOKEN non défini. Crée un token personnel GitHub "
            "(https://github.com/settings/tokens, scope 'repo' ou 'actions:read') "
            "puis exporte-le: export GITHUB_TOKEN=... avant de relancer cette commande."
        )
    return token


class GitHubActionsConnector(BaseConnector):
    name = "github_actions"

    def fetch(self, force: bool = False, repo: str = None, max_runs: int = 100, **kwargs) -> None:
        """
        Récupère les runs de workflow et leurs logs pour un dépôt GitHub.

        Args:
            repo: identifiant '{owner}/{name}'
            max_runs: nombre maximum de runs à récupérer (pagination GitHub Actions)
        """
        token = _require_token()
        if not repo:
            raise ValueError("Le paramètre 'repo' (ex: 'owner/name') est requis.")

        headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
        repo_dir = self.raw_dir / repo.replace("/", "@")
        runs_dir = repo_dir / "runs"
        logs_dir = repo_dir / "logs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        runs = []
        page = 1
        while len(runs) < max_runs:
            response = requests.get(
                f"{API_BASE}/repos/{repo}/actions/runs",
                headers=headers,
                params={"per_page": 100, "page": page},
                timeout=30,
            )
            response.raise_for_status()
            batch = response.json().get("workflow_runs", [])
            if not batch:
                break
            runs.extend(batch)
            page += 1

        runs = runs[:max_runs]
        pd.DataFrame(runs).to_json(runs_dir / "runs.json", orient="records")
        logger.info(f"{len(runs)} runs récupérés pour '{repo}'")

        for run in runs:
            run_id = run["id"]
            log_path = logs_dir / f"{run_id}.zip"
            if log_path.exists() and not force:
                continue
            log_response = requests.get(
                f"{API_BASE}/repos/{repo}/actions/runs/{run_id}/logs", headers=headers, timeout=30
            )
            if log_response.status_code == 200:
                log_path.write_bytes(log_response.content)
            else:
                logger.warning(f"Logs indisponibles pour le run {run_id} (statut {log_response.status_code})")
            time.sleep(0.2)  # ménage la limite de taux de l'API GitHub

        logger.info(f"GitHub Actions '{repo}': runs + logs téléchargés -> {repo_dir}")

    def parse(self, repo: str = None, **kwargs) -> None:
        """
        Normalise les métadonnées de runs vers une table tabulaire (statut,
        durée, conclusion) et les logs associés vers le schéma logs unifié.
        """
        if not repo:
            raise ValueError("Le paramètre 'repo' est requis (même valeur que pour fetch()).")

        repo_dir = self.raw_dir / repo.replace("/", "@")
        runs_path = repo_dir / "runs" / "runs.json"
        if not runs_path.exists():
            raise FileNotFoundError(f"'{runs_path}' introuvable. Lance fetch(repo='{repo}') d'abord.")

        runs_df = pd.read_json(runs_path)
        builds_df = pd.DataFrame(
            {
                "run_id": runs_df["id"],
                "repo": repo,
                "status": runs_df.get("status"),
                "conclusion": runs_df.get("conclusion"),
                "created_at": pd.to_datetime(runs_df.get("created_at"), errors="coerce"),
                "updated_at": pd.to_datetime(runs_df.get("updated_at"), errors="coerce"),
                "run_attempt": runs_df.get("run_attempt"),
                "event": runs_df.get("event"),
                "head_branch": runs_df.get("head_branch"),
                "head_sha": runs_df.get("head_sha"),
            }
        )
        builds_df["duration_s"] = (builds_df["updated_at"] - builds_df["created_at"]).dt.total_seconds()

        out_dir = self.interim_dir / repo.replace("/", "@")
        out_dir.mkdir(parents=True, exist_ok=True)
        builds_df.to_parquet(out_dir / "runs.parquet", index=False)
        logger.info(f"GitHub Actions '{repo}' parsé: {len(builds_df)} runs -> {out_dir}")
