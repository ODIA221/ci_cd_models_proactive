"""
Connecteur TravisTorrent — table CI/CD plate (pas de découpage logs/metrics/traces)

Important (vérifié en explorant le dépôt réel, contrairement à la description
initiale qui l'annonçait comme un CSV propre "519 373 builds") :
github.com/monperrus/travistorrent-java-ci-build-dataset ne contient PAS de
CSV agrégé, mais un dossier par dépôt GitHub couvert ('{owner}@{repo}/'),
chacun rempli de logs de build Travis CI bruts compressés
('{build_number}_{build_id}_{commit_sha}_{job_id}.log.bz2').

Ce connecteur télécharge donc les logs bruts d'un dépôt donné (via
sparse-checkout git, pour éviter de cloner l'intégralité du dépôt qui
contient des centaines de sous-dossiers), puis en extrait heuristiquement
le statut de build / les compteurs de tests par expressions régulières —
les formats de logs variant selon l'outil de test utilisé (RSpec, Minitest,
JUnit, ...), cette extraction est best-effort, pas une vérité terrain garantie.

Le dump tabulaire "propre" original (statut, durée, tests exécutés/échoués,
latence commit->build) décrit dans la littérature est un artefact différent,
distribué via travistorrent.testroots.org / Figshare — voir le TODO DOI dans
registry.yaml, non résolu à ce jour.
"""

from pathlib import Path
import bz2
import logging
import re
import subprocess

import pandas as pd

from src.data.sources.base import BaseConnector

logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/monperrus/travistorrent-java-ci-build-dataset.git"

LOG_FILENAME_RE = re.compile(r"^(?P<build_number>\d+)_(?P<build_id>\d+)_(?P<commit_sha>[0-9a-f]+)_(?P<job_id>\d+)\.log\.bz2$")
EXIT_CODE_RE = re.compile(r"Your build exited with (\d+)")
RSPEC_RE = re.compile(r"(\d+) examples?, (\d+) failures?")
GENERIC_TEST_RE = re.compile(r"(\d+) tests?, \d+ assertions?, (\d+) failures?, (\d+) errors?")


class TravisTorrentConnector(BaseConnector):
    name = "travistorrent"

    def fetch(self, force: bool = False, repo: str = None, **kwargs) -> None:
        """
        Télécharge les logs bruts d'un dépôt GitHub couvert par TravisTorrent
        via sparse-checkout git (évite de cloner l'intégralité du dépôt).

        Args:
            repo: identifiant '{owner}/{name}' (converti en '{owner}@{name}'
                pour matcher la convention de nommage du dépôt de données)
        """
        if not repo:
            raise ValueError(
                "Le paramètre 'repo' (ex: 'AlchemyCMS/alchemy_cms') est requis. "
                "Consulte la liste des dépôts disponibles sur "
                "https://github.com/monperrus/travistorrent-java-ci-build-dataset"
            )

        folder_name = repo.replace("/", "@")
        dest_dir = self.raw_dir / folder_name

        if dest_dir.exists() and any(dest_dir.iterdir()) and not force:
            logger.info(f"Logs déjà présents, téléchargement ignoré: {dest_dir}")
            return

        clone_dir = self.raw_dir / "_clone"
        if not clone_dir.exists():
            subprocess.run(
                ["git", "clone", "--filter=blob:none", "--no-checkout", "--depth", "1", REPO_URL, str(clone_dir)],
                check=True,
            )
            subprocess.run(["git", "sparse-checkout", "init", "--cone"], cwd=clone_dir, check=True)

        subprocess.run(["git", "sparse-checkout", "add", folder_name], cwd=clone_dir, check=True)
        subprocess.run(["git", "checkout"], cwd=clone_dir, check=True)

        src_dir = clone_dir / folder_name
        if not src_dir.exists():
            raise FileNotFoundError(f"Dépôt '{folder_name}' introuvable dans TravisTorrent (vérifie le nom exact).")

        dest_dir.mkdir(parents=True, exist_ok=True)
        for f in src_dir.glob("*.log.bz2"):
            f.rename(dest_dir / f.name)

        logger.info(f"TravisTorrent '{repo}': {len(list(dest_dir.glob('*.log.bz2')))} logs -> {dest_dir}")

    def parse(self, repo: str = None, **kwargs) -> None:
        """
        Extrait heuristiquement statut de build / compteurs de tests des logs
        bruts d'un dépôt, vers une table plate `builds.parquet`.
        """
        if not repo:
            raise ValueError("Le paramètre 'repo' est requis (même valeur que pour fetch()).")

        folder_name = repo.replace("/", "@")
        src_dir = self.raw_dir / folder_name
        if not src_dir.exists():
            raise FileNotFoundError(f"'{src_dir}' introuvable. Lance fetch(repo='{repo}') d'abord.")

        rows = []
        for log_path in sorted(src_dir.glob("*.log.bz2")):
            match = LOG_FILENAME_RE.match(log_path.name)
            if not match:
                logger.warning(f"Nom de fichier inattendu, ignoré: {log_path.name}")
                continue

            with bz2.open(log_path, "rt", errors="replace") as f:
                text = f.read()

            exit_code_matches = EXIT_CODE_RE.findall(text)
            exit_code = int(exit_code_matches[-1]) if exit_code_matches else None

            tests_run, tests_failed = None, None
            rspec_match = RSPEC_RE.search(text)
            generic_match = GENERIC_TEST_RE.search(text)
            if rspec_match:
                tests_run, tests_failed = int(rspec_match.group(1)), int(rspec_match.group(2))
            elif generic_match:
                tests_run, tests_failed = int(generic_match.group(1)), int(generic_match.group(2))

            rows.append(
                {
                    "repo": repo,
                    "build_number": match.group("build_number"),
                    "build_id": match.group("build_id"),
                    "commit_sha": match.group("commit_sha"),
                    "job_id": match.group("job_id"),
                    "build_status": "passed" if exit_code == 0 else ("failed" if exit_code is not None else "unknown"),
                    "exit_code": exit_code,
                    "tests_run": tests_run,
                    "tests_failed": tests_failed,
                }
            )

        if not rows:
            raise RuntimeError(f"Aucun log parsé dans {src_dir}")

        builds_df = pd.DataFrame(rows)

        out_dir = self.interim_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{folder_name}_builds.parquet"
        builds_df.to_parquet(out_path, index=False)
        logger.info(f"TravisTorrent '{repo}' parsé: {len(builds_df)} builds -> {out_path}")
