"""
CLI d'acquisition des sources de données externes pour LogPipeGuard

Aucune source n'est téléchargée par défaut. Seul --source déclenche une
action; --list se contente d'inspecter le statut courant de chaque source.

Exemples:
    python -m src.data.acquire --list
    python -m src.data.acquire --source loghub --dataset hdfs
    python -m src.data.acquire --source loghub --dataset hdfs --parse-only --no-samples
    python -m src.data.acquire --source rcaeval --subset RE2
    python -m src.data.acquire --source travistorrent --repo AlchemyCMS/alchemy_cms
    python -m src.data.acquire --source github_actions --repo owner/name   # nécessite GITHUB_TOKEN
    python -m src.data.acquire --source jenkins --jenkins-url https://ci.example.com --job my-job  # nécessite JENKINS_USER/JENKINS_TOKEN
    python -m src.data.acquire --source gitlab_ci --project namespace/nom  # nécessite GITLAB_TOKEN
    python -m src.data.acquire --source otel_demo --export all             # nécessite le stack Docker démarré
    python -m src.data.acquire --source gaia
"""

import argparse
import logging
import sys

from src.data.sources.registry import get_connector, list_sources

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def print_sources_table() -> None:
    rows = list_sources()
    widths = {"source": 16, "modalités": 22, "statut": 40, "taille estimée": 25, "caveats": 16}
    header = "".join(f"{name:<{w}}" for name, w in widths.items())
    print(header)
    print("-" * len(header))
    for row in rows:
        caveats = []
        if row["requires_token"]:
            caveats.append("token")
        if row["requires_manual_download"]:
            caveats.append("manuel")
        if row["requires_docker"]:
            caveats.append("docker")
        modalities = ",".join(row["modalities"])
        cells = [
            row["name"],
            modalities,
            row["status"],
            row["estimated_size"],
            ",".join(caveats) if caveats else "-",
        ]
        print("".join(f"{_truncate(str(c), w - 1):<{w}}" for c, w in zip(cells, widths.values())))


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquisition des sources de données LogPipeGuard")
    parser.add_argument("--list", action="store_true", help="Liste les sources et leur statut")
    parser.add_argument("--source", type=str, help="Nom de la source (voir --list)")
    parser.add_argument("--force", action="store_true", help="Force le retéléchargement même si déjà présent")
    parser.add_argument("--fetch-only", action="store_true", help="N'exécute que fetch()")
    parser.add_argument("--parse-only", action="store_true", help="N'exécute que parse()")

    # Options spécifiques à certains connecteurs (ignorées par les autres)
    parser.add_argument("--dataset", type=str, help="[loghub] 'hdfs' ou 'bgl'")
    parser.add_argument(
        "--no-samples", action="store_true", help="[loghub] parse le log brut au lieu des échantillons ait-aecid"
    )
    parser.add_argument("--subset", type=str, help="[rcaeval] 'RE1', 'RE2' ou 'RE3'")
    parser.add_argument("--repo", type=str, help="[travistorrent, github_actions] '{owner}/{name}'")
    parser.add_argument("--export", type=str, default="all", help="[otel_demo] 'prometheus', 'jaeger' ou 'all'")
    parser.add_argument("--job", type=str, help="[jenkins] nom du job (premier niveau uniquement)")
    parser.add_argument("--jenkins-url", type=str, help="[jenkins] URL de base de l'instance (ex: 'https://ci.example.com')")
    parser.add_argument("--project", type=str, help="[gitlab_ci] ID numérique ou 'namespace/nom'")
    parser.add_argument("--gitlab-url", type=str, help="[gitlab_ci] URL de base de l'instance (défaut: gitlab.com)")

    args = parser.parse_args()

    if args.list or not args.source:
        print_sources_table()
        if not args.source:
            return

    connector = get_connector(args.source)

    kwargs = {}
    if args.dataset:
        kwargs["dataset"] = args.dataset
    if args.no_samples:
        kwargs["use_samples"] = False
    if args.subset:
        kwargs["subset"] = args.subset
    if args.repo:
        kwargs["repo"] = args.repo
    if args.export:
        kwargs["export"] = args.export
    if args.job:
        kwargs["job"] = args.job
    if args.jenkins_url:
        kwargs["jenkins_url"] = args.jenkins_url
    if args.project:
        kwargs["project"] = args.project
    if args.gitlab_url:
        kwargs["gitlab_url"] = args.gitlab_url

    do_fetch = not args.parse_only
    do_parse = not args.fetch_only

    try:
        if do_fetch:
            logger.info(f"fetch('{args.source}', force={args.force}, {kwargs})")
            connector.fetch(force=args.force, **kwargs)
        if do_parse:
            logger.info(f"parse('{args.source}', {kwargs})")
            connector.parse(**kwargs)
    except Exception as e:
        logger.error(f"Échec sur la source '{args.source}': {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
