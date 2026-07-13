"""
Téléchargement de fichiers pour les connecteurs de sources de données
Idempotent (ignore si le fichier existe déjà) avec barre de progression
"""

from pathlib import Path
import logging

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


def download_file(url: str, dest: Path, force: bool = False, chunk_size: int = 1024 * 1024) -> Path:
    """
    Télécharge un fichier en streaming vers `dest`, avec reprise idempotente.

    Args:
        url: URL du fichier à télécharger
        dest: Chemin de destination local
        force: Si True, retélécharge même si `dest` existe déjà
        chunk_size: Taille des blocs de streaming (octets)

    Returns:
        Le chemin `dest`
    """
    dest = Path(dest)

    if dest.exists() and not force:
        logger.info(f"Fichier déjà présent, téléchargement ignoré: {dest}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(dest.suffix + ".part")

    logger.info(f"Téléchargement: {url} -> {dest}")
    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with open(tmp_dest, "wb") as f, tqdm(
            total=total_size or None, unit="B", unit_scale=True, desc=dest.name
        ) as progress:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    progress.update(len(chunk))

    tmp_dest.rename(dest)
    logger.info(f"Téléchargement terminé: {dest}")
    return dest
