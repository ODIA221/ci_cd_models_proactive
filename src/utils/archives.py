"""
Extraction sécurisée d'archives (zip/tar) téléchargées depuis des sources tierces
Protège contre les chemins malveillants ("zip-slip")
"""

from pathlib import Path
import logging
import tarfile
import zipfile

logger = logging.getLogger(__name__)


def _is_within_directory(directory: Path, target: Path) -> bool:
    directory = directory.resolve()
    target = target.resolve()
    return directory == target or directory in target.parents


def safe_extract(archive_path: Path, dest_dir: Path) -> Path:
    """
    Extrait une archive zip ou tar(.gz) vers `dest_dir` en vérifiant que
    chaque membre reste bien à l'intérieur de `dest_dir`.

    Args:
        archive_path: Chemin de l'archive (.zip, .tar, .tar.gz, .tgz)
        dest_dir: Dossier de destination

    Returns:
        Le chemin `dest_dir`
    """
    archive_path = Path(archive_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    suffix = archive_path.suffix.lower()

    if suffix == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            for member in zf.namelist():
                member_path = dest_dir / member
                if not _is_within_directory(dest_dir, member_path):
                    raise ValueError(f"Chemin d'archive dangereux détecté: {member}")
            zf.extractall(dest_dir)

    elif suffix in (".tar", ".gz", ".tgz") or archive_path.name.endswith(".tar.gz"):
        mode = "r:gz" if suffix in (".gz", ".tgz") or archive_path.name.endswith(".tar.gz") else "r"
        with tarfile.open(archive_path, mode) as tf:
            for member in tf.getmembers():
                member_path = dest_dir / member.name
                if not _is_within_directory(dest_dir, member_path):
                    raise ValueError(f"Chemin d'archive dangereux détecté: {member.name}")
            tf.extractall(dest_dir)

    else:
        raise ValueError(f"Format d'archive non supporté: {archive_path.name}")

    logger.info(f"Archive extraite: {archive_path} -> {dest_dir}")
    return dest_dir
