"""
Connecteur LogHub (HDFS_v1, BGL) — logs seuls
Réutilise le clone local de github.com/ait-aecid/anomaly-detection-log-datasets
(scripts hdfs_parse.py / bgl_parse.py, échantillons pré-calculés) plutôt que
de réimplémenter le parsing depuis zéro.
"""

from pathlib import Path
import logging
import subprocess

import pandas as pd

from src.data.sources.base import BaseConnector, DATA_ROOT
from src.data import schema
from src.utils.downloading import download_file
from src.utils.archives import safe_extract

logger = logging.getLogger(__name__)

REPO_ROOT = DATA_ROOT.parent
AIT_AECID_DIR = REPO_ROOT / "anomaly-detection-log-datasets"

ZENODO_URLS = {
    "hdfs": "https://zenodo.org/records/8196385/files/HDFS_v1.zip",
    "bgl": "https://zenodo.org/records/8196385/files/BGL.zip",
}
# Sous-dossier attendu par les scripts *_parse.py de anomaly-detection-log-datasets
AIT_SUBDIR = {"hdfs": "hdfs_loghub", "bgl": "bgl_loghub"}
RAW_LOG_FILENAME = {"hdfs": "HDFS.log", "bgl": "BGL.log"}
PARSE_SCRIPT = {"hdfs": "hdfs_parse.py", "bgl": "bgl_parse.py"}


class LogHubConnector(BaseConnector):
    name = "loghub"

    def _dataset_raw_dir(self, dataset: str) -> Path:
        return self.raw_dir / dataset

    def _dataset_interim_dir(self, dataset: str) -> Path:
        return self.interim_dir / dataset

    def status(self, dataset: str = "hdfs") -> str:
        if self._dataset_interim_dir(dataset).exists() and any(self._dataset_interim_dir(dataset).iterdir()):
            return "parsed"
        if self._dataset_raw_dir(dataset).exists() and any(self._dataset_raw_dir(dataset).iterdir()):
            return "downloaded"
        if self._has_ait_aecid_samples(dataset):
            return "downloaded (échantillon ait-aecid disponible)"
        return "not_downloaded"

    def _has_ait_aecid_samples(self, dataset: str) -> bool:
        subdir = AIT_AECID_DIR / AIT_SUBDIR[dataset]
        return (subdir / f"{dataset}_train").exists()

    def fetch(self, force: bool = False, dataset: str = "hdfs", **kwargs) -> None:
        """
        Télécharge l'archive brute Zenodo pour `dataset` ('hdfs' ou 'bgl').
        Non nécessaire si tu utilises parse(use_samples=True), qui s'appuie
        sur les échantillons déjà fournis par anomaly-detection-log-datasets.
        """
        if dataset not in ZENODO_URLS:
            raise ValueError(f"Dataset LogHub inconnu: '{dataset}' (attendu: {list(ZENODO_URLS)})")

        dest_zip = self._dataset_raw_dir(dataset) / f"{dataset}.zip"
        download_file(ZENODO_URLS[dataset], dest_zip, force=force)
        safe_extract(dest_zip, self._dataset_raw_dir(dataset) / "extracted")
        logger.info(f"LogHub '{dataset}' téléchargé et extrait: {self._dataset_raw_dir(dataset)}")

    def parse(self, dataset: str = "hdfs", use_samples: bool = True, **kwargs) -> None:
        """
        Args:
            dataset: 'hdfs' ou 'bgl'
            use_samples: si True (par défaut), parse à partir des séquences
                déjà échantillonnées par anomaly-detection-log-datasets
                (rapide, aucun téléchargement requis, mais sans timestamp/message
                bruts — uniquement des identifiants de template par séquence).
                Si False, parse le log brut téléchargé via fetch() (timestamps
                et templates complets, mais nécessite le fichier log complet).
        """
        if use_samples:
            self._parse_from_samples(dataset)
        else:
            self._parse_from_raw(dataset)

    def _parse_from_samples(self, dataset: str) -> None:
        subdir = AIT_AECID_DIR / AIT_SUBDIR[dataset]
        if not subdir.exists():
            raise FileNotFoundError(
                f"'{subdir}' introuvable. Clone requis: "
                f"git clone https://github.com/ait-aecid/anomaly-detection-log-datasets"
            )

        logs_rows = []
        labels_rows = []
        for split, label_value in [("train", 0), ("test_normal", 0), ("test_abnormal", 1)]:
            sample_file = subdir / f"{dataset}_{split}"
            if not sample_file.exists():
                logger.warning(f"Fichier échantillon absent, ignoré: {sample_file}")
                continue

            with open(sample_file, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line or "," not in line:
                        continue
                    run_id, sequence = line.split(",", 1)
                    for event_id in sequence.split():
                        logs_rows.append(
                            {
                                "timestamp": pd.NaT,
                                "source": f"loghub_{dataset}",
                                "run_id": run_id,
                                "service": None,
                                "level": None,
                                "template_id": event_id,
                                "message": None,
                                "label": label_value,
                            }
                        )
                    labels_rows.append(
                        # fault_type est réutilisé ici pour stocker le split d'origine
                        # ('train'/'test_normal'/'test_abnormal'), indispensable pour
                        # évaluer sans fuite de données (entraîner sur 'train' uniquement,
                        # tester sur test_normal+test_abnormal jamais vus à l'entraînement).
                        {"run_id": run_id, "source": f"loghub_{dataset}", "label": label_value, "fault_type": split}
                    )

        logs_df = pd.DataFrame(logs_rows, columns=schema.LOGS_COLUMNS)
        labels_df = pd.DataFrame(labels_rows, columns=schema.LABELS_COLUMNS)

        if logs_df.empty:
            raise RuntimeError(
                f"Aucune séquence trouvée pour '{dataset}' dans {subdir}. "
                "Vérifie que le dépôt anomaly-detection-log-datasets est bien cloné."
            )

        schema.validate_logs_df(logs_df)
        schema.validate_labels_df(labels_df)

        out_dir = self._dataset_interim_dir(dataset)
        out_dir.mkdir(parents=True, exist_ok=True)
        logs_df.to_parquet(out_dir / "logs.parquet", index=False)
        labels_df.to_parquet(out_dir / "labels.parquet", index=False)
        logger.info(
            f"LogHub '{dataset}' (échantillons) parsé: {len(logs_df)} événements, "
            f"{len(labels_df)} séquences -> {out_dir}"
        )

    def _parse_from_raw(self, dataset: str) -> None:
        subdir = AIT_AECID_DIR / AIT_SUBDIR[dataset]
        raw_log = subdir / RAW_LOG_FILENAME[dataset]
        if not raw_log.exists():
            raise FileNotFoundError(
                f"'{raw_log}' introuvable. Lance fetch(dataset='{dataset}') puis copie le fichier log "
                f"extrait vers '{raw_log}' (structure attendue par les scripts ait-aecid)."
            )

        script = AIT_AECID_DIR / PARSE_SCRIPT[dataset]
        subprocess.run(
            ["python3", str(script), "--data_dir", AIT_SUBDIR[dataset]],
            cwd=AIT_AECID_DIR,
            check=True,
        )

        parsed_csv = subdir / "parsed.csv"
        df = pd.read_csv(parsed_csv, sep=";")
        logs_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(df["time"], errors="coerce"),
                "source": f"loghub_{dataset}",
                "run_id": df["seq_id"],
                "service": None,
                "level": None,
                "template_id": df["event_type"],
                "message": None,
                "label": (df["label"].astype(str).str.lower() == "anomaly").astype(int),
            }
        )
        schema.validate_logs_df(logs_df)

        out_dir = self._dataset_interim_dir(dataset)
        out_dir.mkdir(parents=True, exist_ok=True)
        logs_df.to_parquet(out_dir / "logs.parquet", index=False)
        logger.info(f"LogHub '{dataset}' (brut) parsé: {len(logs_df)} événements -> {out_dir}")
