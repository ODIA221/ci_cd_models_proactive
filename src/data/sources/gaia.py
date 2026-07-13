"""
Connecteur GAIA (Generic AIOps Atlas) — logs + métriques + traces
github.com/CloudWise-OpenSource/GAIA-DataSet

Les liens de téléchargement (Baidu Netdisk / Google Drive selon les mises à
jour du dépôt) ne sont pas stables/scriptables de façon fiable. fetch() ne
télécharge donc rien automatiquement: il vérifie qu'un dépôt manuel a bien
été placé au bon endroit et guide l'utilisateur sinon.
"""

from pathlib import Path
import logging

import pandas as pd

from src.data.sources.base import BaseConnector
from src.data import schema

logger = logging.getLogger(__name__)

MANUAL_INSTRUCTIONS = """
GAIA nécessite un téléchargement manuel (liens Baidu Netdisk / Google Drive
non scriptables de façon fiable, cf. README de
https://github.com/CloudWise-OpenSource/GAIA-DataSet):

  1. Cloner/consulter le dépôt pour trouver les liens actifs à la date d'accès:
     git clone https://github.com/CloudWise-OpenSource/GAIA-DataSet
  2. Télécharger manuellement 'MicroSS/' (traces + logs) et 'Companion Data/'
     (métriques + vérité terrain des anomalies injectées).
  3. Placer les fichiers obtenus sous:
     {manual_dir}/MicroSS/...
     {manual_dir}/Companion Data/...
"""


class GAIAConnector(BaseConnector):
    name = "gaia"

    def status(self) -> str:
        if self.interim_dir.exists() and any(self.interim_dir.iterdir()):
            return "parsed"
        manual_dir = self.raw_dir / "manual"
        if manual_dir.exists() and any(manual_dir.iterdir()):
            return "downloaded (manuel)"
        return "not_downloaded (téléchargement manuel requis)"

    def fetch(self, force: bool = False, **kwargs) -> None:
        manual_dir = self.raw_dir / "manual"
        manual_dir.mkdir(parents=True, exist_ok=True)

        if not any(manual_dir.iterdir()):
            raise RuntimeError(MANUAL_INSTRUCTIONS.format(manual_dir=manual_dir))

        logger.info(f"Données GAIA manuelles détectées: {manual_dir}")

    def parse(self, **kwargs) -> None:
        manual_dir = self.raw_dir / "manual"
        micro_ss = manual_dir / "MicroSS"
        companion_data = manual_dir / "Companion Data"

        if not micro_ss.exists() and not companion_data.exists():
            raise FileNotFoundError(
                f"Aucune donnée GAIA trouvée sous {manual_dir}. " + MANUAL_INSTRUCTIONS.format(manual_dir=manual_dir)
            )

        metrics_rows, logs_rows, traces_rows, labels_rows = [], [], [], []

        # Companion Data: métriques + vérité terrain (anomalies injectées avec timestamps début/fin)
        if companion_data.exists():
            for csv_path in companion_data.glob("*.csv"):
                raw = pd.read_csv(csv_path)
                is_ground_truth = any(c.lower() in ("start_time", "end_time", "fault_type") for c in raw.columns)

                if is_ground_truth:
                    for _, row in raw.iterrows():
                        labels_rows.append(
                            {
                                "run_id": row.get("case_id", row.get("id", csv_path.stem)),
                                "source": "gaia",
                                "label": 1,
                                "fault_type": row.get("fault_type"),
                            }
                        )
                elif "timestamp" in raw.columns:
                    value_cols = [c for c in raw.columns if c not in ("timestamp", "service", "cmdb_id")]
                    for _, row in raw.iterrows():
                        for col in value_cols:
                            metrics_rows.append(
                                {
                                    "timestamp": row["timestamp"],
                                    "source": "gaia",
                                    "run_id": None,
                                    "service": row.get("cmdb_id", row.get("service")),
                                    "metric_name": col,
                                    "value": row[col],
                                    "unit": None,
                                }
                            )

        # MicroSS: logs + traces
        if micro_ss.exists():
            for csv_path in micro_ss.rglob("*log*.csv"):
                raw = pd.read_csv(csv_path)
                for _, row in raw.iterrows():
                    logs_rows.append(
                        {
                            "timestamp": row.get("timestamp"),
                            "source": "gaia",
                            "run_id": None,
                            "service": row.get("cmdb_id", row.get("service")),
                            "level": row.get("level"),
                            "template_id": None,
                            "message": row.get("message", row.get("value")),
                            "label": None,
                        }
                    )

            for csv_path in micro_ss.rglob("*trace*.csv"):
                raw = pd.read_csv(csv_path)
                for _, row in raw.iterrows():
                    traces_rows.append(
                        {
                            "timestamp": row.get("timestamp"),
                            "source": "gaia",
                            "run_id": row.get("trace_id"),
                            "span_id": row.get("id", row.get("span_id")),
                            "parent_span_id": row.get("pid", row.get("parent_span_id")),
                            "service": row.get("cmdb_id", row.get("service")),
                            "operation": row.get("service_name", row.get("operation")),
                            "duration_ms": row.get("elapsedTime", row.get("duration")),
                            "status": row.get("success", row.get("status")),
                        }
                    )

        self.interim_dir.mkdir(parents=True, exist_ok=True)

        metrics_df = pd.DataFrame(metrics_rows, columns=schema.METRICS_COLUMNS)
        logs_df = pd.DataFrame(logs_rows, columns=schema.LOGS_COLUMNS)
        traces_df = pd.DataFrame(traces_rows, columns=schema.TRACES_COLUMNS)
        labels_df = pd.DataFrame(labels_rows, columns=schema.LABELS_COLUMNS)

        for df, validate, filename in [
            (metrics_df, schema.validate_metrics_df, "metrics.parquet"),
            (logs_df, schema.validate_logs_df, "logs.parquet"),
            (traces_df, schema.validate_traces_df, "traces.parquet"),
            (labels_df, schema.validate_labels_df, "labels.parquet"),
        ]:
            if df.empty:
                logger.warning(f"Aucune donnée pour '{filename}', fichier non écrit")
                continue
            validate(df)
            df.to_parquet(self.interim_dir / filename, index=False)

        logger.info(f"GAIA parsé -> {self.interim_dir}")
