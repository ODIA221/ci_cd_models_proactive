"""
Connecteur RCAEval — logs + métriques + traces
735 cas de défaillance réels sur Online Boutique (OB), Sock Shop (SS), Train Ticket (TT).
Source la plus riche pour l'entraînement multimodal (chapitre 5 de la thèse).

Nécessite le paquet optionnel RCAEval:
    pip install -r requirements-optional.txt

Référence API/structure (vérifiée sur github.com/phamquiluan/RCAEval, README + utility/__init__.py):
  - Téléchargement: RCAEval.utility.download_re{1,2,3}_dataset(local_path=...)
    -> extrait vers <local_path>/RE{n}/RE{n}-{OB,SS,TT}/<case_dir>/
  - Chaque dossier de cas: '{benchmark}_{service}_{fault}_{instance}'
    contient metrics.json (ou metrics.csv selon la version), inject_time.txt,
    et pour RE2/RE3: logs.csv, traces.csv (traces absentes pour Sock Shop).
  - La structure exacte de metrics.json n'a pas pu être vérifiée sans
    télécharger le jeu complet (plusieurs Go) ; _load_metrics() gère les deux
    formats et devra être ajustée si le format JSON diffère à l'usage réel.
"""

from pathlib import Path
import json
import logging

import pandas as pd

from src.data.sources.base import BaseConnector
from src.data import schema

logger = logging.getLogger(__name__)

VALID_SUBSETS = ("RE1", "RE2", "RE3")
SYSTEMS = ("OB", "SS", "TT")


class RCAEvalConnector(BaseConnector):
    name = "rcaeval"

    def fetch(self, force: bool = False, subset: str = "RE2", **kwargs) -> None:
        """
        Télécharge un sous-ensemble RCAEval (RE1=métriques seules,
        RE2=logs+métriques+traces, RE3=fautes code-level) via le paquet RCAEval.

        Args:
            subset: 'RE1', 'RE2' ou 'RE3'
        """
        if subset not in VALID_SUBSETS:
            raise ValueError(f"Sous-ensemble RCAEval inconnu: '{subset}' (attendu: {VALID_SUBSETS})")

        subset_dir = self.raw_dir / subset
        if subset_dir.exists() and any(subset_dir.iterdir()) and not force:
            logger.info(f"RCAEval {subset} déjà présent, téléchargement ignoré: {subset_dir}")
            return

        try:
            from RCAEval.utility import (
                download_re1_dataset,
                download_re2_dataset,
                download_re3_dataset,
            )
        except ImportError as e:
            raise ImportError(
                "Le paquet RCAEval n'est pas installé. "
                "Installe-le avec: pip install -r requirements-optional.txt"
            ) from e

        download_fn = {"RE1": download_re1_dataset, "RE2": download_re2_dataset, "RE3": download_re3_dataset}[subset]

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Téléchargement RCAEval {subset} -> {self.raw_dir} (peut prendre plusieurs dizaines de minutes)")
        download_fn(local_path=str(self.raw_dir))

    def _load_metrics(self, case_dir: Path) -> pd.DataFrame:
        """Charge metrics.json ou metrics.csv selon ce qui est présent dans le dossier de cas."""
        csv_path = case_dir / "metrics.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path)

        json_path = case_dir / "metrics.json"
        if json_path.exists():
            with open(json_path) as f:
                raw = json.load(f)
            try:
                return pd.DataFrame(raw)
            except ValueError:
                # Format alternatif: {metric_name: {timestamp: value, ...}, ...}
                return pd.DataFrame(raw).reset_index().rename(columns={"index": "time"})

        return pd.DataFrame()

    def parse(self, subset: str = "RE2", **kwargs) -> None:
        """
        Normalise les cas RCAEval téléchargés vers les schémas unifiés
        logs/metrics/traces + labels. Chaque dossier de cas devient un `run_id`.
        """
        subset_dir = self.raw_dir / subset
        if not subset_dir.exists():
            raise FileNotFoundError(f"'{subset_dir}' introuvable. Lance fetch(subset='{subset}') d'abord.")

        case_dirs = sorted(subset_dir.glob(f"{subset}-*/*"))
        case_dirs = [d for d in case_dirs if d.is_dir()]
        if not case_dirs:
            raise RuntimeError(f"Aucun cas trouvé dans {subset_dir} (attendu: {subset}-{{OB,SS,TT}}/<case>/)")

        metrics_rows, logs_rows, traces_rows, labels_rows = [], [], [], []

        for case_dir in case_dirs:
            run_id = case_dir.name
            parts = run_id.split("_")
            fault_type = parts[-2] if len(parts) >= 3 else None
            labels_rows.append({"run_id": run_id, "source": "rcaeval", "label": 1, "fault_type": fault_type})

            metrics_raw = self._load_metrics(case_dir)
            if not metrics_raw.empty and "time" in metrics_raw.columns:
                value_cols = [c for c in metrics_raw.columns if c not in ("time", "service")]
                for _, row in metrics_raw.iterrows():
                    for col in value_cols:
                        metrics_rows.append(
                            {
                                "timestamp": row["time"],
                                "source": "rcaeval",
                                "run_id": run_id,
                                "service": row.get("service"),
                                "metric_name": col,
                                "value": row[col],
                                "unit": None,
                            }
                        )

            logs_csv = case_dir / "logs.csv"
            if logs_csv.exists():
                raw = pd.read_csv(logs_csv)
                for _, row in raw.iterrows():
                    logs_rows.append(
                        {
                            "timestamp": row.get("time", row.get("timestamp")),
                            "source": "rcaeval",
                            "run_id": run_id,
                            "service": row.get("service"),
                            "level": row.get("level"),
                            "template_id": None,
                            "message": row.get("message"),
                            "label": None,
                        }
                    )

            traces_csv = case_dir / "traces.csv"
            if traces_csv.exists():
                raw = pd.read_csv(traces_csv)
                for _, row in raw.iterrows():
                    traces_rows.append(
                        {
                            "timestamp": row.get("time", row.get("timestamp")),
                            "source": "rcaeval",
                            "run_id": row.get("trace_id", run_id),
                            "span_id": row.get("span_id"),
                            "parent_span_id": row.get("parent_span_id"),
                            "service": row.get("service"),
                            "operation": row.get("operation"),
                            "duration_ms": row.get("duration"),
                            "status": row.get("status"),
                        }
                    )

        out_dir = self.interim_dir / subset
        out_dir.mkdir(parents=True, exist_ok=True)

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
            df.to_parquet(out_dir / filename, index=False)

        logger.info(f"RCAEval {subset} parsé: {len(case_dirs)} cas -> {out_dir}")
