"""
Connecteur RCAEval — logs + métriques + traces
735 cas de défaillance réels sur Online Boutique (OB), Sock Shop (SS), Train Ticket (TT).
Source la plus riche pour l'entraînement multimodal (chapitre 5 de la thèse).

Le paquet PyPI `RCAEval` (même avec l'extra [default]) n'expose PAS le
sous-module RCAEval.utility utilisé par le README upstream pour télécharger
les jeux de données — vérifié le 2026-07-22, le wheel publié ne contient que
2 lignes (`RCAEval/__init__.py: def is_ok(): ...`). Les fonctions
download_re{1,2,3}_dataset n'existent que dans le dépôt source GitHub
(github.com/phamquiluan/RCAEval, RCAEval/utility/__init__.py). Ce connecteur
ne dépend donc PAS du paquet RCAEval: il télécharge directement les archives
Zenodo (mêmes URLs que download_re2*_dataset dans le dépôt source), ce qui
évite en prime la chaîne de dépendances RCAEval[default] (numba/llvmlite,
causal-learn, etc. — env 2023 figé, source de conflits avec les
dépendances du reste de ce projet, notamment mlflow via pkg_resources).

Structure réelle vérifiée après téléchargement (diffère de ce que la
documentation upstream laisse penser):
  RE{n}/RE{n}-{OB,SS,TT}/<service>_<fault_type>/<case_num>/
    inject_time.txt        - un entier, epoch en SECONDES
    metrics.csv             - "time" (s) + une colonne par (service_metric) brut
    simple_metrics.csv      - "time" (s) + un sous-ensemble curé (cpu/mem/diskio/sock)
    logs.csv                - "timestamp" (epoch NANOsecondes) + container_name +
                              message + level + log_template + cluster_id
                              (le parsing/templating des logs est déjà fait,
                              pas besoin de Drain3 ici)
    traces.csv              - "startTime" (epoch MICROsecondes) + traceID/spanID/
                              serviceName/operationName/duration/statusCode/parentSpanID
    logts.csv, tracets_err.csv, tracets_lat.csv - séries temporelles déjà
                              agrégées (non utilisées ici, on repart des données
                              brutes pour rester cohérent avec le pipeline logs
                              bag-of-events/bigrammes déjà en place)

Chaque cas ne fournit qu'UNE fenêtre normale (avant inject_time) et UNE
fenêtre anormale (après) — pas de split train/test_normal/test_abnormal
pré-fourni comme pour LogHub. Ce connecteur construit ce split lui-même, au
niveau du CAS (pas de la fenêtre), pour respecter le protocole sans fuite de
src/models/evaluate.py: ~70% des cas -> uniquement leur fenêtre normale
(fault_type='train'), ~30% des cas -> leurs deux fenêtres
('test_normal'/'test_abnormal'). Assignation déterministe (hash du nom de
cas) pour rester reproductible sans état partagé.
"""

from pathlib import Path
import hashlib
import logging

import pandas as pd
import requests

from src.data.sources.base import BaseConnector
from src.data import schema

logger = logging.getLogger(__name__)

VALID_SUBSETS = ("RE1", "RE2", "RE3")
SYSTEMS = ("OB", "SS", "TT")
ZENODO_RECORD = "14590730"
TRAIN_FRACTION = 0.7


class RCAEvalConnector(BaseConnector):
    name = "rcaeval"

    def fetch(self, force: bool = False, subset: str = "RE2", **kwargs) -> None:
        """
        Télécharge un sous-ensemble RCAEval (RE1=métriques seules côté cas
        simples, RE2=logs+métriques+traces, RE3=fautes code-level) directement
        depuis Zenodo (record 14590730), sans dépendre du paquet RCAEval.

        Args:
            subset: 'RE1', 'RE2' ou 'RE3'
        """
        if subset not in VALID_SUBSETS:
            raise ValueError(f"Sous-ensemble RCAEval inconnu: '{subset}' (attendu: {VALID_SUBSETS})")

        subset_dir = self.raw_dir / subset
        subset_dir.mkdir(parents=True, exist_ok=True)

        for sys_name in SYSTEMS:
            dest_dir = subset_dir / f"{subset}-{sys_name}"
            if dest_dir.exists() and any(dest_dir.iterdir()) and not force:
                logger.info(f"RCAEval {subset}-{sys_name} déjà présent, ignoré: {dest_dir}")
                continue

            url = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{subset}-{sys_name}.zip?download=1"
            zip_path = subset_dir / f"{subset}-{sys_name}.zip"
            logger.info(f"Téléchargement {url} -> {zip_path} (peut être volumineux, plusieurs centaines de Mo à quelques Go)")

            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                        f.write(chunk)

            import zipfile
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(subset_dir)
            zip_path.unlink()
            logger.info(f"RCAEval {subset}-{sys_name} téléchargé et extrait: {dest_dir}")

    def _split_for_case(self, case_id: str) -> bool:
        """True si le cas est assigné au split 'train' (déterministe, ~TRAIN_FRACTION)."""
        digest = hashlib.md5(case_id.encode()).hexdigest()
        return (int(digest, 16) % 100) < int(TRAIN_FRACTION * 100)

    def _load_case_metrics(self, case_dir: Path) -> pd.DataFrame:
        for filename in ("simple_metrics.csv", "metrics.csv"):
            path = case_dir / filename
            if path.exists():
                return pd.read_csv(path)
        return pd.DataFrame()

    def parse(self, subset: str = "RE2", **kwargs) -> None:
        """
        Normalise les cas RCAEval téléchargés en un unique features.parquet
        (une ligne par run_id = un (cas, fenêtre normale/anormale), colonnes
        préfixées par modalité: metric_*, event_*/2gram_*, trace_*) +
        labels.parquet. Chaque (cas, fenêtre) devient un run_id synthétique
        '<cas>__normal' / '<cas>__abnormal'.

        Agrégation IMMÉDIATE par fenêtre (pas de tables tidy intermédiaires
        logs/metrics/traces à l'échelle du dataset complet): logs.csv/
        traces.csv font jusqu'à ~10^5-10^6 lignes PAR CAS, et RE2-OB+SS+TT
        comptent ~270+ cas — matérialiser puis reconcaténer les lignes brutes
        de toutes les fenêtres avant d'agréger s'est avéré trop lourd
        (>15 min sans terminer, tué manuellement). Réutilise directement les
        fonctions d'agrégation de src/data/features.py sur chaque fenêtre
        (petite: une seule valeur de run_id à la fois) plutôt que sur le
        dataset entier.
        """
        from src.data.features import build_metrics_agg_matrix, build_sequence_features, build_traces_agg_matrix

        subset_dir = self.raw_dir / subset
        if not subset_dir.exists():
            raise FileNotFoundError(f"'{subset_dir}' introuvable. Lance fetch(subset='{subset}') d'abord.")

        traces_dir = self.interim_dir / subset / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)

        case_dirs = sorted(
            d for d in subset_dir.glob(f"{subset}-*/*/*")
            if d.is_dir() and (d / "inject_time.txt").exists()
        )
        if not case_dirs:
            raise RuntimeError(f"Aucun cas trouvé dans {subset_dir} (attendu: {subset}-{{OB,SS,TT}}/<service>_<fault>/<case_num>/)")

        feature_rows = []
        labels_rows = []
        n_train_cases, n_test_cases = 0, 0

        for i, case_dir in enumerate(case_dirs):
            # case_id inclut le système + service_fault + numéro pour rester unique sur OB/SS/TT
            case_id = "/".join(case_dir.relative_to(subset_dir).parts)
            # Le 2e segment du chemin est toujours <service>_<fault_type>, avec
            # fault_type in {cpu,delay,disk,loss,mem,socket} (aucun ne contient
            # '_', vérifié sur les 271 cas RE2-OB/SS/TT) — rpartition sur le
            # dernier '_' sépare donc service (peut lui-même contenir des '_'
            # ou '-', ex. 'ts-auth-service') du type de faute injectée.
            service_fault_dirname = case_dir.relative_to(subset_dir).parts[1]
            root_cause_service, _, root_cause_fault_type = service_fault_dirname.rpartition("_")
            inject_time = int((case_dir / "inject_time.txt").read_text().strip())
            is_train_case = self._split_for_case(case_id)
            if is_train_case:
                n_train_cases += 1
            else:
                n_test_cases += 1

            windows = [("normal", 1, "train" if is_train_case else "test_normal")]
            if not is_train_case:
                windows.append(("abnormal", -1, "test_abnormal"))

            metrics_raw = self._load_case_metrics(case_dir)
            logs_raw = pd.read_csv(case_dir / "logs.csv") if (case_dir / "logs.csv").exists() else pd.DataFrame()
            traces_raw = pd.read_csv(case_dir / "traces.csv") if (case_dir / "traces.csv").exists() else pd.DataFrame()

            logs_ts_seconds = logs_raw["timestamp"] / 1e9 if "timestamp" in logs_raw.columns else None
            traces_ts_seconds = traces_raw["startTime"] / 1e6 if "startTime" in traces_raw.columns else None

            for window_name, label_value, split in windows:
                run_id = f"{case_id}__{window_name}"
                labels_rows.append({
                    "run_id": run_id,
                    "source": "rcaeval",
                    "label": label_value,
                    "fault_type": split,
                    # Cause racine connue uniquement après injection: n'a pas
                    # de sens sur la fenêtre 'normal' (pré-injection).
                    "root_cause_service": root_cause_service if window_name == "abnormal" else None,
                    "root_cause_fault_type": root_cause_fault_type if window_name == "abnormal" else None,
                })

                feats = {"run_id": run_id}

                if not metrics_raw.empty and "time" in metrics_raw.columns:
                    mask = metrics_raw["time"] < inject_time if window_name == "normal" else metrics_raw["time"] >= inject_time
                    seg = metrics_raw[mask]
                    if not seg.empty:
                        value_cols = [c for c in seg.columns if c != "time"]
                        melted = seg.melt(id_vars="time", value_vars=value_cols, var_name="metric_name", value_name="value")
                        melted["run_id"] = run_id
                        agg = build_metrics_agg_matrix(melted, id_col="run_id")
                        feats.update(agg.loc[run_id].to_dict())

                if logs_ts_seconds is not None:
                    mask = logs_ts_seconds < inject_time if window_name == "normal" else logs_ts_seconds >= inject_time
                    seg = logs_raw[mask]
                    if not seg.empty and "cluster_id" in seg.columns:
                        tidy = pd.DataFrame({"run_id": run_id, "template_id": seg["cluster_id"].to_numpy()})
                        agg = build_sequence_features(tidy, id_col="run_id")
                        feats.update(agg.loc[run_id].to_dict())

                if traces_ts_seconds is not None:
                    mask = traces_ts_seconds < inject_time if window_name == "normal" else traces_ts_seconds >= inject_time
                    seg = traces_raw[mask]
                    if not seg.empty:
                        tidy = pd.DataFrame({
                            "run_id": run_id,
                            "service": seg.get("serviceName"),
                            "duration_ms": seg.get("duration"),
                            "status": seg.get("statusCode"),
                        })
                        agg = build_traces_agg_matrix(tidy, id_col="run_id")
                        feats.update(agg.loc[run_id].to_dict())

                        # Spans bruts (hiérarchie span_id/parent_span_id
                        # préservée) réservés aux fenêtres test_normal/
                        # test_abnormal: nécessaires au module de corrélation
                        # causale (src/causal/), qui n'a besoin d'expliquer
                        # que les runs du split test. Les fenêtres 'train'
                        # (~185 cas) ne sont pas matérialisées span par span
                        # pour rester dans le même ordre de grandeur de temps
                        # de parsing que l'agrégation existante ci-dessus.
                        if split != "train":
                            raw_spans = pd.DataFrame({
                                "timestamp": traces_ts_seconds[mask].to_numpy(),
                                "source": "rcaeval",
                                "run_id": run_id,
                                "span_id": seg.get("spanID"),
                                "parent_span_id": seg.get("parentSpanID"),
                                "service": seg.get("serviceName"),
                                "operation": seg.get("operationName"),
                                "duration_ms": seg.get("duration"),
                                "status": seg.get("statusCode"),
                            })
                            safe_run_id = run_id.replace("/", "__")
                            raw_spans.to_parquet(traces_dir / f"{safe_run_id}.parquet", index=False)

                feature_rows.append(feats)

            if (i + 1) % 20 == 0:
                logger.info(f"RCAEval {subset}: {i + 1}/{len(case_dirs)} cas traités...")

        out_dir = self.interim_dir / subset
        out_dir.mkdir(parents=True, exist_ok=True)

        features_df = pd.DataFrame(feature_rows).set_index("run_id").fillna(0.0)
        labels_df = pd.DataFrame(labels_rows, columns=schema.LABELS_COLUMNS + schema.OPTIONAL_LABELS_COLUMNS)
        schema.validate_labels_df(labels_df)

        features_df.to_parquet(out_dir / "features.parquet")
        labels_df.to_parquet(out_dir / "labels.parquet", index=False)

        logger.info(
            f"RCAEval {subset} parsé: {len(case_dirs)} cas ({n_train_cases} train, {n_test_cases} test), "
            f"{features_df.shape[0]} runs x {features_df.shape[1]} features -> {out_dir}, "
            f"spans bruts (test uniquement) -> {traces_dir}"
        )
