"""
Export Prometheus (OpenTelemetry Demo) -> data/interim/otel_demo/metrics.parquet
Interroge /api/v1/label/__name__/values puis /api/v1/query_range pour chaque métrique.
"""

from pathlib import Path
import logging
import time

import pandas as pd
import requests

from src.data import schema

logger = logging.getLogger(__name__)


def export(endpoint: str, out_dir: Path, lookback_seconds: int = 3600, step: str = "15s") -> Path:
    now = time.time()
    start = now - lookback_seconds

    names_response = requests.get(f"{endpoint}/api/v1/label/__name__/values", timeout=30)
    names_response.raise_for_status()
    metric_names = names_response.json().get("data", [])

    rows = []
    for metric_name in metric_names:
        response = requests.get(
            f"{endpoint}/api/v1/query_range",
            params={"query": metric_name, "start": start, "end": now, "step": step},
            timeout=30,
        )
        if response.status_code != 200:
            logger.warning(f"Requête échouée pour la métrique '{metric_name}': {response.status_code}")
            continue

        for series in response.json().get("data", {}).get("result", []):
            labels = series.get("metric", {})
            service = labels.get("service_name") or labels.get("job")
            for ts, value in series.get("values", []):
                try:
                    rows.append(
                        {
                            "timestamp": pd.to_datetime(float(ts), unit="s"),
                            "source": "otel_demo",
                            "run_id": None,
                            "service": service,
                            "metric_name": metric_name,
                            "value": float(value),
                            "unit": None,
                        }
                    )
                except (TypeError, ValueError):
                    continue

    metrics_df = pd.DataFrame(rows, columns=schema.METRICS_COLUMNS)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics.parquet"

    if metrics_df.empty:
        logger.warning("Aucune métrique exportée depuis Prometheus")
        return out_path

    schema.validate_metrics_df(metrics_df)
    metrics_df.to_parquet(out_path, index=False)
    logger.info(f"Métriques Prometheus exportées: {len(metrics_df)} points -> {out_path}")
    return out_path
