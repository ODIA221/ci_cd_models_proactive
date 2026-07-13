"""
Export Loki (OpenTelemetry Demo) -> data/interim/otel_demo/logs.parquet
Interroge /loki/api/v1/query_range avec une requête LogQL couvrant tous les flux.
"""

from pathlib import Path
import logging
import time

import pandas as pd
import requests

from src.data import schema

logger = logging.getLogger(__name__)


def export(endpoint: str, out_dir: Path, lookback_seconds: int = 3600, logql_query: str = '{service_name=~".+"}') -> Path:
    now_ns = int(time.time() * 1e9)
    start_ns = now_ns - lookback_seconds * int(1e9)

    response = requests.get(
        f"{endpoint}/loki/api/v1/query_range",
        params={"query": logql_query, "start": start_ns, "end": now_ns, "limit": 5000},
        timeout=30,
    )
    response.raise_for_status()

    rows = []
    for stream in response.json().get("data", {}).get("result", []):
        labels = stream.get("stream", {})
        service = labels.get("service_name") or labels.get("job")
        level = labels.get("level") or labels.get("severity")
        for ts_ns, line in stream.get("values", []):
            rows.append(
                {
                    "timestamp": pd.to_datetime(int(ts_ns), unit="ns"),
                    "source": "otel_demo",
                    "run_id": None,
                    "service": service,
                    "level": level,
                    "template_id": None,
                    "message": line,
                    "label": None,
                }
            )

    logs_df = pd.DataFrame(rows, columns=schema.LOGS_COLUMNS)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "logs.parquet"

    if logs_df.empty:
        logger.warning("Aucun log exporté depuis Loki")
        return out_path

    schema.validate_logs_df(logs_df)
    logs_df.to_parquet(out_path, index=False)
    logger.info(f"Logs Loki exportés: {len(logs_df)} lignes -> {out_path}")
    return out_path
