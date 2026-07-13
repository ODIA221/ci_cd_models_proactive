"""
Export Jaeger (OpenTelemetry Demo) -> data/interim/otel_demo/traces.parquet
Interroge /api/services puis /api/traces?service=... pour chaque service connu.
"""

from pathlib import Path
import logging

import pandas as pd
import requests

from src.data import schema

logger = logging.getLogger(__name__)


def export(endpoint: str, out_dir: Path, limit_per_service: int = 500) -> Path:
    services_response = requests.get(f"{endpoint}/api/services", timeout=30)
    services_response.raise_for_status()
    services = services_response.json().get("data", [])

    rows = []
    for service in services:
        response = requests.get(
            f"{endpoint}/api/traces", params={"service": service, "limit": limit_per_service}, timeout=30
        )
        if response.status_code != 200:
            logger.warning(f"Requête échouée pour le service '{service}': {response.status_code}")
            continue

        for trace in response.json().get("data", []):
            trace_id = trace.get("traceID")
            process_by_id = trace.get("processes", {})
            for span in trace.get("spans", []):
                process = process_by_id.get(span.get("processID"), {})
                parent_span_id = None
                for ref in span.get("references", []):
                    if ref.get("refType") == "CHILD_OF":
                        parent_span_id = ref.get("spanID")
                        break

                rows.append(
                    {
                        "timestamp": pd.to_datetime(span.get("startTime"), unit="us", errors="coerce"),
                        "source": "otel_demo",
                        "run_id": trace_id,
                        "span_id": span.get("spanID"),
                        "parent_span_id": parent_span_id,
                        "service": process.get("serviceName", service),
                        "operation": span.get("operationName"),
                        "duration_ms": span.get("duration", 0) / 1000.0,
                        "status": "error" if any(
                            t.get("key") == "error" and t.get("value") for t in span.get("tags", [])
                        ) else "ok",
                    }
                )

    traces_df = pd.DataFrame(rows, columns=schema.TRACES_COLUMNS)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "traces.parquet"

    if traces_df.empty:
        logger.warning("Aucune trace exportée depuis Jaeger")
        return out_path

    schema.validate_traces_df(traces_df)
    traces_df.to_parquet(out_path, index=False)
    logger.info(f"Traces Jaeger exportées: {len(traces_df)} spans -> {out_path}")
    return out_path
