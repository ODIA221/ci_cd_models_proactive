"""
Export de la télémétrie d'une exécution PipeFault sur sa fenêtre
[début de la fenêtre de référence, fin de l'observation] :

- métriques Prometheus (query_range, pas de 15 s) des familles utiles ;
- traces Jaeger de chaque service, y compris le pipeline Jenkins (service
  « jenkins » exporté par le greffon OpenTelemetry) ;
- journaux Loki (OTLP) ;
- côté Jenkins : description par étape (wfapi), journal de console, rapports
  de tests ;
- état Docker des services livrables (nombre de redémarrages, OOMKilled).

Sorties dans data/raw/pipefault/runs/<run_id>/ : metrics.csv, traces.csv,
logs.csv, jenkins_stages.json, console.txt, junit_report.json, docker_state.json.
"""

from pathlib import Path
import json
import os
import subprocess

import pandas as pd
import requests

PROMETHEUS = os.environ.get("PIPEFAULT_PROMETHEUS", "http://localhost:9090")
JAEGER = os.environ.get("PIPEFAULT_JAEGER", "http://localhost:16686/jaeger/ui")
LOKI = os.environ.get("PIPEFAULT_LOKI", "http://localhost:3100")
JENKINS_URL = os.environ.get("PIPEFAULT_JENKINS_URL", "http://localhost:8090")
METRIC_REGEX = os.environ.get(
    "PIPEFAULT_METRICS", "container_.*|traces_span_metrics_.*|process_.*|jvm_.*|nodejs_.*|go_.*|rpc_.*|http_.*")
STEP = 15


def export_metrics(t0: float, t1: float) -> pd.DataFrame:
    rows = []
    r = requests.get(f"{PROMETHEUS}/api/v1/query_range", timeout=120,
                     params={"query": f'{{__name__=~"{METRIC_REGEX}"}}', "start": t0, "end": t1, "step": STEP})
    r.raise_for_status()
    for series in r.json()["data"]["result"]:
        labels = series["metric"]
        service = labels.get("service_name") or labels.get("container_name") or labels.get("job", "")
        name = labels.get("__name__")
        extra = ",".join(f"{k}={v}" for k, v in sorted(labels.items())
                         if k not in ("__name__", "service_name", "container_name", "job", "instance"))
        for ts, value in series["values"]:
            rows.append((float(ts), service, name, extra, float(value)))
    return pd.DataFrame(rows, columns=["timestamp", "service", "metric", "labels", "value"])


def export_traces(t0: float, t1: float) -> pd.DataFrame:
    services = requests.get(f"{JAEGER}/api/services", timeout=60).json().get("data") or []
    rows = []
    for svc in services:
        r = requests.get(f"{JAEGER}/api/traces", timeout=300, params={
            "service": svc, "start": int(t0 * 1e6), "end": int(t1 * 1e6), "limit": 5000})
        for trace in r.json().get("data") or []:
            procs = trace["processes"]
            for sp in trace["spans"]:
                parent = next((ref["spanID"] for ref in sp.get("references", []) if ref["refType"] == "CHILD_OF"), None)
                tags = {t["key"]: t["value"] for t in sp.get("tags", [])}
                rows.append((sp["traceID"], sp["spanID"], parent, procs[sp["processID"]]["serviceName"],
                             sp["operationName"], sp["startTime"], sp["duration"],
                             tags.get("otel.status_code") or tags.get("error"), tags.get("rpc.grpc.status_code")))
    df = pd.DataFrame(rows, columns=["trace_id", "span_id", "parent_span_id", "service", "operation",
                                     "start_time_us", "duration_us", "status", "grpc_status"])
    return df.drop_duplicates(["trace_id", "span_id"])


def export_logs(t0: float, t1: float) -> pd.DataFrame:
    rows, start = [], int(t0 * 1e9)
    end = int(t1 * 1e9)
    while start < end:
        r = requests.get(f"{LOKI}/loki/api/v1/query_range", timeout=120, params={
            "query": '{service_name=~".+"}', "start": start, "end": end, "limit": 5000, "direction": "forward"})
        r.raise_for_status()
        batch = [(int(ts), s["stream"].get("service_name"), s["stream"].get("severity_text") or s["stream"].get("level"), line)
                 for s in r.json()["data"]["result"] for ts, line in s["values"]]
        if not batch:
            break
        rows += batch
        last = max(b[0] for b in batch)
        if len(batch) < 5000 or last + 1 <= start:
            break
        start = last + 1
    return pd.DataFrame(rows, columns=["timestamp_ns", "service", "level", "message"])


def export_jenkins(s, build_number: int, run_dir: Path) -> None:
    base = f"{JENKINS_URL}/job/pipefault/{build_number}"
    (run_dir / "jenkins_stages.json").write_text(json.dumps(s.get(f"{base}/wfapi/describe", timeout=60).json(), indent=2))
    (run_dir / "console.txt").write_text(s.get(f"{base}/consoleText", timeout=60).text)
    # Rapport JUnit agrégé par Jenkins (404 si le pipeline s'arrête avant test-unit).
    report = s.get(f"{base}/testReport/api/json", timeout=60)
    if report.status_code == 200:
        (run_dir / "junit_report.json").write_text(json.dumps(report.json(), indent=2))


def export_docker_state(run_dir: Path) -> None:
    out = subprocess.run(["docker", "ps", "-a", "--filter", "label=com.docker.compose.project=opentelemetry-demo",
                          "--format", "{{.Names}}"], capture_output=True, text=True).stdout.split()
    state = {}
    for name in out:
        info = json.loads(subprocess.run(["docker", "inspect", name], capture_output=True, text=True).stdout)[0]
        state[name] = {"restart_count": info.get("RestartCount"), "oom_killed": info["State"].get("OOMKilled"),
                       "status": info["State"].get("Status"), "image": info["Config"].get("Image")}
    (run_dir / "docker_state.json").write_text(json.dumps(state, indent=2))


def export_run(run_dir: Path, t0: float, t1: float, jenkins=None, build_number: int = None) -> None:
    export_metrics(t0, t1).to_csv(run_dir / "metrics.csv", index=False)
    export_traces(t0, t1).to_csv(run_dir / "traces.csv", index=False)
    export_logs(t0, t1).to_csv(run_dir / "logs.csv", index=False)
    if jenkins is not None and build_number is not None:
        export_jenkins(jenkins, build_number, run_dir)
    export_docker_state(run_dir)
