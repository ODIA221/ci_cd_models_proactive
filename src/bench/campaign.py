"""
Orchestrateur de campagne PipeFault (article 5, sections 4.4 et 6).

  plan   : tire le calendrier COMPLET avant toute exécution (graine fixée),
           écrit data/raw/pipefault/schedule.csv — c'est ce fichier qui est
           versionné et déclaré dans le préenregistrement ;
  run    : exécute les lignes non encore faites (reprise après interruption),
           une à une : remise à l'état de référence -> fenêtre de référence ->
           déclenchement du pipeline Jenkins -> faute d'exécution éventuelle ->
           fenêtre d'observation -> export -> retour à l'état de référence ;
  status : avancement.

Vérité terrain : data/raw/pipefault/labels.jsonl (une ligne par exécution
terminée). Rien n'est écrit dans labels.jsonl avant la fin de l'export.

Usage: python -m src.bench.campaign plan [--seed 20261031]
       python -m src.bench.campaign run [--limit N] [--split dev|test]
       python -m src.bench.campaign status
"""

from pathlib import Path
import argparse
import csv
import json
import logging
import os
import random
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bench import faults

OUT = REPO_ROOT / "data" / "raw" / "pipefault"
SCHEDULE = OUT / "schedule.csv"
LABELS = OUT / "labels.jsonl"
DEMO_DIR = Path(os.environ.get("DEMO_DIR", REPO_ROOT.parent / "opentelemetry-demo"))
JENKINS_URL = os.environ.get("PIPEFAULT_JENKINS_URL", "http://localhost:8090")
JOB = "pipefault"
REF_SECONDS = int(os.environ.get("PIPEFAULT_REF_SECONDS", 300))
OBSERVE_SECONDS = int(os.environ.get("PIPEFAULT_OBSERVE_SECONDS", 600))
RUNTIME_INJECT_DELAY = 60   # s après la fin du pipeline (ou du déploiement)
N_DELIVERED = 3

# Plan préenregistré (article 5, tableau 4.4) : (code, répétitions par type).
TEST_PLAN = {"X1": 30, "X2": 30, "X3": 30, "X4": 30, "X5": 30, "X6": 30,
             "C1": 10, "C2": 10, "C3": 10, "C4": 10, "R1": 20, "R2": 20, "R3": 20, "N0": 80}
# Bloc de développement (pilote), exécuté en premier, jamais analysé en confirmatoire.
DEV_PLAN = {"X1": 2, "X2": 1, "X3": 1, "X4": 2, "X5": 1, "X6": 1, "C1": 1, "C2": 1, "C3": 1, "C4": 1,
            "R1": 2, "R2": 2, "R3": 2, "N0": 32}

log = logging.getLogger("pipefault")


def draw_run(rng: random.Random, code: str) -> dict:
    fault = faults.CATALOG[code]
    if fault.family == "execution":
        flag = rng.choice(sorted(faults.RUNTIME_FLAGS[code]))
        root, variant = faults.RUNTIME_FLAGS[code][flag]
        delivered = rng.sample(faults.DELIVERABLE, N_DELIVERED)
        return {"fault": code, "fault_service": root, "flag": flag, "flag_variant": variant, "services": delivered}
    if fault.family == "normale":
        return {"fault": code, "fault_service": "", "flag": "", "flag_variant": "",
                "services": rng.sample(faults.DELIVERABLE, N_DELIVERED)}
    target = rng.choice(sorted(fault.applies_to))
    others = rng.sample([s for s in faults.DELIVERABLE if s != target], N_DELIVERED - 1)
    delivered = others + [target]
    rng.shuffle(delivered)
    return {"fault": code, "fault_service": target, "flag": "", "flag_variant": "", "services": delivered}


def plan(seed: int) -> list:
    rng = random.Random(seed)
    rows = []
    for split, spec in (("dev", DEV_PLAN), ("test", TEST_PLAN)):
        block = [dict(draw_run(rng, code), split=split) for code, n in spec.items() for _ in range(n)]
        rng.shuffle(block)
        rows += block
    for i, row in enumerate(rows, 1):
        f = faults.CATALOG[row["fault"]]
        row.update(run_id=f"pf{i:04d}", seed=rng.randrange(10**9), family=f.family,
                   origin_stage=f.origin_stage or "", services=",".join(row["services"]))
    return rows


def write_schedule(rows: list) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if SCHEDULE.exists():
        raise SystemExit(f"{SCHEDULE} existe déjà : le calendrier préenregistré ne se retire pas. "
                         "Supprimer le fichier à la main uniquement avant le début de la campagne.")
    cols = ["run_id", "split", "family", "fault", "fault_service", "origin_stage", "flag", "flag_variant",
            "services", "seed"]
    with SCHEDULE.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({c: r[c] for c in cols} for r in rows)


def read_schedule() -> list:
    with SCHEDULE.open() as f:
        return list(csv.DictReader(f))


def done_ids() -> set:
    if not LABELS.exists():
        return set()
    return {json.loads(line)["run_id"] for line in LABELS.read_text().splitlines() if line.strip()}


# --- Pilotage de la démo -----------------------------------------------------

def compose(*args: str) -> None:
    cmd = ["docker", "compose", "-p", "opentelemetry-demo", "--project-directory", str(DEMO_DIR),
           "-f", str(DEMO_DIR / "compose.yaml"), "-f", str(REPO_ROOT / "bench" / "compose.pipefault.yaml"), *args]
    log.info("+ " + " ".join(cmd[-6:]))
    subprocess.run(cmd, check=True, env=bench_env())


def bench_env() -> dict:
    e = dict(os.environ)
    image_version = next(l.split("=", 1)[1].strip() for l in (DEMO_DIR / ".env").read_text().splitlines()
                         if l.startswith("IMAGE_VERSION="))
    e.update(PIPEFAULT_DIR=str(REPO_ROOT), DEMO_DIR=str(DEMO_DIR), DEMO_VERSION=image_version,
             OTEL_COLLECTOR_CONFIG_EXTRAS=str(REPO_ROOT / "bench" / "otel" / "otelcol-pipefault.yml"))
    e.setdefault("PIPEFAULT_JENKINS_PASSWORD", "")
    return e


def reset_reference() -> None:
    """État de référence : images publiées de la démo pour les 4 services
    livrables (aucune surcharge de configuration), drapeaux par défaut."""
    write_flags(None)
    compose("up", "-d", "--no-deps", "--force-recreate", *faults.DELIVERABLE)


FLAGS_FILE = DEMO_DIR / "src" / "flagd" / "demo.flagd.json"
FLAGS_BACKUP = OUT / "demo.flagd.original.json"


def write_flags(flag_variant) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not FLAGS_BACKUP.exists():
        FLAGS_BACKUP.write_text(FLAGS_FILE.read_text())
    original = json.loads(FLAGS_BACKUP.read_text())
    doc = original if flag_variant is None else faults.runtime_flag_patch(original, *flag_variant)
    FLAGS_FILE.write_text(json.dumps(doc, indent=2))


# --- Jenkins -------------------------------------------------------------------

def jenkins_session():
    import requests
    s = requests.Session()
    s.auth = ("admin", os.environ.get("PIPEFAULT_JENKINS_PASSWORD") or (REPO_ROOT / "bench" / ".jenkins_password").read_text().strip())
    crumb = s.get(f"{JENKINS_URL}/crumbIssuer/api/json", timeout=30).json()
    s.headers[crumb["crumbRequestField"]] = crumb["crumb"]
    return s


def run_pipeline(s, row: dict) -> dict:
    params = {"RUN_ID": row["run_id"], "SERVICES": row["services"], "FAULT": row["fault"],
              "FAULT_SERVICE": row["fault_service"] if row["family"] in ("ci", "inter_etapes") else "",
              "SEED": row["seed"]}
    r = s.post(f"{JENKINS_URL}/job/{JOB}/buildWithParameters", params=params, timeout=30)
    r.raise_for_status()
    queue = r.headers["Location"].rstrip("/") + "/api/json"
    while True:
        item = s.get(queue, timeout=30).json()
        if item.get("executable"):
            number = item["executable"]["number"]
            break
        time.sleep(3)
    while True:
        build = s.get(f"{JENKINS_URL}/job/{JOB}/{number}/api/json", timeout=30).json()
        if not build["building"]:
            return {"jenkins_build": number, "jenkins_result": build["result"],
                    "jenkins_duration_s": build["duration"] / 1000}
        time.sleep(10)


def execute(row: dict) -> None:
    from src.bench import export
    run_dir = OUT / "runs" / row["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    reset_reference()
    t_ref_start = time.time()
    log.info(f"{row['run_id']} {row['fault']} {row['fault_service']} : fenêtre de référence {REF_SECONDS} s")
    time.sleep(REF_SECONDS)

    s = jenkins_session()
    t_pipeline = time.time()
    result = run_pipeline(s, row)
    manifest = json.loads((run_dir / "manifest.json").read_text()) if (run_dir / "manifest.json").exists() else {}
    t_anchor = manifest.get("t_deploy_end") or time.time()

    t_inject = None
    if row["family"] == "execution":
        time.sleep(max(0, t_anchor + RUNTIME_INJECT_DELAY - time.time()))
        write_flags((row["flag"], row["flag_variant"]))
        t_inject = time.time()
    elif row["family"] == "inter_etapes":
        t_inject = manifest.get("t_deploy")
    elif row["family"] == "ci":
        t_inject = t_pipeline

    time.sleep(max(0, t_anchor + OBSERVE_SECONDS - time.time()))
    t_end = time.time()
    export.export_run(run_dir, t_ref_start, t_end, jenkins=s, build_number=result["jenkins_build"])
    write_flags(None)

    label = {**{k: row[k] for k in ("run_id", "split", "family", "fault", "fault_service", "origin_stage",
                                    "flag", "flag_variant", "seed")},
             "services": row["services"].split(","), **result,
             "t_ref_start": t_ref_start, "t_pipeline_start": t_pipeline, "t_deploy": manifest.get("t_deploy"),
             "t_inject": t_inject, "t_end": t_end, "test_attempts": manifest.get("test_attempts"),
             "patched_file": manifest.get("patched_file")}
    with LABELS.open("a") as f:
        f.write(json.dumps(label, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Campagne PipeFault")
    parser.add_argument("action", choices=["plan", "run", "status"])
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", choices=["dev", "test"], default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.action == "plan":
        rows = plan(args.seed)
        write_schedule(rows)
        by = {}
        for r in rows:
            by.setdefault((r["split"], r["family"]), 0)
            by[(r["split"], r["family"])] += 1
        print(f"{len(rows)} exécutions -> {SCHEDULE}")
        for k in sorted(by):
            print(f"  {k[0]:4s} {k[1]:13s} {by[k]}")
        return

    rows = read_schedule()
    done = done_ids()
    if args.action == "status":
        print(f"{len(done)}/{len(rows)} exécutions terminées")
        return
    todo = [r for r in rows if r["run_id"] not in done and (args.split is None or r["split"] == args.split)]
    if args.split is None and any(r["split"] == "dev" for r in todo) and any(r["split"] == "test" for r in todo):
        todo = [r for r in todo if r["split"] == "dev"]   # le pilote d'abord, toujours
    for row in todo[: args.limit]:
        execute(row)


if __name__ == "__main__":
    main()
