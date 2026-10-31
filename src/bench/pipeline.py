"""
Étapes du pipeline PipeFault, appelées une à une par bench/jenkins/Jenkinsfile
(une étape Jenkins = un appel `python3 -m src.bench.pipeline <étape>`), pour
que la logique soit testable hors de Jenkins (src/bench/selftest.py).

Étapes : checkout -> build -> test-unit -> image-check -> deploy -> smoke.
La fenêtre d'observation après déploiement n'est PAS une étape : c'est
l'orchestrateur (campaign.py) qui la tient, pour que la durée du pipeline ne
dépende pas de la campagne.

Paramètres (variables d'environnement posées par Jenkins) :
RUN_ID, SERVICES (liste séparée par des virgules), FAULT, FAULT_SERVICE,
SEED, DEMO_DIR (même chemin que sur l'hôte), PIPEFAULT_DIR, WORKSPACE.
Bibliothèque standard uniquement (conteneur Jenkins).
"""

from pathlib import Path
import json
import os
import random
import shutil
import subprocess
import sys
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.bench import faults
from src.bench.synthetic_tests import build_suite, junit_xml

COMPOSE_PROJECT = "opentelemetry-demo"
TEST_ATTEMPTS = 3   # politique de relance identique pour TOUTES les exécutions


def env(name: str, default: str = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        raise SystemExit(f"Variable {name} manquante")
    return value


def params() -> dict:
    return {"run_id": env("RUN_ID"), "services": [s for s in env("SERVICES").split(",") if s],
            "fault": env("FAULT", "N0"), "fault_service": os.environ.get("FAULT_SERVICE") or None,
            "seed": int(env("SEED", "0")), "demo_dir": Path(env("DEMO_DIR")),
            "pipefault_dir": Path(env("PIPEFAULT_DIR")), "workspace": Path(env("WORKSPACE", os.getcwd()))}


def run_dir(p: dict) -> Path:
    d = p["pipefault_dir"] / "data" / "raw" / "pipefault" / "runs" / p["run_id"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def update_manifest(p: dict, **fields) -> None:
    path = run_dir(p) / "manifest.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data.update(fields)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def sh(cmd: list, **kw) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def dotenv(demo_dir: Path) -> dict:
    values = {}
    for line in (demo_dir / ".env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip()
    return values


def stage_checkout(p: dict) -> None:
    src = p["workspace"] / "demo"
    if src.exists():
        shutil.rmtree(src)
    shutil.copytree(p["demo_dir"], src, ignore=shutil.ignore_patterns(".git"))
    patched = None
    if p["fault"] in ("X4", "C4"):
        patched = faults.patch_source(src, p["fault"], p["fault_service"])
        print(f"Modification du code : {patched}", flush=True)
    update_manifest(p, run_id=p["run_id"], services=p["services"], fault=p["fault"],
                    fault_service=p["fault_service"], seed=p["seed"], patched_file=patched,
                    t_pipeline_start=time.time())


def stage_build(p: dict) -> None:
    src = p["workspace"] / "demo"
    dot = dotenv(p["demo_dir"])
    for svc in p["services"]:
        dockerfile = dot[f"{svc.upper().replace('-', '_')}_DOCKERFILE"]
        cmd = ["docker", "build", "-f", str(src / dockerfile), "-t", f"pipefault/{svc}:{p['run_id']}",
               "--label", f"pipefault.run_id={p['run_id']}"]
        if svc == "ad":
            cmd += ["--build-arg", f"OTEL_JAVA_AGENT_VERSION={dot['OTEL_JAVA_AGENT_VERSION']}"]
        if p["fault"] == "C3" and svc == p["fault_service"]:
            # HTTP(S)_PROXY sont des arguments de construction prédéfinis de
            # Docker : toutes les commandes RUN qui téléchargent échouent.
            cmd += ["--no-cache", "--build-arg", f"HTTP_PROXY={faults.UNREACHABLE_PROXY}",
                    "--build-arg", f"HTTPS_PROXY={faults.UNREACHABLE_PROXY}"]
        sh(cmd + [str(src)])
    update_manifest(p, t_build_end=time.time())


def stage_test_unit(p: dict) -> int:
    """Tests SYNTHÉTIQUES du banc (synthetic_tests.py) : ils produisent les
    signaux d'une étape de test (journal, rapport JUnit, durée, relances) mais
    ne testent pas la logique métier — limite assumée et rapportée."""
    out = p["workspace"] / "junit"
    out.mkdir(exist_ok=True)
    attempts = []
    for attempt in range(1, TEST_ATTEMPTS + 1):
        failed = 0
        cases = []
        for svc in p["services"]:
            broken = p["fault"] == "C1" and svc == p["fault_service"]
            flaky = 0.3 if p["fault"] == "C2" and svc == p["fault_service"] else 0.0
            rng = random.Random(p["seed"] * 100 + attempt)
            suite = build_suite(svc, p["workspace"] / "demo", broken=broken, flaky=flaky, rng=rng)
            result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
            failed += len(result.failures) + len(result.errors)
            cases.append((svc, result))
        (out / f"tests-attempt{attempt}.xml").write_text(junit_xml(cases, attempt))
        attempts.append(failed)
        print(f"Tentative {attempt}/{TEST_ATTEMPTS} : {failed} échec(s)", flush=True)
        if failed == 0:
            break
    update_manifest(p, test_attempts=attempts)
    return 0 if attempts[-1] == 0 else 1


def stage_image_check(p: dict) -> None:
    for svc in p["services"]:
        sh(["docker", "image", "inspect", "--format", "{{.Id}} {{.Size}}", f"pipefault/{svc}:{p['run_id']}"])


def compose_cmd(p: dict) -> list:
    files = [p["demo_dir"] / "compose.yaml", p["pipefault_dir"] / "bench" / "compose.pipefault.yaml"]
    cmd = ["docker", "compose", "-p", COMPOSE_PROJECT, "--project-directory", str(p["demo_dir"])]
    for f in files:
        cmd += ["-f", str(f)]
    return cmd


def stage_deploy(p: dict) -> None:
    override = faults.deploy_override(p["services"], p["run_id"], p["fault"], p["fault_service"])
    path = run_dir(p) / "deploy-override.yaml"
    path.write_text(json.dumps(override, indent=2))   # le JSON est du YAML valide
    t_deploy = time.time()
    sh(compose_cmd(p) + ["-f", str(path), "up", "-d", "--no-deps", "--no-build", *p["services"]])
    update_manifest(p, t_deploy=t_deploy, t_deploy_end=time.time(), deploy_override=override)


def stage_smoke(p: dict) -> int:
    import urllib.request
    url = os.environ.get("SMOKE_URL", "http://frontend-proxy:8080/")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                print(f"Smoke {url} -> {r.status}", flush=True)
                update_manifest(p, smoke_status=r.status)
                return 0
        except Exception as e:  # noqa: BLE001
            print(f"Smoke tentative {attempt + 1}: {e}", flush=True)
            time.sleep(5)
    update_manifest(p, smoke_status=None)
    return 1


STAGES = {"checkout": stage_checkout, "build": stage_build, "test-unit": stage_test_unit,
          "image-check": stage_image_check, "deploy": stage_deploy, "smoke": stage_smoke}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in STAGES:
        raise SystemExit(f"Usage: python3 -m src.bench.pipeline <{'|'.join(STAGES)}>")
    sys.exit(STAGES[sys.argv[1]](params()) or 0)
