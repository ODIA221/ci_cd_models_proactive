"""
Vérifications du banc PipeFault qui ne demandent NI Docker en marche NI
Jenkins : tout ce qui peut casser silencieusement une campagne est contrôlé
contre la démo réelle (../opentelemetry-demo).

1. chaque point d'insertion de code (X4, C4) existe et la modification est faite ;
2. chaque surcharge de configuration (X1, X2, X5) vise une variable que le
   service déclare déjà dans compose.yaml, et chaque limite (X3, X6) est
   plus basse que celle d'origine ;
3. chaque drapeau et variante d'exécution existe dans demo.flagd.json ;
4. les tests synthétiques : verts sans faute, rouges sous C1, instables sous C2 ;
5. le calendrier : effectifs du préenregistrement, couples valides, service
   fautif livré, tirage reproductible ;
6. les étapes checkout et test-unit du pipeline, exécutées pour de vrai dans
   un répertoire temporaire ;
7. les fichiers compose/Jenkins : syntaxe, et `docker compose config` si le
   CLI est installé (pas besoin du démon).

Usage: python -m src.bench.selftest
"""

from pathlib import Path
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bench import campaign, faults, pipeline
from src.bench.synthetic_tests import build_suite

DEMO = Path(os.environ.get("DEMO_DIR", REPO_ROOT.parent / "opentelemetry-demo"))
results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append(ok)
    print(f"[{'OK' if ok else 'ÉCHEC'}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def mem_mb(value: str) -> float:
    v = str(value).upper()
    return float(v[:-1]) * (1024 if v.endswith("G") else 1) if v[-1] in "MG" else float(v) / 2**20


def run_suite(svc, demo, **kw) -> int:
    with open(os.devnull, "w") as null:
        r = unittest.TextTestRunner(stream=null).run(build_suite(svc, demo, **kw))
    return len(r.failures) + len(r.errors)


def main() -> None:
    if not DEMO.exists():
        raise SystemExit(f"Démo absente : {DEMO} (./run.sh otel-up la clone)")
    compose = yaml.safe_load((DEMO / "compose.yaml").read_text())["services"]
    flags = json.loads((DEMO / "src" / "flagd" / "demo.flagd.json").read_text())["flags"]

    with tempfile.TemporaryDirectory() as tmp:
        for fault, services in (("X4", faults.DELIVERABLE), ("C4", faults.CATALOG["C4"].applies_to)):
            for svc in services:
                copy = Path(tmp) / f"{fault}-{svc}"
                rel = faults.CODE_ANCHORS[svc][0]
                (copy / rel).parent.mkdir(parents=True)
                shutil.copy(DEMO / rel, copy / rel)
                faults.patch_source(copy, fault, svc)
                check(f"{fault} insérée dans {rel}", f"PIPEFAULT {fault}" in (copy / rel).read_text())

    for code in ("X1", "X2", "X5"):
        for svc in faults.CATALOG[code].applies_to:
            declared = {e.split("=", 1)[0] for e in compose[svc].get("environment", [])}
            for var in faults.CATALOG[code].env(svc):
                check(f"{code}/{svc} : {var} déclarée par le service", var in declared)
    for code in ("X3", "X6"):
        for svc in faults.CATALOG[code].applies_to:
            limits = faults.CATALOG[code].limits(svc)
            if "memory" in limits:
                orig = compose[svc]["deploy"]["resources"]["limits"]["memory"]
                check(f"X6/{svc} : mémoire {limits['memory']} < {orig}", mem_mb(limits["memory"]) < mem_mb(orig))
            else:
                check(f"X3/{svc} : cpus {limits['cpus']}", float(limits["cpus"]) < 1)
    ov = faults.deploy_override(["checkout", "payment", "ad"], "pf0000", "X6", "payment")
    check("surcharge de déploiement : images du run + limite sur le seul service fautif",
          ov["services"]["checkout"]["image"] == "pipefault/checkout:pf0000"
          and "deploy" in ov["services"]["payment"] and "deploy" not in ov["services"]["ad"])

    for code, table in faults.RUNTIME_FLAGS.items():
        for flag, (root, variant) in table.items():
            ok = flag in flags and variant in flags[flag]["variants"]
            check(f"{code} : drapeau {flag}={variant} (racine {root})", ok and root in compose)

    for svc in faults.DELIVERABLE:
        check(f"tests synthétiques verts sans faute ({svc})", run_suite(svc, DEMO) == 0)
        check(f"C1 fait échouer les tests ({svc})", run_suite(svc, DEMO, broken=True) > 0)
    first_fail = sum(run_suite("payment", DEMO, flaky=0.3, rng=random.Random(i)) > 0 for i in range(300)) / 300
    check("C2 : échec d'une tentative ≈ 30 %", 0.22 <= first_fail <= 0.38, f"{first_fail:.0%} sur 300 tirages")

    rows = campaign.plan(20261031)
    again = campaign.plan(20261031)
    check("calendrier reproductible à graine fixée", rows == again)
    for split, spec in (("dev", campaign.DEV_PLAN), ("test", campaign.TEST_PLAN)):
        counts = {}
        for r in rows:
            if r["split"] == split:
                counts[r["fault"]] = counts.get(r["fault"], 0) + 1
        check(f"effectifs {split} = plan préenregistré", counts == spec, f"{sum(counts.values())} exécutions")
    valid = set(faults.valid_pairs())
    bad = [r["run_id"] for r in rows if r["family"] in ("ci", "inter_etapes")
           and ((r["fault"], r["fault_service"]) not in valid or r["fault_service"] not in r["services"].split(","))]
    check("couples (faute, service) valides et service fautif livré", not bad, ", ".join(bad[:5]))
    check(f"{campaign.N_DELIVERED} services livrés par exécution",
          all(len(r["services"].split(",")) == campaign.N_DELIVERED for r in rows))
    inter = sum(r["split"] == "test" and r["family"] == "inter_etapes" for r in rows)
    check("180 exécutions inter-étapes en test (puissance, section 6.5)", inter == 180)

    with tempfile.TemporaryDirectory() as tmp:
        os.environ.update(RUN_ID="pf-selftest", SERVICES="checkout,payment,ad", FAULT="C2", FAULT_SERVICE="ad",
                          SEED="7", DEMO_DIR=str(DEMO), PIPEFAULT_DIR=tmp, WORKSPACE=tmp)
        p = pipeline.params()
        pipeline.stage_checkout(p)
        check("étape checkout : copie de la démo", (Path(tmp) / "demo" / "src" / "ad").exists())
        with open(os.devnull, "w") as null:
            stdout, sys.stdout = sys.stdout, null
            try:
                code = pipeline.stage_test_unit(p)
            finally:
                sys.stdout = stdout
        manifest = json.loads((pipeline.run_dir(p) / "manifest.json").read_text())
        check("étape test-unit : rapports JUnit et tentatives enregistrées",
              len(list((Path(tmp) / "junit").glob("*.xml"))) == len(manifest["test_attempts"]),
              f"tentatives {manifest['test_attempts']}, code {code}")
        os.environ.update(FAULT="X4", FAULT_SERVICE="checkout")
        pipeline.stage_checkout(pipeline.params())
        check("étape checkout : X4 appliquée à la copie, pas à la démo",
              "PIPEFAULT X4" in (Path(tmp) / "demo" / "src/checkout/main.go").read_text()
              and "PIPEFAULT" not in (DEMO / "src/checkout/main.go").read_text())

    for f in ("bench/compose.pipefault.yaml", "bench/jenkins/casc.yaml", "bench/otel/otelcol-pipefault.yml",
              "bench/loki/loki.yaml"):
        try:
            yaml.safe_load((REPO_ROOT / f).read_text())
            check(f"YAML valide : {f}", True)
        except yaml.YAMLError as e:
            check(f"YAML valide : {f}", False, str(e)[:120])
    if shutil.which("docker"):
        env = campaign.bench_env()
        env["PIPEFAULT_JENKINS_PASSWORD"] = "selftest"
        r = subprocess.run(["docker", "compose", "-p", "opentelemetry-demo", "--project-directory", str(DEMO),
                            "-f", str(DEMO / "compose.yaml"), "-f", str(REPO_ROOT / "bench/compose.pipefault.yaml"),
                            "config", "--quiet"], capture_output=True, text=True, env=env)
        check("docker compose config (démo + banc)", r.returncode == 0, r.stderr.strip()[:300])

    n_bad = results.count(False)
    print(f"\n{len(results) - n_bad}/{len(results)} vérifications réussies")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
