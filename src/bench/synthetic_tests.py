"""
Tests synthétiques de l'étape test-unit de PipeFault.

Ils vérifient des propriétés STATIQUES de la copie du code livrée (fichiers
attendus présents, déclarations gRPC présentes, pas de marqueur de conflit),
pas le comportement du service. C'est voulu : comme beaucoup de suites
réelles, ils ne voient pas une faute de configuration ou une régression de
performance — c'est précisément ce qui rend les fautes inter-étapes
« inter-étapes ». C1 inverse une assertion ; C2 fait échouer un test avec une
probabilité donnée (graine fixée par exécution et par tentative).
"""

from pathlib import Path
from xml.sax.saxutils import escape
import random
import unittest

EXPECTED_FILES = {
    "checkout": ["src/checkout/main.go", "src/checkout/go.mod"],
    "payment": ["src/payment/index.js", "src/payment/charge.js", "src/payment/package.json"],
    "recommendation": ["src/recommendation/recommendation_server.py", "src/recommendation/requirements.txt"],
    "ad": ["src/ad/build.gradle", "src/ad/src/main/java/oteldemo/AdService.java"],
}
GRPC_MARKERS = {"checkout": "PlaceOrder", "payment": "chargeServiceHandler",
                "recommendation": "ListRecommendations", "ad": "getAds"}


def build_suite(service: str, demo_dir: Path, broken: bool = False, flaky: float = 0.0,
                rng: random.Random = None) -> unittest.TestSuite:
    rng = rng or random.Random(0)
    flaky_draw = rng.random()

    class ServiceContract(unittest.TestCase):
        def test_expected_files(self):
            for rel in EXPECTED_FILES[service]:
                self.assertTrue((demo_dir / rel).exists(), rel)

        def test_grpc_handler_declared(self):
            main = demo_dir / EXPECTED_FILES[service][-1 if service == "ad" else 0]
            present = GRPC_MARKERS[service] in main.read_text()
            self.assertEqual(present, not broken, "gestionnaire gRPC")   # C1 : assertion inversée

        def test_no_conflict_markers(self):
            for rel in EXPECTED_FILES[service]:
                self.assertNotIn("<<<<<<<", (demo_dir / rel).read_text())

        def test_dependency_resolution(self):
            # C2 : échec aléatoire, comme un test dépendant d'un délai réseau.
            self.assertGreaterEqual(flaky_draw, flaky, "délai dépassé (test instable)")

    ServiceContract.__name__ = f"{service.capitalize()}Contract"
    return unittest.defaultTestLoader.loadTestsFromTestCase(ServiceContract)


def junit_xml(cases: list, attempt: int) -> str:
    """cases = [(service, unittest.TestResult)] -> rapport JUnit minimal."""
    suites = []
    for service, result in cases:
        failed = {t.id(): msg for t, msg in result.failures + result.errors}
        body = []
        for test_id in sorted(failed) + []:
            body.append(f'<testcase classname="{service}" name="{escape(test_id.rsplit(".", 1)[-1])}">'
                        f'<failure>{escape(failed[test_id][-500:])}</failure></testcase>')
        passed = result.testsRun - len(failed)
        body += [f'<testcase classname="{service}" name="passed_{i}"/>' for i in range(passed)]
        suites.append(f'<testsuite name="{service}-attempt{attempt}" tests="{result.testsRun}" '
                      f'failures="{len(failed)}">' + "".join(body) + "</testsuite>")
    return '<?xml version="1.0" encoding="UTF-8"?><testsuites>' + "".join(suites) + "</testsuites>"
