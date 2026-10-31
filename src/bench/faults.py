"""
Catalogue des fautes du banc PipeFault (article 5, section 4.3), écrit contre
le code RÉEL de la démo OpenTelemetry 3.0.0 (dossier frère ../opentelemetry-demo) :
variables d'environnement et limites de compose.yaml, points d'insertion dans
le code des quatre services livrés, drapeaux flagd de src/flagd/demo.flagd.json.

Trois familles :
- CI pures : la faute naît ET se voit dans le pipeline (étape en échec, ou
  relances d'un test instable) ;
- inter-étapes : la faute naît dans le pipeline (code construit ou
  configuration livrée) mais le pipeline reste vert, l'effet n'apparaît qu'à
  l'exécution ;
- exécution pures : drapeau flagd activé pendant la fenêtre d'observation,
  sur un service qui n'est pas forcément livré.

Chaque faute déclare les services auxquels elle s'applique : ce n'est PAS le
même ensemble pour toutes (une erreur de compilation n'existe pas en Node.js
ou en Python, une limite de tas n'a pas d'équivalent simple en Python...).
Le calendrier (campaign.py) ne tire que des couples (faute, service) valides.

Ce module n'importe que la bibliothèque standard : il est aussi exécuté dans
le conteneur Jenkins (src/bench/pipeline.py).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
import re

DELIVERABLE = ["checkout", "payment", "recommendation", "ad"]

# Valeurs de compose.yaml / .env de la démo 3.0.0 (vérifiées le 2026-10-31).
DEFAULT_PORTS = {"checkout": 5050, "payment": 50051, "recommendation": 9001, "ad": 9555}
DEFAULT_MEMORY = {"checkout": "20M", "payment": "140M", "recommendation": "500M", "ad": "300M"}

# Point d'insertion d'une régression (X4) ou d'une erreur de compilation (C4) :
# (fichier relatif à la racine de la démo, motif de la ligne d'ouverture de la
# fonction, gabarit inséré juste après — {ms} = durée de la pause).
CODE_ANCHORS = {
    "checkout": ("src/checkout/main.go",
                 r"func \(cs \*checkout\) PlaceOrder\(ctx context\.Context, req \*pb\.PlaceOrderRequest\) \(\*pb\.PlaceOrderResponse, error\) \{",
                 "\ttime.Sleep({ms} * time.Millisecond) // PIPEFAULT X4\n"),
    "payment": ("src/payment/index.js",
                r"async function chargeServiceHandler\(call, callback\) \{",
                "  await new Promise(resolve => setTimeout(resolve, {ms})); // PIPEFAULT X4\n"),
    "recommendation": ("src/recommendation/recommendation_server.py",
                       r"    def ListRecommendations\(self, request, context\):",
                       "        __import__('time').sleep({ms} / 1000)  # PIPEFAULT X4\n"),
    "ad": ("src/ad/src/main/java/oteldemo/AdService.java",
           r"    public void getAds\(AdRequest req, StreamObserver<AdResponse> responseObserver\) \{",
           "      try { Thread.sleep({ms}); } catch (InterruptedException e) { Thread.currentThread().interrupt(); } // PIPEFAULT X4\n"),
}
COMPILE_ERROR = {
    "checkout": "\tpipefaultUndefinedSymbol() // PIPEFAULT C4\n",
    "ad": "      pipefaultUndefinedSymbol(); // PIPEFAULT C4\n",
}

# Drapeau flagd -> (service racine, variante activée). Variantes de demo.flagd.json.
RUNTIME_FLAGS = {
    "R1": {"intlShippingSlowdown": ("shipping", "5sec"), "imageSlowLoad": ("image-provider", "5sec")},
    "R2": {"adHighCpu": ("ad", "on"), "emailMemoryLeak": ("email", "1000x"), "adManualGc": ("ad", "on")},
    "R3": {"paymentFailure": ("payment", "50%"), "cartFailure": ("cart", "50%"),
           "productCatalogFailure": ("product-catalog", "on"), "adFailure": ("ad", "on"),
           "recommendationCacheFailure": ("recommendation", "on")},
}

UNREACHABLE_PROXY = "http://127.0.0.1:9"   # port « discard » fermé : refus immédiat


@dataclass(frozen=True)
class Fault:
    code: str
    family: str                 # ci | inter_etapes | execution | normale
    label: str
    origin_stage: Optional[str]  # étape d'origine (None pour exécution/normale)
    applies_to: tuple = ()      # services livrables concernés (vide = drapeau)
    expected: str = ""          # effet attendu, vérifié par src/bench/check.py
    env: Callable = field(default=None, compare=False, repr=False)       # service -> {VAR: valeur}
    limits: Callable = field(default=None, compare=False, repr=False)    # service -> {memory|cpus: ...}


def _x1_env(service):
    return {"checkout": {"PAYMENT_ADDR": "payment:1"},
            "recommendation": {"PRODUCT_CATALOG_ADDR": "product-catalog:1"}}[service]


def _x2_env(service):
    # Limite de tas mal réglée dans la configuration livrée (valeurs d'origine :
    # GOMEMLIMIT=16MiB pour checkout, pas de limite explicite pour payment/ad).
    return {"checkout": {"GOMEMLIMIT": "2MiB"},
            "payment": {"NODE_OPTIONS": "--require @opentelemetry/auto-instrumentations-node/register --max-old-space-size=16"},
            "ad": {"_JAVA_OPTIONS": "-Xmx32m"}}[service]


def _x5_env(service):
    # Le service écoute sur un autre port que celui que ses appelants utilisent
    # (leurs variables *_ADDR ne changent pas) : symptôme chez les appelants.
    return {f"{service.upper()}_PORT": str(DEFAULT_PORTS[service] + 1)}


CATALOG = {
    "N0": Fault("N0", "normale", "exécution sans faute", None, tuple(DELIVERABLE), "pipeline vert, pas d'effet"),
    "C1": Fault("C1", "ci", "test unitaire cassé (assertion inversée)", "test-unit", tuple(DELIVERABLE),
                "étape test-unit en échec"),
    "C2": Fault("C2", "ci", "test rendu instable (échec aléatoire p = 0,3, 3 tentatives)", "test-unit",
                tuple(DELIVERABLE), "au moins une tentative de test en échec"),
    "C3": Fault("C3", "ci", "miroir de dépendances inaccessible pendant la construction", "build",
                tuple(DELIVERABLE), "étape build en échec (proxy injoignable)"),
    "C4": Fault("C4", "ci", "erreur de compilation", "build", ("checkout", "ad"), "étape build en échec"),
    "X1": Fault("X1", "inter_etapes", "adresse d'une dépendance erronée dans la configuration livrée", "deploy",
                ("checkout", "recommendation"), "pipeline vert ; erreurs d'appel après déploiement", env=_x1_env),
    "X2": Fault("X2", "inter_etapes", "limite de tas trop basse dans la configuration livrée", "deploy",
                ("checkout", "payment", "ad"), "pipeline vert ; latence, erreurs ou redémarrages", env=_x2_env),
    "X3": Fault("X3", "inter_etapes", "limite CPU du conteneur trop basse dans le manifeste", "deploy",
                tuple(DELIVERABLE), "pipeline vert ; latence accrue", limits=lambda s: {"cpus": "0.05"}),
    "X4": Fault("X4", "inter_etapes", "régression de performance dans le code (pause sur une route fréquente)",
                "build", tuple(DELIVERABLE), "pipeline vert ; latence accrue"),
    "X5": Fault("X5", "inter_etapes", "port d'écoute du service modifié, appelants inchangés", "deploy",
                tuple(DELIVERABLE), "pipeline vert ; erreurs chez les appelants", env=_x5_env),
    "X6": Fault("X6", "inter_etapes", "limite mémoire du conteneur trop basse dans le manifeste", "deploy",
                tuple(DELIVERABLE), "pipeline vert ; redémarrages (OOM)",
                limits=lambda s: {"memory": {"checkout": "6M", "payment": "40M", "recommendation": "40M", "ad": "80M"}[s]}),
    "R1": Fault("R1", "execution", "latence injectée par drapeau (SC1 d'OF4CD)", None, (), "latence du service racine"),
    "R2": Fault("R2", "execution", "saturation de ressources par drapeau (SC2 d'OF4CD)", None, (), "CPU/mémoire du service racine"),
    "R3": Fault("R3", "execution", "erreurs applicatives par drapeau", None, (), "taux d'erreur du service racine"),
}


def patch_source(demo_dir: Path, fault: str, service: str, ms: int = 300) -> Optional[str]:
    """Applique X4 (pause) ou C4 (erreur de compilation) dans une COPIE de la
    démo. Renvoie le fichier modifié, None si la faute ne touche pas au code.
    Échoue bruyamment si le point d'insertion a disparu (autre version de la
    démo) : mieux vaut une campagne arrêtée qu'une faute silencieusement absente."""
    if fault not in ("X4", "C4"):
        return None
    rel, pattern, template = CODE_ANCHORS[service]
    path = demo_dir / rel
    text = path.read_text()
    match = re.search(pattern, text)
    if not match:
        raise RuntimeError(f"Point d'insertion introuvable dans {rel} (version de la démo différente ?)")
    insert = template.replace("{ms}", str(ms)) if fault == "X4" else COMPILE_ERROR[service]
    end = text.index("\n", match.end()) + 1
    path.write_text(text[:end] + insert + text[end:])
    return rel


def deploy_override(services: list, run_id: str, fault: str, fault_service: Optional[str],
                    image_repo: str = "pipefault") -> dict:
    """Fichier compose additionnel « livré » par l'étape deploy : image
    construite par ce run pour chaque service livré, plus la configuration
    fautive éventuelle (X1/X2/X3/X5/X6) pour le service fautif."""
    f = CATALOG[fault]
    out = {"services": {}}
    for svc in services:
        entry = {"image": f"{image_repo}/{svc}:{run_id}",
                 "labels": {"pipefault.run_id": run_id, "pipefault.delivered": "true"}}
        if svc == fault_service:
            if f.env:
                entry["environment"] = f.env(svc)
            if f.limits:
                entry["deploy"] = {"resources": {"limits": f.limits(svc)}}
        out["services"][svc] = entry
    return out


def runtime_flag_patch(flags_doc: dict, flag: str, variant: str) -> dict:
    """Copie de demo.flagd.json avec un drapeau basculé (flagd relit le fichier)."""
    doc = {"$schema": flags_doc.get("$schema"), "flags": {k: dict(v) for k, v in flags_doc["flags"].items()}}
    if variant not in doc["flags"][flag]["variants"]:
        raise ValueError(f"Variante {variant!r} absente du drapeau {flag}")
    doc["flags"][flag]["defaultVariant"] = variant
    return doc


def valid_pairs() -> list:
    """Tous les couples (faute, service fautif) que le calendrier peut tirer."""
    pairs = []
    for code, f in CATALOG.items():
        if f.family == "execution":
            pairs += [(code, f"{flag}") for flag in RUNTIME_FLAGS[code]]
        elif f.family != "normale":
            pairs += [(code, s) for s in f.applies_to]
    return pairs
