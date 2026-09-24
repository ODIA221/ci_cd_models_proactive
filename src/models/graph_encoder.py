"""
Encodeur de graphe (GAT) pour la branche traces RCAEval — remplace les 6
statistiques agrégées de build_traces_agg_matrix (src/data/features.py) par
un embedding appris, mis en cache hors ligne (precompute), consommé ensuite
via --use-gat-traces dans train_rcaeval.py / evaluate_multimodal.py.

Graphe au niveau SERVICE, pas span: un run RCAEval contient ~200k-400k spans
(mesuré sur data/interim/rcaeval/RE2/traces/*.parquet), une matrice
d'adjacence dense span-par-span est donc ingérable (N^2). Un graphe de
dépendances entre les <=35 services distincts d'un run reste petit (dense
tractable) et correspond à la granularité déjà utilisée par le module de
corrélation causale (GET /explain raisonne en "service suspect").

Les spans bruts par run ne sont mis en cache en parquet (rcaeval.py::parse)
que pour le split test — pas train, pour limiter le temps de parsing. Ce
script relit donc directement les traces.csv bruts par cas (même lecture que
fait déjà parse() avec succès sur ~270 cas) pour construire les graphes de
TOUTES les fenêtres, sans jamais les persister à ce niveau de détail: seul
l'embedding final (taille fixe) est mis en cache.
"""

from pathlib import Path
import argparse
import logging
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data.sources.rcaeval import RCAEvalConnector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# call_count, duration_mean, error_rate (le reste du vecteur de nœud est le
# one-hot d'identité de service, taille = len(service_vocab))
N_NODE_STATS = 3


class GATLayer(nn.Module):
    """
    Attention de graphe (Veličković et al.) fait main, sur adjacence DENSE
    (N x N) — tractable ici car N <= ~35 (graphe service, pas span). Pas de
    dépendance torch_geometric/torch_scatter.
    """

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.W = nn.Linear(in_dim, n_heads * out_dim, bias=False)
        self.attn = nn.Parameter(torch.empty(n_heads, 2 * out_dim))
        nn.init.xavier_uniform_(self.attn)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor, return_attention: bool = False):
        """
        x: (N, in_dim), adjacency: (N, N) bool (boucles propres incluses).
        Returns (N, n_heads*out_dim), ou (sortie, alpha (N, N, heads)) si
        return_attention — alpha[i, j] = poids que le nœud i accorde au
        voisin j (normalisé sur j), AVANT dropout (en eval() c'est identique).
        """
        n = x.size(0)
        h = self.W(x).view(n, self.n_heads, self.out_dim)

        h_i = h.unsqueeze(1).expand(n, n, self.n_heads, self.out_dim)
        h_j = h.unsqueeze(0).expand(n, n, self.n_heads, self.out_dim)
        concat = torch.cat([h_i, h_j], dim=-1)
        e = self.leaky_relu((concat * self.attn).sum(dim=-1))  # (N, N, heads)

        e = e.masked_fill(~adjacency.unsqueeze(-1), float("-inf"))
        alpha = torch.softmax(e, dim=1)  # normalisé sur les voisins j
        alpha_dropped = self.dropout(alpha)

        out = torch.einsum("ijh,jhd->ihd", alpha_dropped, h).reshape(n, self.n_heads * self.out_dim)
        if return_attention:
            return out, alpha
        return out


class TraceGraphAutoencoder(nn.Module):
    """
    2 GATLayer empilées (encodeur, embedding par nœud) + décodeur MLP par
    nœud reconstruisant son vecteur de features d'origine. Perte Huber, même
    choix que MultimodalAutoencoder._modality_loss (stabilité numérique face
    à des services d'échelle d'activité très différente au sein d'un run).
    encode_graph() moyenne les embeddings de nœuds -> vecteur de taille fixe
    par run, indépendant du nombre de services impliqués.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 32, embed_dim: int = 16, n_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.gat1 = GATLayer(in_dim, hidden_dim, n_heads=n_heads, dropout=dropout)
        self.gat2 = GATLayer(hidden_dim * n_heads, embed_dim, n_heads=1, dropout=dropout)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )
        self.optimizer = None

    def encode_nodes(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.gat1(x, adjacency))
        return self.gat2(h, adjacency)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode_nodes(x, adjacency))

    def attention_weights(self, x: torch.Tensor, adjacency: torch.Tensor) -> dict:
        """
        Poids d'attention des deux couches pour un graphe (mode eval, sans
        gradient): {"layer1": (N, N) moyenne des têtes, "layer2": (N, N)}.
        ATTENTION: ce modèle est un auto-encodeur entraîné à RECONSTRUIRE
        des graphes normaux — rien dans son objectif ne pousse alpha à
        refléter une influence causale. Ce sont des poids associatifs
        (échelon 1 de Pearl), à valider empiriquement (cf.
        src/models/evaluate_causal.py --ranker), pas à présenter comme causaux.
        """
        self.eval()
        with torch.no_grad():
            h, alpha1 = self.gat1(x, adjacency, return_attention=True)
            _, alpha2 = self.gat2(F.elu(h), adjacency, return_attention=True)
        return {
            "layer1": alpha1.mean(dim=-1).cpu().numpy(),
            "layer2": alpha2.mean(dim=-1).cpu().numpy(),
        }

    def encode_graph(self, x: torch.Tensor, adjacency: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            z = self.encode_nodes(x, adjacency)
            return z.mean(dim=0).cpu().numpy()

    def train_model(self, graphs: list, epochs: int = 100, lr: float = 1e-3):
        """graphs: liste de (x, adjacency) — un par run du split train (normal uniquement)."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        graphs = [(x.to(device), a.to(device)) for x, a in graphs]
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.SmoothL1Loss()

        for epoch in range(epochs):
            self.train()
            self.optimizer.zero_grad()
            total_loss = 0.0
            # Graphes à taille variable (nb de services par run) -> pas de
            # TensorDataset/DataLoader classique: chaque graphe est passé
            # individuellement, la perte moyennée sur le "batch" complet
            # (peu de graphes, ~185, dense sur N<=35: reste rapide).
            for x, adjacency in graphs:
                loss = criterion(self(x, adjacency), x) / len(graphs)
                loss.backward()
                total_loss += loss.item()
            self.optimizer.step()

            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.6f}")


def build_service_graph(spans: pd.DataFrame, service_vocab: list):
    """
    spans: colonnes 'service'/'span_id'/'parent_span_id'/'duration_ms'/'status',
    déjà restreintes à une fenêtre/run. service_vocab: liste globale FIXE et
    triée des services connus (index one-hot stable entre runs).

    Returns (node_features: FloatTensor(N,F), adjacency: BoolTensor(N,N), node_names)
    ou None si aucun service identifiable dans ce segment. N = services
    DISTINCTS présents dans ce run (jamais tout le vocabulaire).
    """
    node_names = sorted(spans["service"].dropna().unique().tolist())
    if not node_names:
        return None
    idx = {name: i for i, name in enumerate(node_names)}
    n = len(node_names)
    vocab_index = {name: i for i, name in enumerate(service_vocab)}

    g = spans.groupby("service")
    call_count = np.log1p(g.size().reindex(node_names, fill_value=0).to_numpy(dtype="float32"))
    duration_mean = np.log1p(g["duration_ms"].mean().reindex(node_names, fill_value=0).to_numpy(dtype="float32"))
    error_rate = (
        g["status"].apply(lambda s: (pd.to_numeric(s, errors="coerce").fillna(0) != 0).mean())
        .reindex(node_names, fill_value=0).to_numpy(dtype="float32")
    )

    one_hot = np.zeros((n, len(service_vocab)), dtype="float32")
    for name, i in idx.items():
        j = vocab_index.get(name)
        if j is not None:
            one_hot[i, j] = 1.0

    node_features = np.concatenate([np.stack([call_count, duration_mean, error_rate], axis=1), one_hot], axis=1)

    # Arêtes: parent_span_id -> span_id, mappées service -> service (table
    # locale au segment: span_id peut être réutilisé entre segments
    # différents, jamais au sein du même). Vectorisé (pas de boucle Python
    # ligne à ligne: un segment peut compter jusqu'à ~400k spans).
    span_to_service = spans.set_index("span_id")["service"]
    span_to_service = span_to_service[~span_to_service.index.duplicated(keep="first")]

    parent_service = spans["parent_span_id"].map(span_to_service)
    own_service = spans["service"]
    valid = own_service.notna() & parent_service.notna()

    adjacency = np.eye(n, dtype=bool)
    if valid.any():
        child_idx = own_service[valid].map(idx).to_numpy()
        parent_idx = parent_service[valid].map(idx).to_numpy()
        adjacency[parent_idx, child_idx] = True
        adjacency[child_idx, parent_idx] = True

    return torch.from_numpy(node_features), torch.from_numpy(adjacency), node_names


def _collect_service_vocab(case_dirs: list) -> list:
    """Pré-scan léger (une seule colonne) pour fixer un vocabulaire de
    services global et stable avant de construire le moindre graphe."""
    services = set()
    for case_dir in case_dirs:
        path = case_dir / "traces.csv"
        if not path.exists():
            continue
        try:
            col = pd.read_csv(path, usecols=["serviceName"])["serviceName"]
        except ValueError:
            continue
        services.update(col.dropna().unique().tolist())
    return sorted(services)


def gat_model_path(subset: str = "RE2") -> Path:
    return REPO_ROOT / "data" / "interim" / "rcaeval" / subset / "trace_gat_model.pt"


def save_gat_model(model: "TraceGraphAutoencoder", service_vocab: list, path: Path, seed: int) -> None:
    """Persiste poids + vocabulaire de services (sans lui, l'index one-hot
    des nœuds n'est pas reconstructible) + hyperparamètres d'architecture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "service_vocab": service_vocab,
        "in_dim": model.gat1.W.in_features,
        "embed_dim": model.embed_dim,
        "seed": seed,
    }, path)


def load_gat_model(subset: str = "RE2"):
    """Returns (model en eval(), service_vocab) ou lève FileNotFoundError si
    le modèle n'a jamais été sauvegardé (runs antérieurs à cette option)."""
    path = gat_model_path(subset)
    if not path.exists():
        raise FileNotFoundError(
            f"Modèle GAT non sauvegardé ({path}). Lance: python -m src.models.graph_encoder --subset {subset} --save-model-only"
        )
    checkpoint = torch.load(path, map_location="cpu")
    model = TraceGraphAutoencoder(in_dim=checkpoint["in_dim"], embed_dim=checkpoint["embed_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["service_vocab"]


def build_and_cache_gat_features(subset: str = "RE2", embed_dim: int = 16, epochs: int = 100,
                                 save_model_only: bool = False, seed: int = 42) -> Path:
    # Graine fixée: sans elle, deux réentraînements donnent des poids
    # d'attention différents pour le même run — inacceptable pour une vue
    # censée "expliquer" une anomalie de façon reproductible.
    torch.manual_seed(seed)
    np.random.seed(seed)
    connector = RCAEvalConnector()
    subset_dir = connector.raw_dir / subset
    if not subset_dir.exists():
        raise FileNotFoundError(
            f"'{subset_dir}' introuvable. Lance d'abord: python -m src.data.acquire --source rcaeval --subset {subset}"
        )

    case_dirs = sorted(
        d for d in subset_dir.glob(f"{subset}-*/*/*")
        if d.is_dir() and (d / "inject_time.txt").exists()
    )
    if not case_dirs:
        raise RuntimeError(f"Aucun cas trouvé dans {subset_dir}")

    logger.info(f"{len(case_dirs)} cas trouvés, construction du vocabulaire de services...")
    service_vocab = _collect_service_vocab(case_dirs)
    logger.info(f"{len(service_vocab)} services distincts: {service_vocab}")

    train_graphs = []
    all_graphs = {}

    for i, case_dir in enumerate(case_dirs):
        case_id = "/".join(case_dir.relative_to(subset_dir).parts)
        inject_time = int((case_dir / "inject_time.txt").read_text().strip())
        is_train_case = connector._split_for_case(case_id)

        traces_path = case_dir / "traces.csv"
        if not traces_path.exists():
            continue
        traces_raw = pd.read_csv(traces_path)
        if "startTime" not in traces_raw.columns:
            continue
        traces_ts_seconds = traces_raw["startTime"] / 1e6

        windows = [("normal", "train" if is_train_case else "test_normal")]
        if not is_train_case:
            windows.append(("abnormal", "test_abnormal"))

        for window_name, split in windows:
            run_id = f"{case_id}__{window_name}"
            mask = traces_ts_seconds < inject_time if window_name == "normal" else traces_ts_seconds >= inject_time
            seg = traces_raw[mask]
            if seg.empty:
                continue
            spans = pd.DataFrame({
                "service": seg.get("serviceName"),
                "span_id": seg.get("spanID"),
                "parent_span_id": seg.get("parentSpanID"),
                "duration_ms": seg.get("duration"),
                "status": seg.get("statusCode"),
            })
            graph = build_service_graph(spans, service_vocab)
            if graph is None:
                continue
            x, adjacency, _ = graph
            all_graphs[run_id] = (x, adjacency)
            if split == "train":
                train_graphs.append((x, adjacency))

        if (i + 1) % 20 == 0:
            logger.info(f"{i + 1}/{len(case_dirs)} cas traités (graphes construits: {len(all_graphs)})")

    logger.info(f"{len(train_graphs)} graphes d'entraînement (normal, split train), {len(all_graphs)} graphes au total")
    if not train_graphs:
        raise RuntimeError("Aucun graphe d'entraînement construit — vérifier que traces.csv contient des spans exploitables")

    in_dim = N_NODE_STATS + len(service_vocab)
    model = TraceGraphAutoencoder(in_dim=in_dim, embed_dim=embed_dim)
    model.train_model(train_graphs, epochs=epochs)

    model_path = gat_model_path(subset)
    save_gat_model(model.cpu(), service_vocab, model_path, seed)
    logger.info(f"Modèle GAT sauvegardé (poids + vocabulaire): {model_path}")
    if save_model_only:
        # Ne réécrit PAS trace_gat_features.parquet: les résultats déjà
        # publiés (chapitre 3/5 de docs/) ont été produits avec l'ancien
        # cache, non reproductible (entraîné sans graine).
        return model_path

    device = next(model.parameters()).device
    rows = {
        run_id: model.encode_graph(x.to(device), adjacency.to(device))
        for run_id, (x, adjacency) in all_graphs.items()
    }

    out_df = pd.DataFrame.from_dict(rows, orient="index", columns=[f"trace_gat_{i}" for i in range(embed_dim)])
    out_df.index.name = "run_id"

    out_path = REPO_ROOT / "data" / "interim" / "rcaeval" / subset / "trace_gat_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path)
    logger.info(f"Embeddings GAT traces mis en cache: {out_path} ({out_df.shape[0]} runs x {embed_dim} colonnes)")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Précalcule des embeddings GAT (graphe de dépendances service) pour la branche traces RCAEval")
    parser.add_argument("--subset", type=str, default="RE2")
    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-model-only", action="store_true",
                        help="Entraîne et sauvegarde le modèle (pour l'extraction d'attention) sans réécrire trace_gat_features.parquet")
    args = parser.parse_args()
    build_and_cache_gat_features(subset=args.subset, embed_dim=args.embed_dim, epochs=args.epochs,
                                 save_model_only=args.save_model_only, seed=args.seed)


if __name__ == "__main__":
    main()
