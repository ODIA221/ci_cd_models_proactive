"""
Encodeur de séquence (LSTM) pour la branche logs RCAEval — remplace les
colonnes event_*/2gram_* (sac d'événements + bigrammes, src/data/features.py,
ordre perdu) par un embedding appris sur la VRAIE séquence ordonnée de
templates de log, mis en cache hors ligne, consommé via --use-lstm-logs dans
train_rcaeval.py / evaluate_multimodal.py.

Le champ cluster_id de logs.csv (template déjà calculé) n'existe que pour 25
des 271 cas RE2 (tous Online Boutique) — cf. correction du commentaire dans
rcaeval.py. Ce script mine donc TOUS les cas via un seul TemplateMiner
partagé (drain3, comme src/data/log_parsing.py), pour un vocabulaire de
templates cohérent d'un run à l'autre — le cluster_id pré-existant est
ignoré pour cette même raison (numérotation incohérente avec le reste du
corpus s'il était réutilisé tel quel).

Sur ce corpus (3 systèmes RE2 aux formats de log très hétérogènes: logs
structurés Go, stack traces Java/Spring, payloads JSON à texte libre),
drain3 finit par créer des centaines de milliers de clusters distincts
(mesuré: 272534) — un vocabulaire de sortie de cette taille rend le
décodeur (softmax par pas de temps) infaisable en pratique (tué après 2h+
sans qu'une seule époque ne termine). Le vocabulaire du MODÈLE est donc
plafonné séparément de celui de drain3: seuls les `vocab_cap` templates les
plus fréquents (comptés sur le split TRAIN uniquement, même discipline que
CICDPreprocessor.fit) gardent un id dédié, tout le reste est regroupé sous
un unique token UNK — pratique standard pour les vocabulaires à longue
traîne en NLP.
"""

from collections import Counter
from pathlib import Path
import argparse
import logging
import pickle
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data.sources.rcaeval import RCAEvalConnector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

PAD_ID = 0
START_ID = 1
UNK_ID = 2
# Les cluster_id drain3 (1-indexés) sont décalés de +1 à l'usage (cluster 1
# -> token 2, etc.) pour réserver 0=PAD/1=START/2=UNK avant le remappage
# vers un vocabulaire compact (cf. _build_vocab_mapping).


class LogSequenceAutoencoder(nn.Module):
    """
    Embedding -> LSTM encodeur (dernier état caché -> embedding de run, dim
    fixe) -> vecteur latent utilisé comme état INITIAL (h0/c0) d'un LSTM
    décodeur en teacher forcing DÉCALÉ (le décodeur ne voit jamais le token
    qu'il doit prédire, seulement ses prédécesseurs + le latent) -> logits
    sur le vocabulaire, cross-entropy par pas (padding ignoré). Sans ce
    décalage, un décodeur qui reçoit directement le token cible en entrée
    peut simplement le recopier et contourner le goulot d'étranglement latent
    — le décalage force l'information à passer par le vecteur compressé.
    Entraîné sur les séquences train (normal uniquement), même pretexte de
    reconstruction que TraceGraphAutoencoder (graph_encoder.py) et
    MultimodalAutoencoder pour les autres modalités.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 16, hidden_dim: int = 32,
                 pad_id: int = PAD_ID, start_id: int = START_ID):
        super().__init__()
        self.pad_id = pad_id
        self.start_id = start_id
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.encoder_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, embed_dim)
        self.to_h0 = nn.Linear(embed_dim, hidden_dim)
        self.to_c0 = nn.Linear(embed_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.to_vocab = nn.Linear(hidden_dim, vocab_size)
        self.optimizer = None

    def encode(self, seqs: torch.Tensor) -> torch.Tensor:
        """seqs: (B, L) ids -> (B, embed_dim)"""
        embedded = self.embedding(seqs)
        _, (h_n, _) = self.encoder_lstm(embedded)
        return self.to_latent(h_n[-1])

    def _shift_right(self, seqs: torch.Tensor) -> torch.Tensor:
        start_col = torch.full((seqs.size(0), 1), self.start_id, dtype=seqs.dtype, device=seqs.device)
        return torch.cat([start_col, seqs[:, :-1]], dim=1)

    def forward(self, seqs: torch.Tensor) -> torch.Tensor:
        latent = self.encode(seqs)
        h0 = self.to_h0(latent).unsqueeze(0)
        c0 = self.to_c0(latent).unsqueeze(0)
        decoder_input = self.embedding(self._shift_right(seqs))
        decoder_out, _ = self.decoder_lstm(decoder_input, (h0, c0))
        return self.to_vocab(decoder_out)

    def encode_batch(self, seqs: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            return self.encode(seqs).cpu().numpy()

    def train_model(self, sequences: torch.Tensor, epochs: int = 50, batch_size: int = 32, lr: float = 1e-3):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        sequences = sequences.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=self.pad_id)

        n = sequences.size(0)
        for epoch in range(epochs):
            self.train()
            perm = torch.randperm(n, device=device)
            total_loss, n_batches = 0.0, 0
            for start in range(0, n, batch_size):
                batch = sequences[perm[start:start + batch_size]]
                self.optimizer.zero_grad()
                logits = self(batch)
                loss = criterion(logits.reshape(-1, logits.size(-1)), batch.reshape(-1))
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / n_batches:.6f}")


def _dedupe_consecutive(ids: list) -> list:
    """Compresse les répétitions consécutives du même template (logs de
    heartbeat/polling identiques des dizaines de milliers de fois de suite,
    observé sur RCAEval) — sinon la troncature aux derniers max_len
    événements ne verrait jamais qu'un seul type d'événement répété."""
    out = []
    for x in ids:
        if not out or out[-1] != x:
            out.append(x)
    return out


def _build_vocab_mapping(sequences: dict, train_run_ids: list, vocab_cap: int) -> dict:
    """token brut (cluster_id+1) -> id compact [3, 3+vocab_cap), basé sur la
    fréquence dans le split TRAIN uniquement (jamais le test — même
    discipline que CICDPreprocessor.fit, évite de dimensionner le
    vocabulaire sur des templates uniquement vus en test). Les tokens hors
    des vocab_cap plus fréquents (train ou test) sont mappés vers UNK_ID à
    l'usage, pas dans ce dict."""
    counts = Counter()
    for run_id in train_run_ids:
        counts.update(sequences[run_id])
    kept = [tok for tok, _ in counts.most_common(vocab_cap)]
    return {tok: i + 3 for i, tok in enumerate(kept)}  # 0=PAD, 1=START, 2=UNK


def _to_padded_tensor(sequences: dict, max_len: int, pad_id: int = PAD_ID):
    """sequences: run_id -> list[int]. Troncature aux DERNIERS max_len tokens
    (biais de récence, cohérent avec le chapitre proactivité), padding à
    GAUCHE (les tokens réels restent alignés à droite)."""
    run_ids = list(sequences.keys())
    arr = np.full((len(run_ids), max_len), pad_id, dtype=np.int64)
    for i, run_id in enumerate(run_ids):
        seq = sequences[run_id][-max_len:]
        arr[i, max_len - len(seq):] = seq
    return run_ids, torch.from_numpy(arr)


def _mine_sequences(subset: str, cache_dir: Path, use_cache: bool = True) -> tuple:
    """Mining Drain3 (étape coûteuse, ~40 min sur RE2) isolée dans sa propre
    fonction avec cache disque: en cas d'échec plus loin dans le pipeline
    (déjà arrivé deux fois pendant le développement — vocabulaire explosé,
    puis décodeur infaisable), on ne veut plus jamais avoir à la refaire
    pour itérer sur ce qui vient après (plafonnement du vocabulaire,
    hyperparamètres du LSTM, etc.)."""
    cache_path = cache_dir / "_log_sequences_cache.pkl"
    if use_cache and cache_path.exists():
        logger.info(f"Séquences de templates chargées depuis le cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

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

    miner_config = TemplateMinerConfig()
    # Un seul miner partagé sur les 3 systèmes RE2 (OB/SS/TT), aux formats de
    # log très hétérogènes (logs structurés Go, stack traces Java/Spring,
    # payloads JSON à description libre type lorem ipsum...) — sans borne, le
    # coût par message explose (recherche dans un arbre Drain qui n'arrive
    # plus à discriminer): 20 cas traités en ~45s avant la borne, ~29 MIN
    # après (observé entre les cas OB et le premier cas Sock Shop).
    # drain_max_clusters (cache LRU des templates trackés pour le matching)
    # borne ce coût — mais PAS le nombre total de templates distincts créés
    # au fil du corpus (drain.clusters_counter, non borné): cf.
    # _build_vocab_mapping pour le plafonnement du vocabulaire du MODÈLE.
    miner_config.drain_max_clusters = 1000
    miner = TemplateMiner(config=miner_config)
    sequences, train_run_ids = {}, []

    for i, case_dir in enumerate(case_dirs):
        case_id = "/".join(case_dir.relative_to(subset_dir).parts)
        inject_time = int((case_dir / "inject_time.txt").read_text().strip())
        is_train_case = connector._split_for_case(case_id)

        logs_path = case_dir / "logs.csv"
        if not logs_path.exists():
            continue
        logs_raw = pd.read_csv(logs_path, usecols=["timestamp", "message"])
        if logs_raw.empty:
            continue
        logs_raw = logs_raw.sort_values("timestamp")
        logs_ts_seconds = logs_raw["timestamp"] / 1e9

        # Mining sur toute la séquence du cas en une fois, dans l'ordre — les
        # fenêtres normal/abnormal sont ensuite de simples sous-plages
        # d'indices, pas besoin de re-miner par fenêtre.
        template_ids = [
            miner.add_log_message(str(m).strip())["cluster_id"] + 1  # +1: réserve 0=PAD, 1=START, 2=UNK
            for m in logs_raw["message"].to_numpy()
        ]
        logs_raw = logs_raw.assign(template_id=template_ids)

        windows = [("normal", "train" if is_train_case else "test_normal")]
        if not is_train_case:
            windows.append(("abnormal", "test_abnormal"))

        for window_name, split in windows:
            run_id = f"{case_id}__{window_name}"
            mask = logs_ts_seconds < inject_time if window_name == "normal" else logs_ts_seconds >= inject_time
            seg_ids = logs_raw.loc[mask, "template_id"].tolist()
            if not seg_ids:
                continue
            seg_ids = _dedupe_consecutive(seg_ids)
            sequences[run_id] = seg_ids
            if split == "train":
                train_run_ids.append(run_id)

        if (i + 1) % 20 == 0:
            logger.info(f"{i + 1}/{len(case_dirs)} cas traités (séquences: {len(sequences)})")

    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump((sequences, train_run_ids), f)
        logger.info(f"Séquences de templates mises en cache: {cache_path}")

    return sequences, train_run_ids


def build_and_cache_lstm_features(subset: str = "RE2", embed_dim: int = 16, hidden_dim: int = 32,
                                   epochs: int = 50, max_len: int = 200, vocab_cap: int = 500,
                                   use_mining_cache: bool = True) -> Path:
    cache_dir = REPO_ROOT / "data" / "interim" / "rcaeval" / subset
    sequences, train_run_ids = _mine_sequences(subset, cache_dir, use_cache=use_mining_cache)
    if not train_run_ids:
        raise RuntimeError("Aucune séquence d'entraînement construite — vérifier logs.csv (colonnes timestamp/message)")

    vocab_map = _build_vocab_mapping(sequences, train_run_ids, vocab_cap)
    sequences = {rid: [vocab_map.get(tok, UNK_ID) for tok in seq] for rid, seq in sequences.items()}
    vocab_size = len(vocab_map) + 3  # 0=PAD, 1=START, 2=UNK
    logger.info(f"{len(sequences)} séquences ({len(train_run_ids)} train), vocabulaire plafonné: {vocab_size} (dont UNK)")

    train_ids, train_tensor = _to_padded_tensor({rid: sequences[rid] for rid in train_run_ids}, max_len)

    model = LogSequenceAutoencoder(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim)
    model.train_model(train_tensor, epochs=epochs)

    all_ids, all_tensor = _to_padded_tensor(sequences, max_len)
    device = next(model.parameters()).device
    embeddings = model.encode_batch(all_tensor.to(device))

    out_df = pd.DataFrame(embeddings, index=all_ids, columns=[f"event_lstm_{i}" for i in range(embed_dim)])
    out_df.index.name = "run_id"

    out_path = REPO_ROOT / "data" / "interim" / "rcaeval" / subset / "event_lstm_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path)
    logger.info(f"Embeddings LSTM logs mis en cache: {out_path} ({out_df.shape[0]} runs x {embed_dim} colonnes)")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Précalcule des embeddings LSTM (séquence de templates de log) pour la branche logs RCAEval")
    parser.add_argument("--subset", type=str, default="RE2")
    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max-len", type=int, default=200)
    parser.add_argument("--vocab-cap", type=int, default=500, help="Nombre de templates les plus fréquents (split train) gardant un id dédié; le reste est regroupé sous un token UNK partagé.")
    parser.add_argument("--no-mining-cache", action="store_true", help="Force le re-mining Drain3 même si un cache de séquences existe déjà.")
    args = parser.parse_args()
    build_and_cache_lstm_features(subset=args.subset, embed_dim=args.embed_dim, epochs=args.epochs, max_len=args.max_len,
                                   vocab_cap=args.vocab_cap, use_mining_cache=not args.no_mining_cache)


if __name__ == "__main__":
    main()
