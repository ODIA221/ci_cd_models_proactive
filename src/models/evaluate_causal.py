"""
Évaluation protocolée du module de corrélation causale (src/causal/) sur
RCAEval RE2 — même discipline méthodologique que evaluate.py/
evaluate_multimodal.py: on mesure un chiffre réel (precision@1/@3 contre le
service fautif injecté), pas juste que le script s'exécute sans erreur.

Ground truth: labels.parquet expose root_cause_service (cf. rcaeval.py
parse(), qui l'extrait du nom de dossier <service>_<fault_type>) sur les
runs 'test_abnormal' uniquement — c'est le seul split où le module de
corrélation a quelque chose à expliquer.

Couverture partielle attendue: Sock Shop (SS) ne fournit pas de traces.csv
dans RCAEval (cf. features.py: "traces manquantes pour Sock Shop"), donc une
partie des runs 'test_abnormal' n'a pas de spans bruts à charger — ces runs
sont comptés séparément (non_evaluable) plutôt que silencieusement ignorés,
pour ne pas gonfler artificiellement la précision rapportée.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import logging
import sys
from datetime import datetime

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.causal import correlation, signals

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def evaluate_run(source_dir: Path, run_id: str, root_cause_service: str, baseline_stats: pd.DataFrame, z_threshold: float) -> dict:
    try:
        traces = correlation.load_run_traces(source_dir, run_id)
    except FileNotFoundError:
        return {"run_id": run_id, "root_cause_service": root_cause_service, "evaluable": False}

    ranked = correlation.rank_suspect_services(traces, run_id, baseline_stats, z_threshold=z_threshold)
    top_services = [r["service"] for r in ranked]

    return {
        "run_id": run_id,
        "root_cause_service": root_cause_service,
        "evaluable": True,
        "predicted_top1_service": top_services[0] if top_services else None,
        "hit_at_1": bool(top_services and top_services[0] == root_cause_service),
        "hit_at_3": root_cause_service in top_services[:3],
        "n_suspect_services": len(ranked),
    }


# --- Comparaison des classements de la couche v2 (src/causal/signals.py) ------
# Question testée: les poids d'attention GAT (alpha) apportent-ils quelque
# chose à la localisation de cause racine, à classement IDENTIQUE par
# ailleurs? D'où l'ablation propagation_attention vs propagation_uniforme.

RANKERS = {
    "score_multimodal": signals.rank_by_anomaly_score,
    "precedence_temporelle": signals.rank_by_onset,
    "propagation_attention": lambda s: signals.rank_by_propagation(s, use_attention=True),
    "propagation_uniforme": lambda s: signals.rank_by_propagation(s, use_attention=False),
    "propagation_degre": lambda s: signals.rank_by_propagation(s, use_attention=False, degree_normalized=True),
}


def attention_uniformity(sig: dict):
    """Entropie normalisée moyenne des lignes d'attention (H / log(degré),
    nœuds de degré >= 2, boucle propre incluse): 1,0 = attention parfaitement
    uniforme sur les voisins, i.e. le GAT ne "choisit" aucun voisin et
    alpha n'apporte aucune information au-delà de la topologie."""
    att = sig.get("attention") if sig else None
    if not att:
        return None
    import numpy as np
    values = []
    for row in np.array(att["layer1"]):
        p = row[row > 1e-6]
        if len(p) >= 2:
            values.append(float(-(p * np.log(p)).sum() / np.log(len(p))))
    return float(np.mean(values)) if values else None


def _signals_or_none(run_id: str, subset: str):
    try:
        return run_id, signals.load_or_compute_run_signals(run_id, subset)
    except Exception as e:  # un cas brut illisible ne doit pas tuer 1 h de calcul
        logging.getLogger(__name__).warning(f"{run_id}: {e}")
        return run_id, None


def compare_rankers(source_dir: Path, subset: str, workers: int, include_normal: bool,
                    baseline_stats: pd.DataFrame, z_threshold: float) -> None:
    labels_df = pd.read_parquet(source_dir / "labels.parquet")
    abnormal = labels_df[labels_df["fault_type"] == "test_abnormal"].set_index("run_id")
    normal_ids = labels_df.loc[labels_df["fault_type"] == "test_normal", "run_id"].tolist() if include_normal else []
    run_ids = abnormal.index.tolist() + normal_ids

    logger.info(f"Calcul des signaux pour {len(run_ids)} runs ({workers} processus, cache: causal_signals/)...")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        computed = dict(pool.map(_signals_or_none, run_ids, [subset] * len(run_ids)))

    rows = []
    for run_id in abnormal.index:
        sig = computed.get(run_id)
        truth = signals.canonical_service(abnormal.loc[run_id, "root_cause_service"])
        base = {
            "run_id": run_id, "systeme": run_id.split("/")[0], "type_faute": abnormal.loc[run_id, "root_cause_fault_type"],
            "root_cause_service": truth, "a_des_traces": bool(sig and sig["modalities_available"]["traces"]),
            "a_attention": bool(sig and sig.get("attention")),
            "attention_entropie_normalisee": attention_uniformity(sig),
        }
        if sig is None:
            rows.append({**base, "ranker": "calcul_impossible", "hit_at_1": False, "hit_at_3": False, "rang": None})
            continue
        for name, ranker in RANKERS.items():
            ranking = ranker(sig)
            rank = ranking.index(truth) + 1 if truth in ranking else None
            rows.append({**base, "ranker": name, "hit_at_1": rank == 1, "hit_at_3": rank is not None and rank <= 3, "rang": rank})
        # Référence: heuristique d'arbre de spans existante (GET /explain).
        try:
            traces = correlation.load_run_traces(source_dir, run_id)
            ranked = [signals.canonical_service(r["service"]) for r in
                      correlation.rank_suspect_services(traces, run_id, baseline_stats, z_threshold=z_threshold)]
            rank = ranked.index(truth) + 1 if truth in ranked else None
            rows.append({**base, "ranker": "arbre_spans_v1", "hit_at_1": rank == 1, "hit_at_3": rank is not None and rank <= 3, "rang": rank})
        except FileNotFoundError:
            pass

    results = pd.DataFrame(rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "experiments"
    results.to_csv(out_dir / f"causal_rankers_{timestamp}.csv", index=False)

    def summarize(df: pd.DataFrame, subset_name: str) -> pd.DataFrame:
        g = df.groupby("ranker").agg(n_runs=("run_id", "nunique"), precision_at_1=("hit_at_1", "mean"), precision_at_3=("hit_at_3", "mean"))
        return g.reset_index().assign(sous_ensemble=subset_name)

    summaries = [
        summarize(results, "tous_runs_anormaux"),
        # Comparaison équitable avec arbre_spans_v1 et l'attention: mêmes runs.
        summarize(results[results["a_attention"]], "runs_avec_attention"),
    ]
    by_fault = (results[results["ranker"] != "calcul_impossible"]
                .groupby(["type_faute", "ranker"]).agg(n_runs=("run_id", "nunique"), precision_at_1=("hit_at_1", "mean"))
                .reset_index())
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(out_dir / f"causal_rankers_summary_{timestamp}.csv", index=False)
    by_fault.to_csv(out_dir / f"causal_rankers_by_fault_{timestamp}.csv", index=False)
    logger.info("\n" + summary.to_string(index=False))
    uniformity = results.drop_duplicates("run_id")["attention_entropie_normalisee"].dropna()
    if len(uniformity):
        logger.info(f"Uniformité de l'attention GAT (1 = uniforme): médiane={uniformity.median():.3f}, min={uniformity.min():.3f} sur {len(uniformity)} runs")
    logger.info("\n" + by_fault.pivot(index="type_faute", columns="ranker", values="precision_at_1").round(2).to_string())

    if normal_ids:
        # Fausses pistes: sur un run SANS faute, combien de services la vue
        # désigne-t-elle comme notablement anormaux (a_i >= 0,5)?
        fp = []
        for run_id in normal_ids + abnormal.index.tolist():
            sig = computed.get(run_id)
            if sig is None:
                continue
            scores = [n["anomaly_score"] for n in sig["nodes"]]
            fp.append({"run_id": run_id, "anormal": run_id in abnormal.index,
                       "n_services": len(scores), "n_services_a_ge_0_5": sum(a >= 0.5 for a in scores),
                       "max_a": max(scores) if scores else 0.0})
        fp_df = pd.DataFrame(fp)
        fp_df.to_csv(out_dir / f"causal_false_leads_{timestamp}.csv", index=False)
        logger.info("Services 'notablement anormaux' par run (médiane, [min-max]):\n" + fp_df.groupby("anormal").agg(
            n_services=("n_services", "median"), notables_median=("n_services_a_ge_0_5", "median"),
            notables_max=("n_services_a_ge_0_5", "max"), max_a_median=("max_a", "median")).to_string())
    logger.info(f"Résultats: experiments/causal_rankers*_{timestamp}.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluation protocolée de la corrélation causale sur RCAEval")
    parser.add_argument("--source-dir", type=str, default="data/interim/rcaeval/RE2", help="Dossier contenant labels.parquet + traces/")
    parser.add_argument("--z-threshold", type=float, default=3.0, help="Seuil de z-score de latence pour marquer un span suspect")
    parser.add_argument("--compare-rankers", action="store_true",
                        help="Compare les classements de la couche v2 (score multimodal, précédence, propagation avec/sans attention GAT) — ~1 h au premier lancement")
    parser.add_argument("--subset", type=str, default="RE2")
    parser.add_argument("--workers", type=int, default=2, help="Processus parallèles (un cas Train Ticket peut occuper ~2 Go de RAM)")
    parser.add_argument("--no-normal", action="store_true", help="Ne calcule pas les signaux des runs normaux (pas de mesure de fausses pistes)")
    args = parser.parse_args()

    source_dir = REPO_ROOT / args.source_dir
    if args.compare_rankers:
        compare_rankers(source_dir, args.subset, args.workers, not args.no_normal,
                        correlation.load_baseline_stats(source_dir), args.z_threshold)
        return

    labels_df = pd.read_parquet(source_dir / "labels.parquet")

    abnormal = labels_df[labels_df["fault_type"] == "test_abnormal"]
    if abnormal.empty:
        raise RuntimeError(f"Aucun run 'test_abnormal' dans {source_dir / 'labels.parquet'}")

    baseline_stats = correlation.load_baseline_stats(source_dir)
    logger.info(f"Baseline de latence chargée: {baseline_stats.shape[0]} (service, operation)")

    results = [
        evaluate_run(source_dir, row.run_id, row.root_cause_service, baseline_stats, args.z_threshold)
        for row in abnormal.itertuples(index=False)
    ]
    results_df = pd.DataFrame(results)

    evaluable = results_df[results_df["evaluable"]]
    n_total = len(results_df)
    n_evaluable = len(evaluable)
    precision_at_1 = evaluable["hit_at_1"].mean() if n_evaluable else float("nan")
    precision_at_3 = evaluable["hit_at_3"].mean() if n_evaluable else float("nan")

    logger.info(f"Couverture: {n_evaluable}/{n_total} runs 'test_abnormal' évaluables (spans bruts disponibles)")
    logger.info(f"precision@1={precision_at_1:.4f}, precision@3={precision_at_3:.4f} (sur les {n_evaluable} runs évaluables)")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = REPO_ROOT / "experiments" / f"causal_eval_{timestamp}.csv"
    out_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(out_path, index=False)
    logger.info(f"Résultats détaillés sauvegardés: {out_path}")


if __name__ == "__main__":
    main()
