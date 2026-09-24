"""
Évaluation protocolée de la proactivité (chapitre 10 de la thèse) — mesure
le délai de détection minimal, pas juste la précision/rappel à fenêtre
complète déjà mesurés par evaluate_multimodal.py.

LogPipeGuard exige d'"anticiper l'anomalie avant qu'elle ne se
matérialise". Opérationnalisation retenue (discutée et validée): minimiser
le DÉLAI DE DÉTECTION après le début réel du problème, pas prédire
l'imprévisible — dans RCAEval les fautes sont injectées de façon exogène
(stress-ng/pumba) à un instant choisi par l'expérimentateur, donc il
n'existe structurellement aucun précurseur causal dans la fenêtre normale
qui précède l'injection.

Un détecteur DISTINCT est entraîné PAR HORIZON plutôt que de réutiliser un
modèle entraîné sur fenêtre complète avec moins de données en entrée: un
agrégat (mean/std/...) sur 15s de trafic n'a pas la même distribution qu'un
agrégat sur 720s, même en l'absence de faute — comparer un modèle "fenêtre
complète" à des entrées tronquées mesurerait un artefact de fenêtrage, pas
un vrai déficit de détection. cf. RCAEvalConnector.build_horizon_features()
(src/data/sources/rcaeval.py) pour le fenêtrage exact (H dernières secondes
avant inject_time = baseline normale, H premières secondes après = évaluation).

Structure temporelle vérifiée sur RE2 (OB/SS/TT): 720s avant / 720s après
inject_time sur tous les cas — d'où la borne haute de la grille d'horizons.
"""

from pathlib import Path
import argparse
import logging
import sys
from datetime import datetime

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data.sources.rcaeval import RCAEvalConnector
from src.models.detection_models import AnomalyDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

HORIZONS = [15, 30, 60, 120, 300, 720]

# (nom, préfixes) — mêmes conventions que evaluate_multimodal.py::FEATURE_SETS.
# 'traces' seul (meilleure modalité isolée à fenêtre complète, AUC=0.78) et
# 'fused' (toutes colonnes) pour rester borné en temps de calcul (6 horizons
# x 2 jeux plutôt que x4).
FEATURE_SETS = [
    ("traces", ("trace_",)),
    ("fused", None),
]


def build_features_for_set(features_df: pd.DataFrame, prefixes) -> pd.DataFrame:
    if prefixes is None:
        return features_df
    cols = [c for c in features_df.columns if c.startswith(prefixes)]
    if not cols:
        raise ValueError(f"Aucune colonne ne correspond aux préfixes {prefixes}")
    return features_df[cols]


def evaluate_horizon(features_df: pd.DataFrame, labels_df: pd.DataFrame, feature_set: str, prefixes) -> dict:
    labels_indexed = labels_df.set_index("run_id")

    try:
        X = build_features_for_set(features_df, prefixes)
    except ValueError as e:
        logger.warning(f"Jeu '{feature_set}' ignoré: {e}")
        return None

    split = labels_indexed.reindex(X.index)["fault_type"]
    y = pd.Series(labels_indexed.reindex(X.index)["label"].to_numpy(), index=X.index)

    train_mask = split == "train"
    test_mask = split.isin(["test_normal", "test_abnormal"])
    X_train, X_test, y_test = X[train_mask], X[test_mask], y[test_mask]

    detector = AnomalyDetector(model_type="isolation_forest", contamination="auto")
    detector.build_model()
    detector.train(X_train.values)

    y_pred = detector.predict(X_test.values)
    precision = precision_score(y_test, y_pred, pos_label=-1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=-1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=-1, zero_division=0)

    try:
        scores = -detector.model.decision_function(X_test.values)
        auc = roc_auc_score((y_test == -1).astype(int), scores)
    except ValueError as e:
        logger.warning(f"AUC non calculable: {e}")
        auc = None

    return {
        "feature_set": feature_set,
        "precision_anomalie": round(precision, 4),
        "rappel_anomalie": round(recall, 4),
        "f1_anomalie": round(f1, 4),
        "auc": round(auc, 4) if auc is not None else None,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_vraies_anomalies": int((y_test == -1).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluation du délai de détection (proactivité) sur RCAEval")
    parser.add_argument("--subset", type=str, default="RE2")
    parser.add_argument("--horizons", type=int, nargs="+", default=HORIZONS)
    args = parser.parse_args()

    connector = RCAEvalConnector()
    results = []

    for horizon in args.horizons:
        logger.info(f"=== Horizon: {horizon}s ===")
        features_df, labels_df = connector.build_horizon_features(horizon=horizon, subset=args.subset)

        for feature_set, prefixes in FEATURE_SETS:
            result = evaluate_horizon(features_df, labels_df, feature_set, prefixes)
            if result is None:
                continue
            result["horizon_seconds"] = horizon
            results.append(result)
            logger.info(
                f"horizon={horizon}s / {feature_set}: précision={result['precision_anomalie']}, "
                f"rappel={result['rappel_anomalie']}, F1={result['f1_anomalie']}, AUC={result['auc']}"
            )

    results_df = pd.DataFrame(results)
    results_df = results_df[["horizon_seconds", "feature_set"] + [c for c in results_df.columns if c not in ("horizon_seconds", "feature_set")]]
    print("\n" + results_df.to_string(index=False))

    # Résumé: plus petit horizon atteignant >=80% du rappel à la fenêtre
    # complète (720s) — réponse chiffrée directe à "à partir de quand peut-on
    # détecter fiablement", par jeu de features.
    for feature_set, _ in FEATURE_SETS:
        subset_results = results_df[results_df["feature_set"] == feature_set].sort_values("horizon_seconds")
        if subset_results.empty:
            continue
        full_recall = subset_results[subset_results["horizon_seconds"] == max(args.horizons)]["rappel_anomalie"]
        if full_recall.empty:
            continue
        target = 0.8 * full_recall.iloc[0]
        reliable = subset_results[subset_results["rappel_anomalie"] >= target]
        if not reliable.empty:
            best = reliable.iloc[0]
            logger.info(
                f"[{feature_set}] Délai de détection minimal (>=80% du rappel à {max(args.horizons)}s, "
                f"soit {target:.3f}): {best['horizon_seconds']}s (rappel={best['rappel_anomalie']})"
            )
        else:
            logger.info(f"[{feature_set}] Aucun horizon testé n'atteint 80% du rappel à fenêtre complète")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = REPO_ROOT / "experiments" / f"evaluation_proactive_{timestamp}.csv"
    out_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(out_path, index=False)
    logger.info(f"Résultats sauvegardés: {out_path}")


if __name__ == "__main__":
    main()
