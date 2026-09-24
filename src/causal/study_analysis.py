"""
Analyse des sessions d'étude utilisateur enregistrées par POST /study/sessions
(experiments/user_study/sessions.jsonl) — produit la Table 3 de l'article
(moyenne ± écart-type par condition) UNIQUEMENT à partir de données réellement
collectées. Sans fichier de sessions, le script s'arrête: il n'existe aucun
chemin de code qui produise des chiffres d'étude sans participants.

Tests (plan intra-sujets, petits effectifs, normalité non supposée) :
Friedman sur les 3 conditions (moyenne par participant), puis Wilcoxon
appariés deux à deux avec correction de Bonferroni. NASA-TLX et SUS sont
collectés hors outil (questionnaires papier/formulaire) et ne sont donc pas
calculés ici.

Usage: python -m src.causal.study_analysis
"""

from pathlib import Path
import itertools
import json
import sys

import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SESSIONS_PATH = REPO_ROOT / "experiments" / "user_study" / "sessions.jsonl"
MEASURES = {"diagnosis_seconds": "Temps de diagnostic (s)", "correct": "Précision cause racine", "confidence_likert": "Confiance (1-5)"}


def main() -> None:
    if not SESSIONS_PATH.exists():
        sys.exit(f"Aucune session enregistrée ({SESSIONS_PATH}). Rien à analyser — aucun chiffre ne sera produit.")
    df = pd.DataFrame([json.loads(line) for line in SESSIONS_PATH.read_text().splitlines() if line.strip()])
    df["correct"] = df["correct"].astype(float)
    n_participants = df["participant_id"].nunique()
    print(f"{len(df)} tâches, {n_participants} participant(s), conditions: {df['condition'].value_counts().to_dict()}\n")

    table = df.groupby("condition")[list(MEASURES)].agg(["mean", "std"]).round(2)
    print(table.to_string(), "\n")

    per_participant = df.groupby(["participant_id", "condition"])[list(MEASURES)].mean().reset_index()
    complete = per_participant.groupby("participant_id")["condition"].nunique()
    complete_ids = complete[complete == 3].index
    if len(complete_ids) < 5:
        print(f"Seulement {len(complete_ids)} participant(s) ont passé les 3 conditions: tests statistiques non "
              "calculés (effectif trop faible pour qu'un test de rang soit informatif).")
        return

    data = per_participant[per_participant["participant_id"].isin(complete_ids)]
    conditions = sorted(data["condition"].unique())
    n_pairs = len(conditions) * (len(conditions) - 1) // 2
    for measure, label in MEASURES.items():
        wide = data.pivot(index="participant_id", columns="condition", values=measure)[conditions]
        chi2, p = stats.friedmanchisquare(*[wide[c] for c in conditions])
        print(f"{label}: Friedman χ²={chi2:.2f}, p={p:.4f} (n={len(wide)})")
        for a, b in itertools.combinations(conditions, 2):
            try:
                _, p_pair = stats.wilcoxon(wide[a], wide[b])
            except ValueError:  # différences toutes nulles
                p_pair = 1.0
            print(f"   {a} vs {b}: Wilcoxon p={p_pair:.4f}, p Bonferroni={min(p_pair * n_pairs, 1.0):.4f}")


if __name__ == "__main__":
    main()
