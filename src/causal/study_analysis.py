"""
Analyse PRÉENREGISTRÉE des sessions d'étude utilisateur enregistrées par
POST /study/sessions (experiments/user_study/sessions.jsonl). Le plan est
décrit dans docs/08-protocole-etude-utilisateur.md ; ce script l'exécute tel
quel, sans option qui changerait l'analyse a posteriori.

Il ne produit des chiffres QU'À PARTIR de données réellement collectées :
sans fichier de sessions, il s'arrête. Aucun chemin de code ne fabrique de
résultats d'étude.

Plan (conditions A = manuel, B = attribution statique, C = exploration) :
- H1, principale : temps de diagnostic plus court en C qu'en A. Modèle mixte
  linéaire sur log(temps), ordonnée aléatoire par participant, position du
  bloc (1er, 2e, 3e) en effet fixe, contraste C-A, test bilatéral à 0,05.
  Tous les modèles ci-dessous incluent la même covariable de position.
- H2, secondaire : exactitude plus élevée en C qu'en B. Régression logistique
  GEE groupée par participant (corrélation échangeable, erreur standard
  corrigée pour petits échantillons de Mancl-DeRouen), contraste C-B.
- H3, secondaire : temps plus court en C qu'en B. Même modèle que H1, contraste C-B.
- H4, secondaire : confiance plus élevée en C qu'en A. Modèle mixte linéaire
  sur la note de Likert, contraste C-A.
- Correction de Holm sur H2-H4.
- Règle préenregistrée : temps plafonné à 900 s (durée maximale annoncée
  par tâche ; au-delà, l'expérimentateur demande la meilleure hypothèse).
- Sensibilité : Friedman puis Wilcoxon appariés (Bonferroni) sur les
  moyennes par participant, comme dans la version initiale du protocole.
NASA-TLX et SUS sont collectés hors outil et analysés à part (descriptif).

Usage: python -m src.causal.study_analysis [--sessions CHEMIN]
"""

from pathlib import Path
import argparse
import itertools
import json
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SESSIONS_PATH = REPO_ROOT / "experiments" / "user_study" / "sessions.jsonl"
MAX_SECONDS = 900
MIN_COMPLETE_PARTICIPANTS = 5
CONDITION_CODES = {"A_manuel": "A", "B_attribution_statique": "B", "C_exploration_causale": "C"}
MEASURES = {"diagnosis_seconds": "Temps de diagnostic (s)", "correct": "Exactitude", "confidence_likert": "Confiance (1-5)"}


def load_sessions(path: Path) -> pd.DataFrame:
    df = pd.DataFrame([json.loads(line) for line in path.read_text().splitlines() if line.strip()])
    df["cond"] = df["condition"].map(CONDITION_CODES)
    df["correct"] = df["correct"].astype(float)
    df["diagnosis_seconds"] = df["diagnosis_seconds"].clip(upper=MAX_SECONDS)
    df["log_time"] = np.log(df["diagnosis_seconds"].clip(lower=1))
    # Position du bloc dans la séance (0, 1, 2) : covariable préenregistrée,
    # neutralise l'apprentissage/la fatigue (plan par blocs de 5 tâches).
    df["position"] = (df["task_index"] // 5).clip(upper=2)
    return df


def holm(p_values: dict) -> dict:
    """Correction de Holm-Bonferroni (pas à pas descendant)."""
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m, running, adjusted = len(ordered), 0.0, {}
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[name] = running
    return adjusted


def fmt_p(p: float) -> str:
    """« < 0,0001 » ou « = 0,0123 », à précéder directement de « p »."""
    return "< 0,0001" if p < 1e-4 else "= " + f"{p:.4f}".replace(".", ",")


def mixed_contrast(df: pd.DataFrame, outcome: str, reference: str, target: str) -> dict:
    import statsmodels.formula.api as smf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.mixedlm(f"{outcome} ~ C(cond, Treatment('{reference}')) + C(position)", df, groups=df["participant_id"]).fit(reml=True)
    term = f"C(cond, Treatment('{reference}'))[T.{target}]"
    low, high = fit.conf_int().loc[term]
    return {"estimate": float(fit.params[term]), "ci_low": float(low), "ci_high": float(high), "p": float(fit.pvalues[term])}


def gee_accuracy_contrast(df: pd.DataFrame, reference: str, target: str) -> dict:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.gee(f"correct ~ C(cond, Treatment('{reference}')) + C(position)", groups="participant_id", data=df,
                      family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit(
            # Correction de Mancl-DeRouen: l'erreur standard robuste du GEE
            # est sous-estimée avec peu de groupes (ici des dizaines de
            # participants au plus) — p trop optimistes sans elle.
            cov_type="bias_reduced")
    term = f"C(cond, Treatment('{reference}'))[T.{target}]"
    low, high = fit.conf_int().loc[term]
    return {"estimate": float(fit.params[term]), "ci_low": float(low), "ci_high": float(high), "p": float(fit.pvalues[term])}


def sensitivity_rank_tests(df: pd.DataFrame) -> None:
    per_participant = df.groupby(["participant_id", "cond"])[list(MEASURES)].mean().reset_index()
    conditions = ["A", "B", "C"]
    n_pairs = 3
    for measure, label in MEASURES.items():
        wide = per_participant.pivot(index="participant_id", columns="cond", values=measure)[conditions]
        chi2, p = stats.friedmanchisquare(*[wide[c] for c in conditions])
        print(f"  {label}: Friedman χ²={chi2:.2f}, p={p:.4f} (n={len(wide)})")
        for a, b in itertools.combinations(conditions, 2):
            try:
                _, p_pair = stats.wilcoxon(wide[a], wide[b])
            except ValueError:  # différences toutes nulles
                p_pair = 1.0
            print(f"     {a} contre {b} : Wilcoxon p={p_pair:.4f}, Bonferroni p={min(p_pair * n_pairs, 1.0):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse préenregistrée de l'étude utilisateur")
    parser.add_argument("--sessions", type=Path, default=SESSIONS_PATH, help="Fichier de sessions (par défaut : celui de l'étude)")
    args = parser.parse_args()

    if not args.sessions.exists():
        sys.exit(f"Aucune session enregistrée ({args.sessions}). Rien à analyser — aucun chiffre ne sera produit.")
    df = load_sessions(args.sessions)
    print(f"{len(df)} tâches, {df['participant_id'].nunique()} participant(s), "
          f"par condition : {df['cond'].value_counts().sort_index().to_dict()}")
    print(f"Temps plafonnés à {MAX_SECONDS} s : {int((df['diagnosis_seconds'] >= MAX_SECONDS).sum())} tâche(s)\n")

    print("Statistiques descriptives (moyenne, écart-type) :")
    print(df.groupby("cond")[list(MEASURES)].agg(["mean", "std"]).round(2).to_string(), "\n")

    complete = df.groupby("participant_id")["cond"].nunique()
    complete_ids = complete[complete == 3].index
    if len(complete_ids) < MIN_COMPLETE_PARTICIPANTS:
        print(f"Seulement {len(complete_ids)} participant(s) ont passé les 3 conditions (minimum "
              f"{MIN_COMPLETE_PARTICIPANTS}) : tests d'hypothèses non calculés.")
        return
    data = df[df["participant_id"].isin(complete_ids)]

    h1 = mixed_contrast(data, "log_time", "A", "C")
    print("H1 (principale) — temps C contre A, modèle mixte sur log(temps) :")
    print(f"  rapport des temps C/A = {np.exp(h1['estimate']):.2f} "
          f"[IC 95 % {np.exp(h1['ci_low']):.2f} ; {np.exp(h1['ci_high']):.2f}], p {fmt_p(h1['p'])}"
          f" -> {'H1 soutenue' if h1['p'] < 0.05 and h1['estimate'] < 0 else 'H1 non soutenue'} (alpha = 0,05)\n")

    secondary = {
        "H2 exactitude C contre B (GEE logistique, rapport de cotes)": gee_accuracy_contrast(data, "B", "C"),
        "H3 temps C contre B (modèle mixte, rapport des temps)": mixed_contrast(data, "log_time", "B", "C"),
        "H4 confiance C contre A (modèle mixte, points de Likert)": mixed_contrast(data, "confidence_likert", "A", "C"),
    }
    adjusted = holm({name: r["p"] for name, r in secondary.items()})
    print("Hypothèses secondaires (correction de Holm) :")
    for name, r in secondary.items():
        transform = np.exp if ("rapport" in name) else (lambda v: v)
        print(f"  {name} : {transform(r['estimate']):.2f} [IC 95 % {transform(r['ci_low']):.2f} ; "
              f"{transform(r['ci_high']):.2f}], p {fmt_p(r['p'])}, p Holm {fmt_p(adjusted[name])}")

    print("\nAnalyse de sensibilité (tests de rangs sur moyennes par participant) :")
    sensitivity_rank_tests(data)


if __name__ == "__main__":
    main()
