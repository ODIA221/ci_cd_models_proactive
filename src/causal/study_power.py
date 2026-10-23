"""
Analyse de puissance par simulation pour l'étude utilisateur (protocole :
docs/08-protocole-etude-utilisateur.md) — remplace l'approximation d'un test
apparié sur moyennes de participants, qui ignorait que chaque participant
passe 5 tâches par condition.

Modèle simulé (critère principal, temps de diagnostic en log) :

    y_pt = b0 + b_cond + u_p + w_t + e_pt
    u_p ~ N(0, rho_p)      effet participant (vitesse propre à chacun)
    w_t ~ N(0, rho_t)      effet tâche (difficulté propre à chaque exécution)
    e_pt ~ N(0, 1 - rho_p - rho_t)

La variance totale d'une observation vaut 1 : l'effet `d` (C contre A) est
donc exprimé en écarts-types d'UNE observation, convention du protocole.
Chaque participant reçoit 15 exécutions tirées sans remise parmi les 60 qui
ont des traces, en 3 blocs de 5 (un par condition), l'ordre des blocs suivant
un carré latin attribué à tour de rôle (même plan que GET /study/tasks).

Analyse simulée = analyse préenregistrée : modèle mixte, ordonnée aléatoire
participant (statsmodels MixedLM), position du bloc en effet fixe, contraste
C contre A, test bilatéral à alpha. L'effet tâche n'est PAS modélisé dans l'ajustement (il est traité
comme du bruit), ce qui rend l'estimation de puissance conservatrice.

Les variances (rho_p, rho_t) sont INCONNUES tant que l'étude pilote n'a pas
eu lieu : le script produit une grille de sensibilité, pas un chiffre unique.

Usage: python -m src.causal.study_power [--sims 300]
"""

from pathlib import Path
import argparse
import itertools
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
N_TASK_POOL = 60          # exécutions anormales de test avec traces (RE2)
TASKS_PER_CONDITION = 5
CONDITIONS = ["A", "B", "C"]


def simulate_once(rng: np.random.Generator, n_participants: int, d: float, rho_p: float, rho_t: float) -> pd.DataFrame:
    task_effect = rng.normal(0, np.sqrt(rho_t), N_TASK_POOL)
    effect = {"A": 0.0, "B": d / 2, "C": d}  # B à mi-chemin : n'affecte pas le contraste C-A
    rows = []
    for p in range(n_participants):
        u = rng.normal(0, np.sqrt(rho_p))
        tasks = rng.choice(N_TASK_POOL, size=TASKS_PER_CONDITION * len(CONDITIONS), replace=False)
        for i, t in enumerate(tasks):
            # Plan par blocs, ordre des blocs en carré latin à tour de rôle (comme /study/tasks)
            cond = CONDITIONS[(i // TASKS_PER_CONDITION + p) % len(CONDITIONS)]
            # Gain = temps plus court : signe négatif sur le log-temps.
            y = -effect[cond] + u + task_effect[t] + rng.normal(0, np.sqrt(1 - rho_p - rho_t))
            rows.append((p, t, cond, i // TASKS_PER_CONDITION, y))
    return pd.DataFrame(rows, columns=["participant", "task", "condition", "position", "log_time"])


def p_value_c_vs_a(df: pd.DataFrame) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.mixedlm("log_time ~ C(condition, Treatment('A')) + C(position)", df, groups=df["participant"]).fit(reml=True)
    return float(fit.pvalues["C(condition, Treatment('A'))[T.C]"])


def power(n_participants: int, d: float, rho_p: float, rho_t: float, alpha: float, sims: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    hits = sum(p_value_c_vs_a(simulate_once(rng, n_participants, d, rho_p, rho_t)) < alpha for _ in range(sims))
    return hits / sims


def main() -> None:
    parser = argparse.ArgumentParser(description="Puissance de l'étude utilisateur par simulation (modèle mixte)")
    parser.add_argument("--sims", type=int, default=300, help="Simulations par cellule de la grille")
    parser.add_argument("--alpha", type=float, default=0.05, help="Seuil du test principal (un seul contraste préenregistré)")
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()

    grid = {
        "n_participants": [6, 9, 12, 15, 18, 24],
        "d": [0.5],
        "rho_p": [0.2, 0.4],
        "rho_t": [0.1, 0.2],
    }
    rows = []
    for i, (n, d, rp, rt) in enumerate(itertools.product(*grid.values())):
        pw = power(n, d, rp, rt, args.alpha, args.sims, args.seed + i)
        rows.append({"n_participants": n, "d": d, "rho_participant": rp, "rho_tache": rt, "alpha": args.alpha,
                     "simulations": args.sims, "puissance": round(pw, 3)})
        print(f"N={n:2d} d={d} rho_p={rp} rho_t={rt} -> puissance={pw:.3f}", flush=True)

    # Sensibilité: effet plus faible (d = 0,3) dans le cas de variances le
    # plus défavorable de la grille — ce qu'un relecteur demandera.
    for n in [12, 18, 24, 30]:
        pw = power(n, 0.3, 0.2, 0.2, args.alpha, args.sims, args.seed + 900 + n)
        rows.append({"n_participants": n, "d": 0.3, "rho_participant": 0.2, "rho_tache": 0.2, "alpha": args.alpha,
                     "simulations": args.sims, "puissance": round(pw, 3)})
        print(f"[sensibilité] N={n:2d} d=0.3 rho_p=0.2 rho_t=0.2 -> puissance={pw:.3f}", flush=True)

    out = REPO_ROOT / "experiments" / "study_power_grid.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    main_rows = [r for r in rows if r["d"] == 0.5]
    table = pd.DataFrame(main_rows).pivot_table(index="n_participants", columns=["rho_participant", "rho_tache"], values="puissance")
    print("\nPuissance (contraste C contre A, d = 0,5, alpha = %.2f)\n" % args.alpha + table.to_string())
    print(f"\nGrille sauvegardée : {out}")


if __name__ == "__main__":
    main()
