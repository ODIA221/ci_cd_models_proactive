"""
Puissance du test de McNemar exact pour les hypothèses confirmatoires de
l'article 5 (tableau 5) : simulation à graine fixée, au seuil le plus strict
de la procédure de Holm (alpha = 0,05/3). Une condition = taille d'échantillon,
taux de paires discordantes, part de ces paires favorable à la méthode testée.

Usage: python -m src.bench.power [--sims 2000] [--seed 20261031]
Sortie: experiments/pipefault_power_grid.csv
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy.stats import binomtest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SIZES = [90, 120, 150, 180, 240]
SCENARIOS = [(0.25, 0.75), (0.30, 0.75), (0.30, 0.70)]
ALPHA = 0.05 / 3


def power(rng: np.random.Generator, n: int, discordant: float, favourable: float, sims: int) -> float:
    hits = 0
    for _ in range(sims):
        d = rng.binomial(n, discordant)
        if d and binomtest(rng.binomial(d, favourable), d, 0.5).pvalue < ALPHA:
            hits += 1
    return hits / sims


def main() -> None:
    parser = argparse.ArgumentParser(description="Puissance du test de McNemar (article 5, tableau 5)")
    parser.add_argument("--sims", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = [{"executions": n, "discordantes": d, "favorables": f, "alpha": ALPHA, "simulations": args.sims,
             "puissance": power(rng, n, d, f, args.sims)} for n in SIZES for d, f in SCENARIOS]
    out = REPO_ROOT / "experiments" / "pipefault_power_grid.csv"
    table = pd.DataFrame(rows)
    table.to_csv(out, index=False)
    print(table.pivot_table(index="executions", columns=["discordantes", "favorables"], values="puissance").round(2).to_string())
    print(f"\nGrille sauvegardée : {out}")


if __name__ == "__main__":
    main()
