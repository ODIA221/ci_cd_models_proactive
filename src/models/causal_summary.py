"""
Tous les chiffres de localisation de cause racine cités dans l'article et dans
docs/07, recalculés à partir des résultats bruts — pour qu'aucun chiffre du
texte ne dépende d'un calcul ponctuel non publié.

Entrées : le dernier experiments/causal_rankers_<horodatage>.csv et son
causal_false_leads_<horodatage>.csv (produits par
./run.sh evaluate-causal --compare-rankers), plus le cache des signaux
(data/interim/rcaeval/RE2/causal_signals/) pour les ex aequo.

Usage: python -m src.models.causal_summary [--stamp AAAAMMJJ_HHMMSS]
"""

from pathlib import Path
import argparse
import json

import pandas as pd
from scipy.stats import binomtest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXPERIMENTS = REPO_ROOT / "experiments"
SIGNALS_DIR = REPO_ROOT / "data" / "interim" / "rcaeval" / "RE2" / "causal_signals"


def mcnemar(wide: pd.DataFrame, a: str, b: str) -> str:
    d = wide[[a, b]].dropna().astype(bool)
    only_a, only_b = int((d[a] & ~d[b]).sum()), int((~d[a] & d[b]).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return f"{a} contre {b} : n = {len(d)}, {only_a} contre {only_b} gagnées seules, p = {p:.3f}"


def holm(pvalues: dict) -> dict:
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    adjusted, running = {}, 0.0
    for i, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - i) * p))
        adjusted[name] = running
    return adjusted


def summarize_baselines(r: pd.DataFrame, wide: pd.DataFrame, path: Path) -> None:
    """Méthodes de référence RCAEval (src/models/rcaeval_baselines.py) sur les
    mêmes cas. Échec ou dépassement de délai = pas de réponse = raté (une
    méthode qui ne rend rien ne diagnostique rien) ; P@k sur les seuls cas
    aboutis donné à côté."""
    b = pd.read_csv(path)
    print(f"\nMéthodes de référence RCAEval ({path.name}) :")
    for name, reason in b[b["statut"] == "non_executee"][["methode", "erreur"]].itertuples(index=False):
        print(f"  {name} : non exécutée ({reason})")
    b = b[b["run_id"].notna()]
    cases = set(r["run_id"])
    rows, pvals = [], {}
    for method, g in b.groupby("methode"):
        g = g[g["run_id"].isin(cases)].set_index("run_id")
        ok = g[g["statut"] == "ok"]
        hit1 = (g["statut"].eq("ok") & g["hit_at_1"].fillna(False).astype(bool)).reindex(wide.index, fill_value=False)
        hit3 = (g["statut"].eq("ok") & g["hit_at_3"].fillna(False).astype(bool)).reindex(wide.index, fill_value=False)
        applicable = g[g["statut"] != "non_applicable"].index
        d = pd.DataFrame({"score_multimodal": wide["score_multimodal"].astype(bool), method: hit1}).loc[applicable]
        only_s, only_b = int((d["score_multimodal"] & ~d[method]).sum()), int((~d["score_multimodal"] & d[method]).sum())
        pvals[method] = binomtest(only_s, only_s + only_b, 0.5).pvalue if only_s + only_b else 1.0
        rows.append({"methode": method, "famille": g["famille"].iloc[0], "n_applicables": len(applicable),
                     "aboutis": len(ok), "echecs": int(g["statut"].isin(["echec", "timeout"]).sum()),
                     "P@1": round(hit1.loc[applicable].mean(), 2), "P@3": round(hit3.loc[applicable].mean(), 2),
                     "P@1_aboutis": round(ok["hit_at_1"].astype(bool).mean(), 2) if len(ok) else None,
                     "score_seul": only_s, "methode_seule": only_b, "p_McNemar": round(pvals[method], 3)})
    adjusted = holm(pvals)
    table = pd.DataFrame(rows)
    table["p_Holm"] = table["methode"].map(lambda m: round(adjusted[m], 3))
    print("  (P@1/P@3 : échecs comptés comme ratés ; McNemar = score_multimodal contre la méthode, cas applicables)")
    print("  " + table.sort_values("P@1", ascending=False).to_string(index=False).replace("\n", "\n  "))


def main() -> None:
    parser = argparse.ArgumentParser(description="Chiffres de localisation cités dans l'article")
    parser.add_argument("--stamp", default=None, help="Horodatage des fichiers (par défaut : le plus récent)")
    args = parser.parse_args()

    stamp = args.stamp or sorted(p.stem.split("causal_rankers_")[1] for p in EXPERIMENTS.glob("causal_rankers_2*.csv"))[-1]
    r = pd.read_csv(EXPERIMENTS / f"causal_rankers_{stamp}.csv")
    print(f"Fichiers : causal_rankers_{stamp}.csv\n")

    print("Table 3/4 — P@1 et P@3 :")
    for subset, data in (("86 exécutions", r), ("60 avec spans", r[r["a_attention"]])):
        g = data.groupby("ranker").agg(n=("run_id", "nunique"), p1=("hit_at_1", "mean"), p3=("hit_at_3", "mean")).round(2)
        print(f"  [{subset}]\n" + g.to_string().replace("\n", "\n  "))

    wide = r.pivot_table(index="run_id", columns="ranker", values="hit_at_1", aggfunc="first")
    print("\nTests de McNemar (P@1) :")
    for a, b in (("arbre_spans_v1", "score_multimodal"), ("arbre_spans_v1", "precedence_temporelle"),
                 ("score_multimodal", "precedence_temporelle"), ("propagation_attention", "propagation_uniforme"),
                 ("propagation_attention", "propagation_degre")):
        print("  " + mcnemar(wide, a, b))

    print("\nP@1 par système :")
    print(r[r["ranker"].isin(["score_multimodal", "precedence_temporelle", "arbre_spans_v1"])]
          .pivot_table(index="systeme", columns="ranker", values="hit_at_1", aggfunc="mean").round(2).to_string())

    print("\nTable 5 — P@1 par type de panne :")
    by_fault = r.pivot_table(index="type_faute", columns="ranker", values="hit_at_1", aggfunc="mean").round(2)
    by_fault.insert(0, "n", r[r["ranker"] == "score_multimodal"].groupby("type_faute").size())
    print(by_fault.to_string())

    u = r.drop_duplicates("run_id")["attention_entropie_normalisee"].dropna()
    print(f"\nUniformité de l'attention : médiane U = {u.median():.3f}, minimum {u.min():.3f} (n = {len(u)})")

    # Ex aequo au sommet de a_i, et P@1 SANS départage (ordre des services tel
    # que stocké avant tri, c'est-à-dire alphabétique : valeur de comparaison).
    abnormal = r.drop_duplicates("run_id").set_index("run_id")["root_cause_service"]
    ties, hits_alpha = 0, 0
    for run_id, truth in abnormal.items():
        sig = json.loads((SIGNALS_DIR / f"{run_id.replace('/', '__')}.json").read_text())
        nodes = sig["nodes"]
        top = max(n["anomaly_score"] for n in nodes)
        ties += sum(n["anomaly_score"] == top for n in nodes) > 1
        alpha_rank = sorted(nodes, key=lambda n: (-n["anomaly_score"], n["service"]))
        hits_alpha += alpha_rank[0]["service"] == truth
    print(f"Ex aequo au sommet de a_i : {ties}/{len(abnormal)} exécutions ; "
          f"P@1 sans départage (ordre alphabétique) = {hits_alpha / len(abnormal):.2f}")

    baseline_files = sorted(EXPERIMENTS.glob("causal_baselines_2*.csv"))
    if baseline_files:
        summarize_baselines(r, wide, baseline_files[-1])

    fl = pd.read_csv(EXPERIMENTS / f"causal_false_leads_{stamp}.csv")
    for anormal, label in ((False, "fenêtres normales"), (True, "fenêtres anormales")):
        g = fl[fl["anormal"] == anormal]
        q1, q3 = g["n_services_a_ge_0_5"].quantile([0.25, 0.75])
        print(f"Fausses pistes, {label} : services avec a_i >= 0,5 médiane {g['n_services_a_ge_0_5'].median():g} "
              f"(IQR {q1:g} à {q3:g}, max {g['n_services_a_ge_0_5'].max()}) ; score max médian {g['max_a'].median():.2f}, "
              f"minimum {g['max_a'].min():.2f}")


if __name__ == "__main__":
    main()
