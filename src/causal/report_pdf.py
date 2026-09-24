"""
Export PDF du rapport de diagnostic (étape 4 du flux d'usage, section 4.3 de
l'article) — même contenu que signals.markdown_report, plus deux graphiques
(score a_i par service, chronologie des premières déviations) pour un
rapport post-incident lisible sans l'interface.

fpdf2 (pur Python) plutôt que WeasyPrint/wkhtmltopdf: aucune dépendance
système à installer. Police Unicode cherchée sur la machine (α, β, ≥, «»,
tirets); à défaut, repli sur Helvetica avec translittération des caractères
hors latin-1 — le PDF est toujours produit, jamais une erreur 500 pour une
question de police.
"""

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # serveur sans affichage
import matplotlib.pyplot as plt  # noqa: E402
from fpdf import FPDF  # noqa: E402

from src.causal import signals  # noqa: E402

_UNICODE_FONTS = [
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
]
_LATIN1_FALLBACK = {"α": "alpha", "β": "beta", "≥": ">=", "≤": "<=", "—": "-", "–": "-", "→": "->",
                    "’": "'", "“": '"', "”": '"', "…": "...", "ᵢ": "i", "×": "x"}

# Même rampe que les vues (gris -> rouge critique) et même bleu de sélection.
_RAMP = ["#f0efec", "#ec835a", "#d03b3b"]
_TEXT_2 = "#52514e"
_GRID = "#e1e0d9"
_HIGHLIGHT = "#2a78d6"


class _ReportPDF(FPDF):
    def __init__(self, run_id: str):
        super().__init__(format="A4")
        self.run_id = run_id
        self.unicode_font = next((p for p in _UNICODE_FONTS if Path(p).exists()), None)
        if self.unicode_font:
            self.add_font("Body", "", self.unicode_font)
            self.add_font("Body", "B", self.unicode_font)  # pas de variante grasse fournie: même fichier
            self.family = "Body"
        else:
            self.family = "Helvetica"
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(15, 15, 15)

    def keep_together(self, height_mm: float):
        """Saut de page si le bloc (titre + figure) ne tient pas en entier:
        évite un titre orphelin en bas de page, figure sur la suivante."""
        if self.get_y() + height_mm > self.h - self.b_margin:
            self.add_page()

    def txt(self, text) -> str:
        text = "" if text is None else str(text).replace("\t", " ")
        if self.unicode_font:
            return text
        for k, v in _LATIN1_FALLBACK.items():
            text = text.replace(k, v)
        return text.encode("latin-1", "replace").decode("latin-1")

    def footer(self):
        self.set_y(-12)
        self.set_font(self.family, "", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, self.txt(f"LogPipeGuard v2 — {self.run_id} — page {self.page_no()}/{{nb}}"), align="C")

    def h1(self, text: str):
        self.set_font(self.family, "B", 15)
        self.set_text_color(11, 11, 11)
        self.multi_cell(0, 7, self.txt(text))
        self.ln(1)

    def h2(self, text: str):
        self.ln(3)
        self.set_font(self.family, "B", 12)
        self.set_text_color(11, 11, 11)
        self.multi_cell(0, 6, self.txt(text))
        self.ln(1)

    def para(self, text: str, size: float = 9.5, color=(40, 40, 40)):
        self.set_font(self.family, "", size)
        self.set_text_color(*color)
        self.multi_cell(0, 4.8, self.txt(text))
        self.ln(0.5)


def _figure_png(fig) -> BytesIO:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_TEXT_2, labelsize=8)
    ax.grid(axis="x", color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _scores_chart(nodes: list, highlight: Optional[str]) -> BytesIO:
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("score", _RAMP)
    top = nodes[:12][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(top) + 0.8))
    ax.barh([n["display_name"] for n in top], [n["anomaly_score"] for n in top],
            color=[cmap(n["anomaly_score"]) for n in top],
            edgecolor=[_HIGHLIGHT if n["service"] == highlight else "none" for n in top], linewidth=1.5, height=0.7)
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color=_TEXT_2, linestyle=":", linewidth=0.8)
    ax.set_xlabel("score d'anomalie a_i (pointillé : seuil « notable » 0,5)", color=_TEXT_2, fontsize=8)
    _style(ax)
    return _figure_png(fig)


def _timeline_chart(nodes: list, probe_seconds: float, highlight: Optional[str]) -> Optional[BytesIO]:
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("score", _RAMP)
    rows = [n for n in nodes if n["onset_s"] is not None][:12]
    if not rows:
        return None
    rows = sorted(rows, key=lambda n: n["onset_s"])[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(rows) + 0.8))
    ax.scatter([n["onset_s"] for n in rows], range(len(rows)), s=90, marker="s",
               c=[cmap(n["anomaly_score"]) for n in rows],
               edgecolors=[_HIGHLIGHT if n["service"] == highlight else _TEXT_2 for n in rows], linewidths=1)
    ax.set_yticks(range(len(rows)), [n["display_name"] for n in rows])
    ax.set_xlim(-probe_seconds * 0.02, probe_seconds * 1.02)
    ax.set_xlabel("secondes après le début de la fenêtre analysée", color=_TEXT_2, fontsize=8)
    _style(ax)
    return _figure_png(fig)


def _table(pdf: _ReportPDF, header: list, rows: list, widths: list):
    pdf.set_font(pdf.family, "", 8)
    pdf.set_draw_color(225, 224, 217)
    with pdf.table(col_widths=widths, text_align="LEFT", line_height=4.2,
                   first_row_as_headings=True) as table:
        head = table.row()
        for h in header:
            head.cell(pdf.txt(h))
        for r in rows:
            row = table.row()
            for value in r:
                row.cell(pdf.txt(value))


def pdf_report(sig: dict, hypothesis: Optional[str] = None, top_n: int = 8) -> bytes:
    pdf = _ReportPDF(sig["run_id"])
    pdf.alias_nb_pages()
    pdf.add_page()
    hyp = signals.canonical_service(hypothesis) if hypothesis else None

    pdf.h1("Rapport de diagnostic")
    window_label = {"post_injection": "après l'injection de la faute",
                    "normal_second_half": "seconde moitié d'une fenêtre normale"}.get(sig["window"]["kind"], sig["window"]["kind"])
    available = ", ".join(m for m, ok in sig["modalities_available"].items() if ok) or "aucune"
    missing = ", ".join(m for m, ok in sig["modalities_available"].items() if not ok)
    pdf.para(f"Exécution : {sig['run_id']}\n"
             f"Fenêtre analysée : {window_label} ({sig['window']['probe_seconds']:.0f} s, référence {sig['window']['base_seconds']:.0f} s)\n"
             f"Modalités disponibles : {available}" + (f" — absentes : {missing} (rien n'est extrapolé)" if missing else "") + "\n"
             f"Généré le {datetime.now():%Y-%m-%d %H:%M}", size=9)

    nodes = sorted(sig["nodes"], key=signals.node_sort_key, reverse=True)
    pdf.h2("Services les plus anormaux")
    rows = []
    for rank, n in enumerate(nodes[:top_n], start=1):
        hints = []
        if n["metrics"]:
            hints.append(f"métrique {n['metrics']['top_metric']}")
        if n["traces"]:
            hints.append(f"traces ({n['traces']['reason']})")
        if n["logs"] and n["logs"]["top_templates"]:
            hints.append(f"log « {n['logs']['top_templates'][0]['template'][:45]} »")
        rows.append([str(rank), n["display_name"], f"{n['anomaly_score']:.2f}", str(n["n_modalities_deviating"]),
                     "—" if n["onset_s"] is None else f"{n['onset_s']:.0f}", " ; ".join(hints)])
    _table(pdf, ["Rang", "Service", "a_i", "Modalités", "Début (s)", "Indice principal"], rows, [10, 34, 11, 17, 16, 92])

    pdf.ln(3)
    pdf.image(_scores_chart(nodes, hyp), w=170)
    timeline = _timeline_chart(nodes, sig["window"]["probe_seconds"], hyp)
    if timeline is not None:
        n_rows = min(12, sum(n["onset_s"] is not None for n in nodes))
        pdf.keep_together(15 + 170 * (0.32 * n_rows + 0.8) / 7)  # hauteur de la figure à 170 mm de large
        pdf.h2("Chronologie des premières déviations")
        pdf.image(timeline, w=170)

    if hyp:
        wi = signals.what_if(sig, hyp)
        pdf.h2(f"Hypothèse retenue : {hyp}")
        coverage = "n/a" if wi["coverage"] is None else f"{wi['coverage']:.0%}"
        pdf.para(f"Services anormaux expliqués par propagation : {', '.join(wi['explained_anomalous']) or '—'}\n"
                 f"Services anormaux NON expliqués : {', '.join(wi['unexplained_anomalous']) or '—'}\n"
                 f"Couverture : {coverage}\n{wi['caveat']}")
        paths = signals.causal_path(sig, hyp)
        if paths:
            pdf.para("Chemins de propagation (appelé → appelants) : " + " | ".join(" → ".join(p) for p in paths[:5]))
        node = next((n for n in nodes if n["service"] == hyp), None)
        if node and node["logs"] and node["logs"]["top_templates"]:
            pdf.h2("Modèles de journaux les plus modifiés")
            _table(pdf, ["Modèle", "β", "avant", "après"],
                   [[t["template"][:90], f"{t['beta']:.2f}", str(t["n_base"]), str(t["n_probe"])]
                    for t in node["logs"]["top_templates"]], [130, 15, 17, 18])

    pdf.h2("Provenance des signaux")
    for key, value in sig["provenance"].items():
        pdf.para(f"{key} : {value}", size=8.5, color=(82, 81, 78))
    pdf.ln(2)
    pdf.para("Les scores et poids d'attention ci-dessus sont des indices associatifs, pas une preuve causale. "
             "Mesuré sur RCAEval RE2 : l'attention GAT est uniforme et n'apporte rien au-delà du degré des nœuds ; "
             "sur des fenêtres SANS faute, 2 services en médiane dépassent quand même a_i ≥ 0,5.",
             size=8.5, color=(82, 81, 78))

    return bytes(pdf.output())
