"""
Sections Streamlit "Exploration causale (v2)" et "Mode étude utilisateur".
Client HTTP pur de l'API (GET /causal/..., /study/...), comme app.py.

Correspondance avec les principes interactifs de l'article (section 3.3) :
- liaison/surbrillance entre vues : le service sélectionné est surligné dans
  la chronologie ET le graphe de propagation ;
- affinement itératif : curseur de seuil sur a_i, appliqué aux trois vues ;
- traçage du chemin causal : chemins de propagation depuis le service
  sélectionné, surlignés dans le graphe (sélection par liste, pas par clic
  direct sur le graphe: Streamlit ne remonte pas les clics Plotly sans
  composant tiers) ;
- "et si" : test STRUCTUREL d'hypothèse de cause racine (couverture des
  services anormaux par propagation), pas un contrefactuel.
"""

import hashlib
import time
from urllib.parse import quote

import pandas as pd
import requests
import streamlit as st

from src.dashboard import causal_views

CONDITIONS = {
    "A_manuel": "Condition A — corrélation manuelle",
    "B_attribution_statique": "Condition B — attribution statique (tableaux)",
    "C_exploration_causale": "Condition C — exploration causale interactive",
}


def _count_interaction() -> None:
    st.session_state["n_interactions"] = st.session_state.get("n_interactions", 0) + 1


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_signals(base_url: str, run_id: str) -> dict:
    response = requests.get(f"{base_url}/causal/{run_id}", timeout=600)
    response.raise_for_status()
    return response.json()


def render_provenance(signals: dict) -> None:
    available = [m for m, ok in signals["modalities_available"].items() if ok]
    missing = [m for m, ok in signals["modalities_available"].items() if not ok]
    st.caption(
        f"Modalités disponibles : {', '.join(available) or 'aucune'}"
        + (f" — **absentes : {', '.join(missing)}** (vues correspondantes vides, rien n'est extrapolé)" if missing else "")
    )
    with st.expander("D'où viennent ces signaux ? (à lire avant d'interpréter)"):
        for key, value in signals["provenance"].items():
            st.markdown(f"- **{key}** : {value}")
        st.markdown("Les nœuds sont des **services** RCAEval (pas des étapes de pipeline CI/CD). "
                    "Les poids d'attention sont **associatifs**, pas une preuve causale.")


def render_interactive_views(base_url: str, signals: dict, key_prefix: str, allow_report: bool = True) -> None:
    run_id = signals["run_id"]
    services = [n["service"] for n in signals["nodes"]]
    labels = {n["service"]: n["display_name"] for n in signals["nodes"]}

    c1, c2 = st.columns([1, 2])
    threshold = c1.slider("Seuil a_i (filtre les trois vues)", 0.0, 1.0, 0.3, 0.05,
                          key=f"{key_prefix}_threshold", on_change=_count_interaction)
    selected = c2.selectbox("Service à examiner (surligné dans toutes les vues)", ["—"] + services,
                            format_func=lambda s: labels.get(s, s), key=f"{key_prefix}_selected",
                            on_change=_count_interaction)
    selected = None if selected == "—" else selected

    path, what_if = None, None
    if selected:
        response = requests.get(f"{base_url}/causal/{run_id}/whatif", params={"hypothesis": selected, "min_score": threshold}, timeout=60)
        if response.status_code == 200:
            what_if = response.json()
            if what_if["paths"]:
                path_labels = [" → ".join(p) for p in what_if["paths"]]
                chosen = st.selectbox("Chemin de propagation depuis ce service (appelé → appelants)", path_labels,
                                      key=f"{key_prefix}_path", on_change=_count_interaction)
                path = what_if["paths"][path_labels.index(chosen)]

    highlight = {selected} if selected else set()
    st.plotly_chart(causal_views.timeline_figure(signals, threshold, highlight), use_container_width=True, key=f"{key_prefix}_tl")

    g1, g2 = st.columns([3, 2])
    with g1:
        fig = causal_views.propagation_figure(signals, threshold, highlight, path)
        if fig is None:
            st.info("Pas de traces pour ce run (ex. Sock Shop dans RCAEval) : graphe de propagation indisponible.")
        else:
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_graph")
    with g2:
        st.plotly_chart(causal_views.modal_figure(signals, threshold), use_container_width=True, key=f"{key_prefix}_modal")

    if what_if:
        st.subheader(f"Et si la cause racine était « {labels.get(selected, selected)} » ?")
        m1, m2, m3 = st.columns(3)
        m1.metric("a_i de l'hypothèse", f"{what_if['hypothesis_anomaly_score']:.2f}" if what_if["hypothesis_anomaly_score"] is not None else "—")
        m2.metric("Services anormaux expliqués", len(what_if["explained_anomalous"]))
        m3.metric("Couverture", f"{what_if['coverage']:.0%}" if what_if["coverage"] is not None else "—")
        if what_if["unexplained_anomalous"]:
            st.warning("Non expliqués par cette hypothèse : " + ", ".join(labels.get(s, s) for s in what_if["unexplained_anomalous"]))
        st.caption(what_if["caveat"])

        node = next(n for n in signals["nodes"] if n["service"] == selected)
        if node["logs"] and node["logs"]["top_templates"]:
            st.markdown("**Templates de logs les plus modifiés** (β = log2 du rapport des taux après/avant)")
            st.dataframe(pd.DataFrame(node["logs"]["top_templates"]), hide_index=True, use_container_width=True)

    if allow_report:
        report = requests.get(f"{base_url}/causal/{run_id}/report", params={"hypothesis": selected} if selected else {}, timeout=60)
        if report.status_code == 200:
            st.download_button("Exporter le rapport de diagnostic (Markdown)", report.text,
                               file_name=f"diagnostic_{run_id.replace('/', '_')}.md", mime="text/markdown")


def render_exploration_section(base_url: str, run_id_options: list) -> None:
    st.header("Exploration causale (v2)")
    if not run_id_options:
        st.info("Lance une détection sur la source RCAEval pour obtenir des runs à explorer.")
        return
    run_id = st.selectbox("Run à explorer", run_id_options, key="explore_run")
    with st.spinner("Calcul des signaux (premier appel sur ce run : jusqu'à ~1-2 min, puis cache)..."):
        try:
            signals = fetch_signals(base_url, run_id)
        except requests.exceptions.RequestException as e:
            st.error(f"Signaux causaux indisponibles : {e}")
            return
    render_provenance(signals)
    # Flux section 4.3: alerte (ici) -> interface v2 avec le run pré-chargé.
    st.link_button("Ouvrir ce run dans l'interface v2 (React + D3)",
                   f"{base_url}/ui/?run={quote(run_id)}",
                   help="Nécessite le build de frontend/ (./run.sh ui-build)")
    render_interactive_views(base_url, signals, key_prefix="explore")


def render_study_mode(base_url: str) -> None:
    """
    Instrumentation d'une étude intra-sujets (section 5.3). Enregistre temps,
    réponse, confiance et nombre d'interactions via POST /study/sessions — la
    correction contre la vérité terrain est faite côté serveur, jamais
    affichée au participant. Ne remplace NI le recrutement NI le
    consentement éclairé NI l'avis d'un comité d'éthique.
    """
    st.header("Mode étude utilisateur")
    st.caption("Enregistre des sessions réelles dans experiments/user_study/sessions.jsonl. "
               "À n'utiliser qu'avec des participants ayant donné leur consentement éclairé.")
    participant = st.text_input("Identifiant participant (pseudonyme, ex. P01)", key="study_participant").strip()
    if not participant:
        return

    try:
        tasks = requests.get(f"{base_url}/study/tasks", params={"participant_id": participant}, timeout=30).json()
    except requests.exceptions.RequestException as e:
        st.error(f"Impossible de charger les tâches : {e}")
        return
    done = st.session_state.setdefault(f"study_done_{participant}", 0)
    if done >= len(tasks):
        st.success("Toutes les tâches sont terminées. Merci ! Faire remplir les questionnaires NASA-TLX et SUS (hors outil).")
        return

    task = tasks[done]
    st.subheader(f"Tâche {done + 1}/{len(tasks)} — {CONDITIONS[task['condition']]}")
    timer_key = f"study_start_{participant}_{done}"
    if timer_key not in st.session_state:
        if st.button("Démarrer le chronomètre et afficher l'alerte", type="primary"):
            st.session_state[timer_key] = time.time()
            st.session_state["n_interactions"] = 0
            st.rerun()
        return

    # Le run_id RCAEval contient le service fautif ("<service>_<faute>/..."):
    # l'afficher donnerait la réponse. Alias opaque seulement — le run_id reste
    # côté serveur Streamlit pour les appels API. Pour la condition A (outils
    # externes), l'expérimentateur doit préparer des copies anonymisées des
    # données (les noms de dossiers RCAEval trahissent aussi la réponse).
    alias = hashlib.sha1(task["run_id"].encode()).hexdigest()[:8]
    st.error(f"Alerte : anomalie détectée sur l'exécution `alerte-{alias}`. Identifiez le service cause racine.")
    with st.spinner("Chargement..."):
        signals = fetch_signals(base_url, task["run_id"])

    if task["condition"] == "A_manuel":
        st.info("Condition A : utilisez vos outils habituels (Grafana/Loki/Jaeger ou fichiers bruts du cas). "
                "Ce tableau de bord ne sert ici qu'à chronométrer et enregistrer votre réponse.")
    elif task["condition"] == "B_attribution_statique":
        st.dataframe(causal_views.static_attribution_table(signals), hide_index=True, use_container_width=True)
        st.dataframe(causal_views.attention_table(signals), hide_index=True, use_container_width=True)
    else:
        render_interactive_views(base_url, signals, key_prefix=f"study_{done}", allow_report=False)

    with st.form(f"study_answer_{done}"):
        services = sorted(n["service"] for n in signals["nodes"])
        answer = st.selectbox("Service cause racine", services)
        confidence = st.slider("Confiance dans ce diagnostic (1 = aucune, 5 = totale)", 1, 5, 3)
        comment = st.text_area("Commentaire (facultatif)")
        if st.form_submit_button("Valider le diagnostic"):
            elapsed = time.time() - st.session_state[timer_key]
            payload = {
                "participant_id": participant, "task_index": task["task_index"], "run_id": task["run_id"],
                "condition": task["condition"], "answer_service": answer, "diagnosis_seconds": elapsed,
                "confidence_likert": confidence, "n_interactions": st.session_state.get("n_interactions", 0),
                "comment": comment or None,
            }
            response = requests.post(f"{base_url}/study/sessions", json=payload, timeout=30)
            if response.status_code == 200:
                st.session_state[f"study_done_{participant}"] = done + 1
                st.rerun()
            else:
                st.error(f"Échec de l'enregistrement : {response.text}")
