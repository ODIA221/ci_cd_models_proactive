"""
Dashboard Streamlit LogPipeGuard — client HTTP pur de l'API FastAPI (src/api/)
N'importe jamais directement AnomalyDetector/CICDPreprocessor: passe toujours
par l'API, pour rester le premier vrai consommateur de la brique commune.
"""

from pathlib import Path
import json

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_CSV = REPO_ROOT / "data" / "raw" / "metrics" / "dataset_metrics.csv"

# Palette validée (skill dataviz): bleu = score normal, rouge critique = anomalie
COLOR_SCORE = "#2a78d6"
COLOR_ANOMALY = "#d03b3b"
COLOR_MUTED_GRID = "#e1e0d9"
COLOR_MUTED_AXIS = "#898781"

st.set_page_config(page_title="LogPipeGuard", layout="wide")


def api_get(base_url: str, path: str):
    response = requests.get(f"{base_url}{path}", timeout=10)
    response.raise_for_status()
    return response.json()


def api_post(base_url: str, path: str, json_body: dict):
    return requests.post(f"{base_url}{path}", json=json_body, timeout=60)


# --- Sidebar ---
st.sidebar.title("LogPipeGuard")
base_url = st.sidebar.text_input("URL de l'API", value="http://localhost:8000").rstrip("/")

try:
    health = api_get(base_url, "/health")
    st.sidebar.success(f"API accessible ({health['status']})")
    api_ok = True
except requests.exceptions.RequestException:
    st.sidebar.error("API inaccessible. Lance-la dans un autre terminal: `./run.sh serve`")
    api_ok = False

selected_model_id = None
if api_ok:
    try:
        models_data = api_get(base_url, "/models")["models"]
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Erreur en listant les modèles: {e}")
        models_data = []

    if models_data:
        options = {f"{m['model_id']} ({m['created_at'][:19]})": m for m in models_data}
        selected_label = st.sidebar.selectbox("Modèle", list(options.keys()))
        selected_model = options[selected_label]
        selected_model_id = selected_model["model_id"]
    else:
        st.sidebar.warning("Aucun modèle entraîné. Lance: `./run.sh demo`")

st.title("LogPipeGuard — détection d'anomalies CI/CD")

if not api_ok:
    st.stop()

# --- Vue d'ensemble ---
st.header("Vue d'ensemble")
col1, col2 = st.columns([1, 2])

with col1:
    if selected_model_id:
        st.subheader("Modèle sélectionné")
        st.metric("Type", selected_model["model_type"])
        st.metric("Nombre de features", len(selected_model["feature_columns"]))
        st.caption(f"Entraîné le {selected_model['created_at'][:19]}")

with col2:
    st.subheader("Sources de données")
    try:
        sources_data = api_get(base_url, "/sources")["sources"]
        st.dataframe(pd.DataFrame(sources_data), use_container_width=True, hide_index=True)
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur en listant les sources: {e}")

if not selected_model_id:
    st.stop()

# --- Détection ---
st.header("Détection")

data_source = st.radio(
    "Données à analyser",
    ["Données d'exemple", "Uploader un CSV", "RCAEval (test, avec chaîne causale)"],
    horizontal=True,
)

df = None
explainable = False  # True seulement pour la source RCAEval (run_id + spans disponibles)
if data_source == "Données d'exemple":
    if SAMPLE_CSV.exists():
        df = pd.read_csv(SAMPLE_CSV)
    else:
        st.error(f"Fichier d'exemple introuvable: {SAMPLE_CSV}")
elif data_source == "Uploader un CSV":
    uploaded = st.file_uploader("Fichier CSV", type="csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
else:
    rcaeval_dir = REPO_ROOT / "data" / "interim" / "rcaeval" / "RE2"
    features_path, labels_path = rcaeval_dir / "features.parquet", rcaeval_dir / "labels.parquet"
    if features_path.exists() and labels_path.exists():
        features_df = pd.read_parquet(features_path)
        labels_df = pd.read_parquet(labels_path).set_index("run_id")
        test_mask = labels_df.reindex(features_df.index)["fault_type"].isin(["test_normal", "test_abnormal"])
        # run_id (index de features.parquet, cf. rcaeval.py) redevient une
        # colonne: c'est ce que l'API extrait pour le faire transiter jusque
        # dans la réponse /predict (cf. main.py) et pouvoir ensuite appeler
        # /explain/{run_id}.
        df = features_df[test_mask].reset_index()
        explainable = True
    else:
        st.error(
            f"Données RCAEval introuvables sous {rcaeval_dir}. "
            "Lance d'abord: python -m src.data.acquire --source rcaeval --subset RE2"
        )

if df is not None:
    max_rows = min(len(df), 5000)
    default_rows = min(500, max_rows)
    n_rows = st.slider("Nombre de lignes à envoyer à l'API", min_value=1, max_value=max_rows, value=default_rows)
    df_subset = df.head(n_rows)

    if st.button("Lancer la détection", type="primary"):
        # to_dict() garde les NaN bruts (non valides en JSON strict) ; to_json()
        # les convertit proprement en null, que l'API sait ensuite réinterpréter
        # comme valeur manquante (imputée par le préprocesseur).
        records = json.loads(df_subset.to_json(orient="records"))
        response = api_post(base_url, "/predict", {"records": records, "model_id": selected_model_id})

        if response.status_code != 200:
            st.error(f"Échec de la prédiction (HTTP {response.status_code}): {response.json().get('detail')}")
        else:
            predictions = response.json()["predictions"]
            results_df = df_subset.reset_index(drop=True).copy()
            results_df["anomaly_score"] = [p["anomaly_score"] for p in predictions]
            results_df["anomalie"] = [p["anomalie"] for p in predictions]
            results_df["run_id"] = [p.get("run_id") for p in predictions]
            st.session_state["results_df"] = results_df
            st.session_state["explainable"] = explainable

# --- Résultats ---
if "results_df" in st.session_state:
    results_df = st.session_state["results_df"]
    st.header("Résultats")

    n_anomalies = int(results_df["anomalie"].sum())
    n_total = len(results_df)
    m1, m2, m3 = st.columns(3)
    m1.metric("Points analysés", n_total)
    m2.metric("Anomalies détectées", n_anomalies)
    m3.metric("Taux d'anomalies", f"{n_anomalies / n_total:.1%}" if n_total else "0%")

    x = results_df["timestamp"] if "timestamp" in results_df.columns else results_df.index
    normal_mask = ~results_df["anomalie"]
    anomaly_mask = results_df["anomalie"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x[normal_mask],
            y=results_df["anomaly_score"][normal_mask],
            mode="markers",
            name="Normal",
            marker=dict(color=COLOR_SCORE, size=6),
            hovertemplate="%{x}<br>score=%{y:.3f}<extra>Normal</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x[anomaly_mask],
            y=results_df["anomaly_score"][anomaly_mask],
            mode="markers",
            name="Anomalie",
            marker=dict(color=COLOR_ANOMALY, size=9, symbol="diamond"),
            hovertemplate="%{x}<br>score=%{y:.3f}<extra>Anomalie</extra>",
        )
    )
    fig.update_layout(
        title="Score d'anomalie dans le temps",
        xaxis_title="Timestamp" if "timestamp" in results_df.columns else "Index",
        yaxis_title="Score d'anomalie (plus élevé = plus anormal)",
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        xaxis=dict(gridcolor=COLOR_MUTED_GRID, linecolor=COLOR_MUTED_AXIS),
        yaxis=dict(gridcolor=COLOR_MUTED_GRID, linecolor=COLOR_MUTED_AXIS),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Anomalies les plus sévères")
    top_anomalies = results_df[results_df["anomalie"]].sort_values("anomaly_score", ascending=False)
    st.dataframe(top_anomalies, use_container_width=True, hide_index=True)

    if st.session_state.get("explainable") and "run_id" in top_anomalies.columns:
        run_id_options = top_anomalies["run_id"].dropna().tolist()
        if run_id_options:
            st.subheader("Chaîne causale")
            selected_run_id = st.selectbox("Anomalie à expliquer (relie l'alerte aux spans de trace suspects)", run_id_options)
            try:
                explain_resp = requests.get(
                    f"{base_url}/explain/{selected_run_id}", params={"model_id": selected_model_id}, timeout=15
                )
            except requests.exceptions.RequestException as e:
                st.error(f"Erreur en appelant /explain: {e}")
            else:
                if explain_resp.status_code != 200:
                    st.info(f"Chaîne causale indisponible: {explain_resp.json().get('detail', explain_resp.text)}")
                else:
                    causal_chain = explain_resp.json()["causal_chain"]
                    if not causal_chain:
                        st.info("Aucun signal suspect (erreur/latence) détecté sur ce run.")
                    else:
                        st.caption("Services suspects, du plus probable au moins probable (cause racine en tête)")
                        for rank, item in enumerate(causal_chain, start=1):
                            st.markdown(
                                f"**{rank}. {item['service']}** — `{item['operation']}` — "
                                f"{item['reason']} (score={item['score']:.1f}, {item['n_suspect_spans']} spans suspects)"
                            )
