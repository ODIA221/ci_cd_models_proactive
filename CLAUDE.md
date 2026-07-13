# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a research/thesis ("thèse") project on anomaly detection in CI/CD pipelines using metrics, logs, and traces. It's a data science codebase combining a small reusable `src/` pipeline with exploratory Jupyter notebooks. There is no README; this file is the primary orientation document.

The code and comments are written in French (variable names are English, docstrings/log messages/comments are French). Match that convention when editing existing files.

## Environment & setup

Two virtualenvs exist in the repo root: `venv/` and `.venv/` (Python 3.12) — both gitignored, and both were originally created at a different absolute path (`~/Documents/models_these`) then moved here, so their `bin/pip`/`bin/python3.12` shebangs point at a path that no longer exists. Use `python3 -m pip install ...` (not `pip install`/`./bin/pip`) inside either venv to avoid the broken shebang, or recreate the venv. Dependencies are listed in `requirements.txt` (pandas, numpy, scikit-learn, torch/torchvision, drain3 for log parsing, plotly, mlflow, joblib, pyyaml, requests, pyarrow). Source-specific heavy dependencies (currently just `RCAEval`) live in `requirements-optional.txt` — install only if you use that connector. Install with:

```bash
python3 -m pip install -r requirements.txt
```

There is no test suite, linter, or build step configured — no pytest, no CI config, no Makefile. `test.py` at the repo root is an ad-hoc data-loading smoke check, not a real test; run it directly with `python test.py` if needed.

**`run.sh` at the repo root is the single entry point** — it creates/updates `.venv` and installs `requirements.txt` before dispatching to a subcommand: `setup`, `demo` (default; runs `src/models/train.py` on the sample data), `sources` (data source registry status), `acquire <args>` (forwards to `src.data.acquire`), `evaluate` (protocoled precision/recall/F1/AUC on labeled data), `otel-up`/`otel-down` (Docker Compose for the OTel demo), `serve` (starts the FastAPI app), `dashboard` (starts the Streamlit UI — requires `serve` running in another terminal), **`start`** (the one-shot "launch everything" path: starts the API in the background, waits for `/health`, opens the dashboard in the browser, and blocks in the foreground running the dashboard), **`stop`** (kills anything bound to ports 8000/8501 — the reliable manual fallback, see gotcha below). Prefer extending `run.sh` over asking users to remember raw `python`/`uvicorn`/`streamlit` invocations.

**Signal-handling gotcha in `start`**: macOS ships bash 3.2 (2007, frozen due to GPLv3 licensing), where a trap on INT/TERM does not reliably fire while bash is blocked in `wait` on a background job — the signal is only processed once `wait` returns on its own. `start` works around this with the standard portable idiom: the trap just sets a flag (`STOP=1`), and a `while kill -0 $PID; do ...; sleep 0.5; done` loop polls that flag instead of relying on `wait` being interrupted. Signal delivery to backgrounded child processes could not be fully verified from within this session's sandboxed tool environment even on a minimal reproduction (independent of this codebase) — if `start`'s Ctrl+C cleanup ever seems to leave the API or dashboard running, `./run.sh stop` is the guaranteed fallback (kills by port, no dependency on trap/signal timing).

## Architecture

Data flows through three stages, mirrored by both `src/` modules and `notebooks/`:

1. **Load** — `src/data/load_data.py`: `CICDDataLoader` reads raw metrics (CSV/Parquet), logs (CSV/JSON), and traces (JSON) from `data/raw/{metrics,logs,traces}/` and writes cleaned output to `data/processed/`. It auto-creates `raw/`/`processed/` subdirectories under whatever `data_path` it's given.
2. **Preprocess** — `src/data/preprocess.py`: `CICDPreprocessor` auto-detects numeric vs. categorical columns (excluding id/timestamp/date-like columns), imputes missing values (`knn`/`mean`/`median`/`drop`), encodes categoricals (`label`/`onehot`), and standardizes numeric features. It has a proper sklearn-style **`fit(df)` / `transform(df)` split** (plus `fit_transform(df)` as a thin `fit().transform()` wrapper, kept for backward compatibility): `fit` detects columns and fits the imputer/encoders/scaler, storing that state on the instance (`numerical_cols`, `categorical_cols`, `numerical_cols_final`, `fitted_columns`); `transform` reapplies that exact fitted state to new data (reindexing to `fitted_columns`, mapping unseen categorical values to `-1`) without ever refitting — this is what makes it safe to run inference on a single incoming row (e.g. an API request) instead of only ever fitting fresh on a full batch.
3. **Model & train** — `src/models/detection_models.py` defines `AnomalyDetector`, a unified wrapper around three unsupervised approaches: `isolation_forest`, `one_class_svm`, and `autoencoder` (a PyTorch `nn.Module` also defined in this file). Predictions follow the sklearn outlier convention: `-1` = anomaly, `1` = normal; `anomaly_score(X)` gives a continuous score (higher = more anomalous — negated `decision_function`, or reconstruction MSE for the autoencoder) for ranking/prioritizing alerts instead of a bare label. `src/models/train.py` wires load → preprocess → train → evaluate → save into `train_pipeline(...)`. Per run it saves **three artifacts** under `models/{model_type}_{timestamp}`: `.joblib` (the sklearn model / autoencoder weights), `_preprocessor.joblib` (the *fitted* `CICDPreprocessor`, so the exact training-time transform can be replayed later), and `_meta.json` (`model_type`, `created_at`, `feature_columns` in order, source filenames) — the API's model registry (below) depends on this triplet. Predictions are saved to `experiments/results_{timestamp}.csv`, enriched with the original timestamp + raw feature values + `anomaly_score`, sorted by score descending (not just bare -1/1 labels).
4. **Evaluation** — `src/models/evaluate.py` is a separate, methodologically stricter script: it trains only on the known-normal split of a *labeled* dataset (currently LogHub HDFS via `data/interim/loghub/hdfs/`) and evaluates on held-out normal+abnormal data, reporting real precision/recall/F1/AUC — unlike `train.py`'s demo run, which has no ground truth and cannot tell you whether detections are meaningful. Feature engineering (`src/data/features.py`, `build_event_count_matrix`) turns per-run_id log sequences into a bag-of-events count matrix — a simple baseline that ignores event order (a known limitation, not state-of-the-art).
5. **API** — `src/api/` (FastAPI, started via `./run.sh serve` / `uvicorn src.api.main:app`) serves a trained model over HTTP: `GET /health`, `GET /models` (lists trained triplets via `model_registry.list_model_triplets()`, which only sees models with a `_meta.json` — older bare `.joblib` files predating this feature are invisible), `GET /sources` (wraps the data-source registry), `POST /predict` (validates incoming JSON keys against the model's `feature_columns`, returns 422 with structured `missing_columns`/`unexpected_columns` on mismatch rather than a stack trace). No `/train` endpoint (training takes minutes — stays a CLI concern) and no auth (research prototype, local use only). Interactive Swagger docs come free at `/docs` (also `/redoc`).
6. **Dashboard** — `src/dashboard/app.py` (Streamlit, started via `./run.sh dashboard`, requires `./run.sh serve` running separately) is a pure HTTP client of the API above — it never imports `AnomalyDetector`/`CICDPreprocessor` directly, keeping the API as the single reusable core. Lets a user pick a trained model, load the sample CSV or upload their own, POST a capped number of rows (slider, default 500/max 5000 — avoids sending the full 41k-row sample at once) to `/predict`, and view an anomaly-score-over-time Plotly chart plus a table of the most severe anomalies. One real gotcha it had to work around: `pd.DataFrame.to_dict()` leaves raw `NaN`s in place, which the standard JSON encoder rejects (`ValueError: Out of range float values are not JSON compliant: nan`) — the sample dataset has ~2400 missing cells, so the dashboard builds the request payload via `json.loads(df.to_json(orient="records"))` instead (pandas' own JSON serializer converts `NaN`→`null` correctly).

**Import-style gotcha**: `train.py`/`evaluate.py` are run as standalone scripts and add the **repo root** to `sys.path` then import via `from src.data.xxx import ...` / `from src.models.xxx import ...` (not the bare `data.xxx`/`models.xxx` used prior to the API's introduction). This matters beyond style: `train.py` `joblib.dump()`s a `CICDPreprocessor` instance, and pickle records the exact module path of its class — if `train.py` and `src/api/` imported the same class under two different qualified names (`data.preprocess.CICDPreprocessor` vs `src.data.preprocess.CICDPreprocessor`), the API would fail to unpickle models trained via the other path with `ModuleNotFoundError`. Keep any code that pickles/joblib-dumps objects on the `src.xxx` import style throughout.

Notebooks in `notebooks/` (`01_EDA_metrics`, `02_log_parsing`, `03_modele_simple`) are exploratory and largely templated with placeholder paths/filenames marked `# MODIFIER ICI` / `# À ADAPTER` — expect them to need path/filename edits before they run, and don't assume they're in sync with the current `src/` API.

## Data layout

- `data/raw/metrics/`, `data/raw/logs/`, `data/raw/traces/` — raw input data (CSV/JSON) per modality.
- `data/processed/` — cleaned/merged output (e.g. `multimodal_data.parquet`).
- `experiments/` — output predictions from training runs (`results_<timestamp>.csv`) and evaluation reports (`evaluation_<timestamp>.csv`).
- `models/` — per training run: `{model_type}_{timestamp}.joblib` (or PyTorch state_dict for the autoencoder), `_preprocessor.joblib`, `_meta.json` — see point 3 above. Created on demand by `train.py`.

Note: `data/raw/traces/train.py` is a stray/misplaced file, not part of the `src/models/train.py` pipeline — don't confuse the two.

## External data source connectors (`src/data/sources/`)

The 3 sample files above are not enough to train a real multimodal model. `src/data/sources/` adds a second, parallel data path: connectors that fetch and normalize external public CI/CD-observability datasets into a schema that plugs into (but doesn't replace) the pipeline above.

**Nothing downloads automatically.** Every source is fetched individually and on purpose:

```bash
python -m src.data.acquire --list                                   # inspect status/caveats, no side effects
python -m src.data.acquire --source loghub --dataset hdfs           # HDFS/BGL logs (LogHub)
python -m src.data.acquire --source rcaeval --subset RE2            # metrics+logs+traces, 735 real failure cases
python -m src.data.acquire --source gaia                            # requires a manual download first (see registry.yaml)
python -m src.data.acquire --source travistorrent --repo owner/name # per-repo raw Travis CI logs
python -m src.data.acquire --source github_actions --repo owner/name # requires GITHUB_TOKEN env var
python -m src.data.acquire --source otel_demo --export all          # requires the OTel demo stack running via Docker
```

`src/data/sources/registry.yaml` is the source of truth for what each connector needs (URLs, estimated size, `requires_token`/`requires_manual_download`/`requires_docker`) — check it before assuming a source "just works." Each connector (`loghub.py`, `rcaeval.py`, `gaia.py`, `travistorrent.py`, `github_actions.py`, `otel_demo/`) implements the `BaseConnector` contract from `base.py`: `fetch()` (idempotent download/collection) and `parse()` (normalization). `registry.py` resolves a source name to its connector class for `acquire.py`.

**Unified schema** (`src/data/schema.py`): every connector normalizes into one tidy/long table per modality — `METRICS_COLUMNS`, `LOGS_COLUMNS`, `TRACES_COLUMNS`, `LABELS_COLUMNS` — joined by `timestamp`/`source`/`run_id`/`service`. This is what makes cross-source fusion possible later without re-normalizing each source again. `pivot_metrics_to_wide()` reshapes tidy metrics into the wide format `CICDDataLoader`/`CICDPreprocessor` already expect; `to_sklearn_labels()` maps 0/1 ground-truth labels to the `-1`/`1` convention `AnomalyDetector` uses. **`load_data.py`/`preprocess.py` are intentionally untouched** — connectors write parquet directly with pandas rather than routing through `CICDDataLoader`, to avoid destabilizing the existing pipeline/notebooks.

Output directory convention (all gitignored except the 4 pre-existing sample files):
- `data/raw/<source>/` — raw downloaded artifacts.
- `data/interim/<source>/{metrics,logs,traces,labels}.parquet` — normalized output of `parse()`, unified schema.
- `data/processed/<source>/` — wide, model-ready exports, produced on demand.

Known caveats worth knowing before using a specific connector:
- **`loghub.py`** reuses the already-cloned `anomaly-detection-log-datasets/` (github.com/ait-aecid/anomaly-detection-log-datasets) toolkit rather than re-implementing HDFS/BGL parsing. `parse(use_samples=True)` (default) reads the pre-shipped sampled sequences in that repo — fast, no download needed, but timestamps/raw messages aren't available in that sampled form (only template/event IDs). `parse(use_samples=False)` parses a fully-downloaded raw log (via `fetch()`) for real timestamps/messages, needs the Zenodo archive.
- **`rcaeval.py`** targets the real `RCAEval.utility.download_re{1,2,3}_dataset()` API (verified against the upstream README/source) — RE2 is the richest subset (metrics+logs+traces). Needs `pip install -r requirements-optional.txt` first.
- **`gaia.py`** cannot auto-download (Baidu Netdisk/Google Drive links aren't stable/scriptable) — `fetch()` prints manual-download instructions and expects the archive placed under `data/raw/gaia/manual/`.
- **`travistorrent.py`**: the `monperrus/travistorrent-java-ci-build-dataset` repo does **not** contain a clean aggregated CSV (contrary to how it's sometimes described) — it's raw, compressed, per-job Travis CI logs organized one folder per GitHub repo (`{owner}@{repo}/*.log.bz2`). `fetch(repo=...)` sparse-checkouts just that folder; `parse(repo=...)` extracts build status/test counts via regex heuristics (best-effort, log format varies by test framework). The "clean" original TravisTorrent dump is a different artifact with an unresolved Figshare DOI (see `TODO` in `registry.yaml`).
- **`github_actions.py`** hard-requires the `GITHUB_TOKEN` env var and fails fast (no API call at all) if it's unset.
- **`otel_demo/`** doesn't download anything — it pings a live Prometheus/Jaeger/Loki stack (expected at `localhost:9090`/`:16686`/`:3100`) and fails fast with setup instructions if unreachable. The stack itself (github.com/open-telemetry/opentelemetry-demo, same demo used in the OF4CD paper) is meant to be cloned as a **sibling directory** and run via `docker compose up`, not vendored into this repo — see `src/data/sources/otel_demo/README.md`.

**Out of scope so far**: actually fusing logs+metrics+traces into one feature matrix for `AnomalyDetector` — this data layer only guarantees consistent join keys across sources; building the fused multimodal model itself is future work.
