# Operations runbook (local)

How to set up, train, serve, and recover the project on one machine. Nothing here needs cloud
accounts or credentials. Monitoring and retraining are explained in
[monitoring.md](monitoring.md); the routine commands are in section 4.

## 1. First run from a clean checkout

Prerequisites: Git, Python 3.11–3.13, and Docker Desktop (for the container path).

```bash
git clone <your-repo-url> churn && cd churn
cp .env.example .env
```

Put the dataset in place. It is not in Git; see [data_profile.md](data_profile.md).

```
data/raw/CustomerChurn.csv     # SHA-256 79e23e3dff5595612d4db00d3f783f07c87162ed987061c5ab516611d3de25d6
```

Then choose **one** of the two paths below.

### Path A: Docker (recommended)

```bash
docker compose build
docker compose --profile tools run --rm trainer                                   # train -> prints VERSION
docker compose --profile tools run --rm trainer python scripts/promote.py --list
docker compose --profile tools run --rm trainer python scripts/promote.py --version <VERSION> --reason "initial baseline"
docker compose up -d
```

- UI: http://localhost:8501 · API docs: http://localhost:8000/docs
- Stop the stack with `docker compose down`.
- On Linux/macOS, set `HOST_UID`/`HOST_GID` in `.env` to the output of `id -u` / `id -g` so files the
  trainer writes into `models/` stay owned by you.

### Path B: Local Python

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -c requirements/constraints.txt -e ".[dev,tracking,frontend]"
.venv/Scripts/python scripts/prepare_data.py
.venv/Scripts/python scripts/train.py
.venv/Scripts/python scripts/promote.py --version <VERSION> --reason "initial baseline"
.venv/Scripts/python -m uvicorn churn.api.app:app --port 8000      # terminal 1
.venv/Scripts/python -m streamlit run frontend/app.py              # terminal 2
```

On Linux/macOS, use `.venv/bin/python`. With `make` installed, `make help` lists shortcuts for the
same commands.

### No dataset? (demo/CI only)

`make smoke`, or `scripts/make_synthetic_data.py` followed by `scripts/train.py --config configs/train_smoke.yaml`,
trains on synthetic rows into `build/smoke-registry`. To serve that model, set
`CHURN_REGISTRY_DIR=build/smoke-registry`. Its metrics mean nothing.

## 2. Containers

| Service | Image | Port | Notes |
|---|---|---|---|
| `api` | `churn-api:local` (python:3.11-slim, multi-stage) | 8000 | Non-root uid 10001. Read-only root filesystem with `/tmp` on tmpfs. All capabilities dropped. `no-new-privileges`. **`models/` is mounted read-only.** 1 GB memory limit. Healthcheck calls `/health`. |
| `frontend` | `churn-frontend:local` (Streamlit + httpx only) | 8501 | Same hardening, 512 MB limit. Starts after the API is healthy. Healthcheck calls `/_stcore/health`. |
| `trainer` | `churn-api:local` | none | Profile `tools`, one-off runs only. It can write to `data/`, `models/`, and `reports/`. |

Measured image sizes (2026-10-27): `churn-api:local` 742 MB, `churn-frontend:local` 818 MB, on a
190 MB `python:3.11-slim` base. Most of that is scientific Python wheels (numpy/pandas/scikit-learn
in the API; Streamlit's pandas/pyarrow in the frontend). Slimming the images further is listed as
future work.

The images contain no data and no model files: `.dockerignore` excludes `data/`, `models/`, `.env`, and
similar. The model is mounted at runtime, so **promoting a new version needs an API restart, not a
rebuild**.

**Checked on 2026-10-27:**
- Training in the `trainer` container (Linux, Python 3.11) gave the same confusion matrices and
  metrics as training on the Windows host (Python 3.13). The only difference was floating-point noise (~1e-14).
- The host-trained model loads and predicts identically in the container.
- Both containers run as uid 10001. Writes to `/app` and `/app/models` fail with "Read-only file system",
  and only `/tmp` is writable.

**Pickle compatibility:** model files are joblib pickles. The image installs exactly the scikit-learn
and numpy versions in `requirements/constraints.txt`. A model trained on the host must have been trained with the same
versions, and `metadata.json → reproducibility` records them. If they differ, retrain inside the
`trainer` container.

## 3. Configuration

All runtime settings are environment variables. See [`.env.example`](../.env.example). It contains no secrets,
and the project needs none.

| Variable | Used by | Default |
|---|---|---|
| `CHURN_REGISTRY_DIR` | API | `models/registry` (`/app/models/registry` in the image) |
| `CHURN_MODEL_VERSION` | API | empty, meaning serve the promoted version |
| `CHURN_MAX_BATCH_SIZE` | API | 100 |
| `CHURN_LOG_LEVEL` | API | INFO |
| `CHURN_MONITORING_CONFIG_PATH` | API | `configs/monitoring.yaml` |
| `CHURN_API_URL` | Frontend | `http://localhost:8000` (`http://api:8000` in compose) |
| `CHURN_API_TIMEOUT` | Frontend | 5 s |
| `API_PORT` / `FRONTEND_PORT` / `PROMETHEUS_PORT` | Compose | 8000 / 8501 / 9090 |

Timeouts: the frontend gives up on an API call after `CHURN_API_TIMEOUT`. Uvicorn closes idle keep-alive
connections after 5 s and allows 10 s for a graceful shutdown.

## 4. Routine tasks

| Task | Command |
|---|---|
| See which version is served | `scripts/promote.py --list` (the `*` row), or `GET /ready` |
| Promote a version | `scripts/promote.py --version V --reason "..."`, then `docker compose restart api` |
| Roll back | Promote the previous version (it's in `current.json → previous_version`), then restart the API. Or set `CHURN_MODEL_VERSION=V` in `.env` and restart. |
| Audit promotions | `models/registry/promotions.jsonl` (append-only) |
| Tail logs | `docker compose logs -f api` (one JSON object per line, no payloads) |
| Check live drift | `curl localhost:8000/monitoring/drift` (since the API started) |
| Check a batch file | `scripts/drift_report.py --input batch.csv` (exit 3 means drift alert) |
| Retrain (judge only) | `scripts/retrain.py --data new.csv`, then read `reports/retraining/<run>/decision.json` |
| Retrain and promote if it passes | `scripts/retrain.py --data new.csv --promote --reason "..."`, then `docker compose restart api` |
| Prometheus (optional) | `docker compose --profile monitoring up -d` → http://localhost:9090 |

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `/ready` returns 503 "no promoted model" | Nothing promoted, or the wrong registry is mounted | Promote a version. Check that `./models/registry/current.json` exists. |
| `/ready` returns 503 "hash mismatch" | Model file changed or corrupted after training | Retrain, or restore the version folder. Don't edit `metadata.json` by hand. |
| `/ready` returns 503 "schema" | Model was trained on a different schema version | Retrain with the current schema, or pin an older version together with its schema |
| UI says "service isn't reachable" | API is down or the URL is wrong | `docker compose ps`; check `CHURN_API_URL` |
| `frontend` never starts in compose | API healthcheck failing | `docker compose logs api` |
| `/monitoring/drift` returns 404 | Served model was trained before reference profiles existed | Retrain (`scripts/retrain.py`) and promote the candidate |
| Drift alert fired | Incoming inputs differ from training | Read the batch report, find out *why* the population changed, and collect outcomes. Don't retrain or roll back on drift alone. |
| Trainer: "Dataset not found" / validation errors | Raw file missing or changed | Check the path and SHA-256. Read `reports/validation_*.json`. |
| Docker: `dockerDesktopLinuxEngine` pipe not found | Docker Desktop engine not running | Open Docker Desktop and wait for "Engine running" |
| Docker Desktop quits on start. Its log (`%LOCALAPPDATA%\Docker\log\host\com.docker.backend.exe.log`) shows `remove ...\dockerInference` or `...\engine.sock: The file cannot be accessed by the system` | Leftover Unix-socket files that Windows refuses to delete. Seen on this machine's Windows build. | Quit Docker Desktop. Rename `%LOCALAPPDATA%\Docker\run` and `%LOCALAPPDATA%\docker-secrets-engine` (e.g. add `.stale-<date>`). Start Docker again; it recreates both folders. Don't choose "Reset to factory defaults" for this. |
| Build fails: `server gave HTTP response to HTTPS client` while loading `python:3.11-slim` metadata | BuildKit's registry lookup through Docker Desktop's internal proxy | `docker pull python:3.11-slim` once, then rebuild |
| Permission denied writing `models/` (Linux) | Container uid differs from host uid | Set `HOST_UID`/`HOST_GID` in `.env` |
