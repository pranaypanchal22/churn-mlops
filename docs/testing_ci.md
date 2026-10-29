# Tests and continuous integration

## Running locally

```bash
.venv/Scripts/python -m pytest                                   # all tests
.venv/Scripts/python -m pytest --cov=churn --cov=frontend        # with coverage
.venv/Scripts/ruff check . && .venv/Scripts/ruff format --check .
```

**The tests never read the Kaggle dataset.** They use small hand-written frames and
[`churn.data.synthetic`](../src/churn/data/synthetic.py): random rows in the same raw layout,
with a planted signal and fake identifiers. A fresh clone can run the full suite without
downloading anything.

## What is covered

| Area | Test file | Examples |
|---|---|---|
| Raw profiling | `test_profile.py` | Target counts. Blank vs non-numeric `Total Charges`. ID and leakage candidates. Source file left untouched. |
| Schema and mapping config | `test_schema_config.py` | Unknown fields, unmapped required fields, a forbidden source sent to a feature, one column mapped twice, bad value-map tokens, version mismatch, config-key typos |
| Validation pipeline | `test_pipeline.py` | Missing or ambiguous columns, unexpected and leakage-looking columns, invalid categories, the blank-`Total Charges` rule (filled only when tenure is 0), non-numeric/range/integer checks, missing target, ID values kept out of reports |
| Training | `test_training.py` | Leakage guard, disjoint stratified splits, unknown and missing categories at predict time, preprocessing fit on train only, threshold rules, a complete artifact, reproducibility, tamper detection, promotion audit log |
| API | `test_api.py` | Health and ready, not-ready cases (nothing promoted, unknown pin, schema mismatch), valid and invalid requests without echoing values, threshold → label, batch limits, model-info, payloads kept out of logs |
| Frontend | `test_frontend.py` | Mapping HTTP failures to error types, payload building and casting, consistency hints, samples valid against the schema, Streamlit app end to end against a live API, the API-down state |
| CLI scripts | `test_cli.py` | Exit codes 0/1/2, reports written, no data written on failure, train → promote flow, `--reason` required |

Results from actual runs:

- GitHub Actions, first run on 2026-10-29 (commit `0f04e6e`): all 5 jobs passed (lint, test 3.11,
  test 3.13, smoke, docker).
  [Run log](https://github.com/pranaypanchal22/churn-mlops/actions/runs/40808515907).

- Windows / Python 3.13.5, after the interactive frontend update: 135 passed. `ruff check` and `ruff format --check` are clean. Line
  coverage is 95% across `churn` and `frontend`.
- Linux / Python 3.11 (`python:3.11-slim` container, same steps as the workflow), at the end of Phase 9: ruff
  clean, full suite passed, smoke run passed (synthetic data → train → promote → serve →
  `/predict`).
- Windows / Python 3.11.18 (temporary conda env, installed with `requirements/constraints.txt`), in Phase 6:
  97 passed (the suite at that time), ruff clean.

`tests/test_monitoring.py` (Phase 8) covers PSI, reference profiles, drift and performance reports,
`/metrics` and `/monitoring/drift`, the drift CLI's exit codes, and every retraining outcome:
dry run, promote, rejection on criteria, and rejection on invalid data.

The parts that aren't covered are mainly MLflow logging (optional, and skipped in tests), a few defensive
branches, and some frontend rendering paths.

## Dependency pinning

- `pyproject.toml` lists **compatible ranges** for direct dependencies.
- [`requirements/constraints.txt`](../requirements/constraints.txt) pins the **exact versions**
  that were tested together. CI and Docker install with `-c requirements/constraints.txt`.
- Transitive dependencies are not pinned. Fully hashed locking (e.g. `pip-tools` or `uv lock`)
  is listed as a future improvement.

## GitHub Actions

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs on pushes to `main`, on pull requests, and on manual dispatch.

| Job | What it does |
|---|---|
| `lint` | Pinned `ruff check` and `ruff format --check` (Python 3.11) |
| `test` | Full pytest suite with coverage on Python **3.11 and 3.13** |
| `smoke` | End to end on synthetic data: generate 1,000 rows → validate → train (`configs/train_smoke.yaml`) → promote into a separate `build/smoke-registry` → start uvicorn → `GET /ready` → `POST /predict` with `tests/fixtures/predict_request.json` → check the response. Prints the API log, even when the job fails. |

Design choices:

- **No secrets, no Kaggle, no cloud.** The token has `contents: read` permission only.
- **The smoke model can never be served by accident.** It uses its own registry folder, which is gitignored.
- Superseded runs on the same branch are cancelled, and every job has a timeout.
- The smoke job's metrics are meaningless (synthetic data) and are never reported.

### Running the CI steps yourself

The `smoke` job is plain shell, so the same steps work locally (Git Bash):

```bash
.venv/Scripts/python scripts/make_synthetic_data.py --rows 1000
.venv/Scripts/python scripts/train.py --config configs/train_smoke.yaml
.venv/Scripts/python scripts/promote.py --registry build/smoke-registry --list
```
