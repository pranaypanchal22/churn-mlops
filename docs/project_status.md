# Project status, acceptance check, and portfolio notes

Status as of 2026-10-27. "Verified" means it was run and the output was observed, not just written.

## Acceptance criteria

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | A new user can follow the README to prepare data, train, test, start API and UI, and predict | **Verified** | Fresh copy of the tracked files plus the dataset. Followed README Option A exactly: build → trainer (identical metrics) → promote → `up` → `/ready`, `/predict` (0.8518), UI health 200. Option B's install-and-test path ran in a clean `python:3.11-slim` container: lint, full suite, smoke. |
| 2 | Training and preprocessing are reproducible from configuration | **Verified** | `configs/train.yaml` + schema + mapping. Repeated host runs, one Linux container run, and the fresh-checkout run all gave the same metrics (differences ≤ 1e-14). |
| 3 | Schema mismatches give actionable errors and are never silently ignored | **Verified** | Wrong mapping on the real file → 2 `missing_required_column` errors listing the available columns and the YAML key to edit, plus 4 `unexpected_column` errors. 30 schema/pipeline tests. |
| 4 | Identifiers and target-derived columns can't leak into training | **Verified** | Forbidden-pattern check at config load, `assert_no_leakage` before fitting, output columns limited to canonical names. Tests in `test_schema_config.py`, `test_pipeline.py`, `test_training.py`. |
| 5 | The API returns validated, versioned predictions | **Verified** | Schema-generated request validation, `model_version` + threshold + note in every response. Not-ready on missing, tampered, or schema-mismatched models. 22 API tests. |
| 6 | The frontend sends requests to the API and shows responses and errors clearly | **Verified** | Checked in the browser at desktop width and 375 px, plus API-down and result states (screenshots in `docs/images/`). 18 frontend tests, including Streamlit's test runner against a live API. |
| 7 | CI runs automated checks without private credentials | **Verified** | First GitHub Actions run on 2026-10-29 ([run 40808515907](https://github.com/pranaypanchal22/churn-mlops/actions/runs/40808515907)) passed all 5 jobs in about 2 minutes: lint, test (3.11), test (3.13), smoke, docker. Read-only token, no secrets. |
| 8 | Docker Compose starts the API and frontend locally | **Verified** | Both healthy. Non-root uid 10001. Read-only filesystem confirmed by failed writes. |
| 9 | Monitoring and retraining are documented and use safe promotion rules | **Verified** | `docs/monitoring.md`. Real-data drift runs (`ok` / `alert`). Retraining dry run → `eligible_not_promoted` with serving unchanged. Tests for promote, reject-on-criteria, and reject-on-invalid-data paths. |
| 10 | Reported dataset facts and metrics come from actual outputs | **Verified** | Every number in the docs was copied from script output or `metadata.json`. Subgroup counts were re-checked before publishing. |

## Implemented vs future work

**Implemented and verified:**
- Data profiling.
- Canonical schema with versioned YAML mappings (two real sources).
- Validation reports.
- Leakage guards.
- Reproducible training with a validation-chosen threshold.
- Hash-checked local registry with an audit log.
- FastAPI service (single and batch prediction, readiness, model info, metrics, live drift).
- Streamlit UI.
- 119 tests.
- GitHub Actions workflow (passing on GitHub).
- Hardened Docker Compose stack.
- JSON logging and Prometheus metrics without payloads.
- PSI drift reports kept separate from labelled performance.
- Gated retraining.
- Optional Prometheus profile with 5 alert rules.
- Model card with subgroup results.

**Optional, written but lightly exercised:** MLflow tracking (it logs to SQLite, and isn't covered by tests).
Prometheus alert rules were loaded and reached `pending`; none has fired for its full `for:` window.

**Future work (not implemented):**
- calibration, per-segment or cost-based thresholds, CV with confidence intervals, tuning;
- fairness audit on a source with `gender`;
- hashed lockfile and smaller images;
- hot model reload;
- durable monitoring store, Grafana, Alertmanager;
- scheduled retraining dry runs;
- any cloud deployment.

## Suggested resume bullets

Placeholders in `[brackets]` are for **you** to fill from your own verified runs (current values are
in `docs/model_evaluation.md` and `docs/testing_ci.md`). Remove any bullet you can't back up in an
interview.

- Built an end-to-end customer-churn ML system (pandas, scikit-learn, FastAPI, Streamlit, Docker),
  from schema-validated ingestion to a monitored prediction API, running fully locally with no cloud
  dependencies.
- Designed a versioned canonical schema with YAML source mappings that reject unmapped, ambiguous,
  and outcome-derived columns with actionable errors. Onboarded a second source layout through
  configuration only.
- Trained and compared Logistic Regression and Random Forest with leakage-safe sklearn pipelines.
  Chose the decision threshold on validation data for a stated recall goal, and reported held-out
  test PR-AUC of [X] and recall of [Y].
- Served versioned predictions through FastAPI with schema-generated validation, readiness checks, and
  hash-verified model loading. Built a Streamlit UI that talks to the API only.
- Automated quality gates with [N] pytest tests ([Z]% coverage) and a GitHub Actions pipeline running
  lint, tests on two Python versions, an end-to-end synthetic smoke run, and Docker builds, with no
  secrets.
- Containerised the stack with non-root, read-only, capability-dropped containers and a read-only
  model mount.
- Implemented privacy-preserving monitoring (JSON logs and Prometheus metrics without payloads,
  aggregate PSI drift), kept separate from labelled performance evaluation.
- Built a gated retraining workflow that promotes a candidate only when [K] policy criteria pass and
  promotion is explicitly requested, and records every decision in an audit log.

Avoid claiming production use, business impact, real customers, or accuracy on other providers.
None of these were measured.
