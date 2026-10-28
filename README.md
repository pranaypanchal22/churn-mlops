# Customer Churn Prediction: MLOps Portfolio Project

End-to-end churn prediction system covering data validation, training, a prediction API, a
frontend, CI, Docker, monitoring, and a controlled retraining workflow.

> Learning and portfolio project. Not for making real customer decisions.

**Status:** Phases 1–6 done (data profiling, schema-aware validation, training and evaluation,
prediction API, Streamlit frontend, tests and CI).
The full README comes in Phase 9.

## Data

The dataset is **not** in Git. To provide it:

1. Download the Kaggle telecom churn archive. It contains `CustomerChurn.csv`.
2. Copy `CustomerChurn.csv`, without changes, to `data/raw/CustomerChurn.csv`.
3. Check the file: its SHA-256 should be
   `79e23e3dff5595612d4db00d3f783f07c87162ed987061c5ab516611d3de25d6`. A different hash means a
   different version, so re-run the profile and review [docs/data_profile.md](docs/data_profile.md).

License status: **unknown, must be verified** on the Kaggle dataset page.

Raw columns are mapped to a canonical schema through a versioned YAML mapping. See
[docs/schema_mapping.md](docs/schema_mapping.md).

## Setup and commands

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -c requirements/constraints.txt -e ".[dev,tracking,frontend]"   # Linux/macOS: .venv/bin/python; tracking (MLflow) is optional
.venv/Scripts/python scripts/profile_data.py      # writes reports/data_profile.{json,md}
.venv/Scripts/python scripts/prepare_data.py      # validate + write data/processed/telco_kaggle_canonical.csv
.venv/Scripts/python scripts/train.py             # registers a candidate in models/registry/ (not served)
.venv/Scripts/python scripts/promote.py --list     # list versions; promote with --version V --reason "..."
.venv/Scripts/python -m uvicorn churn.api.app:app --port 8000   # API; docs at /docs
.venv/Scripts/python -m streamlit run frontend/app.py         # UI at http://localhost:8501 (needs the API)
.venv/Scripts/python -m pytest
.venv/Scripts/ruff check . && .venv/Scripts/ruff format --check .
```

Training method and actual results: [docs/model_evaluation.md](docs/model_evaluation.md).
API endpoints, configuration, and examples: [docs/api.md](docs/api.md).
Frontend behaviour and error states: [docs/frontend.md](docs/frontend.md).
Tests, coverage, and the CI pipeline: [docs/testing_ci.md](docs/testing_ci.md).
