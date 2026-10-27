from pathlib import Path

import pandas as pd
import pytest
import yaml

from churn.data.synthetic import write_synthetic_csv
from churn.models import registry
from churn.models.train import train
from churn.schema.config import load_mapping, load_schema

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "configs" / "schema.yaml"
KAGGLE_MAPPING_PATH = ROOT / "configs" / "mappings" / "telco_kaggle_v1.yaml"
IBM_MAPPING_PATH = ROOT / "configs" / "mappings" / "ibm_telco_v1.yaml"

# Synthetic rows in the Kaggle CustomerChurn.csv layout. Identifiers are fake.
_BASE_ROW = {
    "LoyaltyID": "100001",
    "Customer ID": "0000-TEST",
    "Senior Citizen": "No",
    "Partner": "Yes",
    "Dependents": "No",
    "Tenure": "12",
    "Phone Service": "Yes",
    "Multiple Lines": "No",
    "Internet Service": "Fiber optic",
    "Online Security": "No",
    "Online Backup": "Yes",
    "Device Protection": "No",
    "Tech Support": "No",
    "Streaming TV": "Yes",
    "Streaming Movies": "No",
    "Contract": "Month-to-month",
    "Paperless Billing": "Yes",
    "Payment Method": "Electronic check",
    "Monthly Charges": "80.5",
    "Total Charges": "966.0",
    "Churn": "Yes",
}


def make_raw(n: int = 10) -> pd.DataFrame:
    rows = []
    for i in range(n):
        row = dict(_BASE_ROW)
        row["LoyaltyID"] = str(100001 + i)
        row["Customer ID"] = f"{i:04d}-TEST"
        row["Churn"] = "Yes" if i % 3 == 0 else "No"
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def schema():
    return load_schema(SCHEMA_PATH)


@pytest.fixture
def kaggle_mapping(schema):
    return load_mapping(KAGGLE_MAPPING_PATH, schema)


@pytest.fixture
def ibm_mapping(schema):
    return load_mapping(IBM_MAPPING_PATH, schema)


@pytest.fixture
def raw():
    return make_raw()


def make_smoke_config(directory: Path) -> Path:
    """Small, fast training config on synthetic data (no real dataset needed)."""
    raw = write_synthetic_csv(directory / "raw.csv", n=500, seed=2)
    cfg = {
        "config_version": "test",
        "random_seed": 7,
        "data": {
            "raw_path": str(raw),
            "schema_path": str(SCHEMA_PATH),
            "mapping_path": str(KAGGLE_MAPPING_PATH),
        },
        "split": {"validation_size": 0.2, "test_size": 0.2},
        "models": {
            "logistic_regression": {"max_iter": 500, "class_weight": "balanced"},
            "random_forest": {"n_estimators": 30, "min_samples_leaf": 5},
        },
        "threshold": {"strategy": "precision_at_min_recall", "min_recall": 0.7},
        "explanations": {"permutation_repeats": 2},
        "output": {"registry_dir": str(directory / "registry")},
        "mlflow": {"enabled": False},
    }
    path = directory / "train.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def promoted_registry(tmp_path_factory):
    """A registry with one synthetic-data model already promoted. Returns (dir, version)."""
    result = train(make_smoke_config(tmp_path_factory.mktemp("promoted")))
    reg = result.version_dir.parent
    registry.set_current(reg, result.version, "test fixture")
    return reg, result.version
