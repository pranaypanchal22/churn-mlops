from pathlib import Path

import pandas as pd
import pytest

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
