"""Synthetic raw data in the Kaggle CustomerChurn.csv layout, for tests and CI only.

Values are random with a planted churn signal (month-to-month contracts, short tenure,
fiber, electronic check raise churn odds) so models have something to learn. These rows
say nothing about real customers and must never be used to report model quality.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_synthetic_raw(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    yes_no = np.array(["Yes", "No"])

    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
    tenure = rng.integers(0, 73, n)
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    phone = rng.choice(yes_no, n, p=[0.9, 0.1])
    payment = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        n,
    )
    monthly = np.round(rng.uniform(18.25, 118.75, n), 2)
    total = np.round(monthly * tenure * rng.uniform(0.9, 1.1, n), 2)

    def addon():
        vals = rng.choice(yes_no, n)
        return np.where(internet == "No", "No internet service", vals)

    logit = (
        -1.2
        + 1.4 * (contract == "Month-to-month")
        - 0.04 * tenure
        + 0.8 * (internet == "Fiber optic")
        + 0.5 * (payment == "Electronic check")
    )
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))

    df = pd.DataFrame(
        {
            "LoyaltyID": rng.integers(100000, 999999, n).astype(str),
            "Customer ID": [f"SYN-{i:05d}" for i in range(n)],
            "Senior Citizen": rng.choice(yes_no, n, p=[0.16, 0.84]),
            "Partner": rng.choice(yes_no, n),
            "Dependents": rng.choice(yes_no, n, p=[0.3, 0.7]),
            "Tenure": tenure.astype(str),
            "Phone Service": phone,
            "Multiple Lines": np.where(phone == "No", "No phone service", rng.choice(yes_no, n)),
            "Internet Service": internet,
            "Online Security": addon(),
            "Online Backup": addon(),
            "Device Protection": addon(),
            "Tech Support": addon(),
            "Streaming TV": addon(),
            "Streaming Movies": addon(),
            "Contract": contract,
            "Paperless Billing": rng.choice(yes_no, n),
            "Payment Method": payment,
            "Monthly Charges": monthly.astype(str),
            "Total Charges": total.astype(str),
            "Churn": np.where(churn, "Yes", "No"),
        }
    )
    # Mirror the real file's quirk: brand-new customers have a blank total.
    df.loc[tenure == 0, "Total Charges"] = " "
    return df


def write_synthetic_csv(path: Path, n: int = 600, seed: int = 0) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    make_synthetic_raw(n, seed).to_csv(path, index=False)
    return path
