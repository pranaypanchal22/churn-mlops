"""Input and prediction drift against a training-time reference profile.

Everything here works on aggregate counts (histogram bins and category tallies), never on
stored rows, so live monitoring keeps no customer-level data.

Drift answers "do incoming inputs or scores look different from training?". It does NOT say
whether the model got worse: that needs ground-truth outcomes (see `performance_report`).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from churn.features.build import FeatureSet
from churn.models.evaluate import ranking_metrics, threshold_metrics

PROFILE_VERSION = "1"
OTHER = "__other__"
EPS = 1e-4
# Fixed score bins so prediction distributions are comparable across model versions.
SCORE_EDGES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

DRIFT_CAVEAT = (
    "Drift compares incoming inputs and scores with the training data. It can signal that the "
    "model is being used outside the conditions it was trained on, but it does not measure "
    "accuracy: model performance can only be assessed with observed churn outcomes."
)


class MonitoringConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Common PSI rules of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 large shift.
    psi_warn: float = Field(default=0.1, gt=0)
    psi_alert: float = Field(default=0.25, gt=0)
    # Below this many rows the PSI is too noisy to act on.
    min_samples: int = Field(default=200, ge=1)
    # Absolute change in the share of customers flagged high-risk that triggers a warning.
    flag_rate_warn_delta: float = Field(default=0.10, gt=0, le=1)


def load_monitoring_config(path: Path | None) -> MonitoringConfig:
    if path is None or not Path(path).exists():
        return MonitoringConfig()
    return MonitoringConfig.model_validate(
        yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    )


# --- Reference profile (built once, at training time) ------------------------------------


def _bin_index(values: np.ndarray, edges: list[float]) -> np.ndarray:
    return np.searchsorted(np.asarray(edges, dtype=float), values, side="right")


def _proportions(counts: np.ndarray) -> list[float]:
    total = counts.sum()
    return (counts / total).round(6).tolist() if total else [0.0] * len(counts)


def build_reference_profile(
    X_train: pd.DataFrame,
    features: FeatureSet,
    validation_proba: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Summarise training inputs and validation scores as bins/proportions."""
    numeric = {}
    for col in features.numeric:
        values = X_train[col].to_numpy(dtype=float)
        present = values[~np.isnan(values)]
        # Decile edges from training data; duplicates collapse for discrete columns.
        edges = sorted(set(np.quantile(present, np.linspace(0.1, 0.9, 9)).round(6).tolist()))
        counts = np.bincount(_bin_index(present, edges), minlength=len(edges) + 1)
        numeric[col] = {
            "edges": edges,
            "proportions": _proportions(counts),
            "missing_rate": round(float(np.isnan(values).mean()), 6),
            "min": float(present.min()),
            "max": float(present.max()),
        }

    categorical = {}
    for col in features.categorical:
        shares = X_train[col].value_counts(normalize=True, dropna=False)
        categorical[col] = {"proportions": {str(k): round(float(v), 6) for k, v in shares.items()}}

    score_counts = np.bincount(
        _bin_index(validation_proba, SCORE_EDGES), minlength=len(SCORE_EDGES) + 1
    )
    return {
        "profile_version": PROFILE_VERSION,
        "n_train_rows": int(len(X_train)),
        "numeric": numeric,
        "categorical": categorical,
        "prediction": {
            "source": "validation split",
            "edges": SCORE_EDGES,
            "proportions": _proportions(score_counts),
            "threshold": float(threshold),
            "flag_rate": round(float((validation_proba >= threshold).mean()), 6),
        },
    }


# --- Aggregate counts for incoming data ---------------------------------------------------


def empty_counts(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "n": 0,
        "numeric": {
            col: {"bins": [0] * (len(p["edges"]) + 1), "missing": 0, "below_min": 0, "above_max": 0}
            for col, p in profile["numeric"].items()
        },
        "categorical": {col: {} for col in profile["categorical"]},
        "prediction": {"bins": [0] * (len(SCORE_EDGES) + 1), "flagged": 0, "n": 0},
    }


def update_counts(
    counts: dict[str, Any],
    profile: dict[str, Any],
    X: pd.DataFrame,
    proba: np.ndarray | None = None,
) -> None:
    """Add a batch of model-ready rows (and optional scores) to running counts, in place."""
    counts["n"] += len(X)
    for col, p in profile["numeric"].items():
        values = X[col].to_numpy(dtype=float)
        present = values[~np.isnan(values)]
        c = counts["numeric"][col]
        idx = np.bincount(_bin_index(present, p["edges"]), minlength=len(p["edges"]) + 1)
        c["bins"] = (np.asarray(c["bins"]) + idx).tolist()
        c["missing"] += int(np.isnan(values).sum())
        c["below_min"] += int((present < p["min"]).sum())
        c["above_max"] += int((present > p["max"]).sum())
    for col, p in profile["categorical"].items():
        known = set(p["proportions"])
        tallies = counts["categorical"][col]
        for value, n in X[col].astype(str).value_counts().items():
            key = value if value in known else OTHER
            tallies[key] = tallies.get(key, 0) + int(n)
    if proba is not None:
        pred = counts["prediction"]
        idx = np.bincount(_bin_index(proba, SCORE_EDGES), minlength=len(SCORE_EDGES) + 1)
        pred["bins"] = (np.asarray(pred["bins"]) + idx).tolist()
        pred["flagged"] += int((proba >= profile["prediction"]["threshold"]).sum())
        pred["n"] += int(len(proba))


# --- Comparison ---------------------------------------------------------------------------


def psi(expected: list[float], actual_counts: list[int]) -> float:
    """Population Stability Index between reference proportions and observed counts."""
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    total = float(np.sum(actual_counts))
    if total == 0:
        return 0.0
    a = np.clip(np.asarray(actual_counts, dtype=float) / total, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _status(value: float, cfg: MonitoringConfig) -> str:
    if value >= cfg.psi_alert:
        return "alert"
    return "warn" if value >= cfg.psi_warn else "ok"


_RANK = {"ok": 0, "warn": 1, "alert": 2}


def drift_report(
    profile: dict[str, Any], counts: dict[str, Any], cfg: MonitoringConfig
) -> dict[str, Any]:
    n = counts["n"]
    features: dict[str, Any] = {}
    for col, p in profile["numeric"].items():
        c = counts["numeric"][col]
        value = psi(p["proportions"], c["bins"])
        features[col] = {
            "type": "numeric",
            "psi": round(value, 4),
            "status": _status(value, cfg),
            "missing_rate": round(c["missing"] / n, 4) if n else 0.0,
            "outside_training_range": c["below_min"] + c["above_max"],
        }
    for col, p in profile["categorical"].items():
        tallies = counts["categorical"][col]
        levels = list(p["proportions"]) + [OTHER]
        expected = [p["proportions"].get(k, 0.0) for k in levels]
        observed = [tallies.get(k, 0) for k in levels]
        value = psi(expected, observed)
        features[col] = {
            "type": "categorical",
            "psi": round(value, 4),
            "status": _status(value, cfg),
            "unseen_category_count": tallies.get(OTHER, 0),
        }

    pred_ref, pred = profile["prediction"], counts["prediction"]
    prediction: dict[str, Any] = {"n": pred["n"]}
    if pred["n"]:
        score_psi = psi(pred_ref["proportions"], pred["bins"])
        flag_rate = pred["flagged"] / pred["n"]
        delta = flag_rate - pred_ref["flag_rate"]
        status = _status(score_psi, cfg)
        if abs(delta) >= cfg.flag_rate_warn_delta and status == "ok":
            status = "warn"
        prediction.update(
            {
                "score_psi": round(score_psi, 4),
                "flag_rate": round(flag_rate, 4),
                "reference_flag_rate": pred_ref["flag_rate"],
                "flag_rate_change": round(delta, 4),
                "status": status,
            }
        )

    if n < cfg.min_samples:
        overall = "insufficient_data"
    else:
        statuses = [f["status"] for f in features.values()]
        if "status" in prediction:
            statuses.append(prediction["status"])
        overall = max(statuses, key=_RANK.get) if statuses else "ok"

    drifted = sorted(
        (k for k, f in features.items() if f["status"] != "ok"),
        key=lambda k: -features[k]["psi"],
    )
    return {
        "status": overall,
        "n": n,
        "min_samples": cfg.min_samples,
        "thresholds": {"psi_warn": cfg.psi_warn, "psi_alert": cfg.psi_alert},
        "drifted_features": drifted,
        "features": features,
        "prediction": prediction,
        "performance": "not measured: requires observed churn outcomes",
        "caveat": DRIFT_CAVEAT,
    }


def batch_drift_report(
    profile: dict[str, Any],
    X: pd.DataFrame,
    proba: np.ndarray | None,
    cfg: MonitoringConfig,
) -> dict[str, Any]:
    counts = empty_counts(profile)
    update_counts(counts, profile, X, proba)
    return drift_report(profile, counts, cfg)


def performance_report(
    y_true: np.ndarray, proba: np.ndarray, threshold: float, reference: dict[str, Any]
) -> dict[str, Any]:
    """Actual model quality on labelled data, compared with the held-out test results."""
    current = {**ranking_metrics(y_true, proba), **threshold_metrics(y_true, proba, threshold)}
    keys = ("roc_auc", "average_precision", "precision", "recall", "f1")
    return {
        "n": int(len(y_true)),
        "metrics": {k: round(current[k], 4) for k in keys},
        "reference_test_metrics": {k: round(reference[k], 4) for k in keys},
        "change": {k: round(current[k] - reference[k], 4) for k in keys},
        "confusion_matrix": current["confusion_matrix"],
        "note": "Measured against observed outcomes. Differences on small samples are noisy.",
    }


def copy_counts(counts: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(counts)
