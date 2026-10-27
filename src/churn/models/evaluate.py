"""Metrics, threshold selection, and model-level signals."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def threshold_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 6),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "flagged_rate": float(pred.mean()),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def ranking_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "average_precision": float(average_precision_score(y_true, proba)),
        "brier_score": float(brier_score_loss(y_true, proba)),
        "positive_rate": float(np.mean(y_true)),
        "n": int(len(y_true)),
    }


def pr_curve_frame(y_true: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # precision/recall have one more point than thresholds (the recall=0 endpoint).
    return pd.DataFrame(
        {"threshold": thresholds, "precision": precision[:-1], "recall": recall[:-1]}
    )


def choose_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    strategy: Literal["precision_at_min_recall", "max_f1"],
    min_recall: float = 0.7,
) -> dict[str, Any]:
    """Pick a decision threshold from validation predictions.

    precision_at_min_recall: highest precision among thresholds with recall >= min_recall.
    Ties break toward the higher threshold (fewer contacts). Always feasible for
    min_recall <= 1, because the lowest candidate threshold flags every row.
    """
    curve = pr_curve_frame(y_true, proba)
    if strategy == "max_f1":
        f1 = 2 * curve.precision * curve.recall / (curve.precision + curve.recall)
        row = curve.loc[f1.fillna(0).idxmax()]
    elif strategy == "precision_at_min_recall":
        ok = curve[curve.recall >= min_recall]
        best = ok.precision.max()
        row = ok[ok.precision == best].sort_values("threshold").iloc[-1]
    else:
        raise ValueError(f"unknown threshold strategy '{strategy}'")

    return {
        "strategy": strategy,
        "min_recall": min_recall if strategy == "precision_at_min_recall" else None,
        "threshold": float(row.threshold),
        "selected_on": "validation",
    }


def permutation_signals(
    pipeline, X: pd.DataFrame, y: np.ndarray, repeats: int, seed: int
) -> list[dict[str, float]]:
    """Drop in validation average precision when each input column is shuffled.

    This shows what the model relies on, not what causes churn. Correlated inputs share
    credit, so a low score does not mean a column is irrelevant.
    """
    result = permutation_importance(
        pipeline,
        X,
        y,
        scoring="average_precision",
        n_repeats=repeats,
        random_state=seed,
        n_jobs=1,
    )
    rows = [
        {"feature": col, "mean_drop": float(m), "std": float(s)}
        for col, m, s in zip(
            X.columns, result.importances_mean, result.importances_std, strict=True
        )
    ]
    return sorted(rows, key=lambda r: r["mean_drop"], reverse=True)
