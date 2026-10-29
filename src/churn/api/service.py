"""Loads the served model once and turns validated inputs into predictions."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from churn.features.build import FeatureSet, to_model_frame
from churn.models import registry
from churn.monitoring.drift import MonitoringConfig
from churn.monitoring.live import LiveMonitor
from churn.schema.config import CanonicalSchema

log = logging.getLogger("churn.api")


@dataclass
class ModelService:
    pipeline: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    features: FeatureSet | None = None
    not_ready_reason: str | None = "model not loaded"
    monitor: LiveMonitor = field(default_factory=lambda: LiveMonitor(None, MonitoringConfig()))

    @property
    def ready(self) -> bool:
        return self.pipeline is not None and self.not_ready_reason is None

    @property
    def version(self) -> str | None:
        return self.metadata.get("model_version")

    @property
    def threshold(self) -> float:
        return float(self.metadata["threshold"]["threshold"])

    def load(
        self,
        registry_dir,
        schema: CanonicalSchema,
        pinned_version: str | None = None,
        monitoring: MonitoringConfig | None = None,
    ) -> None:
        """Load the pinned or promoted version. Failures leave the service not ready."""
        try:
            if pinned_version:
                version = pinned_version
            else:
                current = registry.read_current(registry_dir)
                if current is None:
                    raise registry.RegistryError(
                        f"no promoted model in {registry_dir}; run scripts/promote.py"
                    )
                version = current["version"]
            pipeline, metadata = registry.load_version(registry_dir, version)
        except registry.RegistryError as exc:
            self.not_ready_reason = str(exc)
            log.error("model_load_failed", extra={"reason": str(exc)})
            return

        trained_on = metadata["data"]["schema_version"]
        if trained_on != schema.schema_version:
            self.not_ready_reason = (
                f"model {version} was trained on schema {trained_on}, but the API validates "
                f"against schema {schema.schema_version}"
            )
            log.error("schema_mismatch", extra={"reason": self.not_ready_reason})
            return

        self.pipeline, self.metadata = pipeline, metadata
        f = metadata["features"]
        self.features = FeatureSet(
            numeric=f["numeric"], categorical=f["categorical"], categories={}
        )
        self.not_ready_reason = None
        profile = registry.read_reference_profile(registry_dir, version)
        self.monitor = LiveMonitor(profile, monitoring or MonitoringConfig())
        log.info(
            "model_loaded",
            extra={"model_version": version, "drift_monitoring": self.monitor.enabled},
        )

    def predict_proba(self, rows: list[dict[str, Any]], record: bool = True) -> list[float]:
        X = to_model_frame(pd.DataFrame(rows), self.features)
        proba = np.asarray(self.pipeline.predict_proba(X)[:, 1], dtype=float)
        # Only aggregate bin counts are kept; the rows themselves are discarded here.
        if record:
            self.monitor.record(X, proba)
        return [float(p) for p in proba]

    def label(self, probability: float) -> str:
        # Same rule as training: probability >= threshold is the positive class.
        return "high" if probability >= self.threshold else "low"

    def info(self) -> dict[str, Any]:
        m = self.metadata
        return {
            "model_version": m["model_version"],
            "model_type": m["model_type"],
            "created_at_utc": m["created_at_utc"],
            "schema_version": m["data"]["schema_version"],
            "data_mapping": f"{m['data']['mapping_name']}@{m['data']['mapping_version']}",
            "threshold": m["threshold"],
            "risk_labels": {
                "high": "churn_probability >= threshold",
                "low": "churn_probability < threshold",
            },
            "input_schema": m["input_schema"],
            # Context for reading a single score: how common churn was in training data and
            # how many customers the threshold flags. Both come from the training run.
            "reference": {
                "training_churn_rate": m["split"]["positive_rate"]["train"],
                "validation_flagged_rate": m["metrics"]["validation"]["flagged_rate"],
            },
            "metrics": {
                split: {
                    k: m["metrics"][split][k]
                    for k in ("roc_auc", "average_precision", "precision", "recall", "f1")
                }
                for split in ("validation", "test")
            },
            "model_signals": {
                "method": "permutation importance (drop in validation PR-AUC)",
                "caveat": "Model-level association only; not causal and not a per-customer "
                "explanation.",
                "top": m["model_signals"][:5],
            },
            "intended_use": "Portfolio/learning demo trained on one sample telecom dataset; "
            "not validated for real customer decisions or other providers.",
        }
