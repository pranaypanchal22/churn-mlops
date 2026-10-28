"""In-process aggregate monitor for the API.

Keeps running histogram counts of model inputs and scores since the process started. No
individual request is stored, and counts reset on restart; for durable monitoring, scrape
the Prometheus gauges or run scripts/drift_report.py on a batch.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import pandas as pd

from churn.monitoring.drift import (
    MonitoringConfig,
    copy_counts,
    drift_report,
    empty_counts,
    update_counts,
)


class LiveMonitor:
    def __init__(self, profile: dict[str, Any] | None, cfg: MonitoringConfig):
        self.profile = profile
        self.cfg = cfg
        self._lock = threading.Lock()
        self._counts = empty_counts(profile) if profile else None

    @property
    def enabled(self) -> bool:
        return self.profile is not None

    def record(self, X: pd.DataFrame, proba: np.ndarray) -> None:
        if not self.enabled:
            return
        with self._lock:
            update_counts(self._counts, self.profile, X, proba)

    def report(self) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        with self._lock:
            counts = copy_counts(self._counts)
        return drift_report(self.profile, counts, self.cfg)
