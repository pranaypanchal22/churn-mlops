"""Structured logging and Prometheus metrics for the API.

Neither logs nor metrics carry request bodies or feature values. Metric labels are limited
to route templates, status codes, risk labels, and feature *names*, so cardinality stays
bounded and nothing customer-specific is exported.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

# Attributes every LogRecord has; anything else was passed via `extra=` and is emitted.
_STANDARD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        entry.update({k: v for k, v in record.__dict__.items() if k not in _STANDARD_ATTRS})
        if record.exc_info:
            entry["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
        return json.dumps(entry, default=str)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    for name in ("churn", "uvicorn.error"):
        logger = logging.getLogger(name)
        logger.handlers = [handler]
        logger.setLevel(level)
        logger.propagate = False
    # Our access log replaces uvicorn's, which would duplicate every request line.
    logging.getLogger("uvicorn.access").disabled = True


class ApiMetrics:
    """Metric objects bound to their own registry so each app instance (and test) is isolated."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "churn_http_requests_total",
            "HTTP requests",
            ["method", "route", "status"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "churn_http_request_duration_seconds",
            "Request latency",
            ["route"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
            registry=self.registry,
        )
        self.validation_errors = Counter(
            "churn_validation_errors_total",
            "Rejected request fields",
            ["field"],
            registry=self.registry,
        )
        self.predictions = Counter(
            "churn_predictions_total",
            "Predictions served",
            ["risk_label"],
            registry=self.registry,
        )
        self.scores = Histogram(
            "churn_prediction_probability",
            "Distribution of predicted churn probability",
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
            registry=self.registry,
        )
        self.model_ready = Gauge(
            "churn_model_ready", "1 if a model is loaded", registry=self.registry
        )
        self.model_info = Gauge(
            "churn_model_info",
            "Loaded model version (value is always 1)",
            ["model_version", "schema_version"],
            registry=self.registry,
        )
        self.feature_psi = Gauge(
            "churn_feature_psi",
            "PSI of live inputs vs training since process start",
            ["feature"],
            registry=self.registry,
        )
        self.score_psi = Gauge(
            "churn_prediction_psi",
            "PSI of live scores vs validation since process start",
            registry=self.registry,
        )
        self.monitored_rows = Gauge(
            "churn_monitored_rows",
            "Rows counted by the live drift monitor",
            registry=self.registry,
        )

    def update_drift(self, report: dict[str, Any] | None) -> None:
        if not report:
            return
        self.monitored_rows.set(report["n"])
        for name, f in report["features"].items():
            self.feature_psi.labels(feature=name).set(f["psi"])
        if "score_psi" in report["prediction"]:
            self.score_psi.set(report["prediction"]["score_psi"])
