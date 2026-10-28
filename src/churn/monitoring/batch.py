"""Batch monitoring: data quality, drift, and (only with labels) performance for one file."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from churn.data.pipeline import prepare_csv
from churn.features.build import FeatureSet, to_model_frame
from churn.models import registry
from churn.monitoring.drift import (
    MonitoringConfig,
    batch_drift_report,
    performance_report,
)
from churn.schema.config import CanonicalSchema, SourceMapping


class MonitoringError(RuntimeError):
    pass


def monitor_batch(
    input_path: Path,
    schema: CanonicalSchema,
    mapping: SourceMapping,
    registry_dir: Path,
    cfg: MonitoringConfig,
    version: str | None = None,
) -> dict[str, Any]:
    if version is None:
        current = registry.read_current(registry_dir)
        if current is None:
            raise MonitoringError(f"no promoted model in {registry_dir}; pass --version")
        version = current["version"]
    pipeline, metadata = registry.load_version(registry_dir, version)
    profile = registry.read_reference_profile(registry_dir, version)
    if profile is None:
        raise MonitoringError(f"model {version} has no reference profile; retrain it")

    prepared = prepare_csv(input_path, schema, mapping, require_target=False)
    v = prepared.report
    quality = {
        "status": v.status,
        "rows": v.rows_in,
        "errors": [i.model_dump(exclude_none=True) for i in v.errors],
        "warnings": [i.model_dump(exclude_none=True) for i in v.warnings],
        "missing_after_cleaning": {k: n for k, n in v.missing_after_cleaning.items() if n},
    }
    report: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_version": version,
        "input": {"path": v.source.get("path"), "sha256": v.source.get("sha256")},
        "mapping": f"{mapping.mapping_name}@{mapping.mapping_version}",
        "data_quality": quality,
    }
    if prepared.data is None or v.errors:
        report["drift"] = None
        report["performance"] = None
        report["status"] = "invalid_data"
        return report

    f = metadata["features"]
    features = FeatureSet(numeric=f["numeric"], categorical=f["categorical"], categories={})
    X = to_model_frame(prepared.data, features)
    proba = np.asarray(pipeline.predict_proba(X)[:, 1], dtype=float)
    report["drift"] = batch_drift_report(profile, X, proba, cfg)
    report["status"] = report["drift"]["status"]

    target = schema.target.name
    if target in prepared.data and prepared.data[target].notna().all():
        report["performance"] = performance_report(
            prepared.data[target].astype(int).to_numpy(),
            proba,
            metadata["threshold"]["threshold"],
            metadata["metrics"]["test"],
        )
    else:
        report["performance"] = {
            "measured": False,
            "reason": "no outcome labels in this batch; drift alone cannot show whether "
            "the model got worse",
        }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Monitoring report: {report['input']['path']}",
        "",
        f"- Created (UTC): {report['created_at_utc']}",
        f"- Model: `{report['model_version']}` · mapping `{report['mapping']}`",
        f"- **Overall status: {report['status']}**",
        "",
        "## 1. Data quality",
        "",
        f"Validation: **{report['data_quality']['status']}**, "
        f"{report['data_quality']['rows']} rows.",
    ]
    for issue in report["data_quality"]["errors"]:
        lines.append(f"- ERROR [{issue['code']}] {issue['message']}")
    for issue in report["data_quality"]["warnings"]:
        lines.append(f"- warning [{issue['code']}] {issue['message']}")

    drift = report["drift"]
    lines += ["", "## 2. Input and prediction drift (no labels needed)", ""]
    if drift is None:
        lines.append("Not computed: the batch failed validation.")
    else:
        lines += [
            f"Status **{drift['status']}** on {drift['n']} rows "
            f"(PSI warn ≥ {drift['thresholds']['psi_warn']}, alert ≥ "
            f"{drift['thresholds']['psi_alert']}, minimum {drift['min_samples']} rows).",
            "",
            "| Feature | PSI | Status | Notes |",
            "|---|---:|---|---|",
        ]
        for name, feat in sorted(drift["features"].items(), key=lambda kv: -kv[1]["psi"]):
            note = ""
            if feat.get("outside_training_range"):
                note = f"{feat['outside_training_range']} outside training range"
            if feat.get("unseen_category_count"):
                note = f"{feat['unseen_category_count']} unseen categories"
            lines.append(f"| {name} | {feat['psi']:.4f} | {feat['status']} | {note} |")
        p = drift["prediction"]
        if "score_psi" in p:
            lines += [
                "",
                f"Predicted-probability PSI {p['score_psi']:.4f} ({p['status']}). Share flagged "
                f"high-risk {p['flag_rate']:.1%} vs {p['reference_flag_rate']:.1%} on "
                "validation.",
            ]
        lines += ["", f"> {drift['caveat']}"]

    perf = report["performance"]
    lines += ["", "## 3. Model performance (needs outcome labels)", ""]
    if perf is None:
        lines.append("Not computed: the batch failed validation.")
    elif perf.get("measured") is False:
        lines.append(f"Not measured: {perf['reason']}.")
    else:
        lines += [
            f"Measured on {perf['n']} labelled rows.",
            "",
            "| Metric | This batch | Test (training time) | Change |",
            "|---|---:|---:|---:|",
        ]
        for k, val in perf["metrics"].items():
            lines.append(
                f"| {k} | {val:.3f} | {perf['reference_test_metrics'][k]:.3f} | "
                f"{perf['change'][k]:+.3f} |"
            )
        lines += ["", perf["note"]]
    return "\n".join(lines) + "\n"
