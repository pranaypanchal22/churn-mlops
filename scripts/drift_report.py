"""Check a batch of customer data against the served model's training reference.

Usage:
    python scripts/drift_report.py --input <batch.csv>
        [--mapping configs/mappings/telco_kaggle_v1.yaml] [--version <model version>]

Reports (1) data quality, (2) input and prediction drift, and (3) model performance only if
the batch contains churn outcomes. Writes reports/monitoring/<timestamp>_<file>.{json,md}.

Exit codes: 0 ok/warn/insufficient_data, 1 invalid data, 2 config/model problem, 3 drift alert.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from churn.models.registry import RegistryError
from churn.monitoring.batch import MonitoringError, monitor_batch, render_markdown
from churn.monitoring.drift import load_monitoring_config
from churn.schema.config import ConfigError, load_mapping, load_schema


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--mapping", type=Path, default=Path("configs/mappings/telco_kaggle_v1.yaml")
    )
    parser.add_argument("--schema", type=Path, default=Path("configs/schema.yaml"))
    parser.add_argument("--registry", type=Path, default=Path("models/registry"))
    parser.add_argument("--version", help="model version (default: the promoted one)")
    parser.add_argument("--config", type=Path, default=Path("configs/monitoring.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/monitoring"))
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 2
    try:
        schema = load_schema(args.schema)
        mapping = load_mapping(args.mapping, schema)
        report = monitor_batch(
            args.input,
            schema,
            mapping,
            args.registry,
            load_monitoring_config(args.config),
            args.version,
        )
    except (ConfigError, RegistryError, MonitoringError, ValueError) as exc:
        print(f"Cannot run monitoring: {exc}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{args.input.stem}"
    json_path = args.out_dir / f"{stem}.json"
    md_path = args.out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    drift = report["drift"] or {}
    print(f"Status: {report['status']}  (model {report['model_version']})")
    if drift:
        print(f"  rows {drift['n']}, drifted features: {drift['drifted_features'] or 'none'}")
        if "score_psi" in drift["prediction"]:
            p = drift["prediction"]
            print(
                f"  score PSI {p['score_psi']}, flagged {p['flag_rate']:.1%} "
                f"(validation {p['reference_flag_rate']:.1%})"
            )
    perf = report["performance"]
    if perf and perf.get("measured") is not False:
        print(f"  labelled performance: {perf['metrics']} (change vs test {perf['change']})")
    else:
        print("  performance: not measured (no outcome labels)")
    print(f"Report: {md_path}")

    if report["status"] == "invalid_data":
        return 1
    return 3 if report["status"] == "alert" else 0


if __name__ == "__main__":
    raise SystemExit(main())
