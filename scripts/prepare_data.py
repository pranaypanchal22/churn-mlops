"""Map, clean, and validate a raw source file into the canonical schema.

Usage:
    python scripts/prepare_data.py \
        [--input data/raw/CustomerChurn.csv] \
        [--mapping configs/mappings/telco_kaggle_v1.yaml] \
        [--schema configs/schema.yaml]

Always writes the validation report. Writes processed data only when validation passes.
Exit codes: 0 passed, 1 validation failed, 2 input or config problem.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from churn.data.pipeline import file_sha256, prepare_csv
from churn.schema.config import ConfigError, load_mapping, load_schema


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=Path("data/raw/CustomerChurn.csv"))
    parser.add_argument(
        "--mapping", type=Path, default=Path("configs/mappings/telco_kaggle_v1.yaml")
    )
    parser.add_argument("--schema", type=Path, default=Path("configs/schema.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"Input not found: {args.input}. See README > Data.", file=sys.stderr)
        return 2
    try:
        schema = load_schema(args.schema)
        mapping = load_mapping(args.mapping, schema)
    except (ConfigError, ValueError) as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    result = prepare_csv(args.input, schema, mapping)
    report = result.report
    name = mapping.mapping_name

    if result.data is not None and not report.errors:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.out_dir / f"{name}_canonical.csv"
        result.data.to_csv(out_path, index=False)
        report.source["processed_path"] = out_path.as_posix()
        report.source["processed_sha256"] = file_sha256(out_path)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_dir / f"validation_{name}.json"
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    print(
        f"{report.status.upper()}: {name} v{report.mapping_version} "
        f"(schema {report.schema_version}), rows in={report.rows_in} out={report.rows_out}"
    )
    for issue in report.issues:
        print(f"  {issue.severity:7} [{issue.code}] {issue.message}")
    for action in report.cleaning_actions:
        print(f"  cleaning: {action['rule']} -> {action['rows_affected']} row(s)")
    print(f"Report: {report_path}")
    if "processed_path" in report.source:
        print(f"Data:   {report.source['processed_path']}")
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
