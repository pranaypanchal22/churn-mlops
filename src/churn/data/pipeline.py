"""Map a raw source table onto the canonical schema, clean it, and validate it.

Stages:
  1. Resolve columns: each canonical field must match exactly one listed source alias.
  2. Account for every source column (mapped, identifier, excluded, ignored, or unexpected).
  3. Translate values through explicit value maps and parse numerics.
  4. Apply documented fill rules.
  5. Check allowed values, ranges, integer-ness, and missing-value rates.

Every problem is collected into a ValidationReport rather than raised one at a time, so a
single run shows everything that needs fixing. Reported examples are category values or
numbers only; identifier columns are never read beyond their names.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel

from churn.schema.config import CanonicalSchema, FieldSpec, SourceMapping

MAX_EXAMPLES = 5


class Issue(BaseModel):
    severity: Literal["error", "warning"]
    code: str
    message: str
    field: str | None = None
    source_column: str | None = None
    count: int | None = None
    examples: list[str] = []


class ValidationReport(BaseModel):
    status: Literal["passed", "failed"]
    created_at_utc: str
    schema_version: str
    mapping_name: str
    mapping_version: str
    source: dict[str, Any]
    rows_in: int
    rows_out: int | None
    column_resolution: dict[str, str]
    identifier_columns_dropped: list[str]
    excluded_columns_dropped: list[str]
    ignored_columns_dropped: list[str]
    unexpected_columns: list[str]
    missing_before_cleaning: dict[str, int]
    missing_after_cleaning: dict[str, int]
    cleaning_actions: list[dict[str, Any]]
    target_distribution: dict[str, int]
    duplicate_canonical_rows: int | None
    issues: list[Issue]

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]


class DataValidationError(Exception):
    def __init__(self, report: ValidationReport):
        self.report = report
        lines = [f"Data validation failed with {len(report.errors)} error(s):"]
        lines += [f"  [{i.code}] {i.message}" for i in report.errors]
        super().__init__("\n".join(lines))


@dataclass
class PipelineResult:
    data: pd.DataFrame | None
    report: ValidationReport

    def raise_for_errors(self) -> None:
        if self.report.errors:
            raise DataValidationError(self.report)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_raw_csv(path: Path) -> pd.DataFrame:
    # Read as text so blanks and malformed numbers stay visible to validation.
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _examples(values: pd.Series) -> list[str]:
    return [str(v) for v in pd.unique(values)[:MAX_EXAMPLES]]


def _resolve_columns(
    raw_columns: list[str], schema: CanonicalSchema, mapping: SourceMapping
) -> tuple[dict[str, str], list[Issue]]:
    present = set(raw_columns)
    resolution: dict[str, str] = {}
    issues: list[Issue] = []
    required = {f.name for f in schema.fields if f.required} | {schema.target.name}

    for canonical, aliases in mapping.columns.items():
        found = [a for a in aliases if a in present]
        if len(found) == 1:
            resolution[canonical] = found[0]
        elif len(found) > 1:
            issues.append(
                Issue(
                    severity="error",
                    code="ambiguous_column",
                    field=canonical,
                    message=(
                        f"'{canonical}' matches several source columns {found}; keep exactly "
                        f"one alias for this source in columns.{canonical}."
                    ),
                )
            )
        elif canonical in required:
            issues.append(
                Issue(
                    severity="error",
                    code="missing_required_column",
                    field=canonical,
                    message=(
                        f"Required field '{canonical}' not found. Looked for {aliases}. "
                        f"Available columns: {sorted(present)}. If one of these is the same "
                        f"quantity, add it to columns.{canonical} in mapping "
                        f"'{mapping.mapping_name}'."
                    ),
                )
            )
        else:
            issues.append(
                Issue(
                    severity="warning",
                    code="optional_column_absent",
                    field=canonical,
                    message=f"Optional field '{canonical}' not present (looked for {aliases}).",
                )
            )

    for col in mapping.identifier_columns + mapping.excluded_columns + mapping.ignored_columns:
        if col not in present:
            issues.append(
                Issue(
                    severity="warning",
                    code="declared_column_absent",
                    source_column=col,
                    message=f"Column '{col}' is declared in the mapping but not in the file.",
                )
            )

    accounted = set(resolution.values()) | mapping.declared_non_feature_columns
    # Ambiguous aliases are already errors; don't report them a second time as unexpected.
    all_aliases = {a for aliases in mapping.columns.values() for a in aliases}
    unexpected = [c for c in raw_columns if c not in accounted and c not in all_aliases]
    for col in unexpected:
        forbidden = schema.is_forbidden_source(col)
        severity = "error" if mapping.unexpected_columns == "error" or forbidden else "warning"
        hint = (
            " Its name looks like an identifier or target-derived field; add it to "
            "identifier_columns or excluded_columns."
            if forbidden
            else " Map it, or list it under ignored_columns if it is intentionally unused."
        )
        issues.append(
            Issue(
                severity=severity,
                code="unexpected_column",
                source_column=col,
                message=f"Source column '{col}' is not accounted for in the mapping.{hint}",
            )
        )
    return resolution, issues


def _prepare_text(series: pd.Series, mapping: SourceMapping) -> pd.Series:
    out = series.astype("string")
    if mapping.cleaning.strip_whitespace:
        out = out.str.strip()
    if mapping.cleaning.blank_as_missing:
        out = out.mask(out == "")
    return out


def _convert_category(
    name: str, raw: pd.Series, allowed: set[str], vmap: dict[str, str] | None, source: str
) -> tuple[pd.Series, list[Issue]]:
    mapped = raw.map(vmap) if vmap is not None else raw
    invalid = raw.notna() & (mapped.isna() | ~mapped.isin(allowed))
    issues = []
    if invalid.any():
        issues.append(
            Issue(
                severity="error",
                code="invalid_category",
                field=name,
                source_column=source,
                count=int(invalid.sum()),
                examples=_examples(raw[invalid]),
                message=(
                    f"'{source}' has {int(invalid.sum())} value(s) with no mapping to "
                    f"{sorted(allowed)}. Add them to value_maps.{name} if they are "
                    "legitimate; otherwise fix the source."
                ),
            )
        )
    return mapped.where(~invalid).astype("string"), issues


def _convert_numeric(spec: FieldSpec, raw: pd.Series, source: str) -> tuple[pd.Series, list[Issue]]:
    parsed = pd.to_numeric(raw, errors="coerce")
    issues = []
    unparseable = raw.notna() & parsed.isna()
    if unparseable.any():
        issues.append(
            Issue(
                severity="error",
                code="non_numeric_value",
                field=spec.name,
                source_column=source,
                count=int(unparseable.sum()),
                examples=_examples(raw[unparseable]),
                message=f"'{source}' has {int(unparseable.sum())} non-numeric value(s).",
            )
        )
    if spec.dtype == "integer":
        fractional = parsed.notna() & (parsed != parsed.round())
        if fractional.any():
            issues.append(
                Issue(
                    severity="error",
                    code="non_integer_value",
                    field=spec.name,
                    source_column=source,
                    count=int(fractional.sum()),
                    examples=_examples(parsed[fractional]),
                    message=f"'{source}' must be whole numbers.",
                )
            )
            parsed = parsed.where(~fractional)
    return parsed.astype("Float64"), issues


def _range_issues(spec: FieldSpec, values: pd.Series, source: str) -> list[Issue]:
    issues = []
    checks = [
        ("below_min", spec.min, values < spec.min if spec.min is not None else None),
        ("above_max", spec.max, values > spec.max if spec.max is not None else None),
    ]
    for code, bound, mask in checks:
        if mask is None:
            continue
        mask = mask.fillna(False)
        if mask.any():
            issues.append(
                Issue(
                    severity="error",
                    code=code,
                    field=spec.name,
                    source_column=source,
                    count=int(mask.sum()),
                    examples=_examples(values[mask]),
                    message=(
                        f"'{spec.name}' has {int(mask.sum())} value(s) outside "
                        f"[{spec.min}, {spec.max}] ({code.replace('_', ' ')} {bound})."
                    ),
                )
            )
    return issues


def prepare_frame(
    raw: pd.DataFrame,
    schema: CanonicalSchema,
    mapping: SourceMapping,
    source_info: dict[str, Any] | None = None,
) -> PipelineResult:
    """Run the full map/clean/validate pipeline on an already-loaded raw table."""
    resolution, issues = _resolve_columns(raw.columns.tolist(), schema, mapping)
    raw_cols = set(raw.columns)
    report_kwargs: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "schema_version": schema.schema_version,
        "mapping_name": mapping.mapping_name,
        "mapping_version": mapping.mapping_version,
        "source": source_info or {},
        "rows_in": len(raw),
        "column_resolution": resolution,
        "identifier_columns_dropped": [c for c in mapping.identifier_columns if c in raw_cols],
        "excluded_columns_dropped": [c for c in mapping.excluded_columns if c in raw_cols],
        "ignored_columns_dropped": [c for c in mapping.ignored_columns if c in raw_cols],
        "unexpected_columns": [i.source_column for i in issues if i.code == "unexpected_column"],
    }

    structural_error = any(
        i.severity == "error" and i.code in ("missing_required_column", "ambiguous_column")
        for i in issues
    )
    if structural_error:
        # Values can't be validated reliably without a complete column mapping.
        report = ValidationReport(
            status="failed",
            rows_out=None,
            missing_before_cleaning={},
            missing_after_cleaning={},
            cleaning_actions=[],
            target_distribution={},
            duplicate_canonical_rows=None,
            issues=issues,
            **report_kwargs,
        )
        return PipelineResult(data=None, report=report)

    vmaps = mapping.field_value_maps
    text = {name: _prepare_text(raw[src], mapping) for name, src in resolution.items()}
    missing_before = {name: int(s.isna().sum()) for name, s in text.items()}

    out: dict[str, pd.Series] = {}
    for spec in schema.fields:
        if spec.name not in resolution:
            continue
        src = resolution[spec.name]
        if spec.is_numeric:
            out[spec.name], found = _convert_numeric(spec, text[spec.name], src)
        else:
            out[spec.name], found = _convert_category(
                spec.name,
                text[spec.name],
                set(spec.allowed_values or []),
                vmaps.get(spec.name),
                src,
            )
        issues += found

    cleaning_actions = []
    for rule in mapping.cleaning.fill_rules:
        if rule.field not in out or rule.when_field not in out:
            continue
        mask = out[rule.field].isna() & (out[rule.when_field] == rule.when_equals).fillna(False)
        out[rule.field] = out[rule.field].mask(mask, rule.value)
        cleaning_actions.append(
            {
                "rule": (
                    f"fill {rule.field}={rule.value} when {rule.when_field}=={rule.when_equals}"
                ),
                "rows_affected": int(mask.sum()),
            }
        )

    n = len(raw)
    for spec in schema.fields:
        if spec.name not in out:
            continue
        col = out[spec.name]
        src = resolution[spec.name]
        if spec.is_numeric:
            issues += _range_issues(spec, col, src)
            if spec.dtype == "integer":
                out[spec.name] = col.round().astype("Int64")
        n_missing = int(col.isna().sum())
        rate = n_missing / n if n else 0.0
        if n_missing:
            over = rate > spec.max_missing_rate
            issues.append(
                Issue(
                    severity="error" if over else "warning",
                    code="missing_rate_exceeded" if over else "missing_values",
                    field=spec.name,
                    source_column=src,
                    count=n_missing,
                    message=(
                        f"'{spec.name}' has {n_missing} missing value(s) after cleaning "
                        f"({rate:.2%}; limit {spec.max_missing_rate:.2%})."
                    ),
                )
            )

    tgt = schema.target
    target_raw = text[tgt.name]
    target_tokens, found = _convert_category(
        tgt.name,
        target_raw,
        {tgt.positive_value, tgt.negative_value},
        vmaps.get(tgt.name),
        resolution[tgt.name],
    )
    issues += found
    target_missing = target_raw.isna()
    if target_missing.any():
        issues.append(
            Issue(
                severity="error",
                code="missing_target",
                field=tgt.name,
                source_column=resolution[tgt.name],
                count=int(target_missing.sum()),
                message=f"Target '{tgt.name}' is missing in {int(target_missing.sum())} row(s).",
            )
        )
    out[tgt.name] = (target_tokens == tgt.positive_value).astype("Int64").mask(target_tokens.isna())

    data = pd.DataFrame(out)
    allowed_cols = set(schema.field_names) | {tgt.name}
    # Belt and braces: the output can only contain canonical names, so no source
    # identifier or excluded column can slip through under its original name.
    assert set(data.columns) <= allowed_cols

    errors = [i for i in issues if i.severity == "error"]
    report = ValidationReport(
        status="failed" if errors else "passed",
        rows_out=len(data),
        missing_before_cleaning=missing_before,
        missing_after_cleaning={c: int(data[c].isna().sum()) for c in data.columns},
        cleaning_actions=cleaning_actions,
        target_distribution={
            str(k): int(v) for k, v in data[tgt.name].value_counts(dropna=False).items()
        },
        duplicate_canonical_rows=int(data.duplicated().sum()),
        issues=issues,
        **report_kwargs,
    )
    return PipelineResult(data=data, report=report)


def prepare_csv(path: Path, schema: CanonicalSchema, mapping: SourceMapping) -> PipelineResult:
    path = Path(path)
    raw = read_raw_csv(path)
    info = {"path": path.as_posix(), "sha256": file_sha256(path), "rows": len(raw)}
    return prepare_frame(raw, schema, mapping, source_info=info)
