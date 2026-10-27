"""Read-only profiling of a raw churn CSV.

The profile is computed from the file as-is: every column is read as a string so that
blanks and malformed numbers stay visible instead of being coerced away by pandas.
Nothing here writes to or modifies the source file.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

# A column is "numeric-like" if at least this share of its non-blank values parse as numbers.
# Below this it is treated as categorical, and non-parseable values are not flagged.
NUMERIC_LIKE_MIN_SHARE = 0.95

# Columns with at most this many distinct values get full value counts in the report.
MAX_LEVELS_REPORTED = 20

# Name patterns that suggest a column is derived from the churn outcome. Matches are
# candidates for manual review, not automatic verdicts.
LEAKAGE_NAME_PATTERNS = [
    r"churn",
    r"reason",
    r"score",
    r"cltv",
    r"lifetime\s*value",
    r"customer\s*status",
    r"category",
]

ID_NAME_PATTERN = re.compile(r"(^|[\s_])id$|^id[\s_]|customer\s*id|loyalty", re.IGNORECASE)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_blank(series: pd.Series) -> pd.Series:
    return series.str.strip().eq("")


def _profile_column(series: pd.Series, n_rows: int) -> dict[str, Any]:
    blank = _is_blank(series)
    non_blank = series[~blank]
    parsed = pd.to_numeric(non_blank.str.strip(), errors="coerce")
    numeric_share = float(parsed.notna().mean()) if len(non_blank) else 0.0
    n_unique = int(series.nunique(dropna=False))

    info: dict[str, Any] = {
        "n_blank": int(blank.sum()),
        "blank_rate": round(float(blank.mean()), 6),
        "n_unique": n_unique,
        "unique_per_row": n_unique == n_rows,
        "has_surrounding_whitespace": bool((series != series.str.strip()).any()),
    }

    if numeric_share >= NUMERIC_LIKE_MIN_SHARE:
        valid = parsed.dropna()
        bad = non_blank[parsed.isna()]
        is_integer = bool((valid == valid.round()).all())
        info.update(
            {
                "inferred_type": "integer" if is_integer else "float",
                "n_non_numeric": int(len(bad)),
                "non_numeric_examples": sorted(bad.unique().tolist())[:10],
                "min": float(valid.min()),
                "max": float(valid.max()),
                "mean": round(float(valid.mean()), 4),
                "median": float(valid.median()),
                "n_negative": int((valid < 0).sum()),
                "n_zero": int((valid == 0).sum()),
            }
        )
    else:
        info["inferred_type"] = "categorical"

    # Value counts are only useful (and only safe to publish) for low-cardinality columns;
    # high-cardinality columns are typically identifiers.
    if n_unique <= MAX_LEVELS_REPORTED:
        info["value_counts"] = {str(k): int(v) for k, v in series.value_counts().items()}
    return info


def _leakage_candidates(columns: list[str], target: str) -> list[str]:
    hits = []
    for col in columns:
        if col == target:
            continue
        if any(re.search(p, col, re.IGNORECASE) for p in LEAKAGE_NAME_PATTERNS):
            hits.append(col)
    return hits


def _identifier_candidates(df: pd.DataFrame, col_info: dict[str, dict[str, Any]]) -> list[dict]:
    out = []
    for col in df.columns:
        info = col_info[col]
        name_match = bool(ID_NAME_PATTERN.search(col))
        if info["unique_per_row"] or name_match:
            out.append(
                {
                    "column": col,
                    "unique_per_row": info["unique_per_row"],
                    "name_suggests_id": name_match,
                }
            )
    return out


def _total_charges_checks(
    df: pd.DataFrame, total_col: str, tenure_col: str, monthly_col: str
) -> dict[str, Any]:
    """Dataset-specific sanity checks for the cumulative-charges field."""
    total = pd.to_numeric(df[total_col].str.strip(), errors="coerce")
    tenure = pd.to_numeric(df[tenure_col].str.strip(), errors="coerce")
    monthly = pd.to_numeric(df[monthly_col].str.strip(), errors="coerce")
    unparseable = total.isna()

    tenure_of_unparseable = tenure[unparseable].value_counts().sort_index()
    both = total.notna() & tenure.notna() & monthly.notna() & (tenure > 0)
    # TotalCharges should be roughly tenure * monthly; large deviations hint at bad rows,
    # but plan changes over time make exact equality unrealistic.
    ratio = total[both] / (tenure[both] * monthly[both])

    return {
        "column": total_col,
        "n_unparseable": int(unparseable.sum()),
        "tenure_values_where_unparseable": {
            str(int(k)): int(v) for k, v in tenure_of_unparseable.items()
        },
        "n_tenure_zero": int((tenure == 0).sum()),
        "n_total_less_than_monthly_with_tenure_ge_1": int(
            ((total < monthly) & (tenure >= 1)).sum()
        ),
        "ratio_total_to_tenure_x_monthly": {
            "p01": round(float(ratio.quantile(0.01)), 4),
            "median": round(float(ratio.median()), 4),
            "p99": round(float(ratio.quantile(0.99)), 4),
        },
    }


def profile_csv(
    path: Path,
    target: str = "Churn",
    total_charges_col: str | None = "Total Charges",
    tenure_col: str = "Tenure",
    monthly_col: str = "Monthly Charges",
) -> dict[str, Any]:
    path = Path(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    n_rows = len(df)
    columns = df.columns.tolist()

    col_info = {col: _profile_column(df[col], n_rows) for col in columns}
    id_candidates = _identifier_candidates(df, col_info)
    id_cols = [c["column"] for c in id_candidates]

    profile: dict[str, Any] = {
        "source": {
            "path": path.as_posix(),
            "sha256": sha256_of(path),
            "size_bytes": path.stat().st_size,
            "profiled_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        },
        "shape": {"rows": n_rows, "columns": len(columns)},
        "columns": columns,
        "duplicates": {
            "full_row_duplicates": int(df.duplicated().sum()),
            # Customers can share identical attributes, so duplicates after dropping IDs are
            # reported for awareness rather than removed.
            "duplicates_excluding_id_columns": int(df.drop(columns=id_cols).duplicated().sum()),
            "duplicate_values_in_id_columns": {
                c["column"]: int(df[c["column"]].duplicated().sum()) for c in id_candidates
            },
        },
        "identifier_candidates": id_candidates,
        "leakage_candidates_by_name": _leakage_candidates(columns, target),
        "column_profiles": col_info,
    }

    if target in df.columns:
        counts = df[target].value_counts()
        profile["target"] = {
            "column": target,
            "values": {str(k): int(v) for k, v in counts.items()},
            "proportions": {str(k): round(float(v / n_rows), 4) for k, v in counts.items()},
        }
    else:
        profile["target"] = {"column": target, "error": "target column not found"}

    needed = {total_charges_col, tenure_col, monthly_col}
    if total_charges_col and needed.issubset(df.columns):
        profile["total_charges_checks"] = _total_charges_checks(
            df, total_charges_col, tenure_col, monthly_col
        )
    return profile


def render_markdown(profile: dict[str, Any]) -> str:
    src, shape = profile["source"], profile["shape"]
    lines = [
        "# Raw data profile",
        "",
        f"- File: `{src['path']}`",
        f"- SHA-256: `{src['sha256']}`",
        f"- Size: {src['size_bytes']:,} bytes",
        f"- Profiled at (UTC): {src['profiled_at_utc']}",
        f"- Rows: {shape['rows']:,}  |  Columns: {shape['columns']}",
        "",
        "## Target",
        "",
    ]
    tgt = profile["target"]
    if "error" in tgt:
        lines.append(f"**{tgt['error']}** (`{tgt['column']}`)")
    else:
        lines += ["| Value | Count | Share |", "|---|---:|---:|"]
        for k, v in tgt["values"].items():
            lines.append(f"| `{k}` | {v:,} | {tgt['proportions'][k]:.2%} |")

    dup = profile["duplicates"]
    lines += [
        "",
        "## Duplicates",
        "",
        f"- Full-row duplicates: {dup['full_row_duplicates']}",
        f"- Duplicates after dropping identifier columns: {dup['duplicates_excluding_id_columns']}",
        *(
            f"- Repeated values in `{c}`: {n}"
            for c, n in dup["duplicate_values_in_id_columns"].items()
        ),
        "",
        "## Identifier candidates (never model features)",
        "",
    ]
    for c in profile["identifier_candidates"]:
        lines.append(
            f"- `{c['column']}` (unique per row: {c['unique_per_row']}, "
            f"name suggests ID: {c['name_suggests_id']})"
        )
    leak = profile["leakage_candidates_by_name"]
    lines += [
        "",
        "## Target-derived / leakage candidates (by name)",
        "",
        *(f"- `{c}`" for c in leak),
    ]
    if not leak:
        lines.append("- None found in this file.")

    lines += [
        "",
        "## Columns",
        "",
        "| Column | Type | Blank | Unique | Non-numeric | Min | Max | Levels |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for col, info in profile["column_profiles"].items():
        levels = ""
        if "value_counts" in info and info["inferred_type"] == "categorical":
            levels = ", ".join(f"{k} ({v})" for k, v in info["value_counts"].items())
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} |".format(
                col,
                info["inferred_type"],
                info["n_blank"],
                info["n_unique"],
                info.get("n_non_numeric", ""),
                info.get("min", ""),
                info.get("max", ""),
                levels,
            )
        )

    tc = profile.get("total_charges_checks")
    if tc:
        r = tc["ratio_total_to_tenure_x_monthly"]
        lines += [
            "",
            f"## `{tc['column']}` checks",
            "",
            f"- Blank or non-numeric values: {tc['n_unparseable']}",
            f"- Tenure of those rows: {tc['tenure_values_where_unparseable']}",
            f"- Rows with tenure = 0: {tc['n_tenure_zero']}",
            f"- Rows where total < monthly with tenure >= 1: "
            f"{tc['n_total_less_than_monthly_with_tenure_ge_1']}",
            f"- total / (tenure x monthly): p01={r['p01']}, median={r['median']}, p99={r['p99']}",
        ]
    return "\n".join(lines) + "\n"
