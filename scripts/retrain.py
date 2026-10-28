"""Retrain a candidate on new data and promote it only if it passes the acceptance policy.

Usage:
    python scripts/retrain.py --data <new.csv> [--config configs/train.yaml]
                              [--policy configs/promotion.yaml] [--promote --reason "..."]

Without --promote this is a dry run: the candidate is registered and judged, the decision is
recorded, and the served model is untouched. A scheduler should call it WITHOUT --promote.

Exit codes: 0 promoted/eligible/no_change, 1 rejected, 2 config problem.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from churn.data.pipeline import DataValidationError
from churn.models.registry import RegistryError
from churn.models.retrain import retrain
from churn.schema.config import ConfigError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, help="new raw CSV (default: path in --config)")
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--policy", type=Path, default=Path("configs/promotion.yaml"))
    parser.add_argument("--promote", action="store_true", help="promote if all criteria pass")
    parser.add_argument("--reason", help="note stored with the promotion")
    parser.add_argument("--runs-dir", type=Path, default=Path("reports/retraining"))
    args = parser.parse_args(argv)

    if args.data is not None and not args.data.exists():
        print(f"Data not found: {args.data}", file=sys.stderr)
        return 2
    try:
        d = retrain(args.config, args.policy, args.data, args.promote, args.runs_dir, args.reason)
    except (ConfigError, RegistryError, DataValidationError, ValueError, OSError) as exc:
        print(f"Retraining aborted: {exc}", file=sys.stderr)
        return 2

    print(f"Outcome: {d['outcome'].upper()} - {d['summary']}")
    if d.get("candidate_version"):
        print(f"  candidate {d['candidate_version']}  current {d.get('current_version')}")
    cmp = d.get("comparison")
    if cmp:
        cand, cur = cmp["candidate"], cmp["current"] or {}
        for k in ("average_precision", "roc_auc", "recall", "precision", "brier_score"):
            print(f"  {k:18} candidate {cand[k]:.4f}  current {cur.get(k, float('nan')):.4f}")
    for c in d.get("criteria", []):
        mark = "PASS" if c["passed"] else "FAIL"
        print(f"  [{mark}] {c['criterion']}: {c['value']} (limit {c['limit']})")
    print(f"Decision record: {args.runs_dir / d['run_id'] / 'decision.json'}")
    return 1 if d["outcome"] == "rejected" else 0


if __name__ == "__main__":
    raise SystemExit(main())
