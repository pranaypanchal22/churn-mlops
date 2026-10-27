"""Train, evaluate, and register a candidate churn model.

Usage:
    python scripts/train.py [--config configs/train.yaml]

The new version is registered but NOT served. Promote it explicitly with
    python scripts/promote.py --version <version> --reason "..."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from churn.data.pipeline import DataValidationError
from churn.features.build import LeakageError
from churn.models.train import train
from churn.schema.config import ConfigError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    args = parser.parse_args(argv)

    try:
        result = train(args.config)
    except (ConfigError, DataValidationError, LeakageError, FileNotFoundError) as exc:
        print(f"Training aborted: {exc}", file=sys.stderr)
        return 1

    m = result.metadata
    val, test = m["metrics"]["validation"], m["metrics"]["test"]
    print(f"Registered {result.version} ({m['model_type']}) at {result.version_dir}")
    print(f"Threshold {m['threshold']['threshold']:.4f} ({m['threshold']['strategy']})")
    for name, split in (("validation", val), ("test", test)):
        print(
            f"  {name:10} ROC-AUC {split['roc_auc']:.3f}  PR-AUC {split['average_precision']:.3f}"
            f"  P {split['precision']:.3f}  R {split['recall']:.3f}  F1 {split['f1']:.3f}"
        )
    print(f"Report: {result.version_dir / 'evaluation.md'}")
    print("Not promoted. Review, then run scripts/promote.py to serve it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
