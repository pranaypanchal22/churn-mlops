"""Write a synthetic CSV in the CustomerChurn.csv layout (for CI and demos without the dataset).

Usage:
    python scripts/make_synthetic_data.py [--out data/synthetic/CustomerChurn_synthetic.csv]
                                          [--rows 1000] [--seed 0]

The rows are random with a planted signal. Never report model quality from them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from churn.data.synthetic import write_synthetic_csv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", type=Path, default=Path("data/synthetic/CustomerChurn_synthetic.csv")
    )
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    path = write_synthetic_csv(args.out, n=args.rows, seed=args.seed)
    print(f"Wrote {args.rows} synthetic rows to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
