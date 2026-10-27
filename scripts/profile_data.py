"""Profile a raw churn CSV without modifying it.

Usage:
    python scripts/profile_data.py [--input data/raw/CustomerChurn.csv] [--out-dir reports]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from churn.data.profile import profile_csv, render_markdown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=Path("data/raw/CustomerChurn.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    parser.add_argument("--target", default="Churn")
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(
            f"Dataset not found at {args.input}. Download the Kaggle archive and copy "
            "CustomerChurn.csv into data/raw/ (see README > Data).",
            file=sys.stderr,
        )
        return 2

    profile = profile_csv(args.input, target=args.target)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "data_profile.json"
    md_path = args.out_dir / "data_profile.md"
    json_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(profile), encoding="utf-8")

    print(f"Rows: {profile['shape']['rows']}, columns: {profile['shape']['columns']}")
    print(f"Wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
