"""Point the serving model at a registered version. The only command that changes it.

Usage:
    python scripts/promote.py --list
    python scripts/promote.py --version <version> --reason "initial baseline"

Every promotion is appended to <registry>/promotions.jsonl. Automated acceptance checks
for retrained candidates are added in the retraining workflow (Phase 8); this command is
the manual path and always requires a stated reason.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from churn.models import registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--registry", type=Path, default=Path("models/registry"))
    parser.add_argument("--version")
    parser.add_argument("--reason")
    parser.add_argument("--list", action="store_true", help="list versions and exit")
    args = parser.parse_args(argv)

    current = registry.read_current(args.registry)
    if args.list:
        for v in registry.list_versions(args.registry):
            meta = registry.read_metadata(args.registry, v)
            test = meta["metrics"]["test"]
            marker = "*" if current and current["version"] == v else " "
            print(
                f"{marker} {v}  {meta['model_type']:20} test PR-AUC "
                f"{test['average_precision']:.3f}  thr {meta['threshold']['threshold']:.3f}"
            )
        return 0

    if not args.version or not args.reason:
        parser.error("--version and --reason are required to promote")
    try:
        # Loading verifies the artifact hash before anything points at it.
        registry.load_version(args.registry, args.version)
    except registry.RegistryError as exc:
        print(f"Cannot promote: {exc}", file=sys.stderr)
        return 1

    pointer = registry.set_current(args.registry, args.version, args.reason)
    registry.record_decision(
        args.registry,
        {
            "decision": "promoted",
            "mode": "manual",
            "version": args.version,
            "previous_version": pointer["previous_version"],
            "reason": args.reason,
        },
    )
    print(f"Serving version is now {args.version} (was {pointer['previous_version']}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
