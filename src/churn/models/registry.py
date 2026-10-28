"""Local file-based model registry.

Layout:
    <registry_dir>/<version>/model.joblib       fitted sklearn Pipeline
    <registry_dir>/<version>/metadata.json      versions, features, threshold, metrics, hashes
    <registry_dir>/<version>/...                reports saved alongside the model
    <registry_dir>/current.json                 which version is served (changed only by promote)
    <registry_dir>/promotions.jsonl             append-only audit log of promotion decisions

joblib files are pickles: loading one executes code. Only load artifacts this project
produced; the stored SHA-256 is checked before every load to catch tampering or corruption.
"""

from __future__ import annotations

import getpass
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"
REFERENCE_PROFILE_FILE = "reference_profile.json"
CURRENT_FILE = "current.json"
AUDIT_FILE = "promotions.jsonl"


class RegistryError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def new_version_id(fingerprint: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{fingerprint[:8]}"


def save_version(
    registry_dir: Path,
    version: str,
    pipeline: Any,
    metadata: dict[str, Any],
    files: dict[str, str] | None = None,
) -> Path:
    vdir = Path(registry_dir) / version
    if vdir.exists():
        raise RegistryError(f"version {version} already exists; versions are immutable")
    vdir.mkdir(parents=True)
    joblib.dump(pipeline, vdir / MODEL_FILE)
    metadata = {**metadata, "model_version": version, "model_sha256": _sha256(vdir / MODEL_FILE)}
    (vdir / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for name, content in (files or {}).items():
        (vdir / name).write_text(content, encoding="utf-8")
    return vdir


def read_metadata(registry_dir: Path, version: str) -> dict[str, Any]:
    path = Path(registry_dir) / version / METADATA_FILE
    if not path.exists():
        raise RegistryError(f"no metadata for version '{version}' in {registry_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_version(registry_dir: Path, version: str) -> tuple[Any, dict[str, Any]]:
    metadata = read_metadata(registry_dir, version)
    model_path = Path(registry_dir) / version / MODEL_FILE
    if not model_path.exists():
        raise RegistryError(f"model file missing for version '{version}'")
    actual = _sha256(model_path)
    if actual != metadata.get("model_sha256"):
        raise RegistryError(
            f"model file hash mismatch for '{version}' (expected "
            f"{metadata.get('model_sha256')}, got {actual}); refusing to load"
        )
    return joblib.load(model_path), metadata


def list_versions(registry_dir: Path) -> list[str]:
    root = Path(registry_dir)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if (p / METADATA_FILE).exists())


def read_current(registry_dir: Path) -> dict[str, Any] | None:
    path = Path(registry_dir) / CURRENT_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def record_decision(registry_dir: Path, decision: dict[str, Any]) -> None:
    entry = {
        "recorded_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "actor": getpass.getuser(),
        **decision,
    }
    with (Path(registry_dir) / AUDIT_FILE).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def set_current(registry_dir: Path, version: str, reason: str) -> dict[str, Any]:
    """Point serving at `version`. Callers are responsible for checking acceptance criteria."""
    read_metadata(registry_dir, version)  # fail early if the version doesn't exist
    previous = read_current(registry_dir)
    pointer = {
        "version": version,
        "promoted_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "reason": reason,
        "previous_version": previous["version"] if previous else None,
    }
    # Write-then-rename so a reader never sees a half-written pointer.
    tmp = Path(registry_dir) / (CURRENT_FILE + ".tmp")
    tmp.write_text(json.dumps(pointer, indent=2), encoding="utf-8")
    tmp.replace(Path(registry_dir) / CURRENT_FILE)
    return pointer


def read_reference_profile(registry_dir: Path, version: str) -> dict[str, Any] | None:
    """Training-time input/score profile, or None for versions trained before it existed."""
    path = Path(registry_dir) / version / REFERENCE_PROFILE_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
