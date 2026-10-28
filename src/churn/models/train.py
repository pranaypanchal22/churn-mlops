"""Reproducible training run: validate data, compare candidates, pick a threshold, test once.

Data use, in order:
  train      - fit preprocessing + each candidate model
  validation - compare candidates, choose the decision threshold, compute model signals
  test       - one evaluation of the selected model at the chosen threshold; never used
               to make a choice
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import sklearn
import yaml
from pydantic import BaseModel, ConfigDict, Field
from sklearn.model_selection import train_test_split

from churn.data.pipeline import prepare_csv
from churn.features.build import (
    assert_no_leakage,
    build_model_pipeline,
    feature_set_from_schema,
    to_model_frame,
)
from churn.models import registry
from churn.models.evaluate import (
    choose_threshold,
    permutation_signals,
    pr_curve_frame,
    ranking_metrics,
    threshold_metrics,
)
from churn.models.registry import REFERENCE_PROFILE_FILE
from churn.monitoring.drift import build_reference_profile
from churn.schema.config import CanonicalSchema, load_mapping, load_schema


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataCfg(_Strict):
    raw_path: Path
    schema_path: Path
    mapping_path: Path


class SplitCfg(_Strict):
    validation_size: float = Field(gt=0, lt=0.5)
    test_size: float = Field(gt=0, lt=0.5)


class ThresholdCfg(_Strict):
    strategy: Literal["precision_at_min_recall", "max_f1"] = "precision_at_min_recall"
    min_recall: float = Field(default=0.7, gt=0, le=1)


class ExplanationsCfg(_Strict):
    permutation_repeats: int = Field(default=5, ge=1)


class OutputCfg(_Strict):
    registry_dir: Path = Path("models/registry")
    reports_dir: Path = Path("reports")


class MlflowCfg(_Strict):
    enabled: bool = False
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "churn"


class TrainConfig(_Strict):
    config_version: str
    random_seed: int = 42
    data: DataCfg
    split: SplitCfg
    selection_metric: Literal["average_precision", "roc_auc"] = "average_precision"
    models: dict[str, dict[str, Any]]
    threshold: ThresholdCfg = ThresholdCfg()
    explanations: ExplanationsCfg = ExplanationsCfg()
    output: OutputCfg = OutputCfg()
    mlflow: MlflowCfg = MlflowCfg()


def load_train_config(path: Path) -> tuple[TrainConfig, str]:
    text = Path(path).read_text(encoding="utf-8")
    return TrainConfig.model_validate(yaml.safe_load(text)), text


@dataclass
class TrainResult:
    version: str
    version_dir: Path
    metadata: dict[str, Any]


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def split_data(
    X: pd.DataFrame, y: pd.Series, cfg: SplitCfg, seed: int
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=seed
    )
    # validation_size is a fraction of the full dataset, so rescale for the remainder.
    val_frac = cfg.validation_size / (1 - cfg.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_frac, stratify=y_rest, random_state=seed
    )
    return {"train": (X_train, y_train), "validation": (X_val, y_val), "test": (X_test, y_test)}


def input_schema_summary(schema: CanonicalSchema) -> list[dict[str, Any]]:
    return [
        {
            "name": f.name,
            "dtype": f.dtype,
            "allowed_values": f.allowed_values,
            "min": f.min,
            "max": f.max,
            "description": f.description,
        }
        for f in schema.fields
        if f.role == "feature"
    ]


def _log_mlflow(cfg: MlflowCfg, metadata: dict[str, Any], version_dir: Path) -> str | None:
    if not cfg.enabled:
        return None
    try:
        import mlflow
    except ImportError:
        print("mlflow not installed; skipping tracking (pip install -e '.[tracking]').")
        return None
    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)
    with mlflow.start_run(run_name=metadata["model_version"]) as run:
        mlflow.set_tags(
            {
                "model_version": metadata["model_version"],
                "selected_model": metadata["model_type"],
                "schema_version": metadata["data"]["schema_version"],
                "mapping": f"{metadata['data']['mapping_name']}@"
                f"{metadata['data']['mapping_version']}",
                "git_commit": str(metadata["reproducibility"]["git"]["commit"]),
            }
        )
        mlflow.log_params(
            {
                "seed": metadata["reproducibility"]["random_seed"],
                "threshold_strategy": metadata["threshold"]["strategy"],
                "threshold": metadata["threshold"]["threshold"],
                **{f"{metadata['model_type']}.{k}": v for k, v in metadata["model_params"].items()},
            }
        )
        for name, cand in metadata["candidates"].items():
            for k in ("roc_auc", "average_precision", "brier_score"):
                mlflow.log_metric(f"val_{name}_{k}", cand["validation"][k])
        for split in ("validation", "test"):
            sel = metadata["metrics"][split]
            for k in ("roc_auc", "average_precision", "precision", "recall", "f1"):
                mlflow.log_metric(f"{split}_{k}", sel[k])
        # The registry stays the source of truth; MLflow gets a copy of the artifacts.
        mlflow.log_artifacts(str(version_dir))
        return run.info.run_id


def _evaluation_markdown(meta: dict[str, Any]) -> str:
    def fmt(x):
        return f"{x:.3f}" if isinstance(x, float) else str(x)

    lines = [
        f"# Evaluation: {meta['model_version']}",
        "",
        f"- Selected model: **{meta['model_type']}** (by validation {meta['selection_metric']})",
        f"- Threshold: **{meta['threshold']['threshold']:.4f}** "
        f"({meta['threshold']['strategy']}, min_recall={meta['threshold']['min_recall']}, "
        "chosen on validation)",
        f"- Data: {meta['data']['mapping_name']} v{meta['data']['mapping_version']}, "
        f"schema {meta['data']['schema_version']}, sha256 {meta['data']['source_sha256'][:12]}…",
        f"- Split sizes: {meta['split']['sizes']}",
        "",
        "## Candidates (validation)",
        "",
        "| Model | ROC-AUC | PR-AUC | Brier | P@0.5 | R@0.5 | F1@0.5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, c in meta["candidates"].items():
        v, t = c["validation"], c["validation_at_0_5"]
        lines.append(
            f"| {name} | {fmt(v['roc_auc'])} | {fmt(v['average_precision'])} | "
            f"{fmt(v['brier_score'])} | {fmt(t['precision'])} | {fmt(t['recall'])} | "
            f"{fmt(t['f1'])} |"
        )
    lines += [
        "",
        "## Selected model at chosen threshold",
        "",
        "| Split | ROC-AUC | PR-AUC | Precision | Recall | F1 | Flagged | TP | FP | FN | TN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("validation", "test"):
        m = meta["metrics"][split]
        cm = m["confusion_matrix"]
        lines.append(
            f"| {split} | {fmt(m['roc_auc'])} | {fmt(m['average_precision'])} | "
            f"{fmt(m['precision'])} | {fmt(m['recall'])} | {fmt(m['f1'])} | "
            f"{m['flagged_rate']:.1%} | {cm['tp']} | {cm['fp']} | {cm['fn']} | {cm['tn']} |"
        )
    lines += [
        "",
        "Test metrics were computed once, after model and threshold were fixed.",
        "",
        "## Model-level signals (permutation importance, validation PR-AUC drop)",
        "",
        "Associational, not causal. Correlated inputs share credit.",
        "",
        "| Feature | Mean drop | Std |",
        "|---|---:|---:|",
    ]
    for s in meta["model_signals"][:10]:
        lines.append(f"| {s['feature']} | {s['mean_drop']:.4f} | {s['std']:.4f} |")
    return "\n".join(lines) + "\n"


def train(config_path: Path) -> TrainResult:
    cfg, cfg_text = load_train_config(config_path)
    schema = load_schema(cfg.data.schema_path)
    mapping = load_mapping(cfg.data.mapping_path, schema)

    prepared = prepare_csv(cfg.data.raw_path, schema, mapping)
    prepared.raise_for_errors()
    data = prepared.data

    features = feature_set_from_schema(schema)
    assert_no_leakage(features.all, schema, mapping)
    X = to_model_frame(data, features)
    y = data[schema.target.name].astype(int)
    splits = split_data(X, y, cfg.split, cfg.random_seed)
    X_tr, y_tr = splits["train"]
    X_val, y_val = splits["validation"]
    X_te, y_te = splits["test"]

    candidates: dict[str, dict[str, Any]] = {}
    fitted = {}
    for name, params in cfg.models.items():
        pipe = build_model_pipeline(name, params, features, cfg.random_seed)
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_val)[:, 1]
        fitted[name] = (pipe, proba)
        candidates[name] = {
            "params": params,
            "validation": ranking_metrics(y_val.to_numpy(), proba),
            "validation_at_0_5": threshold_metrics(y_val.to_numpy(), proba, 0.5),
        }

    selected = max(candidates, key=lambda n: candidates[n]["validation"][cfg.selection_metric])
    pipe, val_proba = fitted[selected]
    thr = choose_threshold(
        y_val.to_numpy(), val_proba, cfg.threshold.strategy, cfg.threshold.min_recall
    )

    # The only use of the test set: one evaluation after every choice above is fixed.
    test_proba = pipe.predict_proba(X_te)[:, 1]
    metrics = {
        "validation": {
            **ranking_metrics(y_val.to_numpy(), val_proba),
            **threshold_metrics(y_val.to_numpy(), val_proba, thr["threshold"]),
        },
        "test": {
            **ranking_metrics(y_te.to_numpy(), test_proba),
            **threshold_metrics(y_te.to_numpy(), test_proba, thr["threshold"]),
        },
    }
    signals = permutation_signals(
        pipe, X_val, y_val.to_numpy(), cfg.explanations.permutation_repeats, cfg.random_seed
    )

    report = prepared.report
    source_sha = report.source["sha256"]
    cfg_sha = hashlib.sha256(cfg_text.encode()).hexdigest()
    version = registry.new_version_id(hashlib.sha256((source_sha + cfg_sha).encode()).hexdigest())
    metadata: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_type": selected,
        "model_params": cfg.models[selected],
        "selection_metric": cfg.selection_metric,
        "target": {"name": schema.target.name, "positive_label": schema.target.positive_value},
        "features": {
            "numeric": features.numeric,
            "categorical": features.categorical,
            "order": features.all,
        },
        "input_schema": input_schema_summary(schema),
        "threshold": thr,
        "metrics": metrics,
        "candidates": candidates,
        "model_signals": signals,
        "split": {
            "validation_size": cfg.split.validation_size,
            "test_size": cfg.split.test_size,
            "stratified": True,
            "sizes": {k: len(v[1]) for k, v in splits.items()},
            "positive_rate": {k: round(float(v[1].mean()), 4) for k, v in splits.items()},
        },
        "data": {
            "source_path": report.source["path"],
            "source_sha256": source_sha,
            "rows": report.rows_out,
            "schema_version": schema.schema_version,
            "mapping_name": mapping.mapping_name,
            "mapping_version": mapping.mapping_version,
            "identifier_columns_dropped": report.identifier_columns_dropped,
        },
        "reproducibility": {
            "random_seed": cfg.random_seed,
            "train_config_sha256": cfg_sha,
            "git": _git_state(),
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }

    files = {
        "train_config.yaml": cfg_text,
        "validation_report.json": report.model_dump_json(indent=2),
        "pr_curve_validation.csv": pr_curve_frame(y_val.to_numpy(), val_proba).to_csv(index=False),
        "pr_curve_test.csv": pr_curve_frame(y_te.to_numpy(), test_proba).to_csv(index=False),
        # Aggregate-only snapshot of training inputs and validation scores; the API and
        # drift reports compare incoming data against it.
        REFERENCE_PROFILE_FILE: json.dumps(
            build_reference_profile(X_tr, features, val_proba, thr["threshold"]), indent=2
        ),
    }
    vdir = registry.save_version(cfg.output.registry_dir, version, pipe, metadata, files)
    metadata = registry.read_metadata(cfg.output.registry_dir, version)
    (vdir / "evaluation.md").write_text(_evaluation_markdown(metadata), encoding="utf-8")

    # Tracking is a convenience copy; a tracking failure must not lose a registered model.
    try:
        run_id = _log_mlflow(cfg.mlflow, metadata, vdir)
    except Exception as exc:  # noqa: BLE001
        print(f"warning: MLflow logging failed, model is still registered: {exc}")
        run_id = None
    if run_id:
        (vdir / "mlflow_run.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    return TrainResult(version=version, version_dir=vdir, metadata=metadata)
