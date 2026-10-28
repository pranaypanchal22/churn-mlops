"""Controlled retraining: validate -> train candidate -> compare -> gate -> (maybe) promote.

Promotion happens only when every acceptance criterion passes AND the caller explicitly
asked for it. Every run, including rejected and dry runs, writes a decision record and
appends to the registry's audit log.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from churn.data.pipeline import prepare_csv
from churn.features.build import FeatureSet, feature_set_from_schema, to_model_frame
from churn.models import registry
from churn.models.evaluate import ranking_metrics, threshold_metrics
from churn.models.train import load_train_config, split_data, train
from churn.schema.config import load_mapping, load_schema


class PromotionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_version: str
    min_average_precision: float = Field(ge=0, le=1)
    min_recall: float = Field(ge=0, le=1)
    min_precision: float = Field(ge=0, le=1)
    max_average_precision_drop: float = Field(ge=0)
    max_roc_auc_drop: float = Field(ge=0)
    max_brier_increase: float = Field(ge=0)
    require_same_schema_version: bool = True


def load_policy(path: Path) -> tuple[PromotionPolicy, str]:
    text = Path(path).read_text(encoding="utf-8")
    return PromotionPolicy.model_validate(yaml.safe_load(text)), text


def _score(pipeline, metadata: dict[str, Any], data, y: np.ndarray) -> dict[str, Any]:
    f = metadata["features"]
    X = to_model_frame(data, FeatureSet(f["numeric"], f["categorical"], {}))
    proba = pipeline.predict_proba(X)[:, 1]
    thr = metadata["threshold"]["threshold"]
    return {**ranking_metrics(y, proba), **threshold_metrics(y, proba, thr)}


def _check(name: str, passed: bool, value: Any, limit: Any) -> dict[str, Any]:
    return {"criterion": name, "passed": bool(passed), "value": value, "limit": limit}


def evaluate_criteria(
    policy: PromotionPolicy,
    candidate: dict[str, Any],
    current: dict[str, Any] | None,
    candidate_schema: str,
    current_schema: str | None,
) -> list[dict[str, Any]]:
    r = lambda x: round(float(x), 4)  # noqa: E731
    checks = [
        _check(
            "min_average_precision",
            candidate["average_precision"] >= policy.min_average_precision,
            r(candidate["average_precision"]),
            policy.min_average_precision,
        ),
        _check(
            "min_recall",
            candidate["recall"] >= policy.min_recall,
            r(candidate["recall"]),
            policy.min_recall,
        ),
        _check(
            "min_precision",
            candidate["precision"] >= policy.min_precision,
            r(candidate["precision"]),
            policy.min_precision,
        ),
    ]
    if current is not None:
        ap_drop = current["average_precision"] - candidate["average_precision"]
        auc_drop = current["roc_auc"] - candidate["roc_auc"]
        brier_up = candidate["brier_score"] - current["brier_score"]
        checks += [
            _check(
                "max_average_precision_drop",
                ap_drop <= policy.max_average_precision_drop,
                r(ap_drop),
                policy.max_average_precision_drop,
            ),
            _check(
                "max_roc_auc_drop",
                auc_drop <= policy.max_roc_auc_drop,
                r(auc_drop),
                policy.max_roc_auc_drop,
            ),
            _check(
                "max_brier_increase",
                brier_up <= policy.max_brier_increase,
                r(brier_up),
                policy.max_brier_increase,
            ),
        ]
        if policy.require_same_schema_version:
            checks.append(
                _check(
                    "same_schema_version",
                    candidate_schema == current_schema,
                    candidate_schema,
                    current_schema,
                )
            )
    return checks


def retrain(
    train_config_path: Path,
    policy_path: Path,
    data_path: Path | None = None,
    promote: bool = False,
    run_dir_root: Path = Path("reports/retraining"),
    reason: str | None = None,
) -> dict[str, Any]:
    started = datetime.now(UTC)
    run_id = started.strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(run_dir_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg, cfg_text = load_train_config(train_config_path)
    policy, policy_text = load_policy(policy_path)
    reg_dir = cfg.output.registry_dir

    # The candidate's config is the base config with only the data path swapped, saved so
    # the run can be repeated exactly.
    cfg_dict = yaml.safe_load(cfg_text)
    if data_path is not None:
        cfg_dict["data"]["raw_path"] = str(data_path)
    run_cfg_path = run_dir / "train_config.yaml"
    run_cfg_path.write_text(yaml.safe_dump(cfg_dict, sort_keys=False), encoding="utf-8")
    (run_dir / "promotion_policy.yaml").write_text(policy_text, encoding="utf-8")

    decision: dict[str, Any] = {
        "run_id": run_id,
        "started_at_utc": started.isoformat(timespec="seconds"),
        "data_path": cfg_dict["data"]["raw_path"],
        "policy_version": policy.policy_version,
        "promote_requested": promote,
    }

    def finish(outcome: str, **extra) -> dict[str, Any]:
        decision.update(outcome=outcome, **extra)
        decision["finished_at_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
        (run_dir / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
        if Path(reg_dir).exists():
            registry.record_decision(
                reg_dir,
                {
                    "decision": outcome,
                    "mode": "retrain",
                    "run_id": run_id,
                    "version": decision.get("candidate_version"),
                    "previous_version": decision.get("current_version"),
                    "reason": decision.get("summary"),
                },
            )
        return decision

    # 1. Validate incoming data before spending time on training.
    schema = load_schema(cfg_dict["data"]["schema_path"])
    mapping = load_mapping(cfg_dict["data"]["mapping_path"], schema)
    prepared = prepare_csv(Path(cfg_dict["data"]["raw_path"]), schema, mapping)
    (run_dir / "validation_report.json").write_text(
        prepared.report.model_dump_json(indent=2), encoding="utf-8"
    )
    if prepared.report.errors:
        return finish(
            "rejected",
            summary="incoming data failed validation; no candidate trained",
            validation_errors=[i.code for i in prepared.report.errors],
        )

    # 2. Train the candidate (registered, not served).
    result = train(run_cfg_path)
    candidate_version = result.version
    decision["candidate_version"] = candidate_version

    # 3. Score candidate and current model on the candidate's held-out test rows.
    features = feature_set_from_schema(schema)
    data = prepared.data
    X = to_model_frame(data, features)
    y = data[schema.target.name].astype(int)
    X_test, y_test = split_data(X, y, cfg.split, cfg.random_seed)["test"]
    test_rows = data.loc[X_test.index]
    y_arr = y_test.to_numpy()

    cand_pipe, cand_meta = registry.load_version(reg_dir, candidate_version)
    cand_metrics = _score(cand_pipe, cand_meta, test_rows, y_arr)

    current_ptr = registry.read_current(reg_dir)
    cur_metrics, cur_schema = None, None
    if current_ptr is not None:
        decision["current_version"] = current_ptr["version"]
        cur_pipe, cur_meta = registry.load_version(reg_dir, current_ptr["version"])
        cur_metrics = _score(cur_pipe, cur_meta, test_rows, y_arr)
        cur_schema = cur_meta["data"]["schema_version"]

    keys = (
        "roc_auc",
        "average_precision",
        "brier_score",
        "precision",
        "recall",
        "f1",
        "flagged_rate",
    )
    decision["comparison"] = {
        "rows": "candidate test split",
        "n": int(len(y_arr)),
        "candidate": {k: round(cand_metrics[k], 4) for k in keys},
        "current": {k: round(cur_metrics[k], 4) for k in keys} if cur_metrics else None,
        "caveat": "If the new data overlaps the current model's training data, some of these "
        "rows may have been seen by the current model, which favours keeping it.",
    }

    # 4. Gate.
    checks = evaluate_criteria(
        policy, cand_metrics, cur_metrics, cand_meta["data"]["schema_version"], cur_schema
    )
    decision["criteria"] = checks
    failed = [c["criterion"] for c in checks if not c["passed"]]
    if failed:
        return finish("rejected", summary=f"failed criteria: {', '.join(failed)}")

    if current_ptr is not None and candidate_version == current_ptr["version"]:
        return finish("no_change", summary="candidate is already the served version")

    # 5. Promote only on explicit request.
    if not promote:
        return finish(
            "eligible_not_promoted",
            summary="all criteria passed; rerun with --promote (or use scripts/promote.py) "
            "to serve it",
        )
    note = reason or f"retrain run {run_id}: all {len(checks)} criteria passed"
    registry.set_current(reg_dir, candidate_version, note)
    return finish("promoted", summary=note)
