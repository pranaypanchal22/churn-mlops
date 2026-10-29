import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from churn.api.app import create_app
from churn.api.settings import ApiSettings
from churn.data.pipeline import prepare_frame
from churn.data.synthetic import make_synthetic_raw
from churn.features.build import feature_set_from_schema, to_model_frame
from churn.models import registry
from churn.models.train import train
from churn.monitoring.drift import (
    OTHER,
    MonitoringConfig,
    batch_drift_report,
    build_reference_profile,
    performance_report,
    psi,
)
from churn.monitoring.live import LiveMonitor
from tests.conftest import SCHEMA_PATH, make_smoke_config
from tests.test_api import VALID

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
CFG = MonitoringConfig(min_samples=50)


def _script(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


@pytest.fixture(scope="module")
def frame(schema_module):
    schema, mapping = schema_module
    data = prepare_frame(make_synthetic_raw(800, seed=3), schema, mapping).data
    return to_model_frame(data, feature_set_from_schema(schema)), data


@pytest.fixture(scope="module")
def schema_module():
    from churn.schema.config import load_mapping, load_schema
    from tests.conftest import KAGGLE_MAPPING_PATH

    s = load_schema(SCHEMA_PATH)
    return s, load_mapping(KAGGLE_MAPPING_PATH, s)


@pytest.fixture(scope="module")
def profile(frame, schema_module):
    X, _ = frame
    proba = np.linspace(0, 1, len(X))
    return build_reference_profile(X, feature_set_from_schema(schema_module[0]), proba, 0.5)


# --- PSI and reference profile ------------------------------------------------------------


def test_psi_zero_for_identical_and_positive_for_shift():
    assert psi([0.5, 0.5], [50, 50]) == pytest.approx(0, abs=1e-9)
    assert psi([0.5, 0.5], [90, 10]) > 0.25
    assert psi([0.5, 0.5], [0, 0]) == 0.0


def test_reference_profile_is_aggregate_only(profile, frame):
    X, _ = frame
    for p in profile["numeric"].values():
        assert sum(p["proportions"]) == pytest.approx(1, abs=1e-4)
        assert len(p["proportions"]) == len(p["edges"]) + 1
    for p in profile["categorical"].values():
        assert sum(p["proportions"].values()) == pytest.approx(1, abs=1e-4)
    # No row-level values: the profile is far smaller than the data and has no row lists.
    assert len(json.dumps(profile)) < 10_000
    assert profile["n_train_rows"] == len(X)


def test_same_data_shows_no_drift(profile, frame):
    X, _ = frame
    report = batch_drift_report(profile, X, np.linspace(0, 1, len(X)), CFG)
    assert report["status"] == "ok"
    assert report["drifted_features"] == []
    assert all(f["psi"] < 0.01 for f in report["features"].values())


def test_shifted_data_raises_alert_on_shifted_features(profile, frame):
    X, _ = frame
    shifted = X[(X.contract == "month_to_month") & (X.tenure_months < 12)]
    report = batch_drift_report(profile, shifted, np.full(len(shifted), 0.9), CFG)
    assert report["status"] == "alert"
    assert {"contract", "tenure_months"} <= set(report["drifted_features"])
    assert report["prediction"]["flag_rate"] == 1.0
    assert "performance" in report and "outcomes" in report["performance"]


def test_small_batches_are_insufficient_data(profile, frame):
    X, _ = frame
    report = batch_drift_report(profile, X.head(10), None, CFG)
    assert report["status"] == "insufficient_data"


def test_unseen_categories_and_out_of_range_counted(profile, frame):
    X, _ = frame
    odd = X.head(60).copy()
    odd.loc[odd.index[:5], "contract"] = "three_year"
    odd.loc[odd.index[:3], "monthly_charges"] = 999.0
    report = batch_drift_report(profile, odd, None, CFG)
    assert report["features"]["contract"]["unseen_category_count"] == 5
    assert report["features"]["monthly_charges"]["outside_training_range"] == 3
    assert OTHER not in profile["categorical"]["contract"]["proportions"]


def test_performance_report_compares_to_reference():
    y = np.array([0, 1, 1, 0, 1, 0])
    p = np.array([0.1, 0.9, 0.8, 0.3, 0.4, 0.2])
    ref = {"roc_auc": 0.8, "average_precision": 0.6, "precision": 0.5, "recall": 0.7, "f1": 0.6}
    out = performance_report(y, p, 0.5, ref)
    assert out["n"] == 6
    assert out["change"]["roc_auc"] == pytest.approx(out["metrics"]["roc_auc"] - 0.8, abs=1e-3)


def test_live_monitor_disabled_without_profile(frame):
    mon = LiveMonitor(None, CFG)
    mon.record(frame[0], np.zeros(len(frame[0])))
    assert not mon.enabled and mon.report() is None


# --- API integration ----------------------------------------------------------------------


@pytest.fixture
def client(promoted_registry):
    settings = ApiSettings(registry_dir=promoted_registry[0], schema_path=SCHEMA_PATH)
    with TestClient(create_app(settings)) as c:
        yield c


def test_training_saves_reference_profile(promoted_registry):
    reg, version = promoted_registry
    profile = registry.read_reference_profile(reg, version)
    assert profile and profile["prediction"]["source"] == "validation split"


def test_metrics_endpoint_counts_requests_without_payload(client):
    client.post("/predict", json={**VALID, "monthly_charges": 87.65})
    client.post("/predict", json={**VALID, "contract": "nope"})
    text = client.get("/metrics").text
    assert 'churn_http_requests_total{method="POST",route="/predict",status="200"} 1.0' in text
    assert 'churn_validation_errors_total{field="contract"} 1.0' in text
    assert "churn_predictions_total" in text and "churn_model_ready 1.0" in text
    assert "churn_feature_psi" in text
    assert "87.65" not in text and "nope" not in text


def test_monitoring_drift_endpoint_aggregates_predictions(client):
    for _ in range(3):
        client.post("/predict/batch", json={"customers": [VALID] * 5})
    body = client.get("/monitoring/drift").json()
    assert body["n"] == 15
    assert body["status"] == "insufficient_data"
    assert "not measured" in body["performance"]
    # Identical rows are a heavily skewed sample, so the numbers are large but well-formed.
    assert set(body["features"]) == set(VALID)


def test_monitoring_drift_404_without_profile(tmp_path):
    result = train(make_smoke_config(tmp_path))
    (result.version_dir / registry.REFERENCE_PROFILE_FILE).unlink()
    registry.set_current(result.version_dir.parent, result.version, "test")
    settings = ApiSettings(registry_dir=result.version_dir.parent, schema_path=SCHEMA_PATH)
    with TestClient(create_app(settings)) as c:
        assert c.get("/ready").status_code == 200
        assert c.get("/monitoring/drift").status_code == 404


# --- drift_report CLI ---------------------------------------------------------------------


def _drift_cli(promoted_registry, csv, tmp_path):
    return _script("drift_report")(
        [
            "--input",
            str(csv),
            "--registry",
            str(promoted_registry[0]),
            "--out-dir",
            str(tmp_path / "rep"),
        ]
    )


def _latest_report(tmp_path):
    return json.loads(sorted((tmp_path / "rep").glob("*.json"))[-1].read_text())


def test_drift_cli_unlabelled_batch_does_not_claim_performance(promoted_registry, tmp_path):
    raw = make_synthetic_raw(400, seed=11).drop(columns=["Churn"])
    csv = tmp_path / "batch.csv"
    raw.to_csv(csv, index=False)
    assert _drift_cli(promoted_registry, csv, tmp_path) in (0, 3)
    report = _latest_report(tmp_path)
    assert report["performance"]["measured"] is False
    assert report["drift"]["n"] == 400
    assert (
        tmp_path / "rep" / (sorted((tmp_path / "rep").glob("*.json"))[-1].stem + ".md")
    ).exists()


def test_drift_cli_labelled_batch_reports_performance(promoted_registry, tmp_path):
    csv = tmp_path / "labelled.csv"
    make_synthetic_raw(400, seed=12).to_csv(csv, index=False)
    _drift_cli(promoted_registry, csv, tmp_path)
    perf = _latest_report(tmp_path)["performance"]
    assert perf["n"] == 400 and "roc_auc" in perf["metrics"]


def test_drift_cli_shifted_batch_exits_3(promoted_registry, tmp_path):
    raw = make_synthetic_raw(1500, seed=13)
    raw = raw[(raw["Contract"] == "Two year") & (raw["Internet Service"] == "No")]
    csv = tmp_path / "shifted.csv"
    raw.to_csv(csv, index=False)
    cfg = tmp_path / "monitoring.yaml"
    cfg.write_text(f"min_samples: {len(raw)}\n", encoding="utf-8")
    code = _script("drift_report")(
        [
            "--input",
            str(csv),
            "--registry",
            str(promoted_registry[0]),
            "--out-dir",
            str(tmp_path / "rep"),
            "--config",
            str(cfg),
        ]
    )
    report = _latest_report(tmp_path)
    assert report["status"] == "alert" and code == 3
    assert {"contract", "internet_service"} <= set(report["drift"]["drifted_features"])


def test_drift_cli_respects_min_samples(promoted_registry, tmp_path):
    csv = tmp_path / "tiny.csv"
    make_synthetic_raw(30, seed=15).to_csv(csv, index=False)
    assert _drift_cli(promoted_registry, csv, tmp_path) == 0
    assert _latest_report(tmp_path)["status"] == "insufficient_data"


def test_drift_cli_invalid_batch_exits_1(promoted_registry, tmp_path):
    raw = make_synthetic_raw(100, seed=14)
    raw.loc[0, "Contract"] = "Monthly"
    csv = tmp_path / "bad.csv"
    raw.to_csv(csv, index=False)
    assert _drift_cli(promoted_registry, csv, tmp_path) == 1
    assert _latest_report(tmp_path)["status"] == "invalid_data"


# --- Retraining workflow ------------------------------------------------------------------


@pytest.fixture
def retrain_setup(tmp_path):
    """Registry with a promoted baseline, plus a policy file and a fresh data file."""
    cfg_path = make_smoke_config(tmp_path)
    base = train(cfg_path)
    reg = base.version_dir.parent
    registry.set_current(reg, base.version, "baseline")
    policy = tmp_path / "policy.yaml"
    shutil.copy(Path(__file__).resolve().parents[1] / "configs" / "promotion.yaml", policy)
    # Synthetic data is noisy: relax absolute floors so tests exercise the gate logic.
    text = policy.read_text()
    for key, val in (
        ("min_average_precision", "0.2"),
        ("min_recall", "0.5"),
        ("min_precision", "0.2"),
        ("max_average_precision_drop", "1.0"),
        ("max_roc_auc_drop", "1.0"),
        ("max_brier_increase", "1.0"),
    ):
        text = "\n".join(
            f"{key}: {val}" if line.startswith(f"{key}:") else line for line in text.splitlines()
        )
    policy.write_text(text)
    new_data = tmp_path / "new.csv"
    make_synthetic_raw(600, seed=21).to_csv(new_data, index=False)
    return {
        "cfg": cfg_path,
        "reg": reg,
        "base": base.version,
        "policy": policy,
        "data": new_data,
        "runs": tmp_path / "runs",
    }


def _retrain(s, *extra):
    return _script("retrain")(
        [
            "--config",
            str(s["cfg"]),
            "--policy",
            str(s["policy"]),
            "--data",
            str(s["data"]),
            "--runs-dir",
            str(s["runs"]),
            *extra,
        ]
    )


def _decision(s):
    return json.loads(sorted(s["runs"].glob("*/decision.json"))[-1].read_text())


def _audit(s):
    return [json.loads(x) for x in (s["reg"] / "promotions.jsonl").read_text().splitlines()]


def test_retrain_without_promote_never_changes_serving(retrain_setup):
    s = retrain_setup
    assert _retrain(s) == 0
    d = _decision(s)
    assert d["outcome"] == "eligible_not_promoted"
    assert all(c["passed"] for c in d["criteria"])
    assert d["comparison"]["current"] is not None
    assert registry.read_current(s["reg"])["version"] == s["base"]
    assert _audit(s)[-1]["decision"] == "eligible_not_promoted"


def test_retrain_with_promote_switches_and_audits(retrain_setup):
    s = retrain_setup
    assert _retrain(s, "--promote", "--reason", "test promote") == 0
    d = _decision(s)
    assert d["outcome"] == "promoted"
    current = registry.read_current(s["reg"])
    assert current["version"] == d["candidate_version"] != s["base"]
    assert current["previous_version"] == s["base"]
    assert _audit(s)[-1]["decision"] == "promoted"


def test_retrain_rejects_when_criteria_fail(retrain_setup):
    s = retrain_setup
    text = (
        s["policy"].read_text().replace("min_average_precision: 0.2", "min_average_precision: 0.99")
    )
    s["policy"].write_text(text)
    assert _retrain(s, "--promote") == 1
    d = _decision(s)
    assert d["outcome"] == "rejected" and "min_average_precision" in d["summary"]
    assert registry.read_current(s["reg"])["version"] == s["base"]
    assert _audit(s)[-1]["decision"] == "rejected"


def test_retrain_rejects_invalid_data_before_training(retrain_setup):
    s = retrain_setup
    raw = make_synthetic_raw(200, seed=5)
    raw.loc[0, "Contract"] = "Monthly"
    raw.to_csv(s["data"], index=False)
    before = registry.list_versions(s["reg"])
    assert _retrain(s, "--promote") == 1
    d = _decision(s)
    assert d["outcome"] == "rejected" and "invalid_category" in d["validation_errors"]
    assert registry.list_versions(s["reg"]) == before


def test_exploration_batches_are_scored_but_not_monitored(client):
    before = client.get("/monitoring/drift").json()["n"]
    r = client.post(
        "/predict/batch",
        json={"customers": [VALID] * 4},
        headers={"X-Churn-Purpose": "exploration"},
    )
    assert r.status_code == 200 and len(r.json()["predictions"]) == 4
    assert client.get("/monitoring/drift").json()["n"] == before
    metrics = client.get("/metrics").text
    assert "churn_exploration_rows_total 4.0" in metrics
    assert "churn_predictions_total{" not in metrics
