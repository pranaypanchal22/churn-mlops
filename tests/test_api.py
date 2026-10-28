import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from churn.api.app import create_app
from churn.api.settings import ApiSettings
from churn.models import registry
from churn.models.train import train
from tests.conftest import SCHEMA_PATH, make_smoke_config

VALID = {
    "tenure_months": 2,
    "monthly_charges": 70.7,
    "total_charges": 151.65,
    "contract": "month_to_month",
    "payment_method": "electronic_check",
    "paperless_billing": "yes",
    "senior_citizen": "no",
    "partner": "no",
    "dependents": "no",
    "phone_service": "yes",
    "multiple_lines": "no",
    "internet_service": "fiber_optic",
    "online_security": "no",
    "online_backup": "no",
    "device_protection": "no",
    "tech_support": "no",
    "streaming_tv": "no",
    "streaming_movies": "no",
}


def _client(registry_dir, **overrides) -> TestClient:
    settings = ApiSettings(
        registry_dir=registry_dir, schema_path=SCHEMA_PATH, max_batch_size=3, **overrides
    )
    return TestClient(create_app(settings))


@pytest.fixture
def client(promoted_registry):
    with _client(promoted_registry[0]) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ready_reports_loaded_version(client, promoted_registry):
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json() == {"ready": True, "model_version": promoted_registry[1]}


def test_not_ready_without_promoted_model(tmp_path):
    with _client(tmp_path) as c:
        assert c.get("/health").status_code == 200
        r = c.get("/ready")
        assert r.status_code == 503 and "no promoted model" in r.json()["detail"]
        assert c.post("/predict", json=VALID).status_code == 503
        assert c.get("/model-info").status_code == 503


def test_pinned_unknown_version_is_not_ready(promoted_registry):
    with _client(promoted_registry[0], model_version="nope") as c:
        assert c.get("/ready").status_code == 503


def test_schema_version_mismatch_is_not_ready(tmp_path_factory):
    result = train(make_smoke_config(tmp_path_factory.mktemp("mismatch")))
    meta_path = result.version_dir / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["data"]["schema_version"] = "0.0.1"
    meta_path.write_text(json.dumps(meta))
    registry.set_current(result.version_dir.parent, result.version, "test")
    with _client(result.version_dir.parent) as c:
        r = c.get("/ready")
        assert r.status_code == 503 and "schema" in r.json()["detail"]


def test_predict_valid_request(client, promoted_registry):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"churn_probability", "risk_label", "threshold", "model_version", "note"}
    assert 0 <= body["churn_probability"] <= 1
    assert body["model_version"] == promoted_registry[1]
    expected = "high" if body["churn_probability"] >= body["threshold"] else "low"
    assert body["risk_label"] == expected
    assert "estimate" in body["note"]


@pytest.mark.parametrize("threshold,label", [(0.0, "high"), (1.01, "low")])
def test_risk_label_follows_threshold(client, threshold, label):
    thr = client.app.state.service.metadata["threshold"]
    original = thr["threshold"]
    thr["threshold"] = threshold
    try:
        assert client.post("/predict", json=VALID).json()["risk_label"] == label
    finally:
        thr["threshold"] = original


@pytest.mark.parametrize(
    "change,field",
    [
        ({"contract": "monthly"}, "contract"),
        ({"tenure_months": -1}, "tenure_months"),
        ({"tenure_months": 1.5}, "tenure_months"),
        ({"monthly_charges": "abc"}, "monthly_charges"),
        ({"customer_id": "7590-VHVEG"}, "customer_id"),
        ({"gender": "female"}, "gender"),
    ],
)
def test_predict_invalid_request_names_field_without_echoing_value(client, change, field):
    r = client.post("/predict", json={**VALID, **change})
    assert r.status_code == 422
    body = r.json()
    assert body["detail"] == "Invalid request"
    assert field in [e["field"] for e in body["errors"]]
    for value in change.values():
        assert str(value) not in r.text


def test_predict_missing_field(client):
    payload = {k: v for k, v in VALID.items() if k != "contract"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
    assert {"field": "contract", "message": "Field required"} in r.json()["errors"]


def test_batch_predict(client):
    r = client.post("/predict/batch", json={"customers": [VALID, {**VALID, "tenure_months": 60}]})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert [p["index"] for p in preds] == [0, 1]
    assert all(0 <= p["churn_probability"] <= 1 for p in preds)


@pytest.mark.parametrize("n", [0, 4])
def test_batch_size_limits(client, n):
    r = client.post("/predict/batch", json={"customers": [VALID] * n})
    assert r.status_code == 422
    assert r.json()["errors"][0]["field"] == "customers"


def test_batch_error_points_at_item(client):
    r = client.post("/predict/batch", json={"customers": [VALID, {**VALID, "contract": "x"}]})
    assert r.status_code == 422
    assert r.json()["errors"][0]["field"] == "customers.1.contract"


def test_model_info(client, promoted_registry):
    info = client.get("/model-info").json()
    assert info["model_version"] == promoted_registry[1]
    assert info["schema_version"] == "1.0.0"
    assert [f["name"] for f in info["input_schema"]] == list(VALID)
    assert "not causal" in info["model_signals"]["caveat"]
    assert info["max_batch_size"] == 3
    assert info["threshold"]["selected_on"] == "validation"


def test_logs_do_not_contain_payload(client, caplog):
    with caplog.at_level(logging.DEBUG):
        client.post("/predict", json={**VALID, "monthly_charges": 87.65})
        client.post("/predict", json={**VALID, "contract": "secret-value"})
    text = caplog.text + " ".join(str(r.__dict__) for r in caplog.records)
    assert "87.65" not in text and "secret-value" not in text
    assert any(r.getMessage() == "request" for r in caplog.records)


def test_ci_smoke_fixture_matches_schema(client):
    # The CI smoke job posts this file; keep it valid as the schema evolves.
    fixture = json.loads((Path(__file__).parent / "fixtures" / "predict_request.json").read_text())
    assert fixture == VALID
    assert client.post("/predict", json=fixture).status_code == 200
