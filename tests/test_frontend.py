import json
import socket
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn
from frontend.api_client import ApiError, ChurnApiClient
from frontend.form_logic import (
    SAMPLES,
    build_payload,
    consistency_hints,
    describe_field_errors,
    missing_fields,
    ordered_sections,
)
from streamlit.testing.v1 import AppTest

from churn.api.app import create_app
from churn.api.settings import ApiSettings
from tests.conftest import SCHEMA_PATH

APP_PATH = str(Path(__file__).resolve().parents[1] / "frontend" / "app.py")


@pytest.fixture
def input_schema(schema):
    # Same shape the API returns under /model-info -> input_schema.
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


# --- API client -------------------------------------------------------------------------


def _client(handler) -> ChurnApiClient:
    return ChurnApiClient("http://api.test", transport=httpx.MockTransport(handler))


def test_predict_sends_json_payload_and_returns_body():
    seen = {}

    def handler(request: httpx.Request):
        seen["method"], seen["path"] = request.method, request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"churn_probability": 0.4, "risk_label": "low"})

    out = _client(handler).predict({"tenure_months": 3})
    assert seen == {"method": "POST", "path": "/predict", "body": {"tenure_months": 3}}
    assert out["risk_label"] == "low"


@pytest.mark.parametrize(
    ("response", "kind"),
    [
        (
            httpx.Response(
                422,
                json={
                    "detail": "Invalid request",
                    "errors": [{"field": "contract", "message": "bad"}],
                },
            ),
            "invalid",
        ),
        (httpx.Response(503, json={"detail": "Model not ready: none promoted"}), "not_ready"),
        (httpx.Response(500, text="boom"), "server"),
        (httpx.Response(502, text="<html>bad gateway</html>"), "server"),
    ],
)
def test_http_errors_map_to_kinds(response, kind):
    with pytest.raises(ApiError) as exc:
        _client(lambda _: response).predict({})
    assert exc.value.kind == kind
    if kind == "invalid":
        assert exc.value.field_errors == [{"field": "contract", "message": "bad"}]


@pytest.mark.parametrize("error", [httpx.ConnectError("refused"), httpx.ReadTimeout("slow")])
def test_network_failures_are_unavailable(error):
    def handler(request):
        raise error

    with pytest.raises(ApiError) as exc:
        _client(handler).model_info()
    assert exc.value.kind == "unavailable"
    assert "http://api.test" in exc.value.message


# --- Form helpers -----------------------------------------------------------------------


@pytest.mark.parametrize("name", list(SAMPLES))
def test_samples_are_valid_for_the_schema(name, input_schema):
    sample = SAMPLES[name]
    assert set(sample) == {f["name"] for f in input_schema}
    for f in input_schema:
        v = sample[f["name"]]
        if f["allowed_values"]:
            assert v in f["allowed_values"], f["name"]
        else:
            assert f["min"] <= v <= f["max"], f["name"]
    assert consistency_hints(build_payload(sample, input_schema)) == []


def test_build_payload_casts_types_and_drops_non_schema_keys(input_schema):
    values = {
        **SAMPLES[next(iter(SAMPLES))],
        "tenure_months": 5.0,
        "monthly_charges": 70.123,
        "customer_id": "X-1",
    }
    payload = build_payload(values, input_schema)
    assert payload["tenure_months"] == 5 and isinstance(payload["tenure_months"], int)
    assert payload["monthly_charges"] == 70.12
    assert "customer_id" not in payload


def test_missing_fields_reported(input_schema):
    values = {k: v for k, v in SAMPLES[next(iter(SAMPLES))].items() if k != "contract"}
    values["partner"] = None
    assert missing_fields(build_payload(values, input_schema), input_schema) == [
        "contract",
        "partner",
    ]


def test_consistency_hints_flag_impossible_combinations():
    base = dict(SAMPLES[next(iter(SAMPLES))])
    no_internet = {**base, "internet_service": "no"}
    assert any("Internet service" in h for h in consistency_hints(no_internet))
    no_phone = {**base, "phone_service": "no", "multiple_lines": "yes"}
    assert any("Phone service" in h for h in consistency_hints(no_phone))
    low_total = {**base, "tenure_months": 10, "total_charges": 5.0}
    assert any("Total charges" in h for h in consistency_hints(low_total))


def test_sections_cover_every_field(input_schema):
    placed = [f["name"] for fields in ordered_sections(input_schema).values() for f in fields]
    assert sorted(placed) == sorted(f["name"] for f in input_schema)
    extra = input_schema + [{"name": "new_field", "dtype": "float", "min": 0, "max": 1}]
    assert ordered_sections(extra)["Other"][0]["name"] == "new_field"


def test_field_errors_use_friendly_labels():
    out = describe_field_errors([{"field": "tenure_months", "message": "too small"}])
    assert out == ["Tenure (months): too small"]


# --- Streamlit app end to end ----------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_api_url(promoted_registry):
    port = _free_port()
    settings = ApiSettings(registry_dir=promoted_registry[0], schema_path=SCHEMA_PATH)
    server = uvicorn.Server(
        uvicorn.Config(create_app(settings), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


def _app(monkeypatch, url: str) -> AppTest:
    monkeypatch.setenv("CHURN_API_URL", url)
    return AppTest.from_file(APP_PATH, default_timeout=30)


def test_app_shows_friendly_message_when_api_down(monkeypatch):
    at = _app(monkeypatch, f"http://127.0.0.1:{_free_port()}").run()
    assert not at.exception
    assert "isn't reachable" in at.error[0].value
    assert [b.label for b in at.button] == ["Retry"]


def test_app_predicts_through_api(monkeypatch, live_api_url, promoted_registry):
    at = _app(monkeypatch, live_api_url).run()
    assert not at.exception and not at.error
    submit = next(b for b in at.button if b.label == "Estimate churn risk")
    at = submit.click().run()
    assert not at.exception and not at.error
    labels = {m.label: m.value for m in at.metric}
    assert labels["Risk level"] in {"High", "Low"}
    assert labels["Churn probability"].endswith("%")
    assert at.session_state["result"]["model_version"] == promoted_registry[1]


def test_sample_button_fills_form(monkeypatch, live_api_url):
    at = _app(monkeypatch, live_api_url).run()
    second = list(SAMPLES)[1]
    at = next(b for b in at.button if b.label == second).click().run()
    assert at.session_state["field_contract"] == SAMPLES[second]["contract"]
    assert at.session_state["field_tenure_months"] == SAMPLES[second]["tenure_months"]


def test_submit_error_does_not_mention_missing_retry_button(monkeypatch, live_api_url):
    at = _app(monkeypatch, live_api_url).run()
    # API goes away after the form has loaded. Streamlit imports the client as the
    # top-level module `api_client`, so patch that copy rather than frontend.api_client.
    loaded = sys.modules["api_client"]
    monkeypatch.setattr(
        loaded.ChurnApiClient,
        "predict",
        lambda self, payload: (_ for _ in ()).throw(loaded.ApiError("unavailable", "down")),
    )
    at = next(b for b in at.button if b.label == "Estimate churn risk").click().run()
    message = at.error[0].value
    assert "isn't reachable" in message and "Retry" not in message
    assert "Estimate churn risk" in message
