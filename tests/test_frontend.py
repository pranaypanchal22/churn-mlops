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
    WHAT_IF_LEVERS,
    build_payload,
    consistency_hints,
    dependent_updates,
    describe_field_errors,
    describe_risk,
    estimate_total,
    missing_fields,
    ordered_sections,
    tenure_scenarios,
    what_if_scenarios,
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


def _metrics(at) -> dict[str, str]:
    return {m.label: m.value for m in at.metric}


def _drift_rows(url: str) -> int:
    return httpx.get(f"{url}/monitoring/drift").json()["n"]


def _submit(at):
    return at.button(key="submit").click().run()


def test_nothing_is_predicted_until_submit(monkeypatch, live_api_url):
    before = _drift_rows(live_api_url)
    at = _app(monkeypatch, live_api_url).run()
    assert not at.exception and not at.error
    assert "Model's score" not in _metrics(at)
    assert any("appear here" in m.value for m in at.markdown)
    assert at.button(key="submit").label == "Get prediction"
    assert _drift_rows(live_api_url) == before


def test_submit_shows_prediction(monkeypatch, live_api_url, promoted_registry):
    at = _submit(_app(monkeypatch, live_api_url).run())
    assert not at.exception and not at.error
    metrics = _metrics(at)
    assert metrics["Model's score"].endswith("%")
    assert metrics["Risk level"] in {"High", "Low"}
    assert any(promoted_registry[1] in c.value for c in at.caption)
    assert [t.label for t in at.tabs][:2] == ["🔀 What could change it?", "📈 Over time"]


def test_edits_apply_only_after_submit(monkeypatch, live_api_url):
    at = _submit(_app(monkeypatch, live_api_url).run())
    first = _metrics(at)["Model's score"]
    # Edits without pressing the button are not applied (form semantics).
    at.radio(key="field_contract").set_value("two_year")
    at.run()
    assert _metrics(at)["Model's score"] == first
    assert at.session_state["submitted"]["payload"]["contract"] == "month_to_month"
    # Edits followed by the button are.
    at.radio(key="field_contract").set_value("two_year")
    at.slider(key="field_tenure_months").set_value(60)
    at = _submit(at)
    assert not at.error
    assert at.session_state["submitted"]["payload"]["contract"] == "two_year"
    assert _metrics(at)["Model's score"] != first


def test_sample_selector_fills_form_and_clears_old_result(monkeypatch, live_api_url):
    at = _submit(_app(monkeypatch, live_api_url).run())
    assert "Model's score" in _metrics(at)
    last = list(SAMPLES)[-1]
    at.selectbox(key="sample_choice").set_value(last).run()
    assert at.session_state["field_contract"] == SAMPLES[last]["contract"]
    assert at.session_state["field_tenure_months"] == SAMPLES[last]["tenure_months"]
    assert "Model's score" not in _metrics(at)


def test_no_internet_is_made_consistent_on_submit(monkeypatch, live_api_url):
    at = _app(monkeypatch, live_api_url).run()
    at.radio(key="field_internet_service").set_value("no")
    at = _submit(at)
    assert at.session_state["field_online_security"] == "no_internet_service"
    assert at.radio(key="field_online_security").disabled
    assert any("No internet service" in i.value for i in at.info)
    assert not at.error


def test_total_charges_can_be_calculated(monkeypatch, live_api_url):
    at = _app(monkeypatch, live_api_url).run()
    at.toggle(key="auto_total").set_value(True)
    at.slider(key="field_tenure_months").set_value(10)
    at = _submit(at)
    monthly = at.session_state["field_monthly_charges"]
    assert at.session_state["field_total_charges"] == pytest.approx(10 * monthly)
    assert any("tenure × monthly" in i.value for i in at.info)


def test_one_submit_is_one_real_prediction(monkeypatch, live_api_url):
    before = _drift_rows(live_api_url)
    at = _submit(_app(monkeypatch, live_api_url).run())
    assert not at.error
    at.run()  # redraws reuse the stored result instead of calling the API again
    # The ~30 what-if rows are sent as exploration and never reach drift monitoring.
    assert _drift_rows(live_api_url) - before == 1


def test_prediction_failure_says_how_to_retry(monkeypatch, live_api_url):
    at = _app(monkeypatch, live_api_url).run()
    # Streamlit imports the client as the top-level module `api_client`, so patch that
    # copy rather than frontend.api_client.
    loaded = sys.modules["api_client"]
    monkeypatch.setattr(
        loaded.ChurnApiClient,
        "predict",
        lambda self, payload: (_ for _ in ()).throw(loaded.ApiError("unavailable", "down")),
    )
    at = _submit(at)
    message = at.error[0].value
    assert "isn't reachable" in message and "Get prediction" in message
    assert "Retry" not in message


# --- Interactive helpers -----------------------------------------------------------------


def test_dependent_updates_mirror_training_data():
    base = dict(SAMPLES[next(iter(SAMPLES))])
    no_net = dependent_updates({**base, "internet_service": "no"})
    assert set(no_net.values()) == {"no_internet_service"} and len(no_net) == 6
    back = {**base, "internet_service": "dsl", "online_security": "no_internet_service"}
    assert dependent_updates(back) == {"online_security": "no"}
    assert dependent_updates({**base, "phone_service": "no"}) == {
        "multiple_lines": "no_phone_service"
    }
    assert dependent_updates(base) == {}


def test_estimate_total():
    assert estimate_total(12, 70.7) == 848.4
    assert estimate_total(0, 99) == 0.0


def test_what_if_changes_exactly_one_lever(input_schema):
    payload = SAMPLES[next(iter(SAMPLES))]
    scenarios = what_if_scenarios(payload, input_schema)
    assert scenarios
    for s in scenarios:
        diff = {k for k in payload if s["payload"][k] != payload[k]}
        assert diff == {s["field"]} and s["field"] in WHAT_IF_LEVERS
        assert s["value"] != payload[s["field"]]


def test_what_if_skips_contradictions(input_schema):
    no_net = {**SAMPLES[next(iter(SAMPLES))], "internet_service": "no"}
    no_net.update(dependent_updates(no_net))
    fields = {s["field"] for s in what_if_scenarios(no_net, input_schema)}
    assert "online_security" not in fields and "tech_support" not in fields
    assert "contract" in fields


def test_tenure_scenarios_keep_totals_consistent():
    payload = {**SAMPLES[next(iter(SAMPLES))], "tenure_months": 5}
    points = tenure_scenarios(payload, max_tenure=72)
    tenures = [p["tenure_months"] for p in points]
    assert 5 in tenures and min(tenures) == 0 and max(tenures) <= 72
    for pt in points:
        assert pt["payload"]["total_charges"] == estimate_total(
            pt["tenure_months"], payload["monthly_charges"]
        )


@pytest.mark.parametrize(
    ("p", "headline"),
    [(0.9, "well above"), (0.62, "just above"), (0.55, "just below"), (0.2, "well below")],
)
def test_describe_risk_bands(p, headline):
    title, detail = describe_risk(p, 0.6)
    assert headline in title and f"{p:.0%}" in detail


def test_predict_batch_chunks_and_marks_exploration():
    seen = []

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        seen.append((len(body["customers"]), request.headers.get("x-churn-purpose")))
        preds = [
            {"index": i, "churn_probability": 0.5, "risk_label": "low"}
            for i in range(len(body["customers"]))
        ]
        return httpx.Response(200, json={"predictions": preds})

    out = _client(handler).predict_batch([{}] * 7, max_batch_size=3)
    assert len(out) == 7
    assert seen == [(3, "exploration"), (3, "exploration"), (1, "exploration")]
