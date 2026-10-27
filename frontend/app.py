"""Streamlit frontend for the churn API.

Run:  streamlit run frontend/app.py
Env:  CHURN_API_URL (default http://localhost:8000), CHURN_API_TIMEOUT seconds (default 5)

The form is built from the API's /model-info input schema, and every prediction comes from
POST /predict. No model, data, or customer identifiers are loaded or stored here.
"""

import os

import streamlit as st

from api_client import ApiError, ChurnApiClient
from form_logic import (
    SAMPLES,
    build_payload,
    consistency_hints,
    describe_field_errors,
    field_label,
    missing_fields,
    ordered_sections,
    value_label,
)

API_URL = os.environ.get("CHURN_API_URL", "http://localhost:8000")
API_TIMEOUT = float(os.environ.get("CHURN_API_TIMEOUT", "5"))
DEFAULT_SAMPLE = next(iter(SAMPLES))

st.set_page_config(page_title="Churn risk estimator", page_icon="📉", layout="centered")


# Keyed by URL so a changed CHURN_API_URL never reuses a client bound to the old one.
@st.cache_resource
def get_client(base_url: str, timeout: float) -> ChurnApiClient:
    return ChurnApiClient(base_url, timeout=timeout)


@st.cache_data(ttl=60, show_spinner=False)
def get_model_info(base_url: str) -> dict:
    # Errors are raised, not cached, so a retry hits the API again.
    return get_client(base_url, API_TIMEOUT).model_info()


def widget_key(name: str) -> str:
    return f"field_{name}"


def load_sample(name: str) -> None:
    for field, value in SAMPLES[name].items():
        st.session_state[widget_key(field)] = value
    st.session_state.pop("result", None)
    st.session_state["loaded_sample"] = name


def render_input(spec: dict) -> None:
    name, key = spec["name"], widget_key(spec["name"])
    label = field_label(name)
    help_text = spec.get("description") or None
    if spec["dtype"] in ("integer", "float"):
        is_int = spec["dtype"] == "integer"
        cast = int if is_int else float
        st.number_input(
            label,
            key=key,
            min_value=cast(spec["min"]) if spec.get("min") is not None else None,
            max_value=cast(spec["max"]) if spec.get("max") is not None else None,
            step=1 if is_int else 1.0,
            format="%d" if is_int else "%.2f",
            help=help_text,
        )
    else:
        options = spec["allowed_values"]
        if len(options) <= 3:
            st.radio(
                label, options, key=key, format_func=value_label, horizontal=True, help=help_text
            )
        else:
            st.selectbox(label, options, key=key, format_func=value_label, help=help_text)


def show_api_error(err: ApiError) -> None:
    if err.kind == "unavailable":
        st.error(
            f"**The prediction service isn't reachable.** {err.message}\n\n"
            "Start it with `uvicorn churn.api.app:app --port 8000` (or `docker compose up`), "
            "then press **Retry**."
        )
    elif err.kind == "not_ready":
        st.warning(
            f"**The service is running but has no model loaded.** {err.message}\n\n"
            "Train and promote a model (see README), then press **Retry**."
        )
    elif err.kind == "invalid":
        details = describe_field_errors(err.field_errors)
        st.error(
            "**Some inputs were rejected by the API:**\n\n"
            + "\n".join(f"- {d}" for d in details or [err.message])
        )
    else:
        st.error(f"**Something went wrong on the server.** {err.message} Please try again.")


def show_result(result: dict) -> None:
    p, thr = result["churn_probability"], result["threshold"]
    high = result["risk_label"] == "high"
    with st.container(border=True):
        st.subheader("Estimated churn risk")
        c1, c2 = st.columns(2)
        c1.metric("Churn probability", f"{p:.0%}")
        c2.metric("Risk level", "High" if high else "Low")
        st.progress(min(max(p, 0.0), 1.0))
        if high:
            st.markdown(
                f":red[**High risk**]: the estimate is at or above the model's threshold of "
                f"{thr:.0%}. In a retention programme this profile would be on the outreach list."
            )
        else:
            st.markdown(
                f":green[**Low risk**]: the estimate is below the model's threshold of {thr:.0%}."
            )
        st.caption(
            f"{result['note']} Scores are best read as a ranking; they run higher than the "
            f"observed churn rate. Model version `{result['model_version']}`."
        )


st.title("Churn risk estimator")
st.markdown(
    "Enter a customer profile to get an **estimated** probability that they will leave. "
    "This is a portfolio demo trained on one sample telecom dataset. It is not a guarantee "
    "and not a tool for real decisions about real people."
)

try:
    with st.spinner("Connecting to the prediction service…"):
        info = get_model_info(API_URL)
except ApiError as err:
    show_api_error(err)
    if st.button("Retry", type="primary"):
        get_model_info.clear()
        st.rerun()
    st.stop()

schema = info["input_schema"]
if "loaded_sample" not in st.session_state:
    load_sample(DEFAULT_SAMPLE)

st.markdown("**Start from a sample profile** (made-up data), then adjust any field.")
cols = st.columns(len(SAMPLES))
for col, name in zip(cols, SAMPLES, strict=True):
    col.button(name, on_click=load_sample, args=(name,), use_container_width=True)

with st.form("customer"):
    for title, fields in ordered_sections(schema).items():
        st.markdown(f"#### {title}")
        left, right = st.columns(2)
        for i, spec in enumerate(fields):
            with left if i % 2 == 0 else right:
                render_input(spec)
    submitted = st.form_submit_button(
        "Estimate churn risk", type="primary", use_container_width=True
    )

if submitted:
    values = {f["name"]: st.session_state.get(widget_key(f["name"])) for f in schema}
    payload = build_payload(values, schema)
    missing = missing_fields(payload, schema)
    if missing:
        st.error("Please fill in: " + ", ".join(field_label(m) for m in missing) + ".")
    else:
        for hint in consistency_hints(payload):
            st.info(f"Check this input: {hint} The estimate is still shown.")
        try:
            with st.spinner("Estimating…"):
                st.session_state["result"] = get_client(API_URL, API_TIMEOUT).predict(payload)
        except ApiError as err:
            st.session_state.pop("result", None)
            show_api_error(err)

if "result" in st.session_state:
    show_result(st.session_state["result"])

with st.expander("About this model"):
    m = info["metrics"]["test"]
    st.markdown(
        f"- Version `{info['model_version']}` ({info['model_type'].replace('_', ' ')}), "
        f"schema {info['schema_version']}\n"
        f"- Threshold {info['threshold']['threshold']:.2f}, chosen on validation data to "
        f"catch at least {info['threshold']['min_recall']:.0%} of churners\n"
        f"- Held-out test: recall {m['recall']:.2f}, precision {m['precision']:.2f}, "
        f"ROC-AUC {m['roc_auc']:.2f}\n"
        f"- {info['intended_use']}"
    )
    top = ", ".join(field_label(s["feature"]) for s in info["model_signals"]["top"])
    st.markdown(f"Inputs this model relies on most: {top}. {info['model_signals']['caveat']}")
