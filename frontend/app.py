"""Streamlit frontend for the churn API: an interactive churn-risk explorer.

Run:  streamlit run frontend/app.py
Env:  CHURN_API_URL (default http://localhost:8000), CHURN_API_TIMEOUT seconds (default 5)

Enter a customer's details in the form and press "Get prediction"; results appear beside
the form. Each submit calls the API once (/predict, plus one /predict/batch for the what-if
charts). Every number shown comes from the API; no model, data, or customer identifiers are
loaded or stored here.
"""

import os

import pandas as pd
import streamlit as st

from api_client import ApiError, ChurnApiClient
from charts import (
    GAUGE_HEIGHT,
    risk_gauge,
    signals_chart,
    tenure_chart,
    what_if_chart,
    what_if_height,
)
from form_logic import (
    SAMPLES,
    build_payload,
    consistency_hints,
    dependent_updates,
    describe_field_errors,
    describe_risk,
    estimate_total,
    field_help,
    field_label,
    missing_fields,
    ordered_sections,
    tenure_scenarios,
    value_label,
    what_if_scenarios,
)

API_URL = os.environ.get("CHURN_API_URL", "http://localhost:8000")
API_TIMEOUT = float(os.environ.get("CHURN_API_TIMEOUT", "5"))
DEFAULT_SAMPLE = next(iter(SAMPLES))
ADDON_PARENT = {
    "multiple_lines": "phone_service",
    **{
        a: "internet_service"
        for a in (
            "online_security",
            "online_backup",
            "device_protection",
            "tech_support",
            "streaming_tv",
            "streaming_movies",
        )
    },
}

st.set_page_config(page_title="Churn risk explorer", page_icon="📉", layout="wide")


# Keyed by URL so a changed CHURN_API_URL never reuses a client bound to the old one.
@st.cache_resource
def get_client(base_url: str, timeout: float) -> ChurnApiClient:
    return ChurnApiClient(base_url, timeout=timeout)


@st.cache_data(ttl=60, show_spinner=False)
def get_model_info(base_url: str) -> dict:
    # Errors are raised, not cached, so a retry hits the API again.
    return get_client(base_url, API_TIMEOUT).model_info()


def client() -> ChurnApiClient:
    return get_client(API_URL, API_TIMEOUT)


def widget_key(name: str) -> str:
    return f"field_{name}"


def current_values(schema: list[dict]) -> dict:
    return {f["name"]: st.session_state.get(widget_key(f["name"])) for f in schema}


# --- Callbacks ------------------------------------------------------------------------------


def load_sample(name: str) -> None:
    for field, value in SAMPLES[name].items():
        st.session_state[widget_key(field)] = value
    st.session_state["loaded_sample"] = name


def on_sample_change() -> None:
    load_sample(st.session_state["sample_choice"])
    # A new profile invalidates the results shown for the previous one.
    for key in ("submitted", "prediction"):
        st.session_state.pop(key, None)


def on_submit() -> None:
    """Tidy the submitted values, then queue them for one prediction on this run.

    Runs as the submit button's callback, so it may still adjust the form's values
    before the widgets are drawn again.
    """
    schema = st.session_state["schema"]
    values = current_values(schema)
    notes = []
    implied = dependent_updates(values)
    if implied:
        values.update(implied)
        for field, value in implied.items():
            st.session_state[widget_key(field)] = value
        notes.append(
            "Set " + ", ".join(field_label(f) for f in implied) + " to "
            f"'{value_label(next(iter(implied.values())))}' to match the selected service."
        )
    if st.session_state.get("auto_total"):
        spec = next(f for f in schema if f["name"] == "total_charges")
        total = min(
            estimate_total(values["tenure_months"], values["monthly_charges"]), float(spec["max"])
        )
        values["total_charges"] = total
        st.session_state[widget_key("total_charges")] = total
        notes.append(f"Total charges calculated as tenure × monthly charges = {total:,.2f}.")
    st.session_state["submitted"] = {"payload": build_payload(values, schema), "notes": notes}
    st.session_state.pop("prediction", None)


# --- Rendering helpers ---------------------------------------------------------------------


def render_input(spec: dict) -> None:
    name, key = spec["name"], widget_key(spec["name"])
    label = field_label(name)
    help_text = field_help(name, spec.get("description"))

    if name == "tenure_months":
        st.slider(
            label, min_value=int(spec["min"]), max_value=int(spec["max"]), key=key, help=help_text
        )
    elif name == "total_charges":
        auto = st.toggle("Work it out from tenure × monthly charges", key="auto_total")
        st.number_input(
            label,
            key=key,
            min_value=float(spec["min"]),
            max_value=float(spec["max"]),
            step=10.0,
            format="%.2f",
            disabled=auto,
            help=help_text,
        )
    elif spec["dtype"] in ("integer", "float"):
        is_int = spec["dtype"] == "integer"
        cast = int if is_int else float
        st.number_input(
            label,
            key=key,
            min_value=cast(spec["min"]),
            max_value=cast(spec["max"]),
            step=1 if is_int else 1.0,
            format="%d" if is_int else "%.2f",
            help=help_text,
        )
    else:
        parent = ADDON_PARENT.get(name)
        disabled = bool(parent) and st.session_state.get(widget_key(parent)) == "no"
        options = spec["allowed_values"]
        if disabled:
            help_text = f"Not applicable: {field_label(parent)} is 'No'."
        if len(options) <= 3:
            st.radio(
                label,
                options,
                key=key,
                format_func=value_label,
                horizontal=True,
                help=help_text,
                disabled=disabled,
            )
        else:
            st.selectbox(
                label,
                options,
                key=key,
                format_func=value_label,
                help=help_text,
                disabled=disabled,
            )


def show_api_error(
    err: ApiError, next_step: str = "press **Retry**", retry_button: bool = True
) -> None:
    if err.kind == "unavailable":
        st.error(
            f"**The prediction service isn't reachable.** {err.message}\n\n"
            "Start it with `uvicorn churn.api.app:app --port 8000` (or `docker compose up`), "
            f"then {next_step}."
        )
    elif err.kind == "not_ready":
        st.warning(
            f"**The service is running but has no model loaded.** {err.message}\n\n"
            f"Train and promote a model (see README), then {next_step}."
        )
    elif err.kind == "invalid":
        details = describe_field_errors(err.field_errors)
        st.error(
            "**Some inputs were rejected by the API:**\n\n"
            + "\n".join(f"- {d}" for d in details or [err.message])
        )
    else:
        st.error(f"**Something went wrong on the server.** {err.message} Then {next_step}.")
    if not retry_button:
        return
    if st.button("Retry", type="primary"):
        get_model_info.clear()
        st.rerun()


# --- Results ----------------------------------------------------------------------------------


def run_prediction(payload: dict, schema: list[dict], max_batch: int) -> dict:
    """One /predict call for the answer and one batch call for the what-if charts."""
    result = client().predict(payload)
    # Scenarios are marked as exploration, so they never count as real traffic in the
    # API's drift monitoring.
    what_if = what_if_scenarios(payload, schema)
    tenure_max = int(next(f for f in schema if f["name"] == "tenure_months")["max"])
    over_time = tenure_scenarios(payload, min(tenure_max, 72))  # training data covers 0-72
    try:
        scores = client().predict_batch(
            [s["payload"] for s in what_if] + [s["payload"] for s in over_time], max_batch
        )
        scenario_error = None
    except ApiError as err:
        scores, scenario_error = [], str(err)
    return {
        "result": result,
        "what_if": what_if,
        "over_time": over_time,
        "scores": scores,
        "scenario_error": scenario_error,
    }


def show_placeholder(threshold: float) -> None:
    with st.container(border=True):
        st.markdown(
            "#### Your prediction will appear here\n"
            "Fill in the customer's details (or pick a sample), then press "
            "**Get prediction**."
        )
        st.markdown(
            "You'll get:\n"
            "- the model's **score** (0–100%) and whether it's **high or low risk**\n"
            f"- where it sits against the **decision line** ({threshold:.0%})\n"
            "- **what could change it**: one-change-at-a-time comparisons\n"
            "- how the score changes **over time** with the company"
        )


def show_results(prediction: dict, submitted: dict, info: dict, threshold: float) -> None:
    result, payload = prediction["result"], submitted["payload"]
    p = result["churn_probability"]
    high = result["risk_label"] == "high"
    headline, detail = describe_risk(p, threshold)

    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 1, 2])
        c1.metric("Model's score", f"{p:.0%}")
        c2.metric("Risk level", "High" if high else "Low")
        flagged = info.get("reference", {}).get("validation_flagged_rate")
        if flagged is not None:
            c3.metric(
                "Customers this line flags",
                f"{flagged:.0%}",
                help="Share of customers in the validation data at or above the decision line.",
            )
        # Height includes the padding above the bar that holds the decision-line label.
        st.altair_chart(risk_gauge(p, threshold), width="stretch", height=GAUGE_HEIGHT)
        color = "red" if high else "blue"
        st.markdown(f"#### :{color}[{headline}]")
        st.markdown(detail)
        for note in submitted["notes"]:
            st.info(note)
        for hint in consistency_hints(payload):
            st.info(f"Check this input: {hint}")
        st.caption(
            f"{result['note']} Scores rank customers well but run higher than real churn "
            f"rates. Model `{result['model_version']}`."
        )
    st.caption(
        "Showing results for the details you submitted. Changed something? Press "
        "**Get prediction** again."
    )

    what_if, over_time = prediction["what_if"], prediction["over_time"]
    scores, scenario_error = prediction["scores"], prediction["scenario_error"]
    tab_what_if, tab_time, tab_model, tab_how = st.tabs(
        ["🔀 What could change it?", "📈 Over time", "🧠 About the model", "❓ How it works"]
    )

    with tab_what_if:
        st.markdown(
            "Each bar changes **one thing** about this customer and asks the model again. "
            "Hover a bar for details."
        )
        if scenario_error:
            st.warning(f"Couldn't score the scenarios: {scenario_error}")
        elif what_if:
            rows = pd.DataFrame(
                {"label": [s["label"] for s in what_if], "probability": scores[: len(what_if)]}
            )
            st.altair_chart(
                what_if_chart(rows, p, threshold),
                width="stretch",
                height=what_if_height(len(rows)) + 60,
            )
            best = rows.loc[rows["probability"].idxmin()]
            if best["probability"] < p:
                st.success(
                    f"Largest drop: **{best['label']}** moves the score from {p:.0%} to "
                    f"{best['probability']:.0%}."
                )
            else:
                st.info("None of these single changes lowers this customer's score.")
        st.caption(
            "Dashed lines: grey = this customer now, yellow = decision line. This shows how "
            "the model *reacts*, not what would *cause* a customer to stay. A real retention "
            "offer would need testing with real customers."
        )

    with tab_time:
        st.markdown(
            "The same customer and plan at different points in their time with the company "
            "(total charges recalculated). The dashed line is the decision line."
        )
        if scenario_error:
            st.warning(f"Couldn't score the scenarios: {scenario_error}")
        else:
            points = pd.DataFrame(
                {
                    "tenure_months": [s["tenure_months"] for s in over_time],
                    "probability": scores[len(what_if) :],
                }
            )
            st.altair_chart(
                tenure_chart(points, int(payload["tenure_months"]), threshold), width="stretch"
            )
            crossing = points[points["probability"] < threshold]["tenure_months"]
            if high and not crossing.empty:
                st.info(
                    f"For this profile the model's score drops below the line from about "
                    f"**{int(crossing.min())} months** with the company."
                )
        st.caption("Based on one snapshot of customers, not on following people over time.")

    with tab_model:
        m = info["metrics"]["test"]
        ref = info.get("reference", {})
        st.markdown("##### How well it did on customers it never saw")
        k1, k2, k3 = st.columns(3)
        k1.metric("Leavers it caught", f"{m['recall']:.0%}", help="Recall on the test set")
        k2.metric(
            "Flagged who really left", f"{m['precision']:.0%}", help="Precision on the test set"
        )
        if "training_churn_rate" in ref:
            k3.metric("Customers who left (training data)", f"{ref['training_churn_rate']:.0%}")
        st.markdown(
            f"The decision line ({threshold:.0%}) was chosen on separate validation data to "
            f"catch at least {info['threshold']['min_recall']:.0%} of leavers while "
            "contacting as few stayers as possible."
        )
        st.markdown("##### What the model leans on most")
        signals = [{**s, "label": field_label(s["feature"])} for s in info["model_signals"]["top"]]
        st.altair_chart(signals_chart(signals), width="stretch")
        st.caption(info["model_signals"]["caveat"])
        st.markdown("##### Known limits")
        st.markdown(
            "- Trained on **one fictional dataset** from one period.\n"
            "- Catches very few leavers on **one- and two-year contracts**.\n"
            "- Scores are a ranking; they run higher than real churn rates.\n"
            f"- {info['intended_use']}"
        )
        st.caption(
            f"Model `{info['model_version']}` · {info['model_type'].replace('_', ' ')} · "
            f"schema {info['schema_version']}"
        )

    with tab_how:
        st.markdown(
            "1. **You describe a customer** in the form: plan, contract, services, charges.\n"
            "2. **Get prediction sends those 18 details** to the prediction API. No name or "
            "ID is involved.\n"
            "3. **The model returns a score** between 0% and 100%: how strongly this profile "
            "resembles customers who left in the training data.\n"
            f"4. **Scores at or above {threshold:.0%} count as high risk**: the group a "
            "retention team would contact first.\n"
            "5. **The tabs ask the model follow-up questions**: what if one thing were "
            "different, or the customer had been around longer?"
        )
        st.markdown(
            "**Good to know:** a high score is not a prediction that *this person* will "
            "leave. It means customers like them left more often in the past."
        )


# --- Page -------------------------------------------------------------------------------------

st.title("📉 Churn risk explorer")
st.markdown(
    "Enter a customer's details and press **Get prediction** to see how likely they are to "
    "**leave**, and *why the model thinks so*. "
    "<span style='opacity:.7'>Portfolio demo trained on one sample telecom dataset of "
    "fictional customers. Estimates are not guarantees.</span>",
    unsafe_allow_html=True,
)

try:
    with st.spinner("Connecting to the prediction service…"):
        info = get_model_info(API_URL)
except ApiError as err:
    show_api_error(err)
    st.stop()

schema = info["input_schema"]
st.session_state["schema"] = schema
threshold = info["threshold"]["threshold"]
max_batch = int(info.get("max_batch_size", 100))
if "loaded_sample" not in st.session_state:
    load_sample(DEFAULT_SAMPLE)
    st.session_state["sample_choice"] = DEFAULT_SAMPLE
    st.session_state["auto_total"] = False

left, right = st.columns([5, 7], gap="large")

with left:
    st.subheader("1 · Customer details")
    st.selectbox(
        "Start from a sample (made-up customers)",
        list(SAMPLES),
        key="sample_choice",
        on_change=on_sample_change,
        help="Fills in a complete profile. Change any field before submitting.",
    )
    with st.form("customer_form"):
        for title, fields in ordered_sections(schema).items():
            with st.expander(title, expanded=title == "Account"):
                for spec in fields:
                    render_input(spec)
        st.caption("Closed sections already have values. Open them to change anything.")
        st.form_submit_button(
            "Get prediction",
            type="primary",
            width="stretch",
            on_click=on_submit,
            key="submit",
            icon="🔮",
        )
    st.caption("No names, IDs or contact details are asked for or sent.")

with right:
    st.subheader("2 · Prediction")
    submitted = st.session_state.get("submitted")
    if submitted is None:
        show_placeholder(threshold)
    else:
        missing = missing_fields(submitted["payload"], schema)
        if missing:
            st.error("Please fill in: " + ", ".join(field_label(m) for m in missing) + ".")
        else:
            if "prediction" not in st.session_state:
                try:
                    with st.spinner("Asking the model…"):
                        st.session_state["prediction"] = run_prediction(
                            submitted["payload"], schema, max_batch
                        )
                    st.toast("Prediction ready", icon="✅")
                except ApiError as err:
                    show_api_error(
                        err, next_step="press **Get prediction** again", retry_button=False
                    )
            if "prediction" in st.session_state:
                show_results(st.session_state["prediction"], submitted, info, threshold)
