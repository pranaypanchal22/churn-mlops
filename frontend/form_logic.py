"""Presentation helpers for the Streamlit form: labels, samples, payload building, hints.

Nothing here predicts or decides validity. The API is the source of truth; the checks
below only add friendly, non-blocking hints before a request is sent.
"""

from __future__ import annotations

from typing import Any

# Display names for canonical fields. Unknown fields fall back to a title-cased name, so a
# schema change never breaks the form; it only looks less polished until labelled.
FIELD_LABELS = {
    "tenure_months": "Tenure (months)",
    "monthly_charges": "Monthly charges",
    "total_charges": "Total charges to date",
    "contract": "Contract",
    "payment_method": "Payment method",
    "paperless_billing": "Paperless billing",
    "senior_citizen": "Senior citizen",
    "partner": "Has partner",
    "dependents": "Has dependents",
    "phone_service": "Phone service",
    "multiple_lines": "Multiple lines",
    "internet_service": "Internet service",
    "online_security": "Online security",
    "online_backup": "Online backup",
    "device_protection": "Device protection",
    "tech_support": "Tech support",
    "streaming_tv": "Streaming TV",
    "streaming_movies": "Streaming movies",
}

VALUE_LABELS = {
    "yes": "Yes",
    "no": "No",
    "month_to_month": "Month-to-month",
    "one_year": "One year",
    "two_year": "Two year",
    "electronic_check": "Electronic check",
    "mailed_check": "Mailed check",
    "bank_transfer_auto": "Bank transfer (automatic)",
    "credit_card_auto": "Credit card (automatic)",
    "dsl": "DSL",
    "fiber_optic": "Fiber optic",
    "no_phone_service": "No phone service",
    "no_internet_service": "No internet service",
}

SECTIONS = {
    "Account": [
        "tenure_months",
        "contract",
        "payment_method",
        "paperless_billing",
        "monthly_charges",
        "total_charges",
    ],
    "Household": ["senior_citizen", "partner", "dependents"],
    "Phone": ["phone_service", "multiple_lines"],
    "Internet": [
        "internet_service",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
    ],
}

INTERNET_ADDONS = [
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
]

# Made-up demo profiles, not real customers.
SAMPLES = {
    "New month-to-month fiber customer": {
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
    },
    "Mid-tenure one-year customer, no add-ons": {
        "tenure_months": 24,
        "monthly_charges": 85.0,
        "total_charges": 2040.0,
        "contract": "one_year",
        "payment_method": "mailed_check",
        "paperless_billing": "yes",
        "senior_citizen": "yes",
        "partner": "yes",
        "dependents": "no",
        "phone_service": "yes",
        "multiple_lines": "yes",
        "internet_service": "fiber_optic",
        "online_security": "no",
        "online_backup": "no",
        "device_protection": "no",
        "tech_support": "no",
        "streaming_tv": "yes",
        "streaming_movies": "yes",
    },
    "Long-term two-year DSL customer": {
        "tenure_months": 60,
        "monthly_charges": 65.0,
        "total_charges": 3900.0,
        "contract": "two_year",
        "payment_method": "bank_transfer_auto",
        "paperless_billing": "no",
        "senior_citizen": "no",
        "partner": "yes",
        "dependents": "yes",
        "phone_service": "yes",
        "multiple_lines": "yes",
        "internet_service": "dsl",
        "online_security": "yes",
        "online_backup": "yes",
        "device_protection": "yes",
        "tech_support": "yes",
        "streaming_tv": "no",
        "streaming_movies": "no",
    },
}


def field_label(name: str) -> str:
    return FIELD_LABELS.get(name, name.replace("_", " ").capitalize())


def value_label(value: str) -> str:
    return VALUE_LABELS.get(value, str(value).replace("_", " ").capitalize())


def ordered_sections(input_schema: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group schema fields into display sections; unlisted fields go to 'Other'."""
    by_name = {f["name"]: f for f in input_schema}
    out: dict[str, list[dict[str, Any]]] = {}
    placed = set()
    for title, names in SECTIONS.items():
        fields = [by_name[n] for n in names if n in by_name]
        if fields:
            out[title] = fields
            placed.update(f["name"] for f in fields)
    rest = [f for f in input_schema if f["name"] not in placed]
    if rest:
        out["Other"] = rest
    return out


def build_payload(values: dict[str, Any], input_schema: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep only schema fields, cast numbers to the declared type. No other changes."""
    payload: dict[str, Any] = {}
    for f in input_schema:
        name = f["name"]
        if name not in values or values[name] is None:
            continue
        value = values[name]
        if f["dtype"] == "integer":
            value = int(value)
        elif f["dtype"] == "float":
            value = round(float(value), 2)
        payload[name] = value
    return payload


def missing_fields(payload: dict[str, Any], input_schema: list[dict[str, Any]]) -> list[str]:
    return [f["name"] for f in input_schema if f["name"] not in payload]


def consistency_hints(payload: dict[str, Any]) -> list[str]:
    """Combinations that never occur in the training data. Shown as warnings only."""
    hints = []
    if payload.get("internet_service") == "no":
        bad = [a for a in INTERNET_ADDONS if payload.get(a) not in (None, "no_internet_service")]
        if bad:
            hints.append(
                "Internet service is 'No', so internet add-ons are usually 'No internet "
                f"service'. Check: {', '.join(field_label(a) for a in bad)}."
            )
    elif any(payload.get(a) == "no_internet_service" for a in INTERNET_ADDONS):
        hints.append("An add-on says 'No internet service' but internet service is selected.")
    if payload.get("phone_service") == "no" and payload.get("multiple_lines") not in (
        None,
        "no_phone_service",
    ):
        hints.append("Phone service is 'No', so multiple lines is usually 'No phone service'.")
    if (
        payload.get("phone_service") == "yes"
        and payload.get("multiple_lines") == "no_phone_service"
    ):
        hints.append("Multiple lines says 'No phone service' but phone service is 'Yes'.")
    tenure, monthly, total = (
        payload.get(k) for k in ("tenure_months", "monthly_charges", "total_charges")
    )
    if tenure and monthly is not None and total is not None and total < monthly:
        hints.append("Total charges are lower than one month's charges for an existing customer.")
    return hints


def describe_field_errors(errors: list[dict[str, str]]) -> list[str]:
    return [
        f"{field_label(e.get('field', 'input'))}: {e.get('message', 'invalid value')}"
        for e in errors
    ]


# --- Interactive helpers ------------------------------------------------------------------
# These shape *requests*; every number shown to the user still comes back from the API.

FIELD_HELP = {
    "tenure_months": "How long the customer has been with the company.",
    "contract": "Customers on longer contracts are committed for longer.",
    "payment_method": "How the bill is paid. 'Automatic' means paid without customer action.",
    "paperless_billing": "Bills sent electronically instead of by post.",
    "monthly_charges": "What the customer pays each month.",
    "total_charges": "Everything billed so far. Usually close to tenure × monthly charges.",
    "multiple_lines": "More than one phone line on the account.",
    "online_security": "Internet security add-on.",
    "tech_support": "Technical support add-on.",
}

PHONE_ADDONS = ["multiple_lines"]

# Fields a user can plausibly change for a customer (e.g. by offering a different plan).
# Personal attributes such as age or household are deliberately not offered as levers.
WHAT_IF_LEVERS = [
    "contract",
    "payment_method",
    "paperless_billing",
    "online_security",
    "tech_support",
]

TENURE_STEP = 3


def field_help(name: str, schema_description: str | None = None) -> str | None:
    return FIELD_HELP.get(name) or schema_description or None


def dependent_updates(values: dict[str, Any]) -> dict[str, Any]:
    """Values implied by the parent services, mirroring the training data exactly.

    In the data, add-ons are 'no_internet_service' if and only if internet_service is 'no'
    (and likewise for the phone). Returning the implied values lets the form keep inputs
    consistent instead of letting users submit combinations the model never saw.
    """
    updates: dict[str, Any] = {}
    for parent, addons, none_token in (
        ("internet_service", INTERNET_ADDONS, "no_internet_service"),
        ("phone_service", PHONE_ADDONS, "no_phone_service"),
    ):
        if parent not in values:
            continue
        has_service = values[parent] != "no"
        for addon in addons:
            current = values.get(addon)
            if not has_service and current != none_token:
                updates[addon] = none_token
            elif has_service and current == none_token:
                updates[addon] = "no"
    return updates


def estimate_total(tenure_months: float, monthly_charges: float) -> float:
    return round(float(tenure_months) * float(monthly_charges), 2)


def what_if_scenarios(
    payload: dict[str, Any], input_schema: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """One scenario per alternative value of each lever, everything else unchanged."""
    by_name = {f["name"]: f for f in input_schema}
    scenarios = []
    for lever in WHAT_IF_LEVERS:
        spec = by_name.get(lever)
        if spec is None or lever not in payload:
            continue
        for value in spec.get("allowed_values") or []:
            if value == payload[lever]:
                continue
            changed = {**payload, lever: value}
            # Skip changes that would contradict the parent service (e.g. tech support
            # for a customer without internet).
            if dependent_updates(changed):
                continue
            scenarios.append(
                {
                    "field": lever,
                    "value": value,
                    "label": f"{field_label(lever)}: {value_label(value)}",
                    "payload": changed,
                }
            )
    return scenarios


def tenure_scenarios(
    payload: dict[str, Any], max_tenure: int, step: int = TENURE_STEP
) -> list[dict[str, Any]]:
    """Same customer at different tenures, with total charges kept consistent."""
    points = sorted(set(range(0, max_tenure + 1, step)) | {int(payload["tenure_months"])})
    monthly = payload["monthly_charges"]
    return [
        {
            "tenure_months": t,
            "payload": {**payload, "tenure_months": t, "total_charges": estimate_total(t, monthly)},
        }
        for t in points
    ]


def describe_risk(probability: float, threshold: float) -> tuple[str, str]:
    """A plain-language headline and explanation relative to the decision threshold."""
    gap = probability - threshold
    if gap >= 0.15:
        headline = "High risk: well above the line"
    elif gap >= 0:
        headline = "High risk: just above the line"
    elif gap > -0.15:
        headline = "Low risk: just below the line"
    else:
        headline = "Low risk: well below the line"
    side = "at or above" if gap >= 0 else "below"
    detail = (
        f"The model's score is **{probability:.0%}**, which is {side} its decision line of "
        f"**{threshold:.0%}**. "
        + (
            "Customers above the line are the ones a retention team would contact first."
            if gap >= 0
            else "A retention team using this model would not prioritise this customer."
        )
    )
    return headline, detail
