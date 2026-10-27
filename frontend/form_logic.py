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
