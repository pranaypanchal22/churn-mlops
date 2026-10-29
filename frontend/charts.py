"""Altair charts for the Streamlit app. They only draw numbers returned by the API."""

from __future__ import annotations

import altair as alt
import pandas as pd

HIGH = "#e4572e"
LOW = "#2e86ab"
NEUTRAL = "#8d99ae"
LINE = "#f5b700"
# Drawing height plus the padding around it; Streamlit needs the full box height.
GAUGE_HEIGHT = 36 + 26 + 6 + 4


def risk_gauge(probability: float, threshold: float) -> alt.LayerChart:
    """Horizontal 0-100% bar with the score filled in and the decision line marked."""
    color = HIGH if probability >= threshold else LOW
    track = pd.DataFrame({"start": [0.0], "end": [1.0]})
    fill = pd.DataFrame({"start": [0.0], "end": [probability]})
    line = pd.DataFrame({"x": [threshold], "label": [f"Decision line {threshold:.0%}"]})
    # No axis: the score and decision line are labelled, and tick labels clip at some widths.
    x = alt.X("start:Q", scale=alt.Scale(domain=[0, 1]), axis=None)

    base = alt.Chart(track).mark_bar(color="#3a3f4b", height=26, cornerRadius=13)
    bar = alt.Chart(fill).mark_bar(color=color, height=26, cornerRadius=13)
    rule = alt.Chart(line).mark_rule(color=LINE, strokeWidth=3, strokeDash=[4, 3])
    text = alt.Chart(line).mark_text(color=LINE, dy=-22, fontWeight="bold")
    return (
        alt.layer(
            base.encode(x=x, x2="end:Q"),
            bar.encode(x=x, x2="end:Q"),
            rule.encode(x="x:Q"),
            text.encode(x="x:Q", text="label:N"),
        )
        # Top padding leaves room for the decision-line label above the bar.
        .properties(height=36, padding={"top": 26, "left": 8, "right": 12, "bottom": 6})
        .configure_view(stroke=None)
    )


def what_if_height(n_rows: int) -> int:
    return 60 + 34 * n_rows


def what_if_chart(rows: pd.DataFrame, current: float, threshold: float) -> alt.LayerChart:
    """Bars per scenario, coloured by whether they cross the decision line."""
    rows = rows.assign(
        side=["Above line" if p >= threshold else "Below line" for p in rows["probability"]],
        change=[f"{p - current:+.0%}" for p in rows["probability"]],
    )
    y = alt.Y(
        "label:N",
        # Lowest score first, so the change that helps most sits at the top.
        sort=list(rows.sort_values("probability")["label"]),
        title=None,
        axis=alt.Axis(labelLimit=260),
    )
    bars = (
        alt.Chart(rows)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X(
                "probability:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", title="Model's score"),
            ),
            y=y,
            color=alt.Color(
                "side:N",
                scale=alt.Scale(domain=["Above line", "Below line"], range=[HIGH, LOW]),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Change"),
                alt.Tooltip("probability:Q", title="Score", format=".0%"),
                alt.Tooltip("change:N", title="vs now"),
            ],
        )
    )
    labels = (
        alt.Chart(rows)
        .mark_text(align="left", dx=4, color="#cfd3da")
        .encode(x="probability:Q", y=y, text="change:N")
    )
    refs = pd.DataFrame({"x": [current, threshold], "what": ["Now", "Decision line"]})
    rule = (
        alt.Chart(refs)
        .mark_rule(strokeWidth=2, strokeDash=[4, 3])
        .encode(
            x="x:Q",
            color=alt.Color(
                "what:N",
                scale=alt.Scale(domain=["Now", "Decision line"], range=[NEUTRAL, LINE]),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
        )
    )
    return (
        alt.layer(bars, labels, rule)
        .resolve_scale(color="independent")
        .properties(height=what_if_height(len(rows)))
    )


def tenure_chart(points: pd.DataFrame, current_tenure: int, threshold: float) -> alt.LayerChart:
    """Score across tenure for the same customer, with today's position marked."""
    line = (
        alt.Chart(points)
        .mark_line(color=NEUTRAL, strokeWidth=3, point=alt.OverlayMarkDef(size=30))
        .encode(
            x=alt.X("tenure_months:Q", title="Months with the company"),
            y=alt.Y(
                "probability:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", title="Model's score"),
            ),
            tooltip=[
                alt.Tooltip("tenure_months:Q", title="Months"),
                alt.Tooltip("probability:Q", title="Score", format=".0%"),
            ],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"y": [threshold]}))
        .mark_rule(color=LINE, strokeDash=[4, 3], strokeWidth=2)
        .encode(y="y:Q")
    )
    now = points[points["tenure_months"] == current_tenure]
    marker = (
        alt.Chart(now)
        .mark_point(size=220, filled=True, color=HIGH)
        .encode(
            x="tenure_months:Q",
            y="probability:Q",
            tooltip=[alt.Tooltip("probability:Q", title="Now", format=".0%")],
        )
    )
    return alt.layer(rule, line, marker).properties(height=300)


def signals_chart(signals: list[dict]) -> alt.Chart:
    rows = pd.DataFrame(signals)
    return (
        alt.Chart(rows)
        .mark_bar(color=NEUTRAL, cornerRadiusEnd=4)
        .encode(
            x=alt.X("mean_drop:Q", title="How much the model's accuracy drops without it"),
            y=alt.Y("label:N", sort="-x", title=None),
            tooltip=[alt.Tooltip("mean_drop:Q", title="Drop in PR-AUC", format=".3f")],
        )
        .properties(height=30 * len(rows) + 20)
    )
