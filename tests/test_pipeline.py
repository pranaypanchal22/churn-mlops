import pandas as pd
import pytest

from churn.data.pipeline import DataValidationError, prepare_frame
from tests.conftest import make_raw


def _codes(report, severity=None):
    return [i.code for i in report.issues if severity is None or i.severity == severity]


def test_valid_input_produces_canonical_frame(schema, kaggle_mapping, raw):
    result = prepare_frame(raw, schema, kaggle_mapping)
    assert result.report.status == "passed", result.report.errors
    assert list(result.data.columns) == schema.feature_names + ["churn"]
    assert set(result.data["churn"].unique()) == {0, 1}
    assert result.report.schema_version == schema.schema_version
    assert result.report.mapping_version == kaggle_mapping.mapping_version


def test_identifiers_never_reach_output(schema, kaggle_mapping, raw):
    result = prepare_frame(raw, schema, kaggle_mapping)
    assert not {"Customer ID", "LoyaltyID"} & set(result.data.columns)
    assert set(result.report.identifier_columns_dropped) == {"Customer ID", "LoyaltyID"}
    # Identifier values must not appear anywhere in the report.
    dumped = result.report.model_dump_json()
    assert "0000-TEST" not in dumped and "100001" not in dumped


def test_missing_required_column_fails_with_actionable_message(schema, kaggle_mapping, raw):
    result = prepare_frame(raw.drop(columns=["Tenure"]), schema, kaggle_mapping)
    assert result.data is None
    issue = next(i for i in result.report.errors if i.code == "missing_required_column")
    assert issue.field == "tenure_months"
    assert "Available columns" in issue.message and "columns.tenure_months" in issue.message
    with pytest.raises(DataValidationError, match="tenure_months"):
        result.raise_for_errors()


def test_alias_resolves_either_source_name(schema, kaggle_mapping, raw):
    mapping = kaggle_mapping.model_copy(
        update={"columns": {**kaggle_mapping.columns, "tenure_months": ["Tenure", "Tenure Months"]}}
    )
    renamed = raw.rename(columns={"Tenure": "Tenure Months"})
    result = prepare_frame(renamed, schema, mapping)
    assert result.report.status == "passed"
    assert result.report.column_resolution["tenure_months"] == "Tenure Months"


def test_two_aliases_present_is_ambiguous_not_guessed(schema, kaggle_mapping, raw):
    mapping = kaggle_mapping.model_copy(
        update={"columns": {**kaggle_mapping.columns, "tenure_months": ["Tenure", "Tenure Months"]}}
    )
    both = raw.assign(**{"Tenure Months": raw["Tenure"]})
    result = prepare_frame(both, schema, mapping)
    assert "ambiguous_column" in _codes(result.report, "error")
    assert result.data is None


def test_unexpected_column_is_error_by_default(schema, kaggle_mapping, raw):
    result = prepare_frame(raw.assign(Region="North"), schema, kaggle_mapping)
    issue = next(i for i in result.report.errors if i.code == "unexpected_column")
    assert issue.source_column == "Region"
    assert result.report.unexpected_columns == ["Region"]


def test_unexpected_column_can_be_downgraded_to_warning(schema, kaggle_mapping, raw):
    mapping = kaggle_mapping.model_copy(update={"unexpected_columns": "warn"})
    result = prepare_frame(raw.assign(Region="North"), schema, mapping)
    assert result.report.status == "passed"
    assert "unexpected_column" in _codes(result.report, "warning")


def test_unexpected_leakage_column_is_error_even_in_warn_mode(schema, kaggle_mapping, raw):
    mapping = kaggle_mapping.model_copy(update={"unexpected_columns": "warn"})
    result = prepare_frame(raw.assign(**{"Churn Score": "80"}), schema, mapping)
    assert "unexpected_column" in _codes(result.report, "error")
    assert "churn_score" not in result.data.columns and "Churn Score" not in result.data.columns


def test_excluded_columns_are_dropped(schema, ibm_mapping):
    raw = (
        make_raw()
        .rename(
            columns={"Tenure": "Tenure Months", "Churn": "Churn Label", "Customer ID": "CustomerID"}
        )
        .drop(columns=["LoyaltyID"])
    )
    raw = raw.assign(**{"Churn Score": "80", "Churn Reason": "Moved", "Gender": "Female"})
    raw = raw.assign(**{c: "x" for c in ibm_mapping.ignored_columns})
    raw = raw.assign(**{"Churn Value": "1", "CLTV": "4000"})
    result = prepare_frame(raw, schema, ibm_mapping)
    assert result.report.status == "passed", result.report.errors
    assert set(result.report.excluded_columns_dropped) == set(ibm_mapping.excluded_columns)
    assert set(result.data.columns) == set(schema.field_names) | {"churn"}


def test_unmapped_category_value_is_reported_not_guessed(schema, kaggle_mapping, raw):
    raw.loc[0, "Contract"] = "Monthly"
    result = prepare_frame(raw, schema, kaggle_mapping)
    issue = next(i for i in result.report.errors if i.code == "invalid_category")
    assert issue.field == "contract" and issue.count == 1 and issue.examples == ["Monthly"]


def test_blank_total_charges_filled_only_when_tenure_zero(schema, kaggle_mapping, raw):
    raw.loc[0, ["Tenure", "Total Charges"]] = ["0", " "]
    result = prepare_frame(raw, schema, kaggle_mapping)
    assert result.report.status == "passed"
    assert result.data.loc[0, "total_charges"] == 0.0
    assert result.report.cleaning_actions[0]["rows_affected"] == 1
    assert result.report.missing_before_cleaning["total_charges"] == 1
    assert result.report.missing_after_cleaning["total_charges"] == 0


def test_blank_total_charges_with_tenure_stays_missing_and_is_reported(schema, kaggle_mapping, raw):
    raw.loc[0, "Total Charges"] = ""
    result = prepare_frame(raw, schema, kaggle_mapping)
    assert pd.isna(result.data.loc[0, "total_charges"])
    # 1 of 10 rows is above the 1% limit for total_charges.
    assert "missing_rate_exceeded" in _codes(result.report, "error")


def test_nonnumeric_total_charges_is_error(schema, kaggle_mapping, raw):
    raw.loc[0, "Total Charges"] = "n/a"
    result = prepare_frame(raw, schema, kaggle_mapping)
    issue = next(i for i in result.report.errors if i.code == "non_numeric_value")
    assert issue.field == "total_charges" and issue.examples == ["n/a"]


@pytest.mark.parametrize(
    ("column", "value", "code"),
    [
        ("Tenure", "-1", "below_min"),
        ("Monthly Charges", "5000", "above_max"),
        ("Tenure", "1.5", "non_integer_value"),
    ],
)
def test_range_and_type_checks(schema, kaggle_mapping, raw, column, value, code):
    raw.loc[0, column] = value
    result = prepare_frame(raw, schema, kaggle_mapping)
    assert code in _codes(result.report, "error")


def test_missing_target_is_error(schema, kaggle_mapping, raw):
    raw.loc[0, "Churn"] = ""
    result = prepare_frame(raw, schema, kaggle_mapping)
    assert "missing_target" in _codes(result.report, "error")


def test_optional_audit_field_absent_is_warning(schema, kaggle_mapping, raw):
    mapping = kaggle_mapping.model_copy(
        update={"columns": {**kaggle_mapping.columns, "gender": ["Gender"]}}
    )
    result = prepare_frame(raw, schema, mapping)
    assert result.report.status == "passed"
    assert "optional_column_absent" in _codes(result.report, "warning")
