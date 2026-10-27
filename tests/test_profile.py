from pathlib import Path

from churn.data.profile import profile_csv, render_markdown

CSV = """LoyaltyID,Customer ID,Tenure,Monthly Charges,Total Charges,Contract,Churn Reason,Churn
1,A-1,0,20.0, ,Month-to-month,,No
2,A-2,10,50.0,500.0,One year,,No
3,A-3,5,70.0,350.0,Month-to-month,Moved,Yes
4,A-4,2,30.0,abc,Two year,,No
"""


def _write(tmp_path: Path) -> Path:
    path = tmp_path / "raw.csv"
    path.write_text(CSV, encoding="utf-8")
    return path


def test_profile_reports_target_and_shape(tmp_path):
    p = profile_csv(_write(tmp_path))
    assert p["shape"] == {"rows": 4, "columns": 8}
    assert p["target"]["values"] == {"No": 3, "Yes": 1}


def test_blank_and_nonnumeric_total_charges_are_counted(tmp_path):
    p = profile_csv(_write(tmp_path))
    tc = p["total_charges_checks"]
    assert tc["n_unparseable"] == 2
    assert tc["tenure_values_where_unparseable"] == {"0": 1, "2": 1}
    # The whitespace-only value counts as blank, "abc" as non-numeric.
    assert p["column_profiles"]["Total Charges"]["n_blank"] == 1


def test_identifiers_and_leakage_candidates_flagged(tmp_path):
    p = profile_csv(_write(tmp_path))
    ids = {c["column"] for c in p["identifier_candidates"]}
    assert {"LoyaltyID", "Customer ID"} <= ids
    assert p["leakage_candidates_by_name"] == ["Churn Reason"]


def test_profile_does_not_modify_source(tmp_path):
    path = _write(tmp_path)
    before = path.read_bytes()
    render_markdown(profile_csv(path))
    assert path.read_bytes() == before
