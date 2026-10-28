"""Exit codes and outputs of the command-line scripts users and CI actually run."""

import importlib.util
import json
from pathlib import Path

import pytest

from churn.models import registry
from tests.conftest import IBM_MAPPING_PATH, KAGGLE_MAPPING_PATH, SCHEMA_PATH, make_smoke_config

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


@pytest.fixture
def synthetic_csv(tmp_path):
    out = tmp_path / "raw.csv"
    assert _script("make_synthetic_data")(["--out", str(out), "--rows", "200"]) == 0
    return out


def test_profile_missing_input_exits_2(tmp_path, capsys):
    assert _script("profile_data")(["--input", str(tmp_path / "nope.csv")]) == 2
    assert "Dataset not found" in capsys.readouterr().err


def test_profile_writes_reports(synthetic_csv, tmp_path):
    assert (
        _script("profile_data")(["--input", str(synthetic_csv), "--out-dir", str(tmp_path / "r")])
        == 0
    )
    assert (tmp_path / "r" / "data_profile.json").exists()


def _prepare(csv, tmp_path, mapping=KAGGLE_MAPPING_PATH, schema=SCHEMA_PATH):
    return _script("prepare_data")(
        [
            "--input",
            str(csv),
            "--mapping",
            str(mapping),
            "--schema",
            str(schema),
            "--out-dir",
            str(tmp_path / "out"),
            "--report-dir",
            str(tmp_path / "rep"),
        ]
    )


def test_prepare_passes_and_writes_data_and_report(synthetic_csv, tmp_path):
    assert _prepare(synthetic_csv, tmp_path) == 0
    report = json.loads((tmp_path / "rep" / "validation_telco_kaggle.json").read_text())
    assert report["status"] == "passed"
    assert report["source"]["processed_sha256"]
    assert (tmp_path / "out" / "telco_kaggle_canonical.csv").exists()


def test_prepare_wrong_mapping_exits_1_and_writes_no_data(synthetic_csv, tmp_path):
    assert _prepare(synthetic_csv, tmp_path, mapping=IBM_MAPPING_PATH) == 1
    report = json.loads((tmp_path / "rep" / "validation_ibm_telco.json").read_text())
    assert report["status"] == "failed"
    assert not (tmp_path / "out").exists()


def test_prepare_bad_config_exits_2(synthetic_csv, tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("mapping_name: x\n", encoding="utf-8")
    assert _prepare(synthetic_csv, tmp_path, mapping=bad) == 2
    assert "Config error" in capsys.readouterr().err


def test_train_then_promote_cli(tmp_path, capsys):
    cfg = make_smoke_config(tmp_path)
    assert _script("train")(["--config", str(cfg)]) == 0
    assert "Not promoted" in capsys.readouterr().out

    reg = tmp_path / "registry"
    version = registry.list_versions(reg)[0]
    promote = _script("promote")
    with pytest.raises(SystemExit):  # argparse: a reason is mandatory
        promote(["--registry", str(reg), "--version", version])
    assert promote(["--registry", str(reg), "--version", "missing", "--reason", "x"]) == 1
    assert promote(["--registry", str(reg), "--version", version, "--reason", "ci"]) == 0
    assert registry.read_current(reg)["version"] == version

    assert promote(["--registry", str(reg), "--list"]) == 0
    assert f"* {version}" in capsys.readouterr().out


def test_train_aborts_on_invalid_data(tmp_path, capsys):
    cfg = make_smoke_config(tmp_path)
    raw = tmp_path / "raw.csv"
    raw.write_text(raw.read_text().replace("Month-to-month", "Monthly", 1), encoding="utf-8")
    assert _script("train")(["--config", str(cfg)]) == 1
    assert "invalid_category" in capsys.readouterr().err
    assert not (tmp_path / "registry").exists()
