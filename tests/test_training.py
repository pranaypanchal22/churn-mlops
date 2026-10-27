import json

import numpy as np
import pytest

from churn.data.pipeline import prepare_frame
from churn.data.synthetic import make_synthetic_raw
from churn.features.build import (
    LeakageError,
    assert_no_leakage,
    build_model_pipeline,
    feature_set_from_schema,
    to_model_frame,
)
from churn.models import registry
from churn.models.evaluate import choose_threshold, threshold_metrics
from churn.models.train import SplitCfg, split_data, train
from tests.conftest import make_smoke_config


@pytest.fixture
def canonical(schema, kaggle_mapping):
    result = prepare_frame(make_synthetic_raw(400, seed=1), schema, kaggle_mapping)
    result.raise_for_errors()
    return result.data


@pytest.fixture
def features(schema):
    return feature_set_from_schema(schema)


def test_feature_set_excludes_target_audit_and_ids(schema, features):
    assert "churn" not in features.all
    assert "gender" not in features.all
    assert not any(c.split("_")[-1] == "id" for c in features.all)
    assert_no_leakage(features.all, schema)


@pytest.mark.parametrize("bad", ["churn", "gender", "customer_id", "Churn Score", "LoyaltyID"])
def test_leakage_guard_rejects(schema, kaggle_mapping, features, bad):
    with pytest.raises(LeakageError):
        assert_no_leakage(features.all + [bad], schema, kaggle_mapping)


def test_split_is_stratified_and_disjoint(canonical, features):
    X, y = to_model_frame(canonical, features), canonical["churn"].astype(int)

    s = split_data(X, y, SplitCfg(validation_size=0.2, test_size=0.2), seed=0)
    idx = [set(s[k][0].index) for k in ("train", "validation", "test")]
    assert not (idx[0] & idx[1] or idx[0] & idx[2] or idx[1] & idx[2])
    assert sum(len(i) for i in idx) == len(X)
    rates = [s[k][1].mean() for k in s]
    assert max(rates) - min(rates) < 0.05


@pytest.mark.parametrize("model", ["logistic_regression", "random_forest"])
def test_pipeline_handles_unknown_and_missing_categories(canonical, features, model):
    params = {"max_iter": 500} if model == "logistic_regression" else {"n_estimators": 20}
    pipe = build_model_pipeline(model, params, features, seed=0)
    X = to_model_frame(canonical, features)
    pipe.fit(X, canonical["churn"].astype(int))

    odd = X.head(3).copy()
    odd.loc[odd.index[0], "contract"] = "three_year"  # never seen, not in schema
    odd.loc[odd.index[1], "payment_method"] = np.nan
    odd.loc[odd.index[2], "total_charges"] = np.nan
    proba = pipe.predict_proba(odd)[:, 1]
    assert proba.shape == (3,)
    assert np.all((proba >= 0) & (proba <= 1))


def test_preprocessing_is_fit_on_training_rows_only(canonical, features):
    pipe = build_model_pipeline("logistic_regression", {"max_iter": 500}, features, seed=0)
    X = to_model_frame(canonical, features)
    train_X = X.iloc[:200]
    pipe.fit(train_X, canonical["churn"].astype(int).iloc[:200])
    scaler = pipe.named_steps["preprocess"].named_transformers_["num"].named_steps["scale"]
    np.testing.assert_allclose(scaler.mean_, train_X[features.numeric].mean().to_numpy())


def test_threshold_precision_at_min_recall_meets_target():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    proba = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
    choice = choose_threshold(y, proba, "precision_at_min_recall", min_recall=0.8)
    m = threshold_metrics(y, proba, choice["threshold"])
    assert m["recall"] >= 0.8
    assert choice["selected_on"] == "validation"


def test_threshold_metrics_label_rule():
    y = np.array([0, 1, 1, 0])
    proba = np.array([0.2, 0.6, 0.5, 0.5])
    m = threshold_metrics(y, proba, 0.5)
    # >= threshold counts as positive.
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 0, "tp": 2}


@pytest.fixture
def smoke_config(tmp_path):
    return make_smoke_config(tmp_path)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    # One shared run for the read-only artifact checks keeps the suite fast.
    return train(make_smoke_config(tmp_path_factory.mktemp("trained")))


def test_training_registers_complete_artifact(trained):
    meta = trained.metadata
    for name in (
        "model.joblib",
        "metadata.json",
        "train_config.yaml",
        "validation_report.json",
        "pr_curve_validation.csv",
        "evaluation.md",
    ):
        assert (trained.version_dir / name).exists(), name
    assert meta["threshold"]["selected_on"] == "validation"
    assert meta["data"]["schema_version"] == "1.0.0"
    assert set(meta["data"]["identifier_columns_dropped"]) == {"Customer ID", "LoyaltyID"}
    assert (
        meta["features"]["order"] == meta["features"]["numeric"] + meta["features"]["categorical"]
    )
    assert not {"churn", "gender"} & set(meta["features"]["order"])
    for split in ("validation", "test"):
        assert 0 <= meta["metrics"][split]["roc_auc"] <= 1
    # Training registers a version; it never changes what is served.
    assert registry.read_current(trained.version_dir.parent) is None


def test_loaded_artifact_predicts_valid_probabilities(trained, schema, kaggle_mapping):
    pipe, meta = registry.load_version(trained.version_dir.parent, trained.version)
    data = prepare_frame(make_synthetic_raw(20, seed=9), schema, kaggle_mapping).data
    X = to_model_frame(data, feature_set_from_schema(schema))
    assert list(X.columns) == meta["features"]["order"]
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (20,)
    assert np.all((proba >= 0) & (proba <= 1))


def test_training_is_reproducible(trained, tmp_path):
    again = train(make_smoke_config(tmp_path)).metadata
    assert again["metrics"] == trained.metadata["metrics"]
    assert again["threshold"] == trained.metadata["threshold"]


def test_tampered_artifact_is_refused(smoke_config):
    result = train(smoke_config)
    (result.version_dir / "model.joblib").write_bytes(b"tampered")
    with pytest.raises(registry.RegistryError, match="hash mismatch"):
        registry.load_version(result.version_dir.parent, result.version)


def test_set_current_and_audit_log(smoke_config):
    result = train(smoke_config)
    reg = result.version_dir.parent
    pointer = registry.set_current(reg, result.version, "test")
    registry.record_decision(reg, {"decision": "promoted", "version": result.version})
    assert registry.read_current(reg)["version"] == result.version
    assert pointer["previous_version"] is None
    log = [json.loads(line) for line in (reg / "promotions.jsonl").read_text().splitlines()]
    assert log[-1]["version"] == result.version
    with pytest.raises(registry.RegistryError):
        registry.set_current(reg, "does-not-exist", "x")
