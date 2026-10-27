"""Feature selection, leakage guards, and the preprocessing/model pipeline.

All preprocessing lives inside one scikit-learn Pipeline so it is fit on the training split
only and travels with the model artifact. The API and training share `to_model_frame` so
inputs are coerced identically in both places.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.schema.config import CanonicalSchema, SourceMapping


class LeakageError(ValueError):
    """A column that could reveal identity or the outcome was about to be used as a feature."""


@dataclass(frozen=True)
class FeatureSet:
    numeric: list[str]
    categorical: list[str]
    # Category levels come from the schema, not the data, so the API and model card can
    # describe valid inputs without looking at training rows.
    categories: dict[str, list[str]]

    @property
    def all(self) -> list[str]:
        return self.numeric + self.categorical


def feature_set_from_schema(schema: CanonicalSchema) -> FeatureSet:
    numeric, categorical, categories = [], [], {}
    for f in schema.fields:
        if f.role != "feature":
            continue
        if f.is_numeric:
            numeric.append(f.name)
        else:
            categorical.append(f.name)
            categories[f.name] = list(f.allowed_values or [])
    return FeatureSet(numeric=numeric, categorical=categorical, categories=categories)


def assert_no_leakage(
    features: list[str], schema: CanonicalSchema, mapping: SourceMapping | None = None
) -> None:
    """Fail if any proposed feature is the target, an audit field, or a source ID/outcome."""
    problems = []
    allowed = set(schema.feature_names)
    for col in features:
        if col == schema.target.name:
            problems.append(f"'{col}' is the target")
        elif col not in allowed:
            problems.append(f"'{col}' is not a canonical feature (audit fields included)")
        if schema.is_forbidden_source(col):
            problems.append(f"'{col}' matches a forbidden identifier/outcome pattern")
    if mapping is not None:
        declared = set(mapping.identifier_columns) | set(mapping.excluded_columns)
        problems += [
            f"'{c}' is a declared identifier/excluded column" for c in features if c in declared
        ]
    if problems:
        raise LeakageError("Refusing to train with leaking features: " + "; ".join(problems))


def to_model_frame(df: pd.DataFrame, features: FeatureSet) -> pd.DataFrame:
    """Select feature columns in a fixed order with dtypes scikit-learn handles reliably.

    Nullable pandas dtypes (Int64, string with pd.NA) are converted to float / object with
    np.nan, which SimpleImputer and OneHotEncoder understand.
    """
    missing = [c for c in features.all if c not in df.columns]
    if missing:
        raise KeyError(f"input is missing feature columns: {missing}")
    out = pd.DataFrame(index=df.index)
    for col in features.numeric:
        out[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    for col in features.categorical:
        out[col] = df[col].astype("object").where(df[col].notna(), np.nan)
    return out


def build_preprocessor(features: FeatureSet) -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            # Fixed categories from the schema keep the encoded layout stable across
            # retrains; anything else seen at prediction time encodes as all zeros.
            (
                "onehot",
                OneHotEncoder(
                    categories=[features.categories[c] for c in features.categorical],
                    handle_unknown="ignore",
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, features.numeric), ("cat", categorical, features.categorical)],
        verbose_feature_names_out=False,
    )


MODEL_FACTORIES = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
}


def build_model_pipeline(
    model_name: str, params: dict[str, Any], features: FeatureSet, seed: int
) -> Pipeline:
    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"unknown model '{model_name}'; choose from {sorted(MODEL_FACTORIES)}")
    estimator = MODEL_FACTORIES[model_name](**params, random_state=seed)
    return Pipeline([("preprocess", build_preprocessor(features)), ("model", estimator)])
