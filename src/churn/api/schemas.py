"""Request and response models.

The request model is generated from configs/schema.yaml, so the API contract and the data
pipeline share one definition of valid inputs. The service additionally refuses to serve a
model trained against a different schema version.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from churn.schema.config import CanonicalSchema

ESTIMATE_NOTE = (
    "This probability is a statistical estimate from a model trained on historical sample "
    "data. It is not a guarantee of customer behaviour and should not be the sole basis "
    "for decisions about a customer."
)


def build_customer_model(schema: CanonicalSchema) -> type[BaseModel]:
    fields: dict[str, Any] = {}
    for spec in schema.fields:
        if spec.role != "feature":
            continue
        constraints = {"description": spec.description or None}
        if spec.is_numeric:
            base = int if spec.dtype == "integer" else float
            annotation = Annotated[base, Field(ge=spec.min, le=spec.max, **constraints)]
        else:
            annotation = Annotated[Literal[tuple(spec.allowed_values)], Field(**constraints)]
        fields[spec.name] = (annotation, ...)
    # extra="forbid" rejects identifiers or anything else not in the schema rather than
    # silently dropping it, so callers learn not to send personal data.
    return create_model(
        "CustomerFeatures",
        __config__=ConfigDict(extra="forbid", strict=False),
        **fields,
    )


def build_batch_model(customer_model: type[BaseModel], max_size: int) -> type[BaseModel]:
    return create_model(
        "BatchRequest",
        __config__=ConfigDict(extra="forbid"),
        customers=(
            Annotated[list[customer_model], Field(min_length=1, max_length=max_size)],
            ...,
        ),
    )


class Prediction(BaseModel):
    churn_probability: float = Field(ge=0, le=1)
    risk_label: Literal["high", "low"]
    threshold: float
    model_version: str
    note: str = ESTIMATE_NOTE


class BatchPredictionItem(BaseModel):
    index: int
    churn_probability: float = Field(ge=0, le=1)
    risk_label: Literal["high", "low"]


class BatchPrediction(BaseModel):
    predictions: list[BatchPredictionItem]
    threshold: float
    model_version: str
    note: str = ESTIMATE_NOTE


class ErrorItem(BaseModel):
    field: str
    message: str


class ErrorResponse(BaseModel):
    detail: str
    errors: list[ErrorItem] = []
