# Prediction API

FastAPI service that serves the **promoted** model from the local registry. Interactive docs are at
`http://localhost:8000/docs` while it runs.

```bash
.venv/Scripts/python -m uvicorn churn.api.app:app --port 8000
```

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `CHURN_REGISTRY_DIR` | `models/registry` | Registry to load from |
| `CHURN_MODEL_VERSION` | unset | Pin an exact version instead of the promoted one (e.g. during a rollback) |
| `CHURN_SCHEMA_PATH` | `configs/schema.yaml` | Canonical schema; request validation is generated from it |
| `CHURN_MAX_BATCH_SIZE` | `100` | Max items per `/predict/batch` call (1–1000) |
| `CHURN_LOG_LEVEL` | `INFO` | Log level |

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Process is up. Always 200 while the server runs. |
| GET | `/ready` | 200 with `model_version` when a model is loaded; **503** with a reason otherwise |
| GET | `/model-info` | Version, model type, schema/mapping versions, threshold, input schema, metrics, model-level signals |
| POST | `/predict` | One customer |
| POST | `/predict/batch` | `{"customers": [...]}`, 1 to `CHURN_MAX_BATCH_SIZE` items |

The service becomes **not ready** (503 on `/ready`, `/predict*`, and `/model-info`) when:
- no version has been promoted,
- the pinned version doesn't exist,
- the model file's hash doesn't match its metadata, or
- the model was trained on a different schema version than the API validates against.

## Input contract

The input is the 18 canonical features. Field names, types, ranges, and allowed values come from
[`configs/schema.yaml`](../configs/schema.yaml) and are listed under `input_schema` in `/model-info`.
**Every field is required. Unknown fields are rejected**, so sending `customer_id`, `gender`,
or any other extra key returns 422. Callers should not send identifiers or personal data at all.

## Example (real response from the promoted model `20261027T185356Z-9178da7c`)

```bash
curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "tenure_months": 2, "monthly_charges": 70.7, "total_charges": 151.65,
  "contract": "month_to_month", "payment_method": "electronic_check",
  "paperless_billing": "yes", "senior_citizen": "no", "partner": "no", "dependents": "no",
  "phone_service": "yes", "multiple_lines": "no", "internet_service": "fiber_optic",
  "online_security": "no", "online_backup": "no", "device_protection": "no",
  "tech_support": "no", "streaming_tv": "no", "streaming_movies": "no"
}'
```

```json
{
  "churn_probability": 0.8518,
  "risk_label": "high",
  "threshold": 0.6036834115455383,
  "model_version": "20261027T185356Z-9178da7c",
  "note": "This probability is a statistical estimate from a model trained on historical sample data. It is not a guarantee of customer behaviour and should not be the sole basis for decisions about a customer."
}
```

`risk_label` is `high` when `churn_probability >= threshold`, which is the same rule used in
training. The threshold was chosen on validation data (see
[model_evaluation.md](model_evaluation.md)). Scores are uncalibrated and run high, so
`churn_probability` is best read as a ranking score.

### Validation error

The same payload with `"contract": "monthly"` returns HTTP 422:

```json
{"detail": "Invalid request",
 "errors": [{"field": "contract", "message": "Input should be 'month_to_month', 'one_year' or 'two_year'"}]}
```

Error bodies name the field and the rule, but **never echo the submitted value**. Batch errors
point at the item, e.g. `customers.1.contract`.

## Logging

Each request logs the request id (taken from the `X-Request-ID` header or generated, and returned
in the response), method, path, status, and latency. Request bodies and response contents are
never logged, and a test checks this. Structured JSON logs and Prometheus metrics are
added in Phase 8.

## Model signals

`/model-info → model_signals` lists permutation importance from validation data. It describes
which inputs this model relies on overall. It is **not causal** and **not a per-customer
explanation**. The API doesn't offer per-prediction explanations.
