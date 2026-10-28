# Architecture

## Components

```mermaid
flowchart LR
    subgraph Offline["Offline (CLI scripts / trainer container)"]
        RAW[("data/raw/<br/>CustomerChurn.csv")]
        MAP["configs/mappings/*.yaml<br/>+ configs/schema.yaml"]
        PIPE["Validation pipeline<br/>map → clean → validate"]
        TRAIN["Training<br/>LogReg vs RandomForest<br/>threshold on validation"]
        REG[("models/registry/<br/>versions + current.json<br/>+ promotions.jsonl")]
        RETRAIN["Gated retraining<br/>configs/promotion.yaml"]
        DRIFT["Batch drift / performance<br/>report"]
        MLF[("MLflow SQLite<br/>(optional)")]
        RAW --> PIPE
        MAP --> PIPE
        PIPE --> TRAIN --> REG
        TRAIN -.-> MLF
        RETRAIN --> PIPE
        RETRAIN --> TRAIN
        RETRAIN -- "promote only if criteria pass<br/>and --promote" --> REG
        REG --> DRIFT
    end

    subgraph Online["Online (docker compose)"]
        UI["Streamlit frontend<br/>:8501"]
        API["FastAPI<br/>:8000<br/>/predict /ready /metrics<br/>/monitoring/drift"]
        PROM["Prometheus :9090<br/>(optional profile)"]
        UI -- "HTTP JSON" --> API
        PROM -- "scrape /metrics" --> API
    end

    REG -- "read-only mount<br/>(promoted version)" --> API
    USER(("User")) --> UI
```

| Component | Code | Responsibility |
|---|---|---|
| Canonical schema | `configs/schema.yaml`, `src/churn/schema/config.py` | The one contract for feature names, types, ranges, and allowed values. Also lists forbidden source-name patterns. |
| Source mappings | `configs/mappings/*.yaml` | Per-source column aliases, value maps, identifier/excluded/ignored columns, and cleaning rules |
| Validation pipeline | `src/churn/data/pipeline.py` | Turns raw tables into canonical ones and collects every issue into a versioned report |
| Training | `src/churn/features/build.py`, `src/churn/models/{train,evaluate}.py` | Leakage guard, one sklearn Pipeline, candidate comparison, threshold choice, and the single test evaluation |
| Registry | `src/churn/models/registry.py` | Immutable, hash-checked versions, the `current.json` pointer, and an append-only audit log |
| API | `src/churn/api/` | Request validation generated from the schema, predictions, readiness, metrics, and live drift |
| Frontend | `frontend/` | Form built from `/model-info`. Talks to the API over HTTP only. |
| Monitoring | `src/churn/monitoring/`, `monitoring/` | Reference profiles, PSI drift, batch reports, Prometheus rules |
| Retraining | `src/churn/models/retrain.py` | Validate → candidate → compare → gate → explicit promotion |

## Request path

```mermaid
sequenceDiagram
    participant U as User
    participant F as Streamlit
    participant A as FastAPI
    participant S as ModelService
    participant M as LiveMonitor
    U->>F: fill form, submit
    F->>A: POST /predict (18 canonical features only)
    A->>A: validate against schema-generated model<br/>(unknown fields -> 422, value not echoed)
    A->>S: predict_proba(rows)
    S->>S: to_model_frame -> sklearn Pipeline
    S->>M: add to aggregate bin counts (rows discarded)
    S-->>A: probability
    A-->>F: probability, risk label, threshold, model_version, note
    A->>A: JSON access log + Prometheus counters (no payload)
    F-->>U: result card
```

## Model lifecycle

```mermaid
stateDiagram-v2
    [*] --> Registered: train.py / retrain.py
    Registered --> Rejected: retrain criteria fail
    Registered --> Eligible: retrain criteria pass (dry run)
    Eligible --> Served: promote.py --reason / retrain --promote
    Registered --> Served: promote.py --reason (manual, audited)
    Served --> Registered: another version promoted
    note right of Served
      API loads it at start-up
      (restart after promotion)
    end note
```

Nothing becomes the served model without an explicit command, and every decision is appended to
`models/registry/promotions.jsonl`.

## Design decisions

| Decision | Why |
|---|---|
| Raw schema kept separate from the canonical schema, with an explicit YAML mapping | New sources need a reviewed mapping. Same-named columns can mean different things (`dependents` differs between the two files in the archive). |
| All preprocessing inside one sklearn `Pipeline` | Fit on training rows only, and shipped as one artifact, so training and serving can't diverge |
| Category levels come from the schema, with `handle_unknown="ignore"` | The encoded layout stays the same across retrains, and unexpected values at serving time don't crash |
| Threshold chosen on validation with a stated goal | Turning a score into an action is a business choice. The test set is used once, only to report results. |
| File-based registry and promotion pointer | Runs locally without services. Promotion is a visible, audited file change. |
| Models mounted read-only into the API | The serving process can't change what it serves |
| Request validation generated from the schema, plus a schema-version check at load | The API and the training data share one definition of valid input |
| Drift stored as aggregate counts only | Monitoring without keeping customer-level records |
| Drift reported separately from performance | Drift is not degradation, and performance needs outcomes |
| MLflow and Prometheus optional | The core project runs with nothing but Python or Docker |

## Repository layout

```
configs/            schema, source mappings, training, monitoring, promotion policy
src/churn/          library code: schema, data, features, models, api, monitoring
frontend/           Streamlit app (own requirements; no model code)
scripts/            CLI entry points: profile, prepare, train, promote, retrain, drift_report
tests/              pytest suite (synthetic data only)
docker/             API and frontend Dockerfiles
monitoring/         Prometheus scrape config and alert rules
docs/               this documentation
data/ models/ reports/   local only, gitignored
```
