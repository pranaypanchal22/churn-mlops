# Monitoring and retraining

Everything here runs locally with no external services. Prometheus is optional.

## What is monitored, and what each signal can and cannot tell you

| Signal | Source | Needs labels? | What it tells you | What it does **not** tell you |
|---|---|---|---|---|
| API health | `/health`, `/ready`, `churn_model_ready` | No | The process is up and a valid model is loaded | Whether predictions are good |
| Traffic and errors | `churn_http_requests_total{route,status}`, `churn_validation_errors_total{field}` | No | Volume, 4xx/5xx rates, which fields callers get wrong | Anything about customers |
| Latency | `churn_http_request_duration_seconds{route}` | No | p50/p95 per route | |
| Score distribution | `churn_prediction_probability`, `churn_predictions_total{risk_label}` | No | Whether the share flagged high-risk is changing | Whether those flags are correct |
| **Input drift** | `/monitoring/drift`, `churn_feature_psi{feature}`, `scripts/drift_report.py` | No | Incoming data differs from the training data | **Whether the model got worse** |
| **Model performance** | `scripts/drift_report.py` on a batch **with outcomes** | **Yes** | Actual precision/recall/PR-AUC on recent customers | (This is the only real quality signal) |

**Drift is not degradation.** A shift in inputs can leave accuracy unchanged, and accuracy can fall
with no visible input drift (for example, when customer behaviour changes for the same profile).
The system never infers performance from drift. Reports state "performance: not measured" unless
outcomes are present.

## Privacy by design

- Logs are JSON lines carrying `request_id`, `method`, `route`, `status`, `latency_ms`, and
  `model_version`. Request and response bodies are never logged. Uvicorn's own access log is turned off to
  avoid duplicate lines.
- Metric labels are limited to route templates, status codes, risk labels, and feature **names**.
- Live drift keeps **only aggregate bin counts** in memory. Rows are dropped right after scoring.
  Counts reset when the process restarts.
- The reference profile saved with each model holds bin edges and proportions, not rows. It is 4.2 KB
  for the real model.

Tests check that submitted values appear in neither logs nor `/metrics`.

## Drift method

At training time, `reference_profile.json` is saved next to the model. It contains:
- numeric features: decile bin edges from the **training split**, the share of rows in each bin, and the min/max;
- categorical features: the share of each level;
- scores: the share of **validation-split** probabilities in 10 fixed bins, plus the share at or above the threshold.

Incoming data goes through the same bins, and each feature gets a **Population Stability Index**.
Thresholds live in [`configs/monitoring.yaml`](../configs/monitoring.yaml):

| PSI | Status |
|---|---|
| < 0.10 | ok |
| 0.10 – 0.25 | warn |
| ≥ 0.25 | alert |
| fewer than `min_samples` (200) rows | insufficient_data |

A score warning is also raised if the share flagged high-risk moves by 10 percentage points or more. Values outside
the training range and categories never seen in training are counted separately.

These PSI cut-offs are common rules of thumb, not calibrated limits.

## Results on 2026-10-27 (served model `20261028T040654Z-ecd0b68e`)

Batches were drawn from `CustomerChurn.csv` with `scripts/drift_report.py`:

| Batch | Rows | Status | Notes |
|---|---:|---|---|
| Random sample, labels removed | 1,000 | **ok** | No feature drift. Score PSI 0.005. Flagged 32.4% vs 32.9% on validation. Performance: not measured. |
| Customers with tenure ≤ 12 on month-to-month contracts, labels removed | 1,994 | **alert** (exit 3) | PSI: tenure 5.01, total charges 3.92, contract 3.74, plus 10 smaller shifts. Score PSI 2.97. Flagged 66.9% vs 32.9%. Performance: **not measured**. |
| Same random sample with labels | 1,000 | ok | Performance measured: ROC-AUC 0.825, PR-AUC 0.619, precision 0.531, recall 0.672. Change vs test: −0.018, −0.015, −0.018, −0.037. |

Caveats on the labelled row: about 60% of those rows are expected to be in the training split (4,225 of 7,043 rows), so this isn't an
independent estimate. It's a demonstration of the labelled path only. The second batch is a deliberately
narrow segment. The alert is correct because the inputs really differ, but nothing here shows the model performs worse on
that segment. Only outcomes for those customers could show that.

Live check in Docker: 400 rows from the second batch were sent through `/predict/batch`.
`/monitoring/drift` reported `alert` with the same top features (tenure 5.03 PSI). In Prometheus, the target was
`up`, all 5 rules loaded, and both drift alerts went to `pending`, waiting out their 30-minute `for:`
window.

## Running it

```bash
# Batch report: data quality + drift (+ performance if the file has the Churn column)
python scripts/drift_report.py --input path/to/batch.csv
#   -> reports/monitoring/<timestamp>_<file>.{json,md}
#   exit 0 ok/warn/insufficient, 1 invalid data, 2 config/model problem, 3 drift alert

# Live (API running)
curl localhost:8000/monitoring/drift
curl localhost:8000/metrics

# Optional Prometheus with alert rules (http://localhost:9090 -> Alerts)
docker compose --profile monitoring up -d
```

Alert rules ([`monitoring/alerts.yml`](../monitoring/alerts.yml), checked with `promtool`):
model not ready (2 min), 5xx rate > 5% (5 min), p95 `/predict` latency > 500 ms (10 min), feature
PSI > 0.25 or score PSI > 0.25 with at least 200 monitored rows (30 min). No Alertmanager or notification channel is
configured.

## Controlled retraining

```bash
python scripts/retrain.py --data path/to/new.csv                      # dry run: judge only
python scripts/retrain.py --data path/to/new.csv --promote --reason "..."  # promote if it passes
```

Steps, in [`src/churn/models/retrain.py`](../src/churn/models/retrain.py):

1. **Validate** the new data with the same schema and mapping. If it fails, the run is **rejected** before any training.
2. **Train a candidate** with the base config, changing only the data path. The candidate is registered but not served.
3. **Compare** the candidate and the current model on the **same rows**, the candidate's held-out test split. Each
   model uses its own validation-chosen threshold.
4. **Gate** with [`configs/promotion.yaml`](../configs/promotion.yaml):
   - absolute floors: PR-AUC ≥ 0.55, recall ≥ 0.65, precision ≥ 0.45;
   - allowed regression vs current: PR-AUC drop ≤ 0.01, ROC-AUC drop ≤ 0.01, Brier increase ≤ 0.02;
   - same schema version as the current model.
5. **Decide:**
   - `rejected` if any criterion fails;
   - `eligible_not_promoted` if all pass but `--promote` wasn't given;
   - `promoted` only if all pass **and** `--promote` was given;
   - `no_change` if the candidate is already the served version.

Every run writes `reports/retraining/<run_id>/` (`decision.json`, the exact `train_config.yaml`, the
`promotion_policy.yaml`, and `validation_report.json`) and appends a line to
`models/registry/promotions.jsonl`.

A scheduled job should run **without** `--promote`. A person then reviews the decision and promotes
with `scripts/promote.py --version ... --reason "..."`. After a promotion, restart the API
(`docker compose restart api`) so it loads the new version.

Caveat built into each decision record: if the new data overlaps the current model's training data, the
current model has already seen some of the comparison rows, which favours keeping it. That makes the gate conservative.

### Real run on 2026-10-27

`scripts/retrain.py --data data/raw/CustomerChurn.csv` (dry run) trained candidate
`20261028T040654Z-ecd0b68e`. It matched the then-current baseline on every compared metric (PR-AUC 0.6338, ROC-AUC 0.8424,
recall 0.7086, precision 0.5487, Brier 0.1678). All 7 criteria passed, and the outcome was `eligible_not_promoted`, so serving was
unchanged. It was then promoted manually with a reason pointing at the decision record. The point of that
promotion was to give the served model a reference profile for drift monitoring. The model
itself is unchanged.
