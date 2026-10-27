# Model training and evaluation

Methodology and results for the first registered model. Every number here was copied from
`models/registry/20261027T185356Z-9178da7c/` (`evaluation.md` and `metadata.json`), produced by
`python scripts/train.py` with [`configs/train.yaml`](../configs/train.yaml). The model artifacts are
not committed, so re-running training regenerates them. Three runs gave identical metrics, because the
seed, the config, and the data hash are all fixed.

## Method

| Step | Detail |
|---|---|
| Data | `CustomerChurn.csv` (SHA-256 `79e23e3d…`) run through mapping `telco_kaggle` v1.0.0 and schema v1.0.0. Training aborts if validation fails. |
| Features | The 18 canonical features. `assert_no_leakage` rejects the target, the audit field `gender`, any declared identifier or excluded column, and any name matching a forbidden ID/outcome pattern. |
| Split | Stratified 60/20/20 with seed 42: train 4,225, validation 1,409, test 1,409. Positive rate is 26.5% in each split. |
| Preprocessing | One scikit-learn `Pipeline` + `ColumnTransformer`, fit on train only. Numeric: median impute, then standard scale. Categorical: most-frequent impute, then one-hot with categories **fixed from the schema** and `handle_unknown="ignore"`. |
| Candidates | Logistic Regression (C=1, balanced class weights) and Random Forest (300 trees, min_samples_leaf=5, balanced). No hyperparameter search in v1. |
| Selection | Highest **validation** PR-AUC (average precision). |
| Threshold | Chosen on **validation**. See the operating goal below. |
| Test | Scored **once**, after model and threshold were fixed. |
| Signals | Permutation importance on validation: the drop in PR-AUC when one column is shuffled, averaged over 5 repeats. |

## Operating goal and the cost of errors

The intended use is a retention-outreach list, meaning customers to contact with an offer.

- **False positive** (flagged, would have stayed): we spend an incentive or a contact on someone who
  didn't need it. Each one costs a little, and a long list wastes the team's capacity.
- **False negative** (missed churner): we lose the chance to keep a customer. This is usually the
  costlier error per customer, but the real ratio depends on offer cost, customer value, and how
  often outreach actually works. **None of that is in this dataset.**

The stated goal is therefore **catch at least 70% of churners, and among thresholds that do, pick
the one with the highest precision** (`precision_at_min_recall`, `min_recall: 0.70`). This
is a configurable assumption, not a business-validated cost model. With real costs, use a
cost-weighted threshold instead.

## Results

### Candidate comparison (validation)

| Model | ROC-AUC | PR-AUC | Brier | Precision@0.5 | Recall@0.5 | F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** (selected) | 0.837 | 0.643 | 0.168 | 0.512 | 0.789 | 0.621 |
| Random Forest | 0.833 | 0.628 | 0.162 | 0.531 | 0.767 | 0.628 |

The two models are close. The PR-AUC gap of 0.015 comes from one validation split with no
confidence interval, so it **does not show that Logistic Regression is reliably better**. It was selected
by the configured rule, and it is also the simpler and more interpretable model.

### Selected model at the chosen threshold (0.6037)

| Split | ROC-AUC | PR-AUC | Precision | Recall | F1 | Flagged | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 0.837 | 0.643 | 0.568 | 0.703 | 0.628 | 32.9% | 263 | 200 | 111 | 835 |
| Test | 0.842 | 0.634 | 0.549 | 0.709 | 0.618 | 34.3% | 265 | 218 | 109 | 817 |

What this means on the held-out test set: flagging 34.3% of customers caught 265 of 374 churners
(70.9%). About 55% of flagged customers actually churned, against a base rate of 26.5%.
109 churners were missed. Test results are close to validation, so the threshold carried over to unseen
data from the same source.

For comparison, the default 0.5 threshold on validation flags 40.9% of customers, with recall 0.789
and precision 0.512 (576 customers flagged). The tuned threshold flags 463, which is 113 fewer contacts
(−20%), and misses 32 more churners.

The precision/recall/threshold curves are in `pr_curve_validation.csv` and `pr_curve_test.csv`
in the version folder.

### Model-level signals (validation)

| Feature | PR-AUC drop when shuffled |
|---|---:|
| tenure_months | 0.238 ± 0.007 |
| internet_service | 0.111 ± 0.014 |
| contract | 0.068 ± 0.008 |
| total_charges | 0.039 ± 0.010 |
| streaming_movies | 0.026 ± 0.008 |

These show which inputs this model relies on. They are **not causes of churn**, and they don't say
that changing a customer's contract would change their outcome. Correlated inputs share credit:
`tenure_months`, `total_charges`, and `contract` overlap heavily, so each one's individual score
understates the group.

## Limitations

- One dataset from one (fictional, IBM sample) telecom with one observation period. Nothing here shows the
  model works for any other provider, region, or time.
- A single split without confidence intervals. Small metric differences should not be over-read.
- 41 rows are duplicates once IDs are removed, and some may land in both train and test (see the
  data profile). The effect should be small at this scale.
- Probabilities have not been calibrated. Balanced class weights push scores upward, so the
  mean predicted probability on validation is 0.410, well above the 26.5% base rate. Scores are useful for ranking customers, but
  should not be read as literal churn rates. Calibration is planned future work.
- No fairness analysis yet. `gender` isn't in this file. The IBM mapping makes it available for audit.

## Reproduce

```bash
.venv/Scripts/python scripts/train.py --config configs/train.yaml
```
```bash
.venv/Scripts/python scripts/promote.py --list
```
