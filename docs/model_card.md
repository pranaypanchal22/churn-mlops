# Model card: customer churn classifier

Format loosely follows *Model Cards for Model Reporting* (Mitchell et al., 2019). All numbers come
from `models/registry/<version>/metadata.json` and the scripts named below.

## Model details

| | |
|---|---|
| Served version | `20261028T040654Z-ecd0b68e`, promoted 2026-10-27. It is identical in every metric to the first baseline `20261027T185356Z-9178da7c`, and was retrained only to add a drift reference profile. |
| Type | Logistic Regression (C = 1.0, balanced class weights), inside a scikit-learn Pipeline: median imputation and standard scaling for numerics, most-frequent imputation and one-hot encoding (categories fixed by the schema, unknowns ignored) for categoricals |
| Selection | Chosen over Random Forest by validation PR-AUC (0.643 vs 0.628). The gap comes from a single split and is not shown to be meaningful. |
| Decision threshold | 0.6037. The highest-precision threshold that catches at least 70% of churners on **validation** data. |
| Inputs | 18 canonical features ([data_dictionary.md](data_dictionary.md)). No identifiers, no protected attributes. |
| Output | Churn probability (uncalibrated), and `high`/`low` risk relative to the threshold |
| Software | Python 3.13 (trained), scikit-learn 1.9.1, numpy 2.2.6, pandas 2.2.3 |
| Licence / owner | Portfolio project by the repository owner. Dataset licence unverified (see Data). |

## Intended use

- **Intended:** learning and demonstrating an end-to-end MLOps workflow, and illustrating how a
  retention team *might* rank customers for outreach.
- **Intended users:** the author and technical reviewers of the portfolio.
- **Out of scope:**
  - decisions about real customers;
  - pricing, credit, or eligibility decisions;
  - any telecom provider, market, or period other than the one in this sample;
  - automated action without a human in the loop.

## Data

| | |
|---|---|
| Source | Kaggle archive containing `CustomerChurn.csv`. Its columns match IBM's sample "Telco customer churn" data, which describes **fictional** customers. |
| Licence | **Unknown, must be verified** on the Kaggle page before redistribution |
| Size | 7,043 rows. Churn rate 26.54%. |
| Split | Stratified 60/20/20 with seed 42: train 4,225, validation 1,409, test 1,409 |
| Preprocessing | [schema_mapping.md](schema_mapping.md). Identifiers are dropped. Blank `Total Charges` with tenure 0 is set to 0.0. |

## Metrics

Test-set results were computed once, after the model and threshold were fixed
([model_evaluation.md](model_evaluation.md)):

| Split | ROC-AUC | PR-AUC | Precision | Recall | F1 | Flagged |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 0.837 | 0.643 | 0.568 | 0.703 | 0.628 | 32.9% |
| Test | 0.842 | 0.634 | 0.549 | 0.709 | 0.618 | 34.3% |

On the test set, 265 of 374 churners were flagged and 218 non-churners were flagged. Brier score is 0.168. The mean predicted
probability on validation is 0.41, against a base rate of 0.265: probabilities **run high** because of
the balanced class weights, and should be read as a ranking score.

## Subgroup results (test set, threshold 0.6037)

Computed for this card from the served model on the 1,409 test rows. Small groups give noisy
estimates, and no confidence intervals were computed.

| Group | n | Churn rate | Flagged | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| senior_citizen = no | 1,187 | 23.3% | 29.3% | 0.529 | 0.667 | 0.843 |
| senior_citizen = yes | 222 | 44.1% | 60.8% | 0.600 | 0.827 | 0.781 |
| partner = no | 736 | 33.7% | 46.1% | 0.566 | 0.774 | 0.815 |
| partner = yes | 673 | 18.7% | 21.4% | 0.507 | 0.579 | 0.856 |
| dependents = no | 978 | 31.0% | 42.8% | 0.551 | 0.762 | 0.827 |
| dependents = yes | 431 | 16.5% | 14.8% | 0.531 | 0.479 | 0.855 |
| contract = month_to_month | 773 | 42.6% | 61.3% | 0.551 | 0.793 | 0.743 |
| contract = one_year | 300 | 12.0% | 3.0% | 0.444 | 0.111 | 0.745 |
| contract = two_year | 336 | 2.7% | 0.0% | n/a (none flagged) | 0.000 | 0.761 |

What this shows:
- A single global threshold catches most churners on month-to-month contracts but **almost
  none of those on one- or two-year contracts**: 4 of 36 and 0 of 9 respectively.
- Recall is also much lower for customers with dependents or partners.
- Senior citizens are flagged at about twice the rate of others. That partly reflects their higher observed
  churn rate, but `senior_citizen` is an age-related attribute and is a model input.

Anyone adapting this model should decide, explicitly, whether those differences are acceptable for
their use, and consider per-segment thresholds or excluding age-related inputs. `gender` isn't in
this dataset, so no gender analysis was possible. The schema supports it as an audit-only field for
sources that have it.

## Model-level signals

Permutation importance on validation (drop in PR-AUC): tenure 0.238, internet service 0.111,
contract 0.068, total charges 0.039, streaming movies 0.026. **These are associations within this
model, not causes of churn**, and not explanations of individual predictions. Correlated inputs
(tenure, total charges, contract) share credit.

## Monitoring and updates

- Live and batch drift are measured against a reference profile ([monitoring.md](monitoring.md)). Drift
  is **not** treated as evidence of degradation. Measuring performance requires outcomes.
- Retraining is gated by [`configs/promotion.yaml`](../configs/promotion.yaml), and promotion is always
  explicit and audited.

## Caveats and limitations

- One fictional, single-period sample. External validity is unknown.
- Single train/validation/test split, with no confidence intervals or hyperparameter search.
- 41 rows are duplicates once IDs are removed, and some may span splits. 20 conflicting-label groups add irreducible noise.
- Probabilities are uncalibrated.
- The 70% recall goal is an assumption, not a cost model. Real offer costs and customer values aren't in the data.
- The threshold behaves very differently across contract types (see above).
- The model artifact is a joblib pickle. Load only artifacts this project produced. The registry checks
  their SHA-256 before loading.

## Ethical considerations

- Outreach driven by this score would concentrate on month-to-month, single, and senior customers. Whether
  that's desirable, and how those customers are contacted, are human decisions.
- Inputs include an age-related attribute (`senior_citizen`).
- No personal identifiers are used, logged, or stored by the serving path, and live monitoring
  keeps only aggregate counts.
