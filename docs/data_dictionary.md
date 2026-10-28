# Data dictionary

The canonical schema is v1.0.0 ([`configs/schema.yaml`](../configs/schema.yaml)). The source is `CustomerChurn.csv`,
7,043 rows, mapped through [`telco_kaggle_v1.yaml`](../configs/mappings/telco_kaggle_v1.yaml).
Counts come from `scripts/profile_data.py` on that file. Field meanings follow the column names and
the IBM Telco sample dataset. **The dataset ships without an official data dictionary**, so
descriptions are interpretations, not definitions from the provider.

## Target

| Canonical | Source column | Values | Counts |
|---|---|---|---|
| `churn` | `Churn` | `Yes` → `yes` → **1** (left), `No` → `no` → **0** (stayed) | 1 = 1,869 (26.54%) · 0 = 5,174 (73.46%) |

The observation window over which churn was measured isn't documented in the dataset.

## Model features (18)

### Numeric

| Canonical | Source | Type | Schema range | Observed (min / median / max) | Notes |
|---|---|---|---|---|---|
| `tenure_months` | `Tenure` | integer | 0–120 | 0 / 29 / 72 | Months with the provider. 11 rows have 0. |
| `monthly_charges` | `Monthly Charges` | float | 0–1,000 | 18.25 / 70.35 / 118.75 | Current monthly bill. Currency not stated. |
| `total_charges` | `Total Charges` | float | 0–100,000 | 18.8 / 1,397.5 / 8,684.8 (parseable values) | Cumulative billed. 11 blank values, all with tenure 0, are set to 0.0 by a mapping rule. Any other blank is reported. Max allowed missing rate after cleaning is 1%. |

### Categorical

Source values are translated to canonical tokens through explicit value maps. Any other value is a
validation error.

| Canonical | Source | Canonical tokens (source value: count) |
|---|---|---|
| `contract` | `Contract` | `month_to_month` (Month-to-month: 3,875), `one_year` (One year: 1,473), `two_year` (Two year: 1,695) |
| `payment_method` | `Payment Method` | `electronic_check` (2,365), `mailed_check` (1,612), `bank_transfer_auto` (Bank transfer (automatic): 1,544), `credit_card_auto` (Credit card (automatic): 1,522) |
| `paperless_billing` | `Paperless Billing` | `yes` (4,171), `no` (2,872) |
| `senior_citizen` | `Senior Citizen` | `yes` (1,142), `no` (5,901). Some versions of this dataset encode it as 0/1, which would need a different value map. |
| `partner` | `Partner` | `yes` (3,402), `no` (3,641) |
| `dependents` | `Dependents` | `yes` (2,110), `no` (4,933). **Differs** in the archive's IBM file (1,627 `yes`). See [schema_mapping.md](schema_mapping.md). |
| `phone_service` | `Phone Service` | `yes` (6,361), `no` (682) |
| `multiple_lines` | `Multiple Lines` | `yes` (2,971), `no` (3,390), `no_phone_service` (682) |
| `internet_service` | `Internet Service` | `dsl` (2,421), `fiber_optic` (3,096), `no` (1,526) |
| `online_security` | `Online Security` | `yes` (2,019), `no` (3,498), `no_internet_service` (1,526) |
| `online_backup` | `Online Backup` | `yes` (2,429), `no` (3,088), `no_internet_service` (1,526) |
| `device_protection` | `Device Protection` | `yes` (2,422), `no` (3,095), `no_internet_service` (1,526) |
| `tech_support` | `Tech Support` | `yes` (2,044), `no` (3,473), `no_internet_service` (1,526) |
| `streaming_tv` | `Streaming TV` | `yes` (2,707), `no` (2,810), `no_internet_service` (1,526) |
| `streaming_movies` | `Streaming Movies` | `yes` (2,732), `no` (2,785), `no_internet_service` (1,526) |

`no_internet_service` appears exactly when `internet_service = no`, and `no_phone_service` exactly
when `phone_service = no`. This was checked on all rows.

## Audit-only field

| Canonical | Role | Notes |
|---|---|---|
| `gender` | audit, optional | Not in `CustomerChurn.csv`. The IBM file provides `Gender`. It is validated and kept in processed data so subgroup error rates can be compared, but it is **never a model input**: `assert_no_leakage` rejects it. |

## Never used as features

| Source column | Why | How it's enforced |
|---|---|---|
| `Customer ID` | Identifier, unique per row | Declared in `identifier_columns`, dropped first. Matches a forbidden pattern. |
| `LoyaltyID` | Identifier (not unique: 22 values are shared by 44 rows) | Same as above |
| `Churn Value`, `Churn Score`, `CLTV`, `Churn Reason`, `Churn Category`, `Customer Status`, `Satisfaction Score` (in other archive files) | Derived from, or recorded after, the outcome, so they would leak the target | Forbidden patterns in the schema. The mapping must list them as `excluded_columns`, or the run fails. |

## Data quality summary

- No full-row duplicates. 41 rows duplicate another once IDs are removed, and 20 groups of identical
  feature rows have conflicting labels. They are kept and listed as a limitation.
- No missing values except the 11 blank `Total Charges` described above.
- No negative or out-of-range numeric values.
