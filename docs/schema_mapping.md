# Canonical schema and source mappings

Raw source files never reach the model directly. Data goes through this flow:

```
raw CSV (source schema) --[mapping YAML]--> canonical table (model schema) --> training / API
```

| File | Owns |
|---|---|
| [`configs/schema.yaml`](../configs/schema.yaml) | Canonical field names, types, allowed tokens, ranges, required/optional, missing-rate limits, forbidden source-name patterns. Dataset-independent. |
| [`configs/mappings/<source>_v<N>.yaml`](../configs/mappings/) | How one specific source layout maps to the canonical schema: column aliases, value maps, which columns are identifiers / excluded / ignored, cleaning rules. |

Both files are versioned. Every validation report records `schema_version`, `mapping_name`,
`mapping_version`, and the source file's SHA-256. Processed data is written only when
validation passes.

## Canonical schema v1.0.0

- **Target:** `churn`. The mapping produces the tokens `yes`/`no`, which are stored as `1`/`0`.
- **Features (18):** `tenure_months`, `monthly_charges`, `total_charges`, `contract`,
  `payment_method`, `paperless_billing`, `senior_citizen`, `partner`, `dependents`,
  `phone_service`, `multiple_lines`, `internet_service`, `online_security`, `online_backup`,
  `device_protection`, `tech_support`, `streaming_tv`, `streaming_movies`.
- **Audit-only (optional):** `gender`. It is validated and kept in processed data so error rates can be
  compared across groups. It is never a model input.

See [data_dictionary.md](data_dictionary.md) (Phase 9) for per-field definitions.

## Rules the pipeline enforces

| Situation | Result |
|---|---|
| Required field: none of its aliases present | **Error** `missing_required_column`. The message lists the aliases tried and the available columns. |
| Several aliases for one field present | **Error** `ambiguous_column`. The pipeline never picks one. |
| Source column not mapped or declared | **Error** `unexpected_column` (or a warning with `unexpected_columns: warn`) |
| Unexpected column whose name looks like an ID or outcome | **Error**, even in warn mode |
| Mapping sends a forbidden source (`customer id`, `churn*`, `score`, `cltv`, `reason`, …) to a feature | **Config error** at load time |
| One source column mapped to two fields | **Config error** |
| Value not in the value map / allowed tokens | **Error** `invalid_category`, with example values |
| Non-numeric, non-integer, or out-of-range number | **Error** `non_numeric_value` / `non_integer_value` / `below_min` / `above_max` |
| Missing values after cleaning, within the field's limit | Warning `missing_values`. The model pipeline imputes them. |
| Missing values above the limit | **Error** `missing_rate_exceeded` |
| Missing or invalid target | **Error** |
| Optional field absent | Warning |

Identifier columns are dropped first. Only their *names* appear in reports, never their values.

## Cleaning rules (from the mapping)

1. Strip surrounding whitespace, then treat empty strings as missing.
2. `fill_rules`: if `total_charges` is missing **and** `tenure_months == 0`, set it to `0.0`.
   This comes from the profile, where all 11 blank totals have tenure 0. A blank total with tenure > 0
   is *not* filled. It is reported, and fails the run if it pushes the missing rate above 1%.

## Example: same field, different source names

`CustomerChurn.csv` (Kaggle) and `Telco_customer_churn.csv` (IBM extended) describe the same
customers with different layouts:

| Canonical | Kaggle mapping | IBM mapping |
|---|---|---|
| `tenure_months` | `Tenure` | `Tenure Months` |
| `churn` | `Churn` | `Churn Label` |
| `gender` (audit) | not present | `Gender` |
| identifiers | `Customer ID`, `LoyaltyID` | `CustomerID` |
| excluded (leakage) | none | `Churn Value`, `Churn Score`, `CLTV`, `Churn Reason` |
| ignored | none | `Count`, `Country`, `State`, `City`, `Zip Code`, `Lat Long`, `Latitude`, `Longitude` |

Each mapping file is self-contained. A single mapping may also list several aliases
(`tenure_months: [Tenure, Tenure Months]`). Exactly one of them must be present.

**Why mappings are not automatic:** both files pass validation, and 18 of the 19 shared
canonical columns have identical value counts. `dependents` does not: Kaggle has 2,110 `yes`, IBM has 1,627.
Columns with the same name can still carry different definitions, and a fuzzy auto-mapper would hide that.
Onboarding a new source therefore needs a human-reviewed mapping file.

## Onboarding a new dataset

1. Profile it: `python scripts/profile_data.py --input <file> --out-dir reports/<name>`.
2. Copy an existing mapping to `configs/mappings/<name>_v1.yaml`. Then:
   - list each canonical field's source column under `columns`,
   - declare every identifier, outcome-derived, and unused column,
   - write a value map for every categorical field whose source values differ from the canonical tokens,
   - add cleaning rules only when the profile supports them, with a comment explaining why.
3. Run `python scripts/prepare_data.py --input <file> --mapping configs/mappings/<name>_v1.yaml`.
4. Fix every error. Review warnings and distribution differences against the existing source.
5. If the source needs a field or token the schema lacks, change `schema.yaml` and bump
   `schema_version`. Models trained on the old version must then be retrained.
