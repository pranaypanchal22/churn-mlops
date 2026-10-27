# Data profile and findings: `CustomerChurn.csv`

All numbers below come from `python scripts/profile_data.py` run on the file described in
[Source](#source-and-license). Re-run that command to regenerate the machine-readable
report at `reports/data_profile.json`. Generated reports are not committed.

## Source and license

| Item | Value |
|---|---|
| Origin | Kaggle dataset archive, downloaded as `archive (1).zip` |
| Kaggle URL | **Unknown. Must be recorded.** |
| Download date | Not recorded. The local zip timestamp is 2026-10-26, which is probably the download date (unconfirmed). |
| File dates inside archive | 2023-06-08 |
| Underlying data | The columns match IBM's sample "Telco customer churn" dataset (fictional customers). The Kaggle uploader may have changed it, e.g. by adding `LoyaltyID`. |
| License | **Unknown. Must be verified on the Kaggle dataset page before any redistribution.** |
| File used | `CustomerChurn.csv`, 992,814 bytes, SHA-256 `79e23e3dff5595612d4db00d3f783f07c87162ed987061c5ab516611d3de25d6` |

The archive also contains other files (`Telco_customer_churn*.csv/.xlsx`). We don't use them yet.
`Telco_customer_churn.csv` uses different column names (`Tenure Months`, `Churn Label`) and has
target-derived columns (`Churn Value`, `Churn Score`, `CLTV`, `Churn Reason`). That makes it a
useful second source for testing the schema mapping later. It must never feed those columns into training.

The raw file is copied unchanged to `data/raw/CustomerChurn.csv` and is never written to.
Cleaned data goes to `data/interim/` and `data/processed/`. None of these are committed.

## Shape and target

- **7,043 rows, 21 columns.** No full-row duplicates.
- Target column `Churn`: `Yes` = churned (positive class, 1,869 rows, 26.54%), `No` =
  retained (5,174 rows, 73.46%). The classes are imbalanced (about 1 : 2.8), so accuracy alone is
  misleading. Use precision, recall, and PR-AUC.

## Identifiers (never features)

| Column | Notes |
|---|---|
| `Customer ID` | Unique per row (7,043 distinct). This is the primary key. |
| `LoyaltyID` | Integer, **not unique**: 22 values are shared by 44 rows, and in the rows checked the shared IDs belong to different `Customer ID`s. It has no meaning as a feature and is treated as an identifier. |

## Leakage

This file has no target-derived columns (no churn reason, score, CLTV, or status). The
profiler's name-pattern check found none. The canonical schema will still keep an explicit denylist, because
other files in the same archive do contain such columns.

## Column types and values

- **No blanks** in any column except `Total Charges`. No values have stray whitespace, apart from the
  blank `Total Charges` cells, which are a single space.
- Numeric: `Tenure` (integer 0–72, mean 32.37), `Monthly Charges` (18.25–118.75), and
  `Total Charges` (18.8–8,684.8 where parseable). No negative values.
- `Senior Citizen` is `Yes`/`No` here. In other versions of this dataset it is `0`/`1`, so
  the mapping layer must normalize it explicitly.
- Service fields use three levels, e.g. `Yes`/`No`/`No internet service` and
  `Yes`/`No`/`No phone service`. The "No … service" level always matches the parent
  service being `No`: 1,526 rows for internet and 682 for phone, counts consistent across all fields.
- `Contract`: Month-to-month (3,875), One year (1,473), Two year (1,695).
- `Payment Method`: 4 levels, including `(automatic)` suffixes.
- No `Gender` column in this file.

## `Total Charges` findings and cleaning rule

- **11 values are blank (a single space). No other non-numeric values.**
- All 11 have `Tenure = 0`, and they are the only rows with tenure 0. All 11 have
  `Churn = No`.
- For tenure ≥ 1, `Total Charges` is close to `Tenure × Monthly Charges`: ratio p01 = 0.855,
  median = 1.0, p99 = 1.159. No row has a total below its monthly charge.

**Rule adopted (Phase 2):** strip whitespace, then parse as a number. Blank → missing. If tenure is 0 and the total
is missing, set it to `0.0`: a new customer has not been billed yet, which fits the
evidence above. Any other unparseable value is a **validation error**, not a silent
fill. This decision depends on this dataset, and it is recorded in the cleaning config.

## Duplicates after dropping identifiers

- 41 rows duplicate another row once `Customer ID` and `LoyaltyID` are removed.
- 20 groups of identical feature rows have **conflicting** `Churn` labels. That is label noise
  no model can resolve. Rows are kept, because separate customers can share the same attributes,
  and removing them would bias the class balance.
- Duplicate feature rows can land in both the train and test splits. With 41 rows out of 7,043 the
  effect is small, but it is listed as a limitation.

## Implications for modeling

1. Drop `Customer ID` and `LoyaltyID` before training, and check this with a test.
2. Use a stratified split, because the positive class is 26.5%.
3. Choose the threshold on validation data. The default 0.5 is not tuned to the business goal.
4. The "No internet/phone service" levels repeat information. One-hot encoding handles this.
   For linear models, regularization deals with the collinearity.
