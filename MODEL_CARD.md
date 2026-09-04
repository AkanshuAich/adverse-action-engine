# Model card: credit risk classifier

Generated from a training run on 2026-09-04. Regenerate with
`python -m aae.ml.model_card`.

## Model details

| Field | Value |
|---|---|
| Version | `xgb-d6907b2a` |
| Type | Gradient-boosted decision trees (XGBoost), binary classification |
| Output | Calibrated probability of default |
| Trained | 2026-09-04 19:22 UTC |
| Training data | **synthetic** |
| Features | 26 |
| Calibration | Isotonic, accepted |

The version is a hash of the feature set, the hyperparameters, the seed and the
data source, so two runs with identical inputs produce the same version and any
change to model behaviour produces a different one. It is written into every
audit record.

## Intended use

Scoring consumer credit applications and producing the ranked factors behind a
decline, so that a regulator-compliant adverse action notice can be generated
and verified against them.

## Out of scope

- **Any real lending decision.** This model is trained on synthetic
  data and has not been validated against a real portfolio.
- **Automated issuance without review.** Every generated notice is verified
  mechanically and then reviewed by a person before it goes out.
- **Pricing, limit setting, or collections.** The model estimates probability
  of default for an accept/decline decision and has not been calibrated for
  any other use.

## Features

14 numeric, 7 categorical, and
5 derived.

**No protected attribute is a feature.** `CODE_GENDER`, `DAYS_BIRTH`, `NAME_FAMILY_STATUS`
are loaded solely to measure disparate impact and are excluded by construction:
a feature specification referencing one cannot be built. Derived features
declare their input columns and those inputs are checked too, so a ratio such
as employment tenure over age - which contains no protected column by name but
encodes age exactly - is rejected.

Feature names are carried through to the applicant-facing notice in plain
language: "Total annual income", "Loan amount requested", "Annual repayment amount", and so on.

## Performance

Measured on a held-out test split of 4,000 applications.

| Metric | Value | Note |
|---|---|---|
| AUC | 0.8949 | Ranking quality |
| KS | 0.6319 | Separation, the measure credit risk asks for by name |
| Brier score | 0.0544 | From 0.0568 uncalibrated |
| Expected calibration error | 0.0101 | From 0.0300 uncalibrated |
| Positive rate | 8.35% | Class imbalance in the training data |
| Boosting rounds | 30 | Chosen by early stopping |

`scale_pos_weight` is deliberately unused. It lifts ranking metrics on an
imbalanced target and destroys calibration by construction, and a decline rests
on a probability threshold rather than a ranking.

Calibration is guarded: isotonic regression is fitted on part of the
calibration split, judged on the rest, and kept only if it measurably improves
expected calibration error. On this run it was
**accepted**.

## Fairness

Adverse impact measured on the decisions this model produces. The
four-fifths screen is 0.80.

| Protected attribute | Adverse impact ratio | Screen | Equalized odds diff. | Smallest group |
|---|---|---|---|---|
| CODE_GENDER | 0.954 | pass | 0.135 | 1,418 |
| DAYS_BIRTH | 0.807 | pass | 0.868 | 18 |
| NAME_FAMILY_STATUS | 0.919 | pass | 0.256 | 159 |

Group sizes are reported because a ratio computed over a handful of applicants
is not a finding. Selection-rate and error-rate measures are both shown: a
model can reach demographic parity while being far likelier to wrongly decline
a creditworthy applicant from one group, and a selection-rate check alone
would not see it.

### Mitigation position

Measured disparity is documented and monitored, not corrected by group.

The obvious response to an adverse impact ratio below 0.8 is to adjust
thresholds per group until the rates equalise. That would be unlawful. Setting
a different decision threshold for applicants of one sex is disparate
treatment - deciding on a prohibited basis - and it does not stop being so
because the intent was to improve a fairness metric. It would also be plainly
visible in the audit log, which records the threshold applied to every
decision.

The lawful responses are to establish that each feature driving the disparity
is a legitimate, job-related business necessity; to search for a less
discriminatory alternative that meets the same business need; and to document
both. Where the disparity flows through a factor such as income - itself
unequally distributed for reasons outside the lender's control - the model is
reflecting the disparity rather than creating it. That is a finding to record
and act on, not a number to adjust away.

This module therefore reports. Deciding what to do about a finding is a
decision for a credit risk and compliance function, made on the record.


## Limitations

- Trained on synthetic data. Every figure above describes behaviour
  on that distribution and should not be read as a claim about a real
  portfolio.
- The probability is calibrated on the training population. Calibration decays
  as the population drifts; see the drift report.
- SHAP attributions explain the booster's raw log-odds, not the calibrated
  probability. Calibration is monotone, so the ranking and direction of factors
  survive it exactly, but the magnitudes are log-odds contributions and are
  never quoted to an applicant.
- The reason cap means a notice names the strongest factors, not every factor
  that counted.

## Governance

- Every decision is written to an append-only, hash-chained audit log
  recording the model version, the exact feature values, the SHAP attributions,
  the threshold in force, and the human sign-off.
- The append-only property is enforced by Postgres privileges, not by
  application code.
- Generated notices are verified against these attributions before issue, and
  a notice that cannot be verified is escalated to a person rather than sent.
