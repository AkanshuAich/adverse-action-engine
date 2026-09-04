# Validation report

Model `xgb-d6907b2a` · generated 2026-09-04 · regenerate with
`python -m aae.ml.model_card`.

This report states what was checked, what was found, and - as importantly -
what was **not** checked. A validation report that lists only successful tests
tells a reviewer nothing about where the risk actually sits.

## Scope of validation

| Area | Checked | How |
|---|---|---|
| Discriminatory power | Yes | AUC and KS on a held-out split |
| Calibration | Yes | ECE and Brier, before and after |
| Fair lending - disparate treatment | Yes | Protected attributes excluded by construction, enforced by test |
| Fair lending - proxy discrimination | Yes | Derived features declare inputs; inputs checked against the protected set |
| Fair lending - disparate impact | Yes | Adverse impact ratio and equalized odds across every protected attribute |
| Population stability | Yes | PSI and KS per feature, and on the score distribution |
| Explanation fidelity | Yes | SHAP additivity asserted against the model's raw margin |
| Notice groundedness | Yes | Six deterministic checks, gated in CI against a baseline |
| Audit integrity | Yes | Hash chain plus database-enforced append-only, tested concurrently |
| **Real-portfolio performance** | **No** | Trained on synthetic data |
| **Live model quality** | **No** | CI figures use a simulated provider; see below |
| **Adversarial prompt robustness** | **Partial** | Categorical inputs are allowlisted; no free-text field exists yet |

## Findings

### 1. Model performance

AUC 0.8949, KS 0.6319 on 4,000 held-out
applications. Calibration
improved expected calibration error from 0.0300 to 0.0101.

**Assessment:** acceptable for the intended use, on this data.

### 2. Fair lending

No protected attribute fell below the four-fifths screen.

Ratios: CODE_GENDER 0.954, DAYS_BIRTH 0.807, NAME_FAMILY_STATUS 0.919.

**Caveat that limits this finding.** Some group sizes are small - the smallest
band holds 18
applicants - and a ratio over a handful of people is noise rather than
evidence. Ratios are reported with group sizes for that reason, and a finding
on a small group warrants more data before it warrants action.

**Assessment:** no action indicated on this data. The measurement should be
repeated on a real portfolio before any deployment decision.

### 3. Population stability

Score PSI 0.0022 (stable), KS
0.0105, comparing 4,000 training rows against
4,000 later applications.
No feature is outside the stable band.

**Assessment:** stable. Monitoring should run on a schedule once deployed;
this is a point-in-time measurement.

### 4. Notice generation

| Metric | Value |
|---|---|
| cases | 100 |
| groundedness_rate | 0.5400 |
| post_repair_rate | 0.9700 |
| escalation_rate | 0.0300 |
| factor_fidelity | 0.8826 |
| citation_precision | 0.8200 |
| element_coverage | 0.9975 |
| prohibited_content_rate | 0.0000 |

**How to read these.** They measure how the *system* handles a fixed
distribution of model mistakes - whether the verifier catches them and whether
the repair prompt is specific enough to fix them. They are **not** a claim
about any real model's quality. Groundedness is measured on the first attempt,
before repair, because reporting the post-repair figure would credit the
verifier's work to the model.

The figure that must be zero is prohibited content in an issued notice. A
model *proposing* a prohibited reason is the case the check exists for and is
counted separately.

**Assessment:** the controls work on the tested distribution. Live figures
require a run against a real provider and are not produced by CI.

## What is not validated

1. **Real-portfolio behaviour.** Every performance and fairness figure here is
   measured on synthetic data. None of it transfers.
2. **Live model quality.** The CI gate uses a simulated provider so that it is
   reproducible and needs no credential. A real backend must be measured
   separately before deployment.
3. **Small-group fairness findings.** Ratios over small bands are not
   actionable evidence.
4. **Prompt injection through free text.** The payload sent to a model is built
   from an allowlist of named numeric and categorical fields, so there is
   currently no free-text path. Adding one - an applicant's written statement,
   say - would require detection-based defences and a fresh assessment.
5. **Long-run calibration.** Calibration is measured once, at training. It
   decays with drift and is not currently re-measured on live outcomes.

## Conditions for deployment

- Retrain and revalidate on real portfolio data.
- Measure notice quality against the production provider, not the simulator.
- Schedule the drift and fairness reports rather than running them by hand.
- Confirm the human review step is staffed; the escalation path is load-bearing
  and assumes someone is at the other end of it.
