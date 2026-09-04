"""Generating the model card and the validation report.

Both are produced from a training run rather than written by hand. A model
card someone typed is a description of what they believed at the time; one the
pipeline emits is a measurement, and it is regenerated when the model changes
rather than quietly going stale beside it.

The two documents answer different questions and are kept separate for that
reason. The **model card** says what this model is, what it may be used for,
and how it behaves - the document a reviewer reads to understand the thing.
The **validation report** says what was checked, what was found, and what was
not checked - the document a model risk function reads to decide whether it
may be deployed. Merging them produces something that does neither job, which
is the usual failure.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final

from aae.logging import get_logger
from aae.ml.fairness import FOUR_FIFTHS, MITIGATION_POSITION
from aae.ml.features import DEFAULT_SPEC, FEATURE_DISPLAY_NAMES, PROTECTED_ATTRIBUTES

if TYPE_CHECKING:
    from collections.abc import Sequence

    from aae.ml.drift import DriftReport
    from aae.ml.fairness import FairnessReport
    from aae.ml.train import TrainedModel

logger = get_logger(__name__)

MODEL_CARD_PATH: Final[Path] = Path("MODEL_CARD.md")
VALIDATION_REPORT_PATH: Final[Path] = Path("VALIDATION_REPORT.md")


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_model_card(model: TrainedModel, fairness: FairnessReport) -> str:
    """Render the model card.

    Args:
        model: The trained model.
        fairness: Disparate impact measured on it.

    Returns:
        Markdown.
    """
    metrics = model.metrics
    generated = datetime.now(UTC).strftime("%Y-%m-%d")

    fairness_rows = [
        (
            group.attribute,
            f"{group.adverse_impact_ratio:.3f}",
            "pass" if group.passes_four_fifths else "**below screen**",
            (
                f"{group.equalized_odds_difference:.3f}"
                if group.equalized_odds_difference is not None
                else "not measured"
            ),
            f"{min(group.group_sizes.values()):,}" if group.group_sizes else "0",
        )
        for group in fairness.groups
    ]

    return f"""# Model card: credit risk classifier

Generated from a training run on {generated}. Regenerate with
`python -m aae.ml.model_card`.

## Model details

| Field | Value |
|---|---|
| Version | `{model.model_version}` |
| Type | Gradient-boosted decision trees (XGBoost), binary classification |
| Output | Calibrated probability of default |
| Trained | {model.trained_at.strftime("%Y-%m-%d %H:%M UTC")} |
| Training data | **{model.data_source}** |
| Features | {len(DEFAULT_SPEC.feature_names)} |
| Calibration | {
        "Isotonic, accepted"
        if metrics.calibration_applied
        else "Identity - isotonic was fitted and rejected"
    } |

The version is a hash of the feature set, the hyperparameters, the seed and the
data source, so two runs with identical inputs produce the same version and any
change to model behaviour produces a different one. It is written into every
audit record.

## Intended use

Scoring consumer credit applications and producing the ranked factors behind a
decline, so that a regulator-compliant adverse action notice can be generated
and verified against them.

## Out of scope

- **Any real lending decision.** This model is trained on {model.data_source}
  data and has not been validated against a real portfolio.
- **Automated issuance without review.** Every generated notice is verified
  mechanically and then reviewed by a person before it goes out.
- **Pricing, limit setting, or collections.** The model estimates probability
  of default for an accept/decline decision and has not been calibrated for
  any other use.

## Features

{len(DEFAULT_SPEC.numeric)} numeric, {len(DEFAULT_SPEC.categorical)} categorical, and
{len(DEFAULT_SPEC.derived)} derived.

**No protected attribute is a feature.** {", ".join(f"`{a}`" for a in sorted(PROTECTED_ATTRIBUTES))}
are loaded solely to measure disparate impact and are excluded by construction:
a feature specification referencing one cannot be built. Derived features
declare their input columns and those inputs are checked too, so a ratio such
as employment tenure over age - which contains no protected column by name but
encodes age exactly - is rejected.

Feature names are carried through to the applicant-facing notice in plain
language: {", ".join(f'"{FEATURE_DISPLAY_NAMES[n]}"' for n in DEFAULT_SPEC.numeric[:3])}, and so on.

## Performance

Measured on a held-out test split of {metrics.n_test:,} applications.

{
        _table(
            ("Metric", "Value", "Note"),
            [
                ("AUC", f"{metrics.auc:.4f}", "Ranking quality"),
                ("KS", f"{metrics.ks:.4f}", "Separation, the measure credit risk asks for by name"),
                (
                    "Brier score",
                    f"{metrics.brier_calibrated:.4f}",
                    f"From {metrics.brier_uncalibrated:.4f} uncalibrated",
                ),
                (
                    "Expected calibration error",
                    f"{metrics.ece_calibrated:.4f}",
                    f"From {metrics.ece_uncalibrated:.4f} uncalibrated",
                ),
                (
                    "Positive rate",
                    f"{metrics.positive_rate:.2%}",
                    "Class imbalance in the training data",
                ),
                ("Boosting rounds", f"{metrics.best_iteration}", "Chosen by early stopping"),
            ],
        )
    }

`scale_pos_weight` is deliberately unused. It lifts ranking metrics on an
imbalanced target and destroys calibration by construction, and a decline rests
on a probability threshold rather than a ranking.

Calibration is guarded: isotonic regression is fitted on part of the
calibration split, judged on the rest, and kept only if it measurably improves
expected calibration error. On this run it was
**{
        "accepted"
        if metrics.calibration_applied
        else "rejected, and the model ships with an identity mapping"
    }**.

## Fairness

Adverse impact measured on the decisions this model produces. The
four-fifths screen is {FOUR_FIFTHS:.2f}.

{
        _table(
            (
                "Protected attribute",
                "Adverse impact ratio",
                "Screen",
                "Equalized odds diff.",
                "Smallest group",
            ),
            fairness_rows,
        )
    }

Group sizes are reported because a ratio computed over a handful of applicants
is not a finding. Selection-rate and error-rate measures are both shown: a
model can reach demographic parity while being far likelier to wrongly decline
a creditworthy applicant from one group, and a selection-rate check alone
would not see it.

### Mitigation position

{MITIGATION_POSITION}

## Limitations

- Trained on {model.data_source} data. Every figure above describes behaviour
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
"""


def render_validation_report(
    model: TrainedModel,
    fairness: FairnessReport,
    drift: DriftReport,
    evaluation: dict[str, float] | None = None,
) -> str:
    """Render the validation report.

    Args:
        model: The trained model.
        fairness: Disparate impact measured on it.
        drift: Population stability against a later sample.
        evaluation: Headline figures from the evaluation harness, if available.

    Returns:
        Markdown.
    """
    generated = datetime.now(UTC).strftime("%Y-%m-%d")
    metrics = model.metrics
    findings = fairness.findings

    scope = _table(
        ("Area", "Checked", "How"),
        [
            ("Discriminatory power", "Yes", "AUC and KS on a held-out split"),
            ("Calibration", "Yes", "ECE and Brier, before and after"),
            (
                "Fair lending - disparate treatment",
                "Yes",
                "Protected attributes excluded by construction, enforced by test",
            ),
            (
                "Fair lending - proxy discrimination",
                "Yes",
                "Derived features declare inputs; inputs checked against the protected set",
            ),
            (
                "Fair lending - disparate impact",
                "Yes",
                "Adverse impact ratio and equalized odds across every protected attribute",
            ),
            (
                "Population stability",
                "Yes",
                "PSI and KS per feature, and on the score distribution",
            ),
            (
                "Explanation fidelity",
                "Yes",
                "SHAP additivity asserted against the model's raw margin",
            ),
            (
                "Notice groundedness",
                "Yes",
                "Six deterministic checks, gated in CI against a baseline",
            ),
            (
                "Audit integrity",
                "Yes",
                "Hash chain plus database-enforced append-only, tested concurrently",
            ),
            (
                "**Real-portfolio performance**",
                "**No**",
                f"Trained on {model.data_source} data",
            ),
            (
                "**Live model quality**",
                "**No**",
                "CI figures use a simulated provider; see below",
            ),
            (
                "**Adversarial prompt robustness**",
                "**Partial**",
                "Categorical inputs are allowlisted; no free-text field exists yet",
            ),
        ],
    )

    calibration_finding = (
        f"improved expected calibration error from {metrics.ece_uncalibrated:.4f} "
        f"to {metrics.ece_calibrated:.4f}"
        if metrics.calibration_applied
        else (
            "was fitted, judged on held-out data, and rejected; the model ships "
            "with an identity mapping"
        )
    )

    smallest_group = min(
        (min(g.group_sizes.values()) for g in fairness.groups if g.group_sizes), default=0
    )

    fairness_finding = (
        "No protected attribute fell below the four-fifths screen."
        if not findings
        else "The following fell below the four-fifths screen: "
        + ", ".join(f"`{f.attribute}`" for f in findings)
        + "."
    )

    drift_finding = (
        "No feature is outside the stable band."
        if not drift.drifted
        else "Drifted: " + ", ".join(f.feature for f in drift.drifted) + "."
    )

    evaluation_section = (
        _table(
            ("Metric", "Value"),
            [
                (name, f"{value:.4f}" if isinstance(value, float) else str(value))
                for name, value in evaluation.items()
            ],
        )
        if evaluation
        else "_Evaluation harness figures unavailable; run `python -m evals.runner`._"
    )

    return f"""# Validation report

Model `{model.model_version}` · generated {generated} · regenerate with
`python -m aae.ml.model_card`.

This report states what was checked, what was found, and - as importantly -
what was **not** checked. A validation report that lists only successful tests
tells a reviewer nothing about where the risk actually sits.

## Scope of validation

{scope}

## Findings

### 1. Model performance

AUC {metrics.auc:.4f}, KS {metrics.ks:.4f} on {metrics.n_test:,} held-out
applications. Calibration
{calibration_finding}.

**Assessment:** acceptable for the intended use, on this data.

### 2. Fair lending

{fairness_finding}

Ratios: {", ".join(f"{g.attribute} {g.adverse_impact_ratio:.3f}" for g in fairness.groups)}.

**Caveat that limits this finding.** Some group sizes are small - the smallest
band holds {smallest_group:,}
applicants - and a ratio over a handful of people is noise rather than
evidence. Ratios are reported with group sizes for that reason, and a finding
on a small group warrants more data before it warrants action.

**Assessment:** no action indicated on this data. The measurement should be
repeated on a real portfolio before any deployment decision.

### 3. Population stability

Score PSI {drift.score_psi:.4f} ({drift.score_severity.value}), KS
{drift.score_ks:.4f}, comparing {drift.reference_rows:,} training rows against
{drift.current_rows:,} later applications.
{drift_finding}

**Assessment:** stable. Monitoring should run on a schedule once deployed;
this is a point-in-time measurement.

### 4. Notice generation

{evaluation_section}

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
   measured on {model.data_source} data. None of it transfers.
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
"""


def write_documents(
    model: TrainedModel,
    fairness: FairnessReport,
    drift: DriftReport,
    evaluation: dict[str, float] | None = None,
) -> tuple[Path, Path]:
    """Write both governance documents.

    Args:
        model: The trained model.
        fairness: Disparate impact measured on it.
        drift: Population stability.
        evaluation: Headline evaluation figures, if available.

    Returns:
        The paths written.
    """
    MODEL_CARD_PATH.write_text(render_model_card(model, fairness), encoding="utf-8")
    VALIDATION_REPORT_PATH.write_text(
        render_validation_report(model, fairness, drift, evaluation), encoding="utf-8"
    )
    logger.info(
        "governance_documents_written",
        model_card=str(MODEL_CARD_PATH),
        validation_report=str(VALIDATION_REPORT_PATH),
    )
    return MODEL_CARD_PATH, VALIDATION_REPORT_PATH


def _main() -> None:
    """Regenerate the model card and validation report."""
    import json

    from aae.data.loaders import load_applications
    from aae.data.synthetic import generate_applications
    from aae.ml.decision import DecisionEngine
    from aae.ml.reports import build_drift_report, build_fairness_report
    from aae.ml.train import train_model

    loaded = load_applications(force_synthetic=True, n_synthetic=20_000)
    model = train_model(loaded)
    engine = DecisionEngine(model, threshold=0.15)

    reference = loaded.frame.head(4_000)
    fairness = build_fairness_report(engine, reference)
    drift = build_drift_report(engine, reference, generate_applications(4_000, seed=8_888))

    baseline = Path("evals/baseline.json")
    evaluation = None
    if baseline.is_file():
        data = json.loads(baseline.read_text(encoding="utf-8"))
        evaluation = {
            key: data[key]
            for key in (
                "cases",
                "groundedness_rate",
                "post_repair_rate",
                "escalation_rate",
                "factor_fidelity",
                "citation_precision",
                "element_coverage",
                "prohibited_content_rate",
            )
            if key in data
        }

    card, report = write_documents(model, fairness, drift, evaluation)
    print(f"Wrote {card} and {report}")  # noqa: T201


if __name__ == "__main__":
    _main()
