"""Measuring disparate impact.

Protected attributes never enter the model. That is necessary and nowhere near
sufficient: a model that has never seen sex can still decline women at a
markedly higher rate, because the features it does see are correlated with the
ones it does not. Excluding an attribute prevents *disparate treatment*; only
measurement detects *disparate impact*, and the second is the one that
survives an audit.

This module measures the impact and reports it. It deliberately does not
correct it, and that is a substantive position rather than an omission - see
:data:`MITIGATION_POSITION`.

Two families of measure, because they answer different questions.

**Selection-rate measures** - the adverse impact ratio and demographic parity -
ask whether groups are approved at similar rates. They need no outcome labels,
so they can be run on live decisions where repayment is not yet known. The
four-fifths rule, a ratio below 0.8, is the long-standing screening threshold
in US enforcement practice.

**Error-rate measures** - equalized odds - ask whether the model is *wrong* in
the same way across groups. A model can hit demographic parity while being far
more likely to wrongly decline a creditworthy applicant from one group, which
is arguably the worse failure and is invisible to a selection-rate check.
These need outcomes, so they run on historical data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)

from aae.logging import get_logger
from aae.ml.features import PROTECTED_ATTRIBUTES

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger(__name__)

FOUR_FIFTHS: Final[float] = 0.80
"""Adverse impact ratio below which a disparity warrants investigation.

From US enforcement practice rather than statute. Not a safe harbour: a ratio
above 0.8 does not make a practice lawful, and one below it does not make it
unlawful. It is a screening threshold, and it is used here as one.
"""

AGE_BANDS: Final[tuple[tuple[str, int, int], ...]] = (
    ("under 25", 0, 25),
    ("25 to 34", 25, 35),
    ("35 to 44", 35, 45),
    ("45 to 54", 45, 55),
    ("55 and over", 55, 200),
)

MITIGATION_POSITION: Final[str] = """\
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
"""


def age_band(days_birth: float) -> str:
    """Convert a Home Credit birth offset to an age band.

    Args:
        days_birth: Days before application, stored negative.

    Returns:
        The band label.
    """
    years = abs(days_birth) / 365.25
    for label, lower, upper in AGE_BANDS:
        if lower <= years < upper:
            return label
    return AGE_BANDS[-1][0]


@dataclass(frozen=True)
class GroupResult:
    """One protected attribute's results.

    Attributes:
        attribute: Which characteristic was measured.
        selection_rates: Decline rate by group.
        group_sizes: How many applicants fell in each group.
        adverse_impact_ratio: Least-favoured rate over most-favoured. One means
            parity; below :data:`FOUR_FIFTHS` warrants investigation.
        demographic_parity_difference: Largest gap in decline rates.
        equalized_odds_difference: Largest gap in error rates. ``None`` when
            outcomes were not supplied.
        true_positive_rates: Recall by group, where outcomes were supplied.
        false_positive_rates: Wrongly-declined rate by group.
    """

    attribute: str
    selection_rates: dict[str, float]
    group_sizes: dict[str, int]
    adverse_impact_ratio: float
    demographic_parity_difference: float
    equalized_odds_difference: float | None
    true_positive_rates: dict[str, float] = field(default_factory=dict)
    false_positive_rates: dict[str, float] = field(default_factory=dict)

    @property
    def passes_four_fifths(self) -> bool:
        """Whether the adverse impact ratio clears the screening threshold."""
        return self.adverse_impact_ratio >= FOUR_FIFTHS

    @property
    def most_affected_group(self) -> str:
        """The group declined most often."""
        return max(self.selection_rates, key=lambda key: self.selection_rates[key])

    def to_dict(self) -> dict[str, Any]:
        """Render for the committed report.

        Returns:
            JSON-compatible data.
        """
        return {
            "attribute": self.attribute,
            "selection_rates": self.selection_rates,
            "group_sizes": self.group_sizes,
            "adverse_impact_ratio": self.adverse_impact_ratio,
            "demographic_parity_difference": self.demographic_parity_difference,
            "equalized_odds_difference": self.equalized_odds_difference,
            "true_positive_rates": self.true_positive_rates,
            "false_positive_rates": self.false_positive_rates,
            "passes_four_fifths": self.passes_four_fifths,
        }


@dataclass(frozen=True)
class FairnessReport:
    """Disparate impact across every protected attribute measured."""

    groups: tuple[GroupResult, ...]
    applications: int
    overall_decline_rate: float
    model_version: str
    mitigation_position: str = MITIGATION_POSITION

    @property
    def findings(self) -> tuple[GroupResult, ...]:
        """Attributes whose ratio falls below the screening threshold."""
        return tuple(group for group in self.groups if not group.passes_four_fifths)

    def to_dict(self) -> dict[str, Any]:
        """Render for the committed report.

        Returns:
            JSON-compatible data.
        """
        return {
            "model_version": self.model_version,
            "applications": self.applications,
            "overall_decline_rate": self.overall_decline_rate,
            "groups": [group.to_dict() for group in self.groups],
            "findings": [group.attribute for group in self.findings],
            "mitigation_position": self.mitigation_position,
        }


def _rates_by_group(frame: MetricFrame, key: str) -> dict[str, float]:
    return {
        str(group): round(float(value), 4)
        for group, value in frame.by_group[key].items()
        if not pd.isna(value)
    }


def _adverse_impact_ratio(decline_rates: dict[str, float]) -> float:
    """Compute the ratio of favourable outcome rates.

    Expressed on *approval* rates, which is how the four-fifths rule is
    conventionally stated: the least-favoured group's approval rate divided by
    the most-favoured group's.

    Args:
        decline_rates: Decline rate by group.

    Returns:
        The ratio, or 1.0 when there is nothing to compare.
    """
    approvals = [1.0 - rate for rate in decline_rates.values()]
    if len(approvals) < 2:
        return 1.0
    best = max(approvals)
    return round(min(approvals) / best, 4) if best > 0 else 0.0


def evaluate_group(
    attribute: str,
    groups: Sequence[str],
    declined: np.ndarray,
    outcomes: np.ndarray | None = None,
) -> GroupResult:
    """Measure disparate impact for one protected attribute.

    Args:
        attribute: Which characteristic this is.
        groups: Group membership per applicant.
        declined: 1 where the application was declined.
        outcomes: True default labels, where known. Error-rate measures are
            omitted without them rather than guessed at.

    Returns:
        The measured result.
    """
    series = pd.Series(list(groups), name=attribute)

    # Selection-rate measures need no outcomes, so y_true is filled with the
    # predictions: fairlearn requires the argument, and selection_rate ignores it.
    frame = MetricFrame(
        metrics={"selection_rate": selection_rate},
        y_true=declined,
        y_pred=declined,
        sensitive_features=series,
    )
    rates = _rates_by_group(frame, "selection_rate")
    sizes = {str(name): int(count) for name, count in series.value_counts().items()}

    equalized_odds: float | None = None
    tpr: dict[str, float] = {}
    fpr: dict[str, float] = {}

    if outcomes is not None and len(np.unique(outcomes)) > 1:
        error_frame = MetricFrame(
            metrics={
                "true_positive_rate": true_positive_rate,
                "false_positive_rate": false_positive_rate,
            },
            y_true=outcomes,
            y_pred=declined,
            sensitive_features=series,
        )
        tpr = _rates_by_group(error_frame, "true_positive_rate")
        fpr = _rates_by_group(error_frame, "false_positive_rate")
        equalized_odds = round(
            float(equalized_odds_difference(outcomes, declined, sensitive_features=series)), 4
        )

    return GroupResult(
        attribute=attribute,
        selection_rates=rates,
        group_sizes=sizes,
        adverse_impact_ratio=_adverse_impact_ratio(rates),
        demographic_parity_difference=round(
            float(demographic_parity_difference(declined, declined, sensitive_features=series)), 4
        ),
        equalized_odds_difference=equalized_odds,
        true_positive_rates=tpr,
        false_positive_rates=fpr,
    )


def analyse_fairness(
    applications: pd.DataFrame,
    declined: np.ndarray,
    *,
    model_version: str,
    use_outcomes: bool = True,
) -> FairnessReport:
    """Measure disparate impact across every protected attribute present.

    Args:
        applications: Source data, including the protected attributes. They are
            loaded for exactly this purpose and are never model features.
        declined: 1 where the application was declined.
        model_version: Recorded on the report.
        use_outcomes: Include error-rate measures when ``TARGET`` is present.

    Returns:
        The report.
    """
    outcomes = (
        applications["TARGET"].to_numpy()
        if use_outcomes and "TARGET" in applications.columns
        else None
    )

    results: list[GroupResult] = []

    for attribute in sorted(PROTECTED_ATTRIBUTES):
        if attribute not in applications.columns:
            continue
        column = applications[attribute]
        groups = (
            [age_band(value) for value in column] if attribute == "DAYS_BIRTH" else column.tolist()
        )
        results.append(evaluate_group(attribute, groups, declined, outcomes))

    report = FairnessReport(
        groups=tuple(results),
        applications=len(applications),
        overall_decline_rate=round(float(np.mean(declined)), 4),
        model_version=model_version,
    )

    for finding in report.findings:
        logger.warning(
            "adverse_impact_finding",
            attribute=finding.attribute,
            ratio=finding.adverse_impact_ratio,
            most_affected=finding.most_affected_group,
            detail="Below the four-fifths screening threshold; investigate and document.",
        )

    return report
