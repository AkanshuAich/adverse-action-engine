"""Detecting when the world stops matching the training data.

A credit model does not fail loudly. It keeps returning probabilities, and
they keep looking reasonable, while the population it scores drifts away from
the one it was fitted on. The decisions get worse and nothing raises an error.
Monitoring is the only thing that catches this before the default rate does.

Two statistics, both standard in credit risk and both implemented here rather
than imported.

**Population Stability Index** compares binned distributions. It is the
measure a model risk function will ask for by name, and its conventional
thresholds - 0.1 and 0.25 - are widely used precisely because they are
conventional.

**Kolmogorov-Smirnov** compares cumulative distributions and needs no binning,
so it does not inherit the arbitrariness of a bin choice.

Implemented directly because the obvious library, Evidently, pulls nltk
transitively, and nltk currently carries PYSEC-2026-3740 with no fix released.
PSI is a sum over bins and KS is a maximum gap between two step functions.
Taking a vulnerable dependency for forty lines of arithmetic would be a poor
trade, and the same reasoning deferred it in week 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

from aae.domain.errors import DataValidationError
from aae.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger(__name__)

PSI_MODERATE: Final[float] = 0.10
PSI_SIGNIFICANT: Final[float] = 0.25
"""Conventional PSI thresholds.

Not derived from anything: they are the numbers the industry uses, which is
what makes a reported PSI comparable between one institution and another.
"""

DEFAULT_BINS: Final[int] = 10

_EPSILON: Final[float] = 1e-6
"""Floor applied to bin proportions.

PSI takes a logarithm of a ratio, so a bin that is empty in either sample
would otherwise give infinity. The floor makes an empty bin a large finite
contribution, which is the right behaviour: it is evidence of drift, not an
error.
"""


class DriftSeverity(StrEnum):
    """How far a feature has moved."""

    STABLE = "stable"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"

    @classmethod
    def from_psi(cls, psi: float) -> DriftSeverity:
        """Classify a PSI value.

        Args:
            psi: The computed index.

        Returns:
            The severity band.
        """
        if psi >= PSI_SIGNIFICANT:
            return cls.SIGNIFICANT
        if psi >= PSI_MODERATE:
            return cls.MODERATE
        return cls.STABLE


def population_stability_index(
    reference: Sequence[float] | np.ndarray,
    current: Sequence[float] | np.ndarray,
    *,
    bins: int = DEFAULT_BINS,
) -> float:
    """Compare two numeric distributions.

    Bin edges come from the reference sample's quantiles, not from equal-width
    bins. Credit features are heavily skewed - income especially - and
    equal-width bins would put almost every applicant in the first bin and
    report stability regardless of what happened.

    Args:
        reference: The distribution the model was fitted on.
        current: The distribution being scored now.
        bins: How many quantile bins to use.

    Returns:
        The index. Zero means identical; larger means further apart.

    Raises:
        DataValidationError: If either sample is empty after dropping missing
            values.
    """
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(current, dtype=np.float64)
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]

    if left.size == 0 or right.size == 0:
        msg = "Population stability needs non-empty reference and current samples."
        raise DataValidationError(msg)

    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(left, quantiles))
    if edges.size < 2:
        # A constant reference feature cannot drift in distribution shape.
        return 0.0

    edges[0], edges[-1] = -np.inf, np.inf

    expected = np.histogram(left, bins=edges)[0] / left.size
    actual = np.histogram(right, bins=edges)[0] / right.size

    expected = np.maximum(expected, _EPSILON)
    actual = np.maximum(actual, _EPSILON)

    return float(np.sum((actual - expected) * np.log(actual / expected)))


def categorical_stability_index(reference: Sequence[str], current: Sequence[str]) -> float:
    """Compare two categorical distributions.

    The same statistic over category frequencies rather than quantile bins. A
    category present in one sample and absent from the other contributes
    heavily, which is correct: a vanished category is drift.

    Args:
        reference: Values the model was fitted on.
        current: Values being scored now.

    Returns:
        The index.

    Raises:
        DataValidationError: If either sample is empty.
    """
    left = pd.Series(list(reference)).dropna()
    right = pd.Series(list(current)).dropna()

    if left.empty or right.empty:
        msg = "Categorical stability needs non-empty reference and current samples."
        raise DataValidationError(msg)

    categories = sorted(set(left.unique()) | set(right.unique()))
    expected = np.array([max((left == c).mean(), _EPSILON) for c in categories])
    actual = np.array([max((right == c).mean(), _EPSILON) for c in categories])

    return float(np.sum((actual - expected) * np.log(actual / expected)))


def ks_statistic(
    reference: Sequence[float] | np.ndarray, current: Sequence[float] | np.ndarray
) -> float:
    """Compare two numeric distributions without binning.

    Args:
        reference: The distribution the model was fitted on.
        current: The distribution being scored now.

    Returns:
        The largest gap between the two cumulative distributions, in [0, 1].
    """
    left = np.sort(np.asarray(reference, dtype=np.float64))
    right = np.sort(np.asarray(current, dtype=np.float64))
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]

    if left.size == 0 or right.size == 0:
        return 0.0

    grid = np.sort(np.unique(np.concatenate([left, right])))
    cdf_left = np.searchsorted(left, grid, side="right") / left.size
    cdf_right = np.searchsorted(right, grid, side="right") / right.size

    return float(np.max(np.abs(cdf_left - cdf_right)))


@dataclass(frozen=True)
class FeatureDrift:
    """How far one feature has moved.

    Attributes:
        feature: Which feature.
        psi: Population stability index.
        ks: KS statistic, for numeric features only.
        severity: Banded interpretation of the PSI.
        reference_missing: Missing-value rate when the model was fitted.
        current_missing: Missing-value rate now. A feature whose availability
            collapses is drifting even when the values that remain look
            unchanged, and only this catches that.
    """

    feature: str
    psi: float
    ks: float | None
    severity: DriftSeverity
    reference_missing: float
    current_missing: float

    @property
    def missing_rate_shift(self) -> float:
        """Change in how often the feature is absent."""
        return round(self.current_missing - self.reference_missing, 4)

    def to_dict(self) -> dict[str, Any]:
        """Render for the committed report.

        Returns:
            JSON-compatible data.
        """
        return {
            "feature": self.feature,
            "psi": round(self.psi, 4),
            "ks": round(self.ks, 4) if self.ks is not None else None,
            "severity": self.severity.value,
            "reference_missing": round(self.reference_missing, 4),
            "current_missing": round(self.current_missing, 4),
            "missing_rate_shift": self.missing_rate_shift,
        }


@dataclass(frozen=True)
class DriftReport:
    """Drift across every monitored feature, plus the score distribution."""

    features: tuple[FeatureDrift, ...]
    score_psi: float
    score_ks: float
    reference_rows: int
    current_rows: int
    model_version: str

    @property
    def score_severity(self) -> DriftSeverity:
        """How far the predicted score distribution has moved.

        The one to read first. Individual features can drift in ways that
        offset each other; if the scores have not moved, the decisions have
        not either.
        """
        return DriftSeverity.from_psi(self.score_psi)

    @property
    def drifted(self) -> tuple[FeatureDrift, ...]:
        """Features that have moved beyond the stable band."""
        return tuple(f for f in self.features if f.severity is not DriftSeverity.STABLE)

    @property
    def requires_attention(self) -> bool:
        """Whether anything has moved enough to warrant a look."""
        return bool(self.drifted) or self.score_severity is not DriftSeverity.STABLE

    def to_dict(self) -> dict[str, Any]:
        """Render for the committed report.

        Returns:
            JSON-compatible data.
        """
        return {
            "model_version": self.model_version,
            "reference_rows": self.reference_rows,
            "current_rows": self.current_rows,
            "score_psi": round(self.score_psi, 4),
            "score_ks": round(self.score_ks, 4),
            "score_severity": self.score_severity.value,
            "requires_attention": self.requires_attention,
            "drifted_features": [f.feature for f in self.drifted],
            "features": [f.to_dict() for f in self.features],
        }


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    *,
    model_version: str,
    bins: int = DEFAULT_BINS,
) -> DriftReport:
    """Compare a current population against the one the model was fitted on.

    Args:
        reference: Features the model was trained on.
        current: Features being scored now.
        reference_scores: Predicted probabilities on the reference set.
        current_scores: Predicted probabilities on the current set.
        model_version: Recorded on the report.
        bins: Quantile bins for the index.

    Returns:
        The report.
    """
    results: list[FeatureDrift] = []

    for column in reference.columns:
        if column not in current.columns:
            continue

        left, right = reference[column], current[column]
        is_numeric = pd.api.types.is_numeric_dtype(left)

        if is_numeric:
            psi = population_stability_index(
                left.dropna().to_numpy(), right.dropna().to_numpy(), bins=bins
            )
            ks: float | None = ks_statistic(left.dropna().to_numpy(), right.dropna().to_numpy())
        else:
            psi = categorical_stability_index(
                left.dropna().astype(str).tolist(), right.dropna().astype(str).tolist()
            )
            ks = None

        results.append(
            FeatureDrift(
                feature=str(column),
                psi=psi,
                ks=ks,
                severity=DriftSeverity.from_psi(psi),
                reference_missing=float(left.isna().mean()),
                current_missing=float(right.isna().mean()),
            )
        )

    report = DriftReport(
        features=tuple(sorted(results, key=lambda f: f.psi, reverse=True)),
        score_psi=population_stability_index(reference_scores, current_scores, bins=bins),
        score_ks=ks_statistic(reference_scores, current_scores),
        reference_rows=len(reference),
        current_rows=len(current),
        model_version=model_version,
    )

    if report.requires_attention:
        logger.warning(
            "drift_detected",
            score_severity=report.score_severity.value,
            score_psi=round(report.score_psi, 4),
            features=[f.feature for f in report.drifted],
        )

    return report
