"""Probability calibration.

Isotonic regression is fitted with scikit-learn at training time, but the
fitted result is reduced to what it actually is: a monotone step function
described by a list of knots. Storing those knots rather than the estimator
object buys three things.

* **No pickle, anywhere.** The knots are plain numbers in JSON.
* **No scikit-learn at serving time.** Prediction is linear interpolation.
* **Auditability.** A reviewer can read the mapping and see exactly how a raw
  score became the probability quoted in a denial notice. An opaque binary
  cannot be reviewed; a table of knots can.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Self

import numpy as np

from aae.domain.errors import ModelError

if TYPE_CHECKING:
    from sklearn.isotonic import IsotonicRegression

PROBABILITY_EPSILON: Final[float] = 1e-6
"""How far a fitted calibrator's output is kept from 0 and 1.

Applies to fitted calibrators only. The identity calibrator's knots are 0.0 and
1.0 by definition and describe a mapping, not a posterior, so they are left
alone.
"""


@dataclass(frozen=True)
class Calibrator:
    """A fitted isotonic mapping from raw score to calibrated probability.

    Attributes:
        thresholds: Raw-score knot positions, strictly increasing.
        values: Calibrated probability at each knot, monotone non-decreasing.
    """

    thresholds: tuple[float, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate the knot table.

        Raises:
            ModelError: If the knots are empty, mismatched, non-monotone, or
                map outside the unit interval.
        """
        if not self.thresholds or len(self.thresholds) != len(self.values):
            msg = (
                f"Calibrator needs matching non-empty knots; got "
                f"{len(self.thresholds)} thresholds and {len(self.values)} values."
            )
            raise ModelError(msg)

        thresholds = np.asarray(self.thresholds, dtype=np.float64)
        values = np.asarray(self.values, dtype=np.float64)

        if np.any(np.diff(thresholds) <= 0):
            msg = "Calibrator thresholds must be strictly increasing."
            raise ModelError(msg)
        if np.any(np.diff(values) < 0):
            msg = "Calibration must be monotone: a higher score cannot map to a lower risk."
            raise ModelError(msg)
        if values.min() < 0.0 or values.max() > 1.0:
            msg = f"Calibrated values must be probabilities; got [{values.min()}, {values.max()}]."
            raise ModelError(msg)

    @classmethod
    def identity(cls) -> Self:
        """Return a calibrator that leaves scores unchanged.

        Used when fitting a calibrator would make probabilities worse than the
        booster's own. Representing "no calibration" as an identity mapping
        rather than as ``None`` keeps the serving path and the artifact format
        uniform: there is always a calibrator, and it is always auditable.

        Returns:
            The identity mapping over the unit interval.
        """
        return cls(thresholds=(0.0, 1.0), values=(0.0, 1.0))

    @property
    def is_identity(self) -> bool:
        """Whether this calibrator leaves scores unchanged."""
        return self.thresholds == (0.0, 1.0) and self.values == (0.0, 1.0)

    @classmethod
    def from_sklearn(cls, fitted: IsotonicRegression) -> Self:
        """Extract the knot table from a fitted scikit-learn estimator.

        Args:
            fitted: An ``IsotonicRegression`` that has been fitted.

        Returns:
            The equivalent pure-data calibrator.

        Raises:
            ModelError: If the estimator has not been fitted.
        """
        if not hasattr(fitted, "X_thresholds_"):
            msg = "IsotonicRegression must be fitted before extracting knots."
            raise ModelError(msg)
        return cls(
            thresholds=tuple(float(x) for x in fitted.X_thresholds_),
            # Clamped away from the boundaries. Isotonic regression pools
            # adjacent scores, and any pooled region whose outcomes are all one
            # class yields a knot of exactly 0.0 or 1.0 - on this model that was
            # 43% of applications receiving a stated 0% probability of default.
            # No statistical model licenses certainty, an audit record asserting
            # it is indefensible to a reviewer, and log-odds of 0 or 1 are not
            # finite. Clamping is applied here rather than at prediction time so
            # the stored calibrator.json is itself honest and inspectable.
            values=tuple(
                min(max(float(y), PROBABILITY_EPSILON), 1.0 - PROBABILITY_EPSILON)
                for y in fitted.y_thresholds_
            ),
        )

    def predict(self, raw_scores: np.ndarray) -> np.ndarray:
        """Map raw model scores to calibrated probabilities.

        Scores outside the fitted range are clipped to the end knots, matching
        scikit-learn's ``out_of_bounds="clip"`` behaviour.

        Args:
            raw_scores: Uncalibrated probabilities from the booster.

        Returns:
            Calibrated probabilities, same shape as the input.
        """
        scores = np.asarray(raw_scores, dtype=np.float64)
        if self.is_identity:
            # Short-circuited rather than interpolated. Interpolating through
            # the identity knots is algebraically exact but not exact in
            # floating point, and it drifted by ~1e-9 - enough to make "an
            # unapplied calibrator changes nothing" merely approximately true.
            # For a number that ends up quoted in a regulated notice, that
            # guarantee should be exact.
            return scores
        return np.interp(
            scores,
            np.asarray(self.thresholds, dtype=np.float64),
            np.asarray(self.values, dtype=np.float64),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible mapping.

        Returns:
            The knot table.
        """
        return {"thresholds": list(self.thresholds), "values": list(self.values)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Self:
        """Reconstruct from a serialised knot table.

        Args:
            payload: A mapping produced by :meth:`to_dict`.

        Returns:
            The calibrator.

        Raises:
            ModelError: If required keys are absent.
        """
        try:
            return cls(
                thresholds=tuple(float(x) for x in payload["thresholds"]),
                values=tuple(float(y) for y in payload["values"]),
            )
        except KeyError as exc:
            msg = f"Calibrator payload is missing {exc.args[0]!r}."
            raise ModelError(msg) from exc
