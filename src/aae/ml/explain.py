"""Per-decision explanation.

Turns one applicant's score into the ranked factors that produced it. This is
the ground truth the verifier checks generated reasons against: if a notice
names a factor, it must appear here, with a matching direction.

**A subtlety worth stating plainly, because a reviewer will ask.** SHAP
explains the booster's raw output, which is in log-odds, not the calibrated
probability the decision threshold is applied to. Those are different numbers.
It is still sound to explain the decision with them, because calibration is a
*monotone* map: it can move where the threshold sits, but it cannot reorder two
applicants, and it cannot change the sign of a feature's contribution. So the
ranking and direction of factors survive calibration exactly, which is all a
denial reason relies on. The magnitudes are log-odds contributions and are
never quoted to an applicant as probabilities.

Attributions are additive: the base value plus every SHAP value reconstructs
the raw log-odds. That identity is asserted in the tests, because a silently
broken explainer would produce plausible-looking reasons for the wrong
features - the worst possible failure here, since nothing downstream would
notice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd
import shap

from aae.domain.errors import ModelError
from aae.domain.models import Factor, FactorDirection
from aae.logging import get_logger
from aae.ml.features import FEATURE_DISPLAY_NAMES

if TYPE_CHECKING:
    from xgboost import XGBClassifier

logger = get_logger(__name__)

DEFAULT_TOP_K: Final[int] = 5

ADDITIVITY_TOLERANCE: Final[float] = 1e-4
"""Slack allowed when checking base + sum(shap) == raw log-odds.

The booster predicts in float32, so exact equality is unavailable; measured
reconstruction error on this model is around 1e-6.
"""


def _to_python_value(raw: object) -> float | str | None:
    """Convert a pandas cell to a JSON-safe domain value.

    Args:
        raw: A single value taken from the feature frame.

    Returns:
        ``None`` for missing values, ``float`` for numerics, ``str`` otherwise.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    if isinstance(raw, (int, float, np.integer, np.floating)):
        value = float(raw)
        return None if np.isnan(value) else value
    return str(raw)


class DecisionExplainer:
    """Explains individual predictions from a trained booster.

    Constructing the underlying SHAP explainer walks the whole tree ensemble,
    so build this once per model and reuse it across requests.
    """

    def __init__(self, booster: XGBClassifier, feature_names: tuple[str, ...]) -> None:
        """Build the explainer.

        Args:
            booster: The fitted XGBoost classifier.
            feature_names: Model input order, from the feature specification.
        """
        self._feature_names = feature_names
        self._explainer = shap.TreeExplainer(booster)

    @property
    def base_value(self) -> float:
        """The model's average output in log-odds, before any feature moves it.

        Read live from the explainer, never cached, because ``expected_value``
        is *mutated* by the first call that processes data. Read immediately
        after construction it returns a preliminary figure; on this model that
        was -2.506688 against a settled -2.526448. Caching it at construction
        introduced a constant 0.0198 error into every additivity check - small
        enough to look like a broken explainer rather than a broken constant.

        Only meaningful once the explainer has seen data. Every caller here
        computes SHAP values first.
        """
        return float(np.asarray(self._explainer.expected_value).reshape(-1)[0])

    def shap_values(self, features: pd.DataFrame) -> np.ndarray:
        """Compute raw SHAP values.

        Args:
            features: Rows in model input order.

        Returns:
            An array of shape ``(n_rows, n_features)``.

        Raises:
            ModelError: If the returned shape does not match the feature set,
                which would silently misalign every attribution.
        """
        values = np.asarray(self._explainer.shap_values(features[list(self._feature_names)]))
        expected = (len(features), len(self._feature_names))
        if values.shape != expected:
            msg = f"SHAP returned {values.shape}, expected {expected}."
            raise ModelError(msg)
        return values

    def explain_row(
        self,
        features: pd.DataFrame,
        row: int = 0,
        *,
        top_k: int = DEFAULT_TOP_K,
    ) -> tuple[Factor, ...]:
        """Explain a single application.

        Factors are ranked by the absolute size of their contribution, so the
        result is an honest account of what moved the score, in both
        directions, rather than only the unfavourable half.

        Args:
            features: Feature frame containing the row.
            row: Positional index of the row to explain.
            top_k: How many factors to return.

        Returns:
            Factors in rank order, rank 1 being the strongest contributor.
        """
        values = self.shap_values(features.iloc[[row]])[0]
        source = features.iloc[row]

        order = np.argsort(np.abs(values))[::-1][:top_k]

        return tuple(
            Factor(
                factor_id=self._feature_names[index],
                display_name=FEATURE_DISPLAY_NAMES.get(
                    self._feature_names[index], self._feature_names[index]
                ),
                value=_to_python_value(source.iloc[int(index)]),
                shap_value=float(values[index]),
                # Positive SHAP raises the log-odds of default, so it counts
                # against the applicant.
                direction=(
                    FactorDirection.ADVERSE if values[index] > 0 else FactorDirection.FAVOURABLE
                ),
                rank=rank,
            )
            for rank, index in enumerate(order, start=1)
        )

    def check_additivity(self, features: pd.DataFrame, raw_log_odds: np.ndarray) -> float:
        """Verify that attributions reconstruct the model output.

        Args:
            features: Rows that were scored.
            raw_log_odds: The booster's raw output for those rows.

        Returns:
            The largest absolute reconstruction error across the rows.
        """
        # Order matters: the SHAP values must be computed before base_value is
        # read, because that call is what settles it. See the property.
        contributions = self.shap_values(features).sum(axis=1)
        reconstructed = self.base_value + contributions
        return float(np.max(np.abs(reconstructed - raw_log_odds)))
