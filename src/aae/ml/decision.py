"""Assembling a complete credit decision.

Brings the model, the calibrator, and the explainer together into a single
:class:`~aae.domain.models.CreditDecision` carrying everything needed to
reconstruct and defend the outcome later: the exact inputs scored, the
calibrated probability, the threshold and model version applied, and the
ranked factors behind it.

The decision object is deliberately self-contained. Nothing downstream - the
notice generator, the verifier, the audit record, the underwriter console -
needs to re-run the model or re-read the source data. That is what makes a
decision reproducible years after the fact, when the model has been retrained
twice and the applicant's circumstances have changed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

from aae.domain.errors import DataValidationError
from aae.domain.models import CreditDecision, Decision
from aae.logging import get_logger
from aae.ml.explain import DEFAULT_TOP_K, DecisionExplainer, _to_python_value
from aae.ml.features import ID_COLUMN, build_features

if TYPE_CHECKING:
    from aae.ml.train import TrainedModel

logger = get_logger(__name__)

DEFAULT_THRESHOLD: Final[float] = 0.5


class DecisionEngine:
    """Scores applications and explains the result.

    Holds the explainer, whose construction walks the entire tree ensemble, so
    build one engine per model and reuse it across requests.
    """

    def __init__(
        self,
        model: TrainedModel,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        """Build the engine.

        Args:
            model: A trained, calibrated model.
            threshold: Probability of default at or above which credit is
                declined. Versioned alongside the model in the audit record,
                because moving a threshold changes outcomes as surely as
                retraining does.
            top_k: How many factors to record per decision.

        Raises:
            ValueError: If the threshold is not a probability.
        """
        if not 0.0 <= threshold <= 1.0:
            msg = f"Threshold must be a probability; got {threshold}."
            raise ValueError(msg)

        self._model = model
        self._threshold = threshold
        self._top_k = top_k
        self._explainer = DecisionExplainer(model.booster, model.spec.feature_names)

    @property
    def model(self) -> TrainedModel:
        """The trained model backing this engine."""
        return self._model

    @property
    def model_version(self) -> str:
        """The version string recorded on every decision this engine makes."""
        return self._model.model_version

    @property
    def threshold(self) -> float:
        """The decline threshold in force."""
        return self._threshold

    def decide(self, application: pd.DataFrame, row: int = 0) -> CreditDecision:
        """Score and explain one application.

        Args:
            application: A frame containing the application. Only ``row`` is
                scored; passing a frame rather than a series keeps the same
                feature-construction path as training, which is what stops
                training and serving from drifting apart.
            row: Positional index of the application to decide.

        Returns:
            The complete decision.

        Raises:
            DataValidationError: If the frame is empty or lacks the id column.
        """
        if len(application) == 0:
            msg = "Cannot decide on an empty application frame."
            raise DataValidationError(msg)
        if ID_COLUMN not in application.columns:
            msg = f"Application frame is missing the {ID_COLUMN!r} column."
            raise DataValidationError(msg)

        features = build_features(application, self._model.spec)
        single = features.iloc[[row]]

        probability = float(self._model.predict_proba(single)[0])
        decision = Decision.DECLINE if probability >= self._threshold else Decision.APPROVE
        factors = self._explainer.explain_row(features, row=row, top_k=self._top_k)

        feature_values: dict[str, float | str | None] = {
            name: _to_python_value(features.iloc[row][name])
            for name in self._model.spec.feature_names
        }

        credit_decision = CreditDecision(
            application_id=str(application.iloc[row][ID_COLUMN]),
            probability_default=probability,
            decision=decision,
            threshold=self._threshold,
            model_version=self._model.model_version,
            feature_values=feature_values,
            factors=factors,
            scored_at=datetime.now(UTC),
        )

        logger.info(
            "application_decided",
            application_id=credit_decision.application_id,
            decision=decision.value,
            probability=round(probability, 4),
            model_version=self._model.model_version,
            top_factor=factors[0].factor_id if factors else None,
        )
        return credit_decision

    def decide_batch(self, applications: pd.DataFrame) -> tuple[CreditDecision, ...]:
        """Score and explain every row in a frame.

        Args:
            applications: Frame of applications.

        Returns:
            One decision per row, in input order.
        """
        return tuple(self.decide(applications, row=index) for index in range(len(applications)))

    def raw_log_odds(self, features: pd.DataFrame) -> np.ndarray:
        """Return the booster's uncalibrated margin, in log-odds.

        Asked for directly rather than recovered by inverting the sigmoid on a
        predicted probability. The round trip through probability is lossy at
        the extremes: the booster emits float32, and near 0 or 1 the quantised
        probability maps to a badly wrong log-odds. Measured on this model,
        that route produced an apparent SHAP additivity error of 2e-2 against a
        true error near 1e-6 - it looked like a broken explainer and was
        actually a broken measurement.

        Args:
            features: Rows in model input order.

        Returns:
            Raw log-odds, one per row.
        """
        margin = self._model.booster.predict(
            features[list(self._model.spec.feature_names)], output_margin=True
        )
        return np.asarray(margin, dtype=np.float64)

    @property
    def explainer(self) -> DecisionExplainer:
        """The underlying explainer, for tests and diagnostics."""
        return self._explainer
