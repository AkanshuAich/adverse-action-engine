"""Explanation and decision assembly.

The additivity test is the important one. A silently broken explainer would
still return plausible-looking factors, just for the wrong features - and
nothing downstream could tell, because a denial reason naming the wrong factor
reads exactly like one naming the right factor. Additivity is the only property
that catches it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from aae.data.loaders import load_applications
from aae.domain.errors import DataValidationError
from aae.domain.models import Decision, FactorDirection
from aae.ml.decision import DecisionEngine
from aae.ml.explain import ADDITIVITY_TOLERANCE
from aae.ml.features import (
    DEFAULT_SPEC,
    FEATURE_DISPLAY_NAMES,
    PROTECTED_ATTRIBUTES,
    build_features,
)
from aae.ml.train import train_model


@pytest.fixture(scope="module")
def engine_and_data():
    loaded = load_applications(force_synthetic=True, n_synthetic=12_000)
    model = train_model(loaded)
    # A low threshold so the sample reliably contains declines to explain.
    return DecisionEngine(model, threshold=0.15), loaded.frame


class TestDisplayNames:
    def test_every_feature_has_a_plain_language_name(self):
        """A feature without a name would reach a customer letter as a column."""
        missing = [f for f in DEFAULT_SPEC.feature_names if f not in FEATURE_DISPLAY_NAMES]
        assert missing == []

    def test_no_orphan_names(self):
        orphans = [k for k in FEATURE_DISPLAY_NAMES if k not in DEFAULT_SPEC.feature_names]
        assert orphans == []

    def test_names_are_not_just_the_column(self):
        for column, name in FEATURE_DISPLAY_NAMES.items():
            assert name != column
            assert "_" not in name


class TestAdditivity:
    def test_attributions_reconstruct_the_model_output(self, engine_and_data):
        """Base plus the SHAP values must equal the log-odds they explain."""
        engine, frame = engine_and_data
        features = build_features(frame.head(300))
        error = engine.explainer.check_additivity(features, engine.raw_log_odds(features))
        assert error < ADDITIVITY_TOLERANCE

    def test_base_value_is_read_after_the_explainer_has_seen_data(self, engine_and_data):
        """SHAP mutates ``expected_value`` on its first call that processes data.

        Reading it at construction returns a preliminary figure. On this model
        that was -2.506688 against a settled -2.526448, which injected a
        constant 0.0198 error into every reconstruction - large enough to be
        wrong, small enough to look like a broken explainer rather than a
        stale constant. This asserts the settled value is what gets used.
        """
        engine, frame = engine_and_data
        features = build_features(frame.head(50))

        contributions = engine.explainer.shap_values(features).sum(axis=1)
        settled_base = engine.explainer.base_value
        margin = engine.raw_log_odds(features)

        assert np.allclose(settled_base + contributions, margin, atol=ADDITIVITY_TOLERANCE)

    def test_shap_shape_matches_the_feature_set(self, engine_and_data):
        engine, frame = engine_and_data
        features = build_features(frame.head(20))
        assert engine.explainer.shap_values(features).shape == (
            20,
            len(DEFAULT_SPEC.feature_names),
        )


class TestFactors:
    def test_ranked_by_absolute_contribution(self, engine_and_data):
        engine, frame = engine_and_data
        factors = engine.decide(frame.head(1)).factors
        magnitudes = [abs(f.shap_value) for f in factors]
        assert magnitudes == sorted(magnitudes, reverse=True)
        assert [f.rank for f in factors] == list(range(1, len(factors) + 1))

    def test_positive_contribution_is_adverse(self, engine_and_data):
        """Positive SHAP raises the log-odds of default, so it counts against."""
        engine, frame = engine_and_data
        for decision in engine.decide_batch(frame.head(25)):
            for factor in decision.factors:
                expected = (
                    FactorDirection.ADVERSE if factor.shap_value > 0 else FactorDirection.FAVOURABLE
                )
                assert factor.direction is expected

    def test_no_factor_is_ever_a_protected_attribute(self, engine_and_data):
        """A denial reason naming sex, age, or marital status would be unlawful."""
        engine, frame = engine_and_data
        for decision in engine.decide_batch(frame.head(40)):
            for factor in decision.factors:
                assert factor.factor_id not in PROTECTED_ATTRIBUTES

    def test_top_k_is_respected(self, engine_and_data):
        engine, frame = engine_and_data
        assert len(DecisionEngine(engine.model, top_k=3).decide(frame.head(1)).factors) == 3

    def test_declines_are_driven_by_adverse_factors(self, engine_and_data):
        """A decline whose top factors were all favourable would be incoherent."""
        engine, frame = engine_and_data
        declines = [
            d for d in engine.decide_batch(frame.head(120)) if d.decision is Decision.DECLINE
        ]
        assert declines, "expected at least one decline in the sample"
        for decision in declines:
            assert decision.adverse_factors()
            assert decision.factors[0].direction is FactorDirection.ADVERSE


class TestCalibrationDoesNotDisturbExplanations:
    def test_calibration_preserves_applicant_ordering(self, engine_and_data):
        """The explainer explains raw log-odds; the decision uses calibrated odds.

        That is only sound because calibration is monotone, so it cannot
        reorder two applicants. This asserts the property the module docstring
        relies on rather than taking it on faith.
        """
        engine, frame = engine_and_data
        features = build_features(frame.head(400))
        raw = engine.raw_log_odds(features)
        calibrated = engine.model.predict_proba(features)

        raw_order = np.argsort(raw)
        # Monotone maps preserve order, allowing for ties introduced by the
        # calibrator's flat regions.
        calibrated_sorted = calibrated[raw_order]
        assert np.all(np.diff(calibrated_sorted) >= -1e-12)


class TestDecisionAssembly:
    def test_threshold_governs_the_outcome(self, engine_and_data):
        engine, frame = engine_and_data
        for decision in engine.decide_batch(frame.head(50)):
            expected = (
                Decision.DECLINE
                if decision.probability_default >= decision.threshold
                else Decision.APPROVE
            )
            assert decision.decision is expected

    def test_probability_is_never_degenerate(self, engine_and_data):
        """Exactly 0 or 1 asserts certainty no model has, and breaks log-odds."""
        engine, frame = engine_and_data
        probabilities = [d.probability_default for d in engine.decide_batch(frame.head(200))]
        assert min(probabilities) > 0.0
        assert max(probabilities) < 1.0

    def test_records_every_feature_value(self, engine_and_data):
        engine, frame = engine_and_data
        decision = engine.decide(frame.head(1))
        assert set(decision.feature_values) == set(DEFAULT_SPEC.feature_names)

    def test_feature_values_are_json_serialisable(self, engine_and_data):
        """They are hashed into the audit payload, which requires plain JSON."""
        engine, frame = engine_and_data
        decision = engine.decide(frame.head(1))
        round_tripped = json.loads(json.dumps(decision.feature_values))
        assert set(round_tripped) == set(decision.feature_values)

    def test_records_model_version_and_threshold(self, engine_and_data):
        engine, frame = engine_and_data
        decision = engine.decide(frame.head(1))
        assert decision.model_version == engine.model_version
        assert decision.threshold == engine.threshold

    def test_empty_frame_is_rejected(self, engine_and_data):
        engine, frame = engine_and_data
        with pytest.raises(DataValidationError, match="empty application frame"):
            engine.decide(frame.head(0))

    def test_missing_id_column_is_rejected(self, engine_and_data):
        engine, frame = engine_and_data
        with pytest.raises(DataValidationError, match="SK_ID_CURR"):
            engine.decide(frame.head(1).drop(columns=["SK_ID_CURR"]))

    def test_rejects_a_threshold_that_is_not_a_probability(self, engine_and_data):
        engine, _ = engine_and_data
        with pytest.raises(ValueError, match="must be a probability"):
            DecisionEngine(engine.model, threshold=1.5)

    def test_scoring_is_deterministic(self, engine_and_data):
        engine, frame = engine_and_data
        first = engine.decide(frame.head(1))
        second = engine.decide(frame.head(1))
        assert first.probability_default == second.probability_default
        assert [f.factor_id for f in first.factors] == [f.factor_id for f in second.factors]
