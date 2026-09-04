"""Calibration, persistence, and the synthetic generator.

The persistence tests exist to keep a specific promise: no pickle, ever. The
calibration tests assert that a quoted probability means what it says, which is
what a threshold-based denial rests on.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

from aae.data.loaders import DataSource, load_applications
from aae.data.schema import validate_applications
from aae.data.synthetic import BASE_DEFAULT_RATE, generate_applications
from aae.domain.errors import DataValidationError, ModelError
from aae.ml.calibration import PROBABILITY_EPSILON, Calibrator
from aae.ml.features import PROTECTED_ATTRIBUTES, build_features
from aae.ml.registry import METADATA_FILENAME, load_model, save_model
from aae.ml.train import train_model

PICKLE_SUFFIXES = (".pkl", ".pickle", ".joblib", ".npy", ".pt")


@pytest.fixture(scope="module")
def trained():
    # Large enough that isotonic has real data to fit; see MIN_CALIBRATION_ROWS.
    loaded = load_applications(force_synthetic=True, n_synthetic=15_000)
    return loaded, train_model(loaded)


class TestSyntheticGenerator:
    def test_is_reproducible(self):
        left = generate_applications(500, seed=42)
        right = generate_applications(500, seed=42)
        assert left.equals(right)

    def test_default_rate_is_near_the_real_base_rate(self):
        """Calibrating the intercept by `logit(p) - mean(risk)` overshoots badly.

        Sigmoid is non-linear, so the mean of the transform is not the
        transform of the mean; the shortcut produced a 20% base rate instead of
        8%. The intercept is solved numerically instead.
        """
        frame = generate_applications(20_000, seed=11)
        assert frame["TARGET"].mean() == pytest.approx(BASE_DEFAULT_RATE, abs=0.012)

    def test_reproduces_real_world_missingness(self):
        frame = generate_applications(3_000, seed=5)
        assert 0.45 < frame["EXT_SOURCE_1"].isna().mean() < 0.65
        assert frame["EXT_SOURCE_2"].isna().sum() == 0

    def test_encodes_the_unemployment_sentinel(self):
        """365243 is a real Home Credit quirk, not a bug to be cleaned away."""
        frame = generate_applications(2_000, seed=5)
        assert (frame["DAYS_EMPLOYED"] == 365_243).sum() > 0

    def test_produces_measurable_disparity_through_income(self):
        """Fairness analysis needs something real to find.

        Group membership never enters the outcome directly; it shifts income,
        and income shifts risk. That mediation is what indirect discrimination
        looks like in practice.
        """
        frame = generate_applications(20_000, seed=13)
        rates = frame.groupby("CODE_GENDER")["TARGET"].mean()
        assert abs(rates["F"] - rates["M"]) > 0.005

    def test_disparity_can_be_switched_off(self):
        frame = generate_applications(20_000, seed=13, disparity_strength=0.0)
        incomes = frame.groupby("CODE_GENDER")["AMT_INCOME_TOTAL"].median()
        assert incomes["F"] == pytest.approx(incomes["M"], rel=0.15)


class TestSchemaValidation:
    def test_valid_data_passes(self):
        assert len(validate_applications(generate_applications(200, seed=2))) == 200

    def test_negative_income_is_rejected(self):
        frame = generate_applications(200, seed=2)
        frame.loc[frame.index[0], "AMT_INCOME_TOTAL"] = -1.0
        with pytest.raises(DataValidationError):
            validate_applications(frame)

    def test_out_of_range_bureau_score_is_rejected(self):
        frame = generate_applications(200, seed=2)
        frame.loc[frame.index[0], "EXT_SOURCE_2"] = 1.4
        with pytest.raises(DataValidationError):
            validate_applications(frame)

    def test_positive_days_birth_is_rejected(self):
        """Days are stored as negative offsets; a positive value is corrupt."""
        frame = generate_applications(200, seed=2)
        frame.loc[frame.index[0], "DAYS_BIRTH"] = 12_000
        with pytest.raises(DataValidationError):
            validate_applications(frame)

    def test_duplicate_application_ids_are_rejected(self):
        frame = generate_applications(200, seed=2)
        frame.loc[frame.index[1], "SK_ID_CURR"] = frame.loc[frame.index[0], "SK_ID_CURR"]
        with pytest.raises(DataValidationError):
            validate_applications(frame)


class TestCalibrator:
    def test_matches_sklearn_apart_from_the_boundary_clamp(self):
        """Agreement is exact except where scikit-learn returns 0 or 1.

        Those are clamped deliberately, so the tolerance here is the clamp
        width rather than floating-point noise.
        """
        rng = np.random.default_rng(0)
        raw = rng.uniform(0, 1, 2_000)
        outcomes = (rng.uniform(0, 1, 2_000) < raw).astype(int)

        fitted = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        fitted.fit(raw, outcomes)
        ours = Calibrator.from_sklearn(fitted)

        probe = np.linspace(0, 1, 500)
        assert np.allclose(ours.predict(probe), fitted.predict(probe), atol=PROBABILITY_EPSILON)

    def test_never_returns_a_degenerate_probability(self):
        """Isotonic emits exact 0 and 1 from pooled single-class regions.

        On the real model that was 43% of applications receiving a stated 0%
        chance of default. No model licenses certainty, an audit record
        asserting it is indefensible, and log-odds of 0 or 1 are not finite.
        """
        rng = np.random.default_rng(1)
        raw = rng.uniform(0, 1, 3_000)
        # Perfectly separable, so isotonic produces flat 0 and 1 regions.
        outcomes = (raw > 0.5).astype(int)

        fitted = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        fitted.fit(raw, outcomes)

        assert fitted.predict(np.array([0.1, 0.9])).tolist() == [0.0, 1.0]

        ours = Calibrator.from_sklearn(fitted)
        calibrated = ours.predict(np.linspace(0, 1, 500))
        assert calibrated.min() > 0.0
        assert calibrated.max() < 1.0
        assert np.isfinite(np.log(calibrated / (1 - calibrated))).all()

    def test_is_monotone(self):
        calibrator = Calibrator(thresholds=(0.0, 0.5, 1.0), values=(0.0, 0.3, 0.9))
        probe = np.linspace(0, 1, 200)
        assert np.all(np.diff(calibrator.predict(probe)) >= -1e-12)

    def test_clips_outside_the_fitted_range(self):
        calibrator = Calibrator(thresholds=(0.2, 0.8), values=(0.1, 0.7))
        assert calibrator.predict(np.array([-5.0]))[0] == pytest.approx(0.1)
        assert calibrator.predict(np.array([5.0]))[0] == pytest.approx(0.7)

    def test_round_trips_through_json(self):
        calibrator = Calibrator(thresholds=(0.1, 0.4, 0.9), values=(0.0, 0.25, 0.8))
        restored = Calibrator.from_dict(json.loads(json.dumps(calibrator.to_dict())))
        assert restored == calibrator

    def test_rejects_a_non_monotone_mapping(self):
        """A higher score mapping to lower risk would invert the decision."""
        with pytest.raises(ModelError, match="monotone"):
            Calibrator(thresholds=(0.1, 0.2), values=(0.8, 0.3))

    def test_rejects_values_outside_the_unit_interval(self):
        with pytest.raises(ModelError, match="probabilities"):
            Calibrator(thresholds=(0.1, 0.2), values=(0.5, 1.7))

    def test_rejects_unfitted_estimator(self):
        with pytest.raises(ModelError, match="must be fitted"):
            Calibrator.from_sklearn(IsotonicRegression())


class TestTraining:
    def test_model_discriminates(self, trained):
        _, model = trained
        assert model.metrics.auc > 0.70
        assert model.metrics.ks > 0.30

    def test_calibration_never_makes_probabilities_worse(self, trained):
        """The guarantee is not "calibration helps" - it is "it never hurts".

        Isotonic overfits on small samples and a boosted model is often already
        near-calibrated, so fitting unconditionally degrades expected
        calibration error about as often as it improves it. The calibrator is
        therefore judged on held-out data and dropped when it does not earn its
        place, which makes this assertion hold at any dataset size.
        """
        _, model = trained
        if model.metrics.calibration_applied:
            assert model.metrics.calibration_improved
        else:
            assert model.metrics.ece_calibrated == pytest.approx(
                model.metrics.ece_uncalibrated, abs=1e-12
            )

    def test_rejected_calibration_leaves_scores_untouched(self):
        """A rejected calibrator must be the identity, not a silent no-op flag."""
        loaded = load_applications(force_synthetic=True, n_synthetic=1_200)
        model = train_model(loaded)
        if not model.metrics.calibration_applied:
            assert model.calibrator.is_identity
            probe = np.linspace(0.0, 1.0, 50)
            assert np.allclose(model.calibrator.predict(probe), probe)

    def test_trained_model_uses_no_protected_attribute(self, trained):
        """The guarantee restated at the model level, not just the spec level."""
        _, model = trained
        assert not set(model.spec.feature_names) & PROTECTED_ATTRIBUTES
        assert not set(model.booster.feature_names_in_) & PROTECTED_ATTRIBUTES

    def test_version_is_deterministic(self, trained):
        loaded, model = trained
        assert train_model(loaded).model_version == model.model_version

    def test_probabilities_are_in_range(self, trained):
        loaded, model = trained
        probabilities = model.predict_proba(build_features(loaded.frame.head(300)))
        assert probabilities.min() >= 0.0
        assert probabilities.max() <= 1.0

    def test_data_provenance_is_recorded(self, trained):
        """Training on synthetic data is a material fact for a model card."""
        loaded, model = trained
        assert loaded.source is DataSource.SYNTHETIC
        assert model.data_source == "synthetic"


class TestRegistry:
    def test_writes_no_pickle(self, trained, tmp_path: Path):
        _, model = trained
        directory = save_model(model, tmp_path / "model")
        written = list(directory.iterdir())
        assert {f.name for f in written} == {"booster.ubj", "calibrator.json", "metadata.json"}
        assert not any(f.suffix in PICKLE_SUFFIXES for f in written)

    def test_round_trip_predictions_are_identical(self, trained, tmp_path: Path):
        loaded, model = trained
        features = build_features(loaded.frame.head(400))
        before = model.predict_proba(features)

        restored = load_model(save_model(model, tmp_path / "model"))
        assert np.array_equal(before, restored.predict_proba(features))

    def test_metadata_is_human_readable_json(self, trained, tmp_path: Path):
        _, model = trained
        directory = save_model(model, tmp_path / "model")
        metadata = json.loads((directory / METADATA_FILENAME).read_text(encoding="utf-8"))
        assert metadata["model_version"] == model.model_version
        assert metadata["feature_order"] == list(model.spec.feature_names)

    def test_missing_artifact_is_reported_clearly(self, trained, tmp_path: Path):
        _, model = trained
        directory = save_model(model, tmp_path / "model")
        (directory / "calibrator.json").unlink()
        with pytest.raises(ModelError, match=r"calibrator\.json"):
            load_model(directory)

    def test_unknown_artifact_schema_is_refused(self, trained, tmp_path: Path):
        _, model = trained
        directory = save_model(model, tmp_path / "model")
        path = directory / METADATA_FILENAME
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["artifact_schema_version"] = 99
        path.write_text(json.dumps(metadata), encoding="utf-8")

        with pytest.raises(ModelError, match="schema version"):
            load_model(directory)

    def test_changed_derived_features_block_loading(self, trained, tmp_path: Path):
        """Scoring with mismatched feature arithmetic must fail loudly.

        The derived-feature logic is code, not data. If it changed since
        training, the stored booster expects different inputs than the current
        code produces, and silently scoring would be wrong in a way nobody sees.
        """
        _, model = trained
        directory = save_model(model, tmp_path / "model")
        path = directory / METADATA_FILENAME
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["feature_spec"]["derived_names"] = ["SOMETHING_ELSE"]
        path.write_text(json.dumps(metadata), encoding="utf-8")

        with pytest.raises(ModelError, match="Derived feature set has changed"):
            load_model(directory)
