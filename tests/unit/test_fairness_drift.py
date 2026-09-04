"""Fairness measurement and drift detection.

Both are monitoring, and monitoring that has never been shown to fire is
indistinguishable from monitoring that cannot. Every test here either
constructs a disparity and asserts it is detected, or constructs parity and
asserts nothing is reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aae.domain.errors import DataValidationError
from aae.ml.drift import (
    PSI_MODERATE,
    PSI_SIGNIFICANT,
    DriftSeverity,
    categorical_stability_index,
    detect_drift,
    ks_statistic,
    population_stability_index,
)
from aae.ml.fairness import (
    FOUR_FIFTHS,
    MITIGATION_POSITION,
    age_band,
    analyse_fairness,
    evaluate_group,
)


class TestAgeBands:
    @pytest.mark.parametrize(
        ("years", "expected"),
        [
            (22, "under 25"),
            (30, "25 to 34"),
            (40, "35 to 44"),
            (50, "45 to 54"),
            (70, "55 and over"),
        ],
    )
    def test_bands(self, years: int, expected: str):
        assert age_band(-years * 365.25) == expected

    def test_handles_the_negative_storage_convention(self):
        """Home Credit stores age as days before application, negative."""
        assert age_band(-14_600) == age_band(14_600)


class TestDisparateImpact:
    def test_parity_scores_one(self):
        groups = ["A"] * 100 + ["B"] * 100
        declined = np.array([1] * 20 + [0] * 80 + [1] * 20 + [0] * 80)
        result = evaluate_group("GROUP", groups, declined)

        assert result.adverse_impact_ratio == 1.0
        assert result.passes_four_fifths

    def test_a_large_disparity_fails_the_four_fifths_screen(self):
        # Group B declined four times as often.
        groups = ["A"] * 100 + ["B"] * 100
        declined = np.array([1] * 10 + [0] * 90 + [1] * 40 + [0] * 60)
        result = evaluate_group("GROUP", groups, declined)

        assert result.adverse_impact_ratio < FOUR_FIFTHS
        assert not result.passes_four_fifths
        assert result.most_affected_group == "B"

    def test_the_ratio_is_expressed_on_approval_rates(self):
        """Which is how the four-fifths rule is conventionally stated."""
        groups = ["A"] * 100 + ["B"] * 100
        declined = np.array([1] * 20 + [0] * 80 + [1] * 40 + [0] * 60)
        result = evaluate_group("GROUP", groups, declined)
        # Approval rates 0.8 and 0.6, so the ratio is 0.75.
        assert result.adverse_impact_ratio == pytest.approx(0.75, abs=0.001)

    def test_a_single_group_cannot_be_disparate(self):
        result = evaluate_group("GROUP", ["A"] * 50, np.array([1] * 25 + [0] * 25))
        assert result.adverse_impact_ratio == 1.0

    def test_group_sizes_are_reported(self):
        """A ratio computed over eight applicants is not a finding."""
        groups = ["A"] * 90 + ["B"] * 10
        result = evaluate_group("GROUP", groups, np.array([0] * 100))
        assert result.group_sizes == {"A": 90, "B": 10}

    def test_error_rates_need_outcomes(self):
        groups = ["A"] * 50 + ["B"] * 50
        declined = np.array([1] * 25 + [0] * 25 + [1] * 25 + [0] * 25)
        without = evaluate_group("GROUP", groups, declined)
        assert without.equalized_odds_difference is None
        assert without.true_positive_rates == {}

    def test_error_rates_are_measured_when_outcomes_are_supplied(self):
        """Demographic parity can hold while error rates differ sharply."""
        rng = np.random.default_rng(0)
        groups = ["A"] * 200 + ["B"] * 200
        declined = np.array([1] * 100 + [0] * 100 + [1] * 100 + [0] * 100)
        outcomes = np.concatenate(
            [
                rng.permutation([1] * 90 + [0] * 110),
                rng.permutation([1] * 20 + [0] * 180),
            ]
        )
        result = evaluate_group("GROUP", groups, declined, outcomes)

        assert result.adverse_impact_ratio == 1.0
        assert result.equalized_odds_difference is not None
        assert result.equalized_odds_difference > 0.0
        assert set(result.true_positive_rates) == {"A", "B"}


class TestFairnessReport:
    def test_measures_every_protected_attribute_present(self):
        frame = pd.DataFrame(
            {
                "CODE_GENDER": ["F", "M"] * 50,
                "DAYS_BIRTH": [-12_000, -18_000] * 50,
                "NAME_FAMILY_STATUS": ["Married", "Single / not married"] * 50,
            }
        )
        report = analyse_fairness(frame, np.array([1, 0] * 50), model_version="v1")
        assert {group.attribute for group in report.groups} == {
            "CODE_GENDER",
            "DAYS_BIRTH",
            "NAME_FAMILY_STATUS",
        }

    def test_reports_findings_below_the_screen(self):
        frame = pd.DataFrame({"CODE_GENDER": ["F"] * 100 + ["M"] * 100})
        declined = np.array([1] * 50 + [0] * 50 + [1] * 5 + [0] * 95)
        report = analyse_fairness(frame, declined, model_version="v1")

        assert [group.attribute for group in report.findings] == ["CODE_GENDER"]

    def test_states_the_mitigation_position(self):
        """Adjusting thresholds per group would itself be disparate treatment."""
        report = analyse_fairness(
            pd.DataFrame({"CODE_GENDER": ["F", "M"] * 10}),
            np.array([1, 0] * 10),
            model_version="v1",
        )
        assert "unlawful" in report.mitigation_position
        # Whitespace-normalised: the position is a wrapped paragraph, so the
        # phrase spans a line break in the source.
        assert "disparate treatment" in " ".join(MITIGATION_POSITION.split())

    def test_serialises_for_the_committed_report(self):
        report = analyse_fairness(
            pd.DataFrame({"CODE_GENDER": ["F", "M"] * 10}),
            np.array([1, 0] * 10),
            model_version="v1",
        )
        payload = report.to_dict()
        assert payload["model_version"] == "v1"
        assert isinstance(payload["groups"], list)
        assert "mitigation_position" in payload


class TestPopulationStability:
    def test_identical_distributions_score_zero(self):
        rng = np.random.default_rng(0)
        sample = rng.normal(size=5_000)
        assert population_stability_index(sample, sample) == pytest.approx(0.0, abs=1e-9)

    def test_a_shifted_distribution_is_significant(self):
        rng = np.random.default_rng(0)
        reference = rng.normal(0, 1, 5_000)
        shifted = rng.normal(3, 1, 5_000)
        assert population_stability_index(reference, shifted) > PSI_SIGNIFICANT

    def test_the_moderate_band_is_reachable(self):
        """The thresholds must distinguish a real but modest shift."""
        rng = np.random.default_rng(0)
        reference = rng.normal(0, 1, 20_000)
        nudged = rng.normal(0.5, 1, 20_000)
        psi = population_stability_index(reference, nudged)
        assert PSI_MODERATE <= psi < PSI_SIGNIFICANT

    def test_the_index_grows_with_the_size_of_the_shift(self):
        """The property that makes it usable as a monitoring signal."""
        rng = np.random.default_rng(0)
        reference = rng.normal(0, 1, 20_000)
        values = [
            population_stability_index(reference, rng.normal(shift, 1, 20_000))
            for shift in (0.0, 0.25, 0.5, 1.0, 2.0)
        ]
        assert values == sorted(values)

    def test_uses_quantile_bins_so_skew_does_not_hide_drift(self):
        """Equal-width bins on income put everyone in the first bin."""
        rng = np.random.default_rng(0)
        reference = rng.lognormal(11, 0.5, 10_000)
        halved = reference * 0.45
        assert population_stability_index(reference, halved) > PSI_SIGNIFICANT

    def test_an_empty_sample_is_an_error(self):
        with pytest.raises(DataValidationError, match="non-empty"):
            population_stability_index(np.array([]), np.array([1.0, 2.0]))

    def test_a_constant_feature_cannot_drift_in_shape(self):
        assert population_stability_index([5.0] * 100, [5.0] * 100) == 0.0

    def test_an_empty_bin_gives_a_large_finite_value_not_infinity(self):
        rng = np.random.default_rng(0)
        reference = rng.uniform(0, 1, 2_000)
        disjoint = rng.uniform(10, 11, 2_000)
        psi = population_stability_index(reference, disjoint)
        assert np.isfinite(psi)
        assert psi > PSI_SIGNIFICANT


class TestCategoricalStability:
    def test_identical_frequencies_score_zero(self):
        values = ["a"] * 60 + ["b"] * 40
        assert categorical_stability_index(values, values) == pytest.approx(0.0, abs=1e-9)

    def test_a_vanished_category_is_drift(self):
        reference = ["a"] * 50 + ["b"] * 50
        current = ["a"] * 100
        assert categorical_stability_index(reference, current) > PSI_SIGNIFICANT

    def test_an_empty_sample_is_an_error(self):
        with pytest.raises(DataValidationError, match="non-empty"):
            categorical_stability_index([], ["a"])


class TestKolmogorovSmirnov:
    def test_identical_samples_score_zero(self):
        sample = [1.0, 2.0, 3.0, 4.0]
        assert ks_statistic(sample, sample) == 0.0

    def test_disjoint_samples_score_one(self):
        assert ks_statistic([0.0, 1.0], [10.0, 11.0]) == 1.0

    def test_an_empty_sample_scores_zero_rather_than_raising(self):
        assert ks_statistic([], [1.0]) == 0.0


class TestDriftReport:
    @staticmethod
    def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(0)
        reference = pd.DataFrame(
            {
                "income": rng.lognormal(11, 0.4, 3_000),
                "score": rng.uniform(0, 1, 3_000),
                "occupation": rng.choice(["a", "b", "c"], 3_000),
            }
        )
        return reference, reference.copy()

    def test_an_unchanged_population_is_stable(self):
        reference, current = self._frames()
        scores = np.random.default_rng(1).uniform(0, 1, len(reference))
        report = detect_drift(reference, current, scores, scores, model_version="v1")

        assert report.score_severity is DriftSeverity.STABLE
        assert report.drifted == ()
        assert not report.requires_attention

    def test_a_shifted_feature_is_named(self):
        reference, current = self._frames()
        current = current.assign(income=current["income"] * 0.4)
        scores = np.random.default_rng(1).uniform(0, 1, len(reference))

        report = detect_drift(reference, current, scores, scores, model_version="v1")
        assert "income" in [feature.feature for feature in report.drifted]
        assert report.requires_attention

    def test_features_are_ordered_by_how_far_they_moved(self):
        reference, current = self._frames()
        current = current.assign(income=current["income"] * 0.4)
        scores = np.random.default_rng(1).uniform(0, 1, len(reference))

        report = detect_drift(reference, current, scores, scores, model_version="v1")
        psis = [feature.psi for feature in report.features]
        assert psis == sorted(psis, reverse=True)

    def test_a_collapse_in_availability_is_recorded(self):
        """A feature that stops arriving has drifted, whatever the values say."""
        reference, current = self._frames()
        current = current.copy()
        current.loc[current.index[:2_400], "score"] = np.nan
        scores = np.random.default_rng(1).uniform(0, 1, len(reference))

        report = detect_drift(reference, current, scores, scores, model_version="v1")
        feature = next(f for f in report.features if f.feature == "score")
        assert feature.missing_rate_shift > 0.5

    def test_score_drift_is_reported_separately(self):
        """Features can drift in offsetting ways; the scores are what matter."""
        reference, current = self._frames()
        rng = np.random.default_rng(1)
        report = detect_drift(
            reference,
            current,
            rng.uniform(0, 0.2, len(reference)),
            rng.uniform(0.8, 1.0, len(current)),
            model_version="v1",
        )
        assert report.score_severity is DriftSeverity.SIGNIFICANT
        assert report.score_ks == pytest.approx(1.0, abs=0.01)

    def test_serialises_for_the_committed_report(self):
        reference, current = self._frames()
        scores = np.random.default_rng(1).uniform(0, 1, len(reference))
        payload = detect_drift(reference, current, scores, scores, model_version="v1").to_dict()

        assert payload["model_version"] == "v1"
        assert payload["reference_rows"] == len(reference)
        assert isinstance(payload["features"], list)

    @pytest.mark.parametrize(
        ("psi", "expected"),
        [
            (0.0, DriftSeverity.STABLE),
            (0.09, DriftSeverity.STABLE),
            (0.10, DriftSeverity.MODERATE),
            (0.24, DriftSeverity.MODERATE),
            (0.25, DriftSeverity.SIGNIFICANT),
            (2.0, DriftSeverity.SIGNIFICANT),
        ],
    )
    def test_severity_bands(self, psi: float, expected: DriftSeverity):
        assert DriftSeverity.from_psi(psi) is expected
