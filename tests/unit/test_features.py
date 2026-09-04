"""Fair-lending guarantees on the feature layer.

These are the most important tests in the repository. A credit model that uses
sex, age, or marital status is unlawful, and the usual failure is not a
deliberate choice — it is a protected column surviving a refactor, or a ratio
that encodes age without naming it. Both are asserted against here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aae.data.synthetic import generate_applications
from aae.domain.errors import FairLendingViolationError
from aae.ml.features import (
    DEFAULT_SPEC,
    DERIVED_FEATURES,
    PROTECTED_ATTRIBUTES,
    DerivedFeature,
    FeatureSpec,
    build_features,
)


@pytest.fixture(scope="module")
def applications() -> pd.DataFrame:
    return generate_applications(n_rows=1_500, seed=7)


class TestProtectedAttributesAreNeverFeatures:
    def test_the_protected_set_names_the_three_prohibited_bases(self):
        """Sex, age, and marital status are all prohibited bases under Reg B.

        Marital status is the one most often missed, because it reads as an
        ordinary demographic field rather than a protected characteristic.
        """
        assert {"CODE_GENDER", "DAYS_BIRTH", "NAME_FAMILY_STATUS"} == PROTECTED_ATTRIBUTES

    def test_default_spec_excludes_every_protected_attribute(self):
        assert not set(DEFAULT_SPEC.feature_names) & PROTECTED_ATTRIBUTES

    def test_built_features_exclude_every_protected_attribute(self, applications: pd.DataFrame):
        features = build_features(applications)
        assert not set(features.columns) & PROTECTED_ATTRIBUTES

    @pytest.mark.parametrize("protected", sorted(PROTECTED_ATTRIBUTES))
    def test_a_spec_naming_a_protected_attribute_cannot_be_built(self, protected: str):
        with pytest.raises(FairLendingViolationError, match="cannot be model features"):
            FeatureSpec(numeric=(protected,), categorical=(), derived=())

    def test_protected_attributes_are_still_loaded_for_fairness_measurement(
        self, applications: pd.DataFrame
    ):
        """Excluding them from features must not mean dropping them entirely.

        Measuring disparate impact requires the attribute; using it as an input
        does not. Conflating the two makes bias unmeasurable.
        """
        assert set(applications.columns) >= PROTECTED_ATTRIBUTES


class TestProxyDiscriminationIsBlocked:
    def test_a_derived_feature_depending_on_age_is_rejected(self):
        """The classic proxy: employment tenure over age encodes age exactly.

        The column ``DAYS_BIRTH`` never appears in the output, so a name-based
        check would pass this. Declaring inputs is what catches it.
        """
        tenure_ratio = DerivedFeature(
            name="EMPLOYED_LIFE_RATIO",
            inputs=("DAYS_EMPLOYED", "DAYS_BIRTH"),
            compute=lambda frame: frame["DAYS_EMPLOYED"] / frame["DAYS_BIRTH"],
            description="Share of life spent employed.",
        )
        with pytest.raises(FairLendingViolationError, match="by proxy"):
            FeatureSpec(numeric=(), categorical=(), derived=(tenure_ratio,))

    def test_the_error_names_the_offending_feature_and_column(self):
        bad = DerivedFeature(
            name="INCOME_PER_YEAR_OF_AGE",
            inputs=("AMT_INCOME_TOTAL", "DAYS_BIRTH"),
            compute=lambda frame: frame["AMT_INCOME_TOTAL"],
            description="Income scaled by age.",
        )
        with pytest.raises(FairLendingViolationError) as excinfo:
            FeatureSpec(numeric=(), categorical=(), derived=(bad,))
        message = str(excinfo.value)
        assert "INCOME_PER_YEAR_OF_AGE" in message
        assert "DAYS_BIRTH" in message

    def test_no_shipped_derived_feature_touches_a_protected_column(self):
        for derived in DERIVED_FEATURES:
            assert not set(derived.inputs) & PROTECTED_ATTRIBUTES, derived.name


class TestFeatureSpec:
    def test_feature_order_is_stable(self):
        """SHAP, the model input, and the audit record must agree on order."""
        assert (
            DEFAULT_SPEC.feature_names
            == FeatureSpec(
                numeric=DEFAULT_SPEC.numeric,
                categorical=DEFAULT_SPEC.categorical,
                derived=DERIVED_FEATURES,
            ).feature_names
        )

    def test_required_columns_include_derived_inputs(self):
        required = set(DEFAULT_SPEC.required_columns())
        for derived in DEFAULT_SPEC.derived:
            assert set(derived.inputs) <= required

    def test_derived_features_cannot_shadow_raw_columns(self):
        clash = DerivedFeature(
            name="AMT_CREDIT",
            inputs=("AMT_CREDIT",),
            compute=lambda frame: frame["AMT_CREDIT"],
            description="Shadows a raw column.",
        )
        with pytest.raises(ValueError, match="shadow raw columns"):
            FeatureSpec(numeric=("AMT_CREDIT",), categorical=(), derived=(clash,))


class TestBuildFeatures:
    def test_output_columns_match_the_spec_exactly(self, applications: pd.DataFrame):
        features = build_features(applications)
        assert list(features.columns) == list(DEFAULT_SPEC.feature_names)

    def test_categoricals_keep_their_level_names(self, applications: pd.DataFrame):
        """XGBoost consumes categories natively, so SHAP can name a real level.

        One-hot encoding would turn a reason into "OCCUPATION_TYPE_3", which is
        not something you can put in a letter to a customer.
        """
        features = build_features(applications)
        levels = set(features["NAME_CONTRACT_TYPE"].cat.categories)
        assert "Cash loans" in levels

    def test_missing_source_columns_raise(self, applications: pd.DataFrame):
        with pytest.raises(KeyError, match="missing required columns"):
            build_features(applications.drop(columns=["AMT_CREDIT"]))

    def test_division_by_zero_becomes_missing_not_infinite(self):
        """An infinity would be read by the model as an extreme applicant.

        Missing is the honest encoding of "this ratio is undefined".
        """
        frame = generate_applications(n_rows=50, seed=3)
        frame = frame.copy()
        frame.loc[frame.index[:10], "AMT_INCOME_TOTAL"] = 0.0

        features = build_features(frame)
        ratios = features["CREDIT_INCOME_RATIO"]
        assert not np.isinf(ratios.to_numpy(dtype=float)).any()
        assert ratios.iloc[:10].isna().all()

    def test_row_count_is_preserved(self, applications: pd.DataFrame):
        assert len(build_features(applications)) == len(applications)
