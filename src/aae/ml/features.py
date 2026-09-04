"""Feature specification and construction.

The fair-lending guarantee lives here, and it is structural rather than a
filter applied late that someone could forget to call. A :class:`FeatureSpec`
refuses to be constructed if it references a protected attribute, so a model
trained through this module cannot use one.

Two distinct traps are closed:

1. **Direct use.** Sex, age, and marital status are protected characteristics
   under ECOA/Regulation B and under RBI fair-practice expectations. They must
   never be model inputs.
2. **Proxy use through derived features.** This is the subtler failure. A ratio
   such as ``DAYS_EMPLOYED / DAYS_BIRTH`` contains no protected column by name,
   yet encodes age directly. Every derived feature therefore declares its input
   columns, and those inputs are validated against the protected set too.

Protected attributes are still *loaded* — the fairness analysis in
:mod:`aae.ml.fairness` needs them to measure disparate impact. They are simply
never features. Measuring bias requires the attribute; using it does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

from aae.domain.errors import FairLendingViolationError

if TYPE_CHECKING:
    from collections.abc import Callable

PROTECTED_ATTRIBUTES: Final[frozenset[str]] = frozenset(
    {
        "CODE_GENDER",  # sex
        "DAYS_BIRTH",  # age
        "NAME_FAMILY_STATUS",  # marital status
    }
)
"""Columns that must never influence a credit decision, directly or by proxy.

Marital status is included deliberately: it is protected under Regulation B and
is the one people most often forget, because it looks like an ordinary
demographic field rather than a prohibited basis.
"""

TARGET_COLUMN: Final[str] = "TARGET"
ID_COLUMN: Final[str] = "SK_ID_CURR"

_EPSILON: Final[float] = 1e-9


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide without producing infinities on zero denominators.

    Args:
        numerator: Values to divide.
        denominator: Values to divide by.

    Returns:
        The ratio, with non-finite results replaced by NaN so the model treats
        them as missing rather than as an extreme value.
    """
    ratio = numerator / denominator.replace(0, np.nan)
    return ratio.replace([np.inf, -np.inf], np.nan)


def _credit_to_income(frame: pd.DataFrame) -> pd.Series:
    return _safe_ratio(frame["AMT_CREDIT"], frame["AMT_INCOME_TOTAL"])


def _annuity_to_income(frame: pd.DataFrame) -> pd.Series:
    return _safe_ratio(frame["AMT_ANNUITY"], frame["AMT_INCOME_TOTAL"])


def _credit_term(frame: pd.DataFrame) -> pd.Series:
    return _safe_ratio(frame["AMT_ANNUITY"], frame["AMT_CREDIT"])


def _goods_price_gap(frame: pd.DataFrame) -> pd.Series:
    return _safe_ratio(frame["AMT_CREDIT"] - frame["AMT_GOODS_PRICE"], frame["AMT_GOODS_PRICE"])


def _income_per_family_member(frame: pd.DataFrame) -> pd.Series:
    return _safe_ratio(frame["AMT_INCOME_TOTAL"], frame["CNT_FAM_MEMBERS"])


@dataclass(frozen=True)
class DerivedFeature:
    """A feature computed from other columns.

    ``inputs`` is not documentation: it is validated against
    :data:`PROTECTED_ATTRIBUTES`, which is what prevents a derived feature from
    smuggling a protected characteristic back into the model.
    """

    name: str
    inputs: tuple[str, ...]
    compute: Callable[[pd.DataFrame], pd.Series]
    description: str

    def __call__(self, frame: pd.DataFrame) -> pd.Series:
        """Compute the feature over a frame.

        Args:
            frame: Source data containing at least :attr:`inputs`.

        Returns:
            The computed column.
        """
        return self.compute(frame)


DERIVED_FEATURES: Final[tuple[DerivedFeature, ...]] = (
    DerivedFeature(
        name="CREDIT_INCOME_RATIO",
        inputs=("AMT_CREDIT", "AMT_INCOME_TOTAL"),
        compute=_credit_to_income,
        description="Loan size relative to annual income.",
    ),
    DerivedFeature(
        name="ANNUITY_INCOME_RATIO",
        inputs=("AMT_ANNUITY", "AMT_INCOME_TOTAL"),
        compute=_annuity_to_income,
        description="Repayment burden as a share of income.",
    ),
    DerivedFeature(
        name="CREDIT_TERM",
        inputs=("AMT_ANNUITY", "AMT_CREDIT"),
        compute=_credit_term,
        description="Implied repayment rate; the inverse of loan term.",
    ),
    DerivedFeature(
        name="GOODS_PRICE_GAP",
        inputs=("AMT_CREDIT", "AMT_GOODS_PRICE"),
        compute=_goods_price_gap,
        description="How far the loan exceeds the value of the goods financed.",
    ),
    DerivedFeature(
        name="INCOME_PER_FAMILY_MEMBER",
        inputs=("AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"),
        compute=_income_per_family_member,
        description="Household income spread across dependants.",
    ),
)


@dataclass(frozen=True)
class FeatureSpec:
    """The complete, validated set of model inputs.

    Construction fails if any column — raw or an input to a derived feature —
    is a protected attribute. There is no way to build a spec that violates
    fair lending, so there is no way to train a model through this module that
    does.
    """

    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    derived: tuple[DerivedFeature, ...] = field(default=DERIVED_FEATURES)

    def __post_init__(self) -> None:
        """Reject any protected attribute, used directly or as a derived input.

        Raises:
            FairLendingViolationError: If a protected attribute is referenced.
        """
        direct = (set(self.numeric) | set(self.categorical)) & PROTECTED_ATTRIBUTES
        if direct:
            msg = (
                f"Protected attributes cannot be model features: {sorted(direct)}. "
                "They may be loaded for fairness measurement, never used as inputs."
            )
            raise FairLendingViolationError(msg)

        for derived in self.derived:
            leaked = set(derived.inputs) & PROTECTED_ATTRIBUTES
            if leaked:
                msg = (
                    f"Derived feature {derived.name!r} depends on protected "
                    f"attributes {sorted(leaked)}, which would encode a prohibited "
                    "basis by proxy even though the column itself is excluded."
                )
                raise FairLendingViolationError(msg)

        duplicated = {name for name in self.derived_names if name in self.raw_names}
        if duplicated:
            msg = f"Derived features shadow raw columns: {sorted(duplicated)}"
            raise ValueError(msg)

    @property
    def raw_names(self) -> tuple[str, ...]:
        """Raw source columns read from the dataset."""
        return (*self.numeric, *self.categorical)

    @property
    def derived_names(self) -> tuple[str, ...]:
        """Names of the computed columns."""
        return tuple(d.name for d in self.derived)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Every column the model sees, in a stable order.

        Order is fixed so that SHAP output, ONNX input, and the audit record all
        agree on which position means which feature.
        """
        return (*self.numeric, *self.derived_names, *self.categorical)

    def required_columns(self) -> tuple[str, ...]:
        """Columns that must be present in the source data to build features."""
        needed: set[str] = set(self.raw_names)
        for derived in self.derived:
            needed.update(derived.inputs)
        return tuple(sorted(needed))


DEFAULT_NUMERIC: Final[tuple[str, ...]] = (
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "REGION_POPULATION_RELATIVE",
    "REGION_RATING_CLIENT",
)

DEFAULT_CATEGORICAL: Final[tuple[str, ...]] = (
    "NAME_CONTRACT_TYPE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
)

FEATURE_DISPLAY_NAMES: Final[dict[str, str]] = {
    # Monetary
    "AMT_INCOME_TOTAL": "Total annual income",
    "AMT_CREDIT": "Loan amount requested",
    "AMT_ANNUITY": "Annual repayment amount",
    "AMT_GOODS_PRICE": "Value of the goods being financed",
    # Tenure
    "DAYS_EMPLOYED": "Length of current employment",
    "DAYS_REGISTRATION": "Time since registration details were last updated",
    "DAYS_ID_PUBLISH": "Time since the identity document was issued",
    # Bureau scores
    "EXT_SOURCE_1": "Credit bureau score (first source)",
    "EXT_SOURCE_2": "Credit bureau score (second source)",
    "EXT_SOURCE_3": "Credit bureau score (third source)",
    # Household
    "CNT_CHILDREN": "Number of children",
    "CNT_FAM_MEMBERS": "Household size",
    "REGION_POPULATION_RELATIVE": "Population density of the area of residence",
    "REGION_RATING_CLIENT": "Regional risk rating",
    # Derived
    "CREDIT_INCOME_RATIO": "Loan amount relative to income",
    "ANNUITY_INCOME_RATIO": "Repayment burden relative to income",
    "CREDIT_TERM": "Repayment rate relative to loan size",
    "GOODS_PRICE_GAP": "Loan amount above the value of the goods financed",
    "INCOME_PER_FAMILY_MEMBER": "Income per household member",
    # Categorical
    "NAME_CONTRACT_TYPE": "Type of loan applied for",
    "NAME_INCOME_TYPE": "Source of income",
    "NAME_EDUCATION_TYPE": "Highest level of education",
    "NAME_HOUSING_TYPE": "Housing situation",
    "OCCUPATION_TYPE": "Occupation",
    "FLAG_OWN_CAR": "Car ownership",
    "FLAG_OWN_REALTY": "Property ownership",
}
"""Plain-language names for every model input.

These are not developer conveniences: they are the words that appear in a
denial notice sent to a person. A column name such as ``EXT_SOURCE_2`` is
meaningless to an applicant and would fail the plain-language expectations
that adverse action notices are held to. A test asserts every feature has an
entry, so a new feature cannot reach a customer letter unnamed.
"""


DEFAULT_SPEC: Final[FeatureSpec] = FeatureSpec(
    numeric=DEFAULT_NUMERIC,
    categorical=DEFAULT_CATEGORICAL,
    derived=DERIVED_FEATURES,
)
"""The production feature set. Validated at import time."""


def build_features(frame: pd.DataFrame, spec: FeatureSpec = DEFAULT_SPEC) -> pd.DataFrame:
    """Construct the model input matrix.

    Categoricals are returned as pandas ``category`` dtype, which XGBoost
    consumes natively; no one-hot expansion is needed and the original level
    names survive into SHAP output, so a reason can name a real category.

    Args:
        frame: Source data containing :meth:`FeatureSpec.required_columns`.
        spec: The validated feature specification.

    Returns:
        A frame with exactly :attr:`FeatureSpec.feature_names` as columns, in
        that order.

    Raises:
        FairLendingViolationError: If the output would contain a protected
            attribute. This is belt and braces over the spec validation, and
            guards against a caller passing a hand-built spec.
        KeyError: If a required source column is missing.
    """
    missing = [column for column in spec.required_columns() if column not in frame.columns]
    if missing:
        msg = f"Source data is missing required columns: {missing}"
        raise KeyError(msg)

    out = pd.DataFrame(index=frame.index)

    for column in spec.numeric:
        out[column] = pd.to_numeric(frame[column], errors="coerce")

    for derived in spec.derived:
        out[derived.name] = derived(frame)

    for column in spec.categorical:
        out[column] = frame[column].astype("category")

    leaked = set(out.columns) & PROTECTED_ATTRIBUTES
    if leaked:  # pragma: no cover - unreachable via DEFAULT_SPEC
        msg = f"Constructed features contain protected attributes: {sorted(leaked)}"
        raise FairLendingViolationError(msg)

    return out[list(spec.feature_names)]
