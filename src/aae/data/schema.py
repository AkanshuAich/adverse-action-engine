"""Schema validation at the data boundary.

Every frame entering the pipeline is validated here, so malformed data fails
loudly at the edge rather than producing a plausible-looking credit decision
built on nonsense. A denial notice is a regulated communication; the inputs
behind it must be checked, not assumed.

The schema is deliberately permissive about *extra* columns (the real Home
Credit file has 122) and strict about the ones the model actually consumes.
"""

from __future__ import annotations

from typing import Final

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError, SchemaErrors

from aae.domain.errors import DataValidationError
from aae.ml.features import ID_COLUMN, TARGET_COLUMN

VALID_GENDERS: Final[tuple[str, ...]] = ("F", "M", "XNA")
"""``XNA`` appears in the real dataset and must not crash validation."""

EMPLOYMENT_SENTINEL: Final[int] = 365243
"""Home Credit encodes "not currently employed" as this value, not as null."""


def _probability_column(*, nullable: bool) -> pa.Column:
    """Build a column constrained to the unit interval.

    Args:
        nullable: Whether missing values are permitted.

    Returns:
        A pandera column definition.
    """
    return pa.Column(
        float,
        checks=[pa.Check.ge(0.0), pa.Check.le(1.0)],
        nullable=nullable,
        coerce=True,
    )


APPLICATION_SCHEMA: Final[pa.DataFrameSchema] = pa.DataFrameSchema(
    columns={
        ID_COLUMN: pa.Column(int, unique=True, coerce=True),
        TARGET_COLUMN: pa.Column(int, checks=pa.Check.isin([0, 1]), coerce=True),
        # --- protected attributes: validated, loaded, never features ---------
        "CODE_GENDER": pa.Column(str, checks=pa.Check.isin(VALID_GENDERS)),
        # Stored as days before application, so always negative.
        "DAYS_BIRTH": pa.Column(int, checks=pa.Check.lt(0), coerce=True),
        "NAME_FAMILY_STATUS": pa.Column(str),
        # --- monetary --------------------------------------------------------
        "AMT_INCOME_TOTAL": pa.Column(float, checks=pa.Check.gt(0), coerce=True),
        "AMT_CREDIT": pa.Column(float, checks=pa.Check.gt(0), coerce=True),
        "AMT_ANNUITY": pa.Column(float, checks=pa.Check.gt(0), nullable=True, coerce=True),
        "AMT_GOODS_PRICE": pa.Column(float, checks=pa.Check.gt(0), nullable=True, coerce=True),
        # --- tenure ----------------------------------------------------------
        # No upper-bound check: the employment sentinel is a legitimate value.
        "DAYS_EMPLOYED": pa.Column(int, coerce=True),
        "DAYS_REGISTRATION": pa.Column(float, checks=pa.Check.le(0), coerce=True),
        "DAYS_ID_PUBLISH": pa.Column(float, checks=pa.Check.le(0), coerce=True),
        # --- bureau scores ---------------------------------------------------
        "EXT_SOURCE_1": _probability_column(nullable=True),
        "EXT_SOURCE_2": _probability_column(nullable=False),
        "EXT_SOURCE_3": _probability_column(nullable=True),
        # --- household -------------------------------------------------------
        "CNT_CHILDREN": pa.Column(int, checks=pa.Check.ge(0), coerce=True),
        "CNT_FAM_MEMBERS": pa.Column(float, checks=pa.Check.ge(1), coerce=True),
        # --- categorical -----------------------------------------------------
        "REGION_POPULATION_RELATIVE": pa.Column(float, checks=pa.Check.gt(0), coerce=True),
        "REGION_RATING_CLIENT": pa.Column(int, checks=pa.Check.isin([1, 2, 3]), coerce=True),
        "NAME_CONTRACT_TYPE": pa.Column(str),
        "NAME_INCOME_TYPE": pa.Column(str),
        "NAME_EDUCATION_TYPE": pa.Column(str),
        "NAME_HOUSING_TYPE": pa.Column(str),
        "OCCUPATION_TYPE": pa.Column(str, nullable=True),
        "FLAG_OWN_CAR": pa.Column(str, checks=pa.Check.isin(["Y", "N"])),
        "FLAG_OWN_REALTY": pa.Column(str, checks=pa.Check.isin(["Y", "N"])),
    },
    # The real file carries 122 columns; we validate the ones we consume.
    strict=False,
    coerce=True,
    name="home_credit_application",
)


def validate_applications(frame: pd.DataFrame, *, lazy: bool = True) -> pd.DataFrame:
    """Validate an application frame against the schema.

    Args:
        frame: Raw application data.
        lazy: Collect every failure before raising rather than stopping at the
            first. Defaults to true because when data is wrong you want the
            whole list, not one column at a time.

    Returns:
        The validated frame, with dtypes coerced.

    Raises:
        DataValidationError: If validation fails. The message carries the
            underlying pandera failure report.
    """
    try:
        return APPLICATION_SCHEMA.validate(frame, lazy=lazy)
    except (SchemaError, SchemaErrors) as exc:
        msg = f"Application data failed schema validation:\n{exc}"
        raise DataValidationError(msg) from exc
