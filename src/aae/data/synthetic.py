"""Synthetic loan applications shaped like the Home Credit dataset.

The real dataset requires a Kaggle account and is 166 MB, which makes it a poor
dependency for tests and impossible for CI. This generator produces data with
the same column names, dtypes, missingness, and broad statistical structure, so
the entire pipeline is exercisable without it. Swapping in the real
``application_train.csv`` is a path change in :mod:`aae.data.loaders`.

The generative process is deliberately not random noise. Three properties are
modelled because the pipeline downstream depends on them:

* **A learnable signal.** ``EXT_SOURCE_*`` dominate, as in the real data, so a
  trained model produces meaningful SHAP attributions rather than noise. A
  denial notice built on noise would be untestable.
* **Realistic missingness.** ``EXT_SOURCE_1`` is absent for most applicants in
  the real data. Code that silently assumes complete columns needs to fail here
  rather than in production.
* **Mild disparate impact, arising through a legitimate mediator.** Income
  differs slightly across groups, and income drives risk. No protected
  attribute enters the outcome directly. This is what indirect discrimination
  actually looks like, and it gives the fairness analysis something real to
  find instead of a clean bill of health that proves nothing.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

DEFAULT_SEED: Final[int] = 20260904
BASE_DEFAULT_RATE: Final[float] = 0.081
"""Roughly the Home Credit positive-class rate, which is heavily imbalanced."""

_CONTRACT_TYPES: Final[tuple[str, ...]] = ("Cash loans", "Revolving loans")
_INCOME_TYPES: Final[tuple[str, ...]] = (
    "Working",
    "Commercial associate",
    "Pensioner",
    "State servant",
)
_EDUCATION_TYPES: Final[tuple[str, ...]] = (
    "Secondary / secondary special",
    "Higher education",
    "Incomplete higher",
    "Lower secondary",
)
_FAMILY_STATUS: Final[tuple[str, ...]] = (
    "Married",
    "Single / not married",
    "Civil marriage",
    "Separated",
    "Widow",
)
_HOUSING_TYPES: Final[tuple[str, ...]] = (
    "House / apartment",
    "With parents",
    "Municipal apartment",
    "Rented apartment",
)
_OCCUPATIONS: Final[tuple[str, ...]] = (
    "Laborers",
    "Sales staff",
    "Core staff",
    "Managers",
    "Drivers",
    "Accountants",
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _solve_intercept(risk: np.ndarray, target_rate: float, tolerance: float = 1e-6) -> float:
    """Find the intercept that yields the requested mean event probability.

    The obvious shortcut, ``logit(target) - risk.mean()``, is wrong: sigmoid is
    non-linear, so the mean of the transformed values is not the transform of
    the mean. With a realistic risk spread that shortcut overshoots the base
    rate by more than a factor of two, which would make the dataset far more
    balanced than real lending data and quietly flatter every model metric.

    Args:
        risk: Per-row log-odds contributions, excluding the intercept.
        target_rate: Desired mean probability.
        tolerance: Absolute convergence tolerance on the realised rate.

    Returns:
        The intercept to add to ``risk``.
    """
    low, high = -50.0, 50.0
    for _ in range(200):
        middle = (low + high) / 2.0
        realised = float(_sigmoid(middle + risk).mean())
        if abs(realised - target_rate) < tolerance:
            return middle
        if realised < target_rate:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def generate_applications(
    n_rows: int = 20_000,
    seed: int = DEFAULT_SEED,
    *,
    disparity_strength: float = 0.35,
) -> pd.DataFrame:
    """Generate a synthetic ``application_train``-shaped frame.

    Args:
        n_rows: Number of applications to generate.
        seed: Seed for reproducibility. The same seed always yields the same
            frame, so model metrics are comparable across runs.
        disparity_strength: How strongly group membership shifts income. Zero
            produces a fair dataset; the default produces measurable indirect
            disparity through income as a mediator. Group never enters the
            outcome directly.

    Returns:
        A frame with the Home Credit column names, including ``TARGET`` and the
        protected attributes needed for fairness analysis.
    """
    rng = np.random.default_rng(seed)

    # --- protected attributes -------------------------------------------------
    # Generated first because income depends on them, which is precisely the
    # mediation path that produces indirect discrimination.
    gender = rng.choice(["F", "M"], size=n_rows, p=[0.66, 0.34])
    age_years = rng.gamma(shape=9.0, scale=4.6, size=n_rows) + 21.0
    age_years = np.clip(age_years, 21.0, 69.0)
    days_birth = -(age_years * 365.25).astype(np.int64)
    family_status = rng.choice(_FAMILY_STATUS, size=n_rows, p=[0.64, 0.15, 0.10, 0.07, 0.04])

    # --- income, mediating group membership ----------------------------------
    income_noise = rng.normal(0.0, 0.45, size=n_rows)
    group_shift = np.where(gender == "M", disparity_strength, -disparity_strength * 0.5)
    log_income = 11.9 + group_shift + income_noise + 0.012 * (age_years - 40.0)
    amt_income = np.exp(log_income).round(-2)

    # --- loan structure -------------------------------------------------------
    credit_multiple = np.clip(rng.gamma(shape=4.0, scale=0.9, size=n_rows), 0.4, 12.0)
    amt_credit = (amt_income * credit_multiple).round(-3)
    goods_price = (amt_credit * rng.uniform(0.80, 1.0, size=n_rows)).round(-3)
    term_months = rng.integers(12, 120, size=n_rows)
    amt_annuity = (amt_credit / term_months * rng.uniform(1.0, 1.35, size=n_rows)).round(-1)

    # --- employment and registration -----------------------------------------
    employed_years = np.clip(rng.gamma(3.0, 2.2, size=n_rows), 0.0, age_years - 18.0)
    days_employed = -(employed_years * 365.25).astype(np.int64)
    # The real dataset encodes "not employed" as the sentinel 365243. Reproduced
    # so downstream code must handle it rather than treating it as 1000 years.
    pensioner = rng.random(n_rows) < 0.18
    days_employed = np.where(pensioner, 365243, days_employed)

    days_registration = -(rng.uniform(200, 9000, size=n_rows)).astype(np.int64)
    days_id_publish = -(rng.uniform(100, 6000, size=n_rows)).astype(np.int64)

    # --- external credit bureau scores ---------------------------------------
    latent_quality = rng.normal(0.0, 1.0, size=n_rows)
    ext_2 = np.clip(_sigmoid(latent_quality * 1.3 + rng.normal(0, 0.35, n_rows)), 0.001, 0.999)
    ext_3 = np.clip(_sigmoid(latent_quality * 1.1 + rng.normal(0, 0.45, n_rows)), 0.001, 0.999)
    ext_1 = np.clip(_sigmoid(latent_quality * 0.9 + rng.normal(0, 0.6, n_rows)), 0.001, 0.999)
    # EXT_SOURCE_1 is missing for most applicants in the real data.
    ext_1 = np.where(rng.random(n_rows) < 0.56, np.nan, ext_1)
    ext_3 = np.where(rng.random(n_rows) < 0.20, np.nan, ext_3)

    children = rng.poisson(0.45, size=n_rows).clip(0, 8)
    family_members = children + np.where(
        np.isin(family_status, ["Married", "Civil marriage"]), 2, 1
    )

    # --- outcome --------------------------------------------------------------
    # Group membership is deliberately absent from this expression. Any measured
    # disparity flows through income, which is a legitimate underwriting factor.
    credit_income_ratio = amt_credit / np.maximum(amt_income, 1.0)
    annuity_income_ratio = amt_annuity * 12.0 / np.maximum(amt_income, 1.0)

    risk = (
        -1.15 * (ext_2 - 0.5) * 4.0
        - 0.85 * (np.nan_to_num(ext_3, nan=0.5) - 0.5) * 4.0
        + 0.22 * np.clip(credit_income_ratio, 0, 15)
        + 1.10 * np.clip(annuity_income_ratio, 0, 3)
        - 0.30 * np.log1p(np.maximum(employed_years, 0.0))
        # Lower income carries higher risk. This term is what makes the
        # disparity real: group shifts income, and income shifts risk, so the
        # measured outcome gap flows entirely through a legitimate underwriting
        # factor. Without it the generator would produce a fair dataset and the
        # fairness analysis would have nothing to detect.
        - 0.55 * (log_income - log_income.mean())
        + rng.normal(0.0, 0.8, size=n_rows)
    )
    target = (rng.random(n_rows) < _sigmoid(_solve_intercept(risk, BASE_DEFAULT_RATE) + risk)).astype(
        np.int8
    )

    frame = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(100_001, 100_001 + n_rows, dtype=np.int64),
            "TARGET": target,
            # Protected attributes: loaded for fairness measurement, never features.
            "CODE_GENDER": gender,
            "DAYS_BIRTH": days_birth,
            "NAME_FAMILY_STATUS": family_status,
            # Model inputs.
            "AMT_INCOME_TOTAL": amt_income,
            "AMT_CREDIT": amt_credit,
            "AMT_ANNUITY": amt_annuity,
            "AMT_GOODS_PRICE": goods_price,
            "DAYS_EMPLOYED": days_employed,
            "DAYS_REGISTRATION": days_registration,
            "DAYS_ID_PUBLISH": days_id_publish,
            "EXT_SOURCE_1": ext_1,
            "EXT_SOURCE_2": ext_2,
            "EXT_SOURCE_3": ext_3,
            "CNT_CHILDREN": children,
            "CNT_FAM_MEMBERS": family_members,
            "REGION_POPULATION_RELATIVE": rng.choice(
                [0.00496, 0.00963, 0.01885, 0.02866, 0.03554, 0.04622], size=n_rows
            ),
            "REGION_RATING_CLIENT": rng.choice([1, 2, 3], size=n_rows, p=[0.16, 0.74, 0.10]),
            "NAME_CONTRACT_TYPE": rng.choice(_CONTRACT_TYPES, size=n_rows, p=[0.905, 0.095]),
            "NAME_INCOME_TYPE": np.where(
                pensioner,
                "Pensioner",
                rng.choice(_INCOME_TYPES[:2] + _INCOME_TYPES[3:], size=n_rows),
            ),
            "NAME_EDUCATION_TYPE": rng.choice(
                _EDUCATION_TYPES, size=n_rows, p=[0.71, 0.24, 0.034, 0.016]
            ),
            "NAME_HOUSING_TYPE": rng.choice(_HOUSING_TYPES, size=n_rows, p=[0.89, 0.05, 0.04, 0.02]),
            "OCCUPATION_TYPE": rng.choice(_OCCUPATIONS, size=n_rows),
            "FLAG_OWN_CAR": rng.choice(["Y", "N"], size=n_rows, p=[0.34, 0.66]),
            "FLAG_OWN_REALTY": rng.choice(["Y", "N"], size=n_rows, p=[0.69, 0.31]),
        }
    )
    # A small share of annuities are genuinely absent in the real data.
    frame.loc[frame.sample(frac=0.003, random_state=seed).index, "AMT_ANNUITY"] = np.nan
    return frame
