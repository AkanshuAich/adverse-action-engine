"""Loading application data.

Two sources, one validated output. The real Home Credit extract is used when
present; otherwise a synthetic frame with the same shape stands in, so tests,
CI, and a fresh clone all work without a 166 MB Kaggle download.

Which source was used is returned alongside the data rather than logged and
forgotten: a model card must state what the model was trained on, and
"synthetic" is a material fact about a credit model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

import pandas as pd

from aae.data.schema import validate_applications
from aae.data.synthetic import DEFAULT_SEED, generate_applications
from aae.logging import get_logger

logger = get_logger(__name__)

DEFAULT_DATA_DIR: Final[Path] = Path("data")
APPLICATION_FILENAME: Final[str] = "application_train.csv"


class DataSource(StrEnum):
    """Where an application frame came from."""

    HOME_CREDIT = "home_credit"
    SYNTHETIC = "synthetic"


@dataclass(frozen=True)
class LoadedApplications:
    """A validated application frame together with its provenance."""

    frame: pd.DataFrame
    source: DataSource
    row_count: int
    default_rate: float

    def describe(self) -> str:
        """Summarise the load for logs and the model card.

        Returns:
            A one-line human-readable summary.
        """
        return (
            f"{self.row_count:,} applications from {self.source.value} "
            f"(default rate {self.default_rate:.2%})"
        )


def load_applications(
    data_dir: Path = DEFAULT_DATA_DIR,
    *,
    n_synthetic: int = 20_000,
    seed: int = DEFAULT_SEED,
    force_synthetic: bool = False,
) -> LoadedApplications:
    """Load application data, preferring the real extract when available.

    Args:
        data_dir: Directory searched for ``application_train.csv``.
        n_synthetic: Rows to generate when falling back to synthetic data.
        seed: Seed for the synthetic generator.
        force_synthetic: Ignore any real file and always generate. Used by tests
            so their behaviour does not depend on whether a developer happens
            to have downloaded the dataset.

    Returns:
        The validated frame and its provenance.

    Raises:
        DataValidationError: If the loaded data fails schema validation.
    """
    csv_path = data_dir / APPLICATION_FILENAME

    if not force_synthetic and csv_path.is_file():
        logger.info("loading_real_data", path=str(csv_path))
        frame = pd.read_csv(csv_path)
        source = DataSource.HOME_CREDIT
    else:
        if not force_synthetic:
            logger.info(
                "real_data_absent_using_synthetic",
                expected_path=str(csv_path),
                hint="Place application_train.csv there to train on real data.",
            )
        frame = generate_applications(n_rows=n_synthetic, seed=seed)
        source = DataSource.SYNTHETIC

    validated = validate_applications(frame)
    loaded = LoadedApplications(
        frame=validated,
        source=source,
        row_count=len(validated),
        default_rate=float(validated["TARGET"].mean()),
    )
    logger.info("applications_loaded", summary=loaded.describe())
    return loaded
