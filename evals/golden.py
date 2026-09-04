"""The golden set.

A fixed list of applications, committed as data rather than regenerated from a
seed. A set that moves when the generator changes is not golden: a metric
comparison across two commits would then be comparing different populations
and attributing the difference to the change under review.

Only declined applications are useful here, because an approval has no adverse
action to explain. They are selected with a margin below the decline threshold
so that retraining the model does not silently flip half the set to approvals -
if some do flip, the report says how many were usable rather than quietly
measuring a smaller population.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import pandas as pd

from aae.data.schema import SCORING_SCHEMA
from aae.data.synthetic import generate_applications
from aae.logging import get_logger

if TYPE_CHECKING:
    from aae.ml.decision import DecisionEngine

logger = get_logger(__name__)

GOLDEN_DIR: Final[Path] = Path(__file__).parent / "golden"
GOLDEN_PATH: Final[Path] = GOLDEN_DIR / "applications.csv"

DEFAULT_SIZE: Final[int] = 100
SELECTION_MARGIN: Final[float] = 0.10
"""How far past the decline threshold an application must score to be included.

Without a margin, a marginal case flips to approval on the next retrain and
the golden set silently shrinks.
"""


def load_golden(path: Path = GOLDEN_PATH) -> pd.DataFrame:
    """Load the committed golden set.

    Args:
        path: Where the set lives.

    Returns:
        The applications, validated against the scoring schema.

    Raises:
        FileNotFoundError: If the set has not been built.
    """
    if not path.is_file():
        msg = f"Golden set not found at {path}. Build it with `python -m evals.golden --rebuild`."
        raise FileNotFoundError(msg)

    frame = pd.read_csv(path)
    logger.info("golden_set_loaded", path=str(path), cases=len(frame))
    return frame


def build_golden(
    engine: DecisionEngine,
    *,
    size: int = DEFAULT_SIZE,
    pool: int = 4_000,
    seed: int = 4242,
    path: Path = GOLDEN_PATH,
) -> pd.DataFrame:
    """Select declined applications and write them out.

    Args:
        engine: Used to score candidates and keep the declined ones.
        size: How many cases to keep.
        pool: How many candidates to score.
        seed: Seed for the candidate pool. Distinct from the training seed so
            the golden set is not drawn from rows the model was fitted on.
        path: Where to write.

    Returns:
        The selected applications.
    """
    candidates = generate_applications(n_rows=pool, seed=seed)
    columns = [column for column in SCORING_SCHEMA.columns if column in candidates.columns]

    keep: list[int] = []
    cutoff = engine.threshold + SELECTION_MARGIN

    for index in range(len(candidates)):
        decision = engine.decide(candidates, row=index)
        if decision.probability_default >= cutoff and decision.adverse_factors():
            keep.append(index)
        if len(keep) >= size:
            break

    selected = candidates.iloc[keep][columns].reset_index(drop=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(path, index=False)

    logger.info(
        "golden_set_built",
        path=str(path),
        cases=len(selected),
        scanned=keep[-1] + 1 if keep else 0,
        cutoff=round(cutoff, 3),
    )
    return selected


def _main() -> None:
    """Rebuild the golden set from the command line."""
    import argparse

    from aae.data.loaders import load_applications
    from aae.ml.decision import DecisionEngine
    from aae.ml.train import train_model

    parser = argparse.ArgumentParser(description="Rebuild the evaluation golden set.")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--threshold", type=float, default=0.15)
    args = parser.parse_args()

    loaded = load_applications(force_synthetic=True, n_synthetic=20_000)
    engine = DecisionEngine(train_model(loaded), threshold=args.threshold)
    frame = build_golden(engine, size=args.size)
    print(f"Wrote {len(frame)} declined applications to {GOLDEN_PATH}")


if __name__ == "__main__":
    _main()
