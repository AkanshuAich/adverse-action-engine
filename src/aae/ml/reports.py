"""Generating the monitoring reports.

Two artifacts a model risk function would ask for, produced by running the
code rather than written by hand. A fairness report that someone typed out is
a description of what they believed; one the pipeline emits is a measurement,
and it can be regenerated when the model changes.

Committed to the repository so the trend is visible in git history. That is
the same reasoning as the evaluation reports: a number that lives only in
somebody's terminal cannot be compared with last month's.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from aae.logging import get_logger
from aae.ml.drift import detect_drift
from aae.ml.fairness import analyse_fairness
from aae.ml.features import build_features

if TYPE_CHECKING:
    import pandas as pd

    from aae.ml.decision import DecisionEngine
    from aae.ml.drift import DriftReport
    from aae.ml.fairness import FairnessReport

logger = get_logger(__name__)

REPORT_DIR: Final[Path] = Path("reports")


def build_fairness_report(engine: DecisionEngine, applications: pd.DataFrame) -> FairnessReport:
    """Score a population and measure disparate impact across it.

    Args:
        engine: The decision engine in force.
        applications: Data including the protected attributes, which are loaded
            for exactly this purpose and are never model features.

    Returns:
        The report.
    """
    declined = np.array(
        [
            1 if engine.decide(applications, row=index).decision.value == "decline" else 0
            for index in range(len(applications))
        ]
    )
    return analyse_fairness(applications, declined, model_version=engine.model_version)


def build_drift_report(
    engine: DecisionEngine, reference: pd.DataFrame, current: pd.DataFrame
) -> DriftReport:
    """Compare a current population against the training distribution.

    Args:
        engine: The decision engine in force.
        reference: Applications the model was fitted on.
        current: Applications being scored now.

    Returns:
        The report.
    """
    reference_features = build_features(reference, engine.model.spec)
    current_features = build_features(current, engine.model.spec)

    return detect_drift(
        reference_features,
        current_features,
        engine.model.predict_proba(reference_features),
        engine.model.predict_proba(current_features),
        model_version=engine.model_version,
    )


def write_report(name: str, payload: dict[str, Any], directory: Path = REPORT_DIR) -> Path:
    """Write a report as JSON.

    Args:
        name: File stem.
        payload: The report content.
        directory: Where to write.

    Returns:
        The path written.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    path.write_text(
        json.dumps({"generated_at": datetime.now(UTC).isoformat(), **payload}, indent=2),
        encoding="utf-8",
    )
    logger.info("report_written", path=str(path))
    return path


def render_fairness(report: FairnessReport) -> str:
    """Render a fairness report for a terminal.

    Args:
        report: The report.

    Returns:
        A printable summary.
    """
    lines = [
        f"Model {report.model_version} over {report.applications:,} applications",
        f"Overall decline rate {report.overall_decline_rate:.1%}",
        "",
    ]
    for group in report.groups:
        verdict = "passes" if group.passes_four_fifths else "BELOW FOUR-FIFTHS"
        ratio = f"{group.adverse_impact_ratio:.3f}"
        lines.append(f"  {group.attribute}  adverse impact ratio {ratio}  [{verdict}]")
        for name, rate in sorted(group.selection_rates.items()):
            size = group.group_sizes.get(name, 0)
            lines.append(f"      {name:<24} decline {rate:.1%}  (n={size:,})")
        if group.equalized_odds_difference is not None:
            lines.append(f"      equalized odds difference {group.equalized_odds_difference:.3f}")
        lines.append("")

    lines.append(
        f"Findings: {', '.join(g.attribute for g in report.findings) or 'none below the screen'}"
    )
    return "\n".join(lines)


def render_drift(report: DriftReport) -> str:
    """Render a drift report for a terminal.

    Args:
        report: The report.

    Returns:
        A printable summary.
    """
    lines = [
        f"Model {report.model_version}: {report.reference_rows:,} reference rows "
        f"vs {report.current_rows:,} current",
        f"Score PSI {report.score_psi:.4f} ({report.score_severity.value}), "
        f"KS {report.score_ks:.4f}",
        "",
    ]
    for feature in report.features[:10]:
        marker = " <-- drifted" if feature.severity.value != "stable" else ""
        lines.append(
            f"  {feature.feature:<28} PSI {feature.psi:7.4f}  {feature.severity.value}{marker}"
        )
    lines.append("")
    lines.append(
        "Requires attention" if report.requires_attention else "No feature outside the stable band"
    )
    return "\n".join(lines)


def _main() -> None:
    """Generate both monitoring reports."""
    import argparse

    from aae.data.loaders import load_applications
    from aae.data.synthetic import generate_applications
    from aae.ml.decision import DecisionEngine
    from aae.ml.train import train_model

    parser = argparse.ArgumentParser(description="Generate the monitoring reports.")
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--sample", type=int, default=4_000)
    args = parser.parse_args()

    loaded = load_applications(force_synthetic=True, n_synthetic=args.rows)
    engine = DecisionEngine(train_model(loaded), threshold=args.threshold)

    reference = loaded.frame.head(args.sample)
    fairness = build_fairness_report(engine, reference)
    print("\n=== Fairness ===\n")  # noqa: T201
    print(render_fairness(fairness))  # noqa: T201
    write_report("fairness", fairness.to_dict())

    # A later population, drawn from a different seed, standing in for the
    # applicants a deployed model would be scoring some months on.
    current = generate_applications(n_rows=args.sample, seed=8_888)
    drift = build_drift_report(engine, reference, current)
    print("\n=== Drift ===\n")  # noqa: T201
    print(render_drift(drift))  # noqa: T201
    write_report("drift", drift.to_dict())


if __name__ == "__main__":
    _main()
