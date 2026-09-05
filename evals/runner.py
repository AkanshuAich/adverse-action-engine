"""Running the golden set and reporting what happened.

The gate compares against a committed baseline rather than against fixed
thresholds. Absolute thresholds either sit so low they never fire or get
raised until someone turns them off; a baseline says "this change made it
worse", which is the question a reviewer actually has.

Two rules are absolute rather than relative. No issued notice may contain a
prohibited reference, and no metric may fall more than the tolerance below
baseline. The first is a legal requirement and has no acceptable non-zero
value. The second is a regression.

Throttling exists because free tiers cap requests per minute. It is off for
the simulator, which has no such limit and would otherwise turn a two-second
run into a ten-minute one.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import pandas as pd

from aae.data.loaders import load_applications
from aae.domain.errors import GenerationError
from aae.generation.graph import NoticeGenerator
from aae.generation.payload import build_payload
from aae.jurisdiction.india_rbi import INDIA_RBI
from aae.logging import get_logger
from aae.ml.decision import DecisionEngine
from aae.ml.train import train_model
from aae.retrieval.corpus import india_rbi_corpus
from aae.verification.prose import check_prose
from aae.verification.rules import check_prohibited_content
from aae.verification.verifier import NoticeVerifier
from evals.golden import load_golden
from evals.metrics import CaseOutcome, EvalMetrics
from evals.readability import PLAIN_LANGUAGE_FLOOR, score_readability
from evals.simulator import SimulatedProvider, describe_profile, profile_from_name

if TYPE_CHECKING:
    from aae.generation.providers.base import StructuredProvider

logger = get_logger(__name__)

REPORT_DIR: Final[Path] = Path(__file__).parent / "reports"
BASELINE_PATH: Final[Path] = Path(__file__).parent / "baseline.json"

GATED_METRICS: Final[tuple[str, ...]] = (
    "groundedness_rate",
    "post_repair_rate",
    "factor_fidelity",
    "citation_precision",
    "element_coverage",
)
"""Metrics a change may not degrade. Higher is better for every one."""

MAX_PROVIDER_FAILURE_RATE: Final[float] = 0.10
"""How many cases may be abandoned before the run stops meaning anything.

Metrics are computed over the cases that finished. If a tenth of the golden
set never reached the verifier, the survivors are not a random sample of it -
a rate limit falls hardest on the longest prompts, which are the hard cases -
and comparing their scores to a baseline measured over the whole set is
comparing two different populations.

Observed: a five-case trial run against a live backend lost four cases to a
rate limit and reported GATE PASSED on the strength of the one that survived.
"""

REGRESSION_TOLERANCE: Final[float] = 0.03
"""How far a gated metric may fall below baseline before the gate fires.

Wide enough to absorb the noise of a stochastic pipeline, narrow enough that a
real degradation shows. With the simulator the pipeline is deterministic, so
any movement at all is a genuine change.
"""


@dataclass(frozen=True)
class GateResult:
    """Whether a run may be merged."""

    passed: bool
    failures: tuple[str, ...]

    def render(self) -> str:
        """Render the verdict for a terminal or a CI log.

        Returns:
            One line per failure, or a single line saying the gate passed.
        """
        if self.passed:
            return "GATE PASSED"
        return "GATE FAILED\n" + "\n".join(f"  - {failure}" for failure in self.failures)


def check_gate(
    metrics: EvalMetrics,
    baseline: dict[str, Any] | None,
    *,
    tolerance: float = REGRESSION_TOLERANCE,
    max_provider_failure_rate: float = MAX_PROVIDER_FAILURE_RATE,
) -> GateResult:
    """Decide whether a run is acceptable.

    A run can fail without any metric moving. If most cases never reached the
    verifier the numbers describe a handful of survivors, so completeness is
    checked before the metrics are believed at all.

    Args:
        metrics: The run just completed.
        baseline: The committed reference, or ``None`` on a first run.
        tolerance: How far a gated metric may fall.
        max_provider_failure_rate: Share of cases that may be abandoned before
            the run is treated as unmeasured rather than merely worse.

    Returns:
        The verdict and every reason it failed.
    """
    failures: list[str] = []

    if metrics.prohibited_content_rate > 0.0:
        failures.append(
            f"prohibited_content_rate is {metrics.prohibited_content_rate:.4f}; it must be "
            "zero. A notice referencing a protected characteristic reached an applicant."
        )

    if metrics.cases == 0:
        failures.append("no cases were evaluated; the golden set produced nothing to measure")

    attempted = metrics.cases + metrics.provider_failures
    if attempted and metrics.provider_failures / attempted > max_provider_failure_rate:
        failures.append(
            f"{metrics.provider_failures} of {attempted} cases were abandoned at the "
            f"provider, above the {max_provider_failure_rate:.0%} ceiling. The metrics "
            "below describe only the cases that finished, which is not the golden set: "
            "they must not be compared against a baseline measured over all of it."
        )

    if baseline is None:
        logger.warning("no_baseline", detail="Recording this run as the baseline.")
        return GateResult(passed=not failures, failures=tuple(failures))

    for name in GATED_METRICS:
        current = getattr(metrics, name)
        previous = baseline.get(name)
        if previous is None:
            continue
        if current < previous - tolerance:
            failures.append(
                f"{name} fell from {previous:.4f} to {current:.4f}, beyond the "
                f"{tolerance:.2f} tolerance"
            )

    return GateResult(passed=not failures, failures=tuple(failures))


@dataclass
class EvalRunner:
    """Runs the whole pipeline over the golden set.

    Attributes:
        engine: Scores and explains applications.
        generator: Produces and repairs notices.
        throttle_seconds: Pause between cases, for rate-limited backends.
    """

    engine: DecisionEngine
    generator: NoticeGenerator
    throttle_seconds: float = 0.0

    def run(self, applications: pd.DataFrame) -> tuple[EvalMetrics, list[CaseOutcome]]:
        """Evaluate every application in the set.

        Args:
            applications: The golden set.

        Returns:
            The aggregate metrics and every measured case.
        """
        corpus = list(india_rbi_corpus())
        required = len(INDIA_RBI.required_elements)
        cases: list[CaseOutcome] = []
        provider_failures = 0
        skipped_approvals = 0

        for index in range(len(applications)):
            decision = self.engine.decide(applications, row=index)

            if decision.decision.value != "decline" or not decision.adverse_factors():
                # An approval has no adverse action to explain. Counted rather
                # than silently dropped, so a retrain that flips the set is
                # visible in the report.
                skipped_approvals += 1
                continue

            payload = build_payload(decision, INDIA_RBI, corpus)
            started = time.perf_counter()
            try:
                outcome = self.generator.generate(decision, payload)
            except GenerationError as exc:
                provider_failures += 1
                logger.warning(
                    "case_abandoned",
                    application_id=decision.application_id,
                    error=str(exc),
                )
                continue
            latency_ms = (time.perf_counter() - started) * 1000.0

            cases.append(
                CaseOutcome.from_outcome(
                    outcome,
                    application_id=decision.application_id,
                    elements_required=required,
                    prohibited_in_issued=self._prohibited_in_issued(outcome, payload),
                    readability=(
                        score_readability(outcome.body).flesch_reading_ease
                        if outcome.body
                        else None
                    ),
                    latency_ms=latency_ms,
                )
            )

            if self.throttle_seconds:
                time.sleep(self.throttle_seconds)

        if skipped_approvals:
            logger.warning(
                "golden_cases_skipped",
                approvals=skipped_approvals,
                detail="These applications no longer decline; the golden set may need rebuilding.",
            )

        metrics = EvalMetrics.from_cases(
            cases,
            provider_failures=provider_failures,
            readability_floor=PLAIN_LANGUAGE_FLOOR,
        )
        return metrics, cases

    @staticmethod
    def _prohibited_in_issued(outcome: Any, payload: Any) -> bool:
        """Re-check an issued notice for prohibited content, independently.

        Deliberately not inferred from the pipeline having passed it. The
        claim under test is that the gate works, and asking the gate whether
        it worked would answer itself.

        Args:
            outcome: The generation outcome.
            payload: What the model was shown.

        Returns:
            Whether a prohibited reference survived into an issued notice.
        """
        if not outcome.issued or outcome.notice is None:
            return False

        structural = check_prohibited_content(outcome.notice, INDIA_RBI)
        prose = (
            check_prose(outcome.body, outcome.notice, payload, INDIA_RBI) if outcome.body else ()
        )
        return bool(structural) or any(
            violation.code.value == "prohibited_content" for violation in prose
        )


def build_provider(name: str, profile: str) -> StructuredProvider:
    """Construct the provider named on the command line.

    Args:
        name: ``simulated`` or a real provider from settings.
        profile: Failure profile, used only by the simulator.

    Returns:
        The provider.
    """
    if name == "simulated":
        return SimulatedProvider(profile_from_name(profile))

    from aae.config import LLMProvider, get_settings
    from aae.generation.providers.registry import build_provider as build_real

    settings = get_settings().model_copy(update={"llm_provider": LLMProvider(name)})
    return build_real(settings)


def write_report(
    metrics: EvalMetrics, cases: list[CaseOutcome], provider: str, profile: str
) -> Path:
    """Write a timestamped report and return its path.

    Committed to the repository so the trend across prompt and model versions
    is visible in git history rather than living in someone's terminal.

    Args:
        metrics: The aggregate result.
        cases: Every measured case.
        provider: Which backend was used.
        profile: Which failure profile, if simulated.

    Returns:
        Where the report was written.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = REPORT_DIR / f"{stamp}-{provider}.json"

    path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "provider": provider,
                "failure_profile": (
                    dict(describe_profile(profile_from_name(profile)))
                    if provider == "simulated"
                    else None
                ),
                "metrics": metrics.to_dict(),
                "escalated_applications": [case.application_id for case in cases if case.escalated],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def render(metrics: EvalMetrics) -> str:
    """Render the metrics as a table for a terminal.

    Args:
        metrics: The aggregate result.

    Returns:
        A printable table.
    """
    rows = [
        ("cases evaluated", f"{metrics.cases}"),
        ("groundedness rate (first attempt)", f"{metrics.groundedness_rate:.1%}"),
        ("post-repair rate (issued)", f"{metrics.post_repair_rate:.1%}"),
        ("escalation rate", f"{metrics.escalation_rate:.1%}"),
        ("factor fidelity", f"{metrics.factor_fidelity:.1%}"),
        ("citation precision", f"{metrics.citation_precision:.1%}"),
        ("element coverage", f"{metrics.element_coverage:.1%}"),
        ("prohibited content in issued", f"{metrics.prohibited_content_rate:.4f}"),
        ("prohibited attempts caught", f"{metrics.prohibited_attempts_caught}"),
        ("mean attempts", f"{metrics.mean_attempts:.2f}"),
        ("readability (Flesch)", f"{metrics.readability_mean:.1f}"),
        ("below plain-language floor", f"{metrics.readability_below_floor:.1%}"),
        ("latency p50", f"{metrics.latency_p50_ms:.0f} ms"),
        ("latency p95", f"{metrics.latency_p95_ms:.0f} ms"),
        ("provider failures", f"{metrics.provider_failures}"),
    ]
    width = max(len(name) for name, _ in rows)
    lines = [f"  {name.ljust(width)}  {value}" for name, value in rows]

    if metrics.violations_by_code:
        lines.append("")
        lines.append("  violations caught, by check:")
        lines.extend(
            f"    {code.ljust(width - 2)}  {count}"
            for code, count in metrics.violations_by_code.items()
        )
    return "\n".join(lines)


def main() -> int:
    """Run the harness from the command line.

    Returns:
        Zero if the gate passed, one otherwise.
    """
    parser = argparse.ArgumentParser(description="Run the evaluation harness.")
    parser.add_argument("--provider", default="simulated")
    parser.add_argument("--profile", default="default", help="Simulator failure profile.")
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--throttle", type=float, default=0.0, help="Seconds between cases.")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N cases.")
    parser.add_argument(
        "--update-baseline", action="store_true", help="Record this run as the new baseline."
    )
    parser.add_argument("--no-report", action="store_true", help="Skip writing a report file.")
    args = parser.parse_args()

    applications = load_golden()
    if args.limit:
        applications = applications.head(args.limit)

    engine = DecisionEngine(
        train_model(load_applications(force_synthetic=True, n_synthetic=20_000)),
        threshold=args.threshold,
    )
    generator = NoticeGenerator(
        provider=build_provider(args.provider, args.profile),
        verifier=NoticeVerifier(INDIA_RBI, india_rbi_corpus()),
    )

    metrics, cases = EvalRunner(engine, generator, args.throttle).run(applications)

    print(f"\nEvaluation: {args.provider}\n")
    print(render(metrics))

    baseline = (
        json.loads(BASELINE_PATH.read_text(encoding="utf-8")) if BASELINE_PATH.is_file() else None
    )
    gate = check_gate(metrics, baseline)
    print(f"\n{gate.render()}\n")

    if not args.no_report:
        print(f"Report written to {write_report(metrics, cases, args.provider, args.profile)}")

    if args.update_baseline:
        BASELINE_PATH.write_text(json.dumps(metrics.to_dict(), indent=2), encoding="utf-8")
        print(f"Baseline updated at {BASELINE_PATH}")

    return 0 if gate.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
