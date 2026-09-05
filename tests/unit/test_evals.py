"""The evaluation harness.

A gate nobody has tested is a gate nobody should trust. These assert that it
fires on the things it exists to catch - a prohibited reference reaching an
applicant, a metric falling below baseline - and that it does not fire on
noise.

The simulator gets its own tests because the metrics are only meaningful if it
is deterministic. A number that moves between runs cannot distinguish a
regression from a coin flip.
"""

from __future__ import annotations

import pytest
from evals.metrics import CaseOutcome, EvalMetrics
from evals.readability import (
    PLAIN_LANGUAGE_FLOOR,
    count_syllables,
    score_readability,
)
from evals.runner import (
    MAX_PROVIDER_FAILURE_RATE,
    REGRESSION_TOLERANCE,
    check_gate,
)
from evals.simulator import PERFECT, FailureProfile, SimulatedProvider, parse_prompt

from aae.domain.errors import ProviderError
from aae.generation.schemas import RenderedBody, SelectedNotice

PLAIN = (
    "We could not approve your loan. Your income is too low for the amount "
    "you asked for. Please call us if you want to talk about this."
)
DENSE = (
    "Notwithstanding the aforementioned determination, the institution's "
    "underwriting methodology necessitated a comprehensive reevaluation of the "
    "applicant's aggregated financial circumstances, incorporating multifactorial "
    "considerations pertaining to indebtedness sustainability."
)


def _case(**overrides: object) -> CaseOutcome:
    defaults: dict[str, object] = {
        "application_id": "APP-1",
        "issued": True,
        "escalated": False,
        "attempts": 1,
        "first_attempt_passed": True,
        "first_attempt_reasons": 2,
        "first_attempt_citations": 1,
        "first_attempt_codes": (),
        "all_violation_codes": (),
        "elements_required": 4,
        "prohibited_in_issued": False,
        "readability": 60.0,
        "latency_ms": 100.0,
    }
    return CaseOutcome(**{**defaults, **overrides})  # type: ignore[arg-type]


def _metrics(cases: list[CaseOutcome], **kwargs: object) -> EvalMetrics:
    return EvalMetrics.from_cases(
        cases,
        readability_floor=PLAIN_LANGUAGE_FLOOR,
        **kwargs,  # type: ignore[arg-type]
    )


class TestReadability:
    def test_plain_language_scores_above_the_floor(self):
        assert score_readability(PLAIN).is_plain_language

    def test_dense_prose_scores_below_the_floor(self):
        assert not score_readability(DENSE).is_plain_language

    def test_plain_beats_dense(self):
        plain = score_readability(PLAIN).flesch_reading_ease
        dense = score_readability(DENSE).flesch_reading_ease
        assert plain > dense

    @pytest.mark.parametrize(
        ("word", "expected"),
        [
            ("the", 1),
            ("hope", 1),
            ("table", 2),
            ("little", 2),
            ("income", 2),
            ("application", 4),
            ("a", 1),
        ],
    )
    def test_syllable_counts(self, word: str, expected: int):
        assert count_syllables(word) == expected

    def test_every_word_has_at_least_one_syllable(self):
        assert count_syllables("rhythm") >= 1

    def test_empty_text_does_not_divide_by_zero(self):
        score = score_readability("")
        assert score.flesch_reading_ease == 0.0
        assert score.words == 0

    def test_counts_are_reported_alongside_the_score(self):
        score = score_readability("One sentence here. And a second one.")
        assert score.sentences == 2
        assert score.words == 7
        assert score.words_per_sentence == pytest.approx(3.5)


class TestMetrics:
    def test_groundedness_counts_only_first_attempt_successes(self):
        """Crediting repaired notices to the model would flatter it."""
        metrics = _metrics(
            [
                _case(first_attempt_passed=True),
                _case(first_attempt_passed=False, attempts=2),
                _case(first_attempt_passed=False, attempts=3),
                _case(first_attempt_passed=True),
            ]
        )
        assert metrics.groundedness_rate == 0.5
        assert metrics.post_repair_rate == 1.0

    def test_escalation_rate_excludes_provider_failures(self):
        """An outage must not look like the model degrading."""
        metrics = _metrics([_case(), _case(issued=False, escalated=True)], provider_failures=5)
        assert metrics.escalation_rate == 0.5
        assert metrics.provider_failures == 5

    def test_factor_fidelity_is_reasons_grounded_over_reasons_given(self):
        metrics = _metrics(
            [
                _case(
                    first_attempt_reasons=4,
                    first_attempt_codes=("factor_grounding",),
                    first_attempt_passed=False,
                )
            ]
        )
        assert metrics.factor_fidelity == 0.75

    def test_citation_precision_is_citations_valid_over_citations_given(self):
        metrics = _metrics(
            [
                _case(
                    first_attempt_citations=2,
                    first_attempt_codes=("citation_validity",),
                    first_attempt_passed=False,
                )
            ]
        )
        assert metrics.citation_precision == 0.5

    def test_prohibited_content_is_measured_over_issued_notices_only(self):
        """A caught proposal is the check working, not a failure."""
        metrics = _metrics(
            [
                _case(all_violation_codes=("prohibited_content",)),
                _case(),
            ]
        )
        assert metrics.prohibited_content_rate == 0.0
        assert metrics.prohibited_attempts_caught == 1

    def test_prohibited_content_in_an_issued_notice_is_counted(self):
        metrics = _metrics([_case(prohibited_in_issued=True), _case()])
        assert metrics.prohibited_content_rate == 0.5

    def test_an_empty_run_reports_zeroes_rather_than_dividing_by_zero(self):
        metrics = _metrics([])
        assert metrics.cases == 0
        assert metrics.groundedness_rate == 0.0

    def test_latency_percentiles_are_real_measurements(self):
        """Interpolating invents a latency that never occurred."""
        cases = [_case(latency_ms=float(value)) for value in range(1, 101)]
        metrics = _metrics(cases)
        assert metrics.latency_p50_ms in {50.0, 51.0}
        assert metrics.latency_p95_ms in {95.0, 96.0}

    def test_serialises_to_json_compatible_data(self):
        payload = _metrics([_case()]).to_dict()
        assert payload["cases"] == 1
        assert isinstance(payload["violations_by_code"], dict)


class TestGate:
    def test_passes_when_nothing_regressed(self):
        metrics = _metrics([_case()])
        baseline = metrics.to_dict()
        assert check_gate(metrics, baseline).passed

    def test_fires_when_prohibited_content_reaches_an_applicant(self):
        """Zero is the only acceptable value; there is no tolerance for this."""
        metrics = _metrics([_case(prohibited_in_issued=True)])
        result = check_gate(metrics, metrics.to_dict())
        assert not result.passed
        assert any("prohibited_content_rate" in failure for failure in result.failures)

    def test_fires_on_a_regression_beyond_tolerance(self):
        baseline = _metrics([_case() for _ in range(10)]).to_dict()
        degraded = _metrics([_case(first_attempt_passed=index > 4) for index in range(10)])
        result = check_gate(degraded, baseline)
        assert not result.passed
        assert any("groundedness_rate fell" in failure for failure in result.failures)

    def test_tolerates_movement_within_the_tolerance(self):
        baseline = _metrics([_case() for _ in range(100)]).to_dict()
        # One case in a hundred, well inside the tolerance band.
        slightly_worse = _metrics([_case(first_attempt_passed=index > 0) for index in range(100)])
        assert REGRESSION_TOLERANCE > 0.01
        assert check_gate(slightly_worse, baseline).passed

    def test_a_first_run_with_no_baseline_passes(self):
        assert check_gate(_metrics([_case()]), None).passed

    def test_an_empty_run_fails(self):
        """Measuring nothing must not read as measuring success."""
        result = check_gate(_metrics([]), None)
        assert not result.passed
        assert any("no cases" in failure for failure in result.failures)

    def test_fires_when_most_of_the_run_never_happened(self):
        """A green gate over four abandoned cases is worse than a red one.

        Observed against a live backend: a rate limit abandoned four of five
        cases and the gate passed on the one that survived. Metrics are
        computed over cases that finished, so a run that mostly did not finish
        reports the scores of its survivors and calls them the system's.
        """
        metrics = _metrics([_case()], provider_failures=4)
        result = check_gate(metrics, metrics.to_dict())

        assert not result.passed
        assert any("abandoned at the provider" in failure for failure in result.failures)

    def test_tolerates_the_occasional_abandoned_case(self):
        """A single flaky call should not block a merge."""
        cases = [_case() for _ in range(100)]
        metrics = _metrics(cases, provider_failures=1)

        assert MAX_PROVIDER_FAILURE_RATE > 0.01
        assert check_gate(metrics, metrics.to_dict()).passed

    def test_a_complete_run_is_not_penalised(self):
        metrics = _metrics([_case() for _ in range(10)], provider_failures=0)

        assert check_gate(metrics, metrics.to_dict()).passed

    def test_the_verdict_names_every_failure(self):
        metrics = _metrics([_case(prohibited_in_issued=True)])
        rendered = check_gate(metrics, None).render()
        assert rendered.startswith("GATE FAILED")
        assert "prohibited" in rendered


class TestSimulator:
    @staticmethod
    def _prompt() -> str:
        return (
            "Jurisdiction: India - RBI Fair Practices Code\n"
            "Maximum principal reasons: 4\n"
            "Required elements: principal_reasons, regulatory_basis\n\n"
            "Factors that counted against this application, strongest first:\n"
            "[\n"
            '  {\n    "factor_id": "EXT_SOURCE_2",\n    "name": "Bureau score",\n'
            '    "applicant_value": 0.21,\n    "rank": 1\n  }\n'
            "]\n\n"
            "Provisions you may cite. Quote from these exactly:\n"
            "[\n"
            '  {\n    "document_id": "rbi-fair-practices-code",\n    "section": "2.3",\n'
            '    "title": "Communication of rejection",\n'
            '    "text": "In case of rejection of a loan application the lender '
            'should convey in writing to the applicant the reasons."\n  }\n'
            "]\n"
        )

    def test_is_deterministic(self):
        """A metric that moves between runs cannot detect a regression."""
        first = SimulatedProvider().complete(system="s", user=self._prompt(), schema=SelectedNotice)
        second = SimulatedProvider().complete(
            system="s", user=self._prompt(), schema=SelectedNotice
        )
        assert first.model_dump() == second.model_dump()

    def test_a_perfect_profile_produces_a_compliant_selection(self):
        provider = SimulatedProvider(PERFECT)
        selection = provider.complete(system="s", user=self._prompt(), schema=SelectedNotice)

        assert [reason.factor_id for reason in selection.principal_reasons] == ["EXT_SOURCE_2"]
        assert selection.citations[0].section == "2.3"
        assert selection.factual_claims == []

    def test_a_perfect_profile_quotes_the_provision_verbatim(self):
        selection = SimulatedProvider(PERFECT).complete(
            system="s", user=self._prompt(), schema=SelectedNotice
        )
        assert selection.citations[0].quoted_span in self._prompt()

    def test_renders_a_body(self):
        body = SimulatedProvider(PERFECT).complete(
            system="s",
            user="Verified reasons to state, all of which must appear:\n1. Your score was low.",
            schema=RenderedBody,
        )
        assert "Your score was low." in body.body

    def test_a_prompt_without_factors_is_refused(self):
        """The simulator reads the prompt as a model would.

        A prompt that omits the factor list is a defect in prompt construction,
        and it should surface as a loud failure rather than as poor metrics.
        """
        with pytest.raises(ProviderError, match="no factor identifiers"):
            SimulatedProvider().complete(
                system="s", user="Nothing useful here.", schema=SelectedNotice
            )

    def test_a_prompt_without_provisions_is_refused(self):
        prompt = 'Factors:\n[{"factor_id": "EXT_SOURCE_2"}]'
        with pytest.raises(ProviderError, match="no citable provisions"):
            SimulatedProvider().complete(system="s", user=prompt, schema=SelectedNotice)

    def test_transport_failures_can_be_simulated(self):
        provider = SimulatedProvider(FailureProfile(transport_failure=1.0))
        with pytest.raises(ProviderError, match="transport failure"):
            provider.complete(system="s", user=self._prompt(), schema=SelectedNotice)

    def test_parses_the_facts_a_model_would_use(self):
        facts = parse_prompt(self._prompt())
        assert facts.factor_ids == ("EXT_SOURCE_2",)
        assert facts.max_reasons == 4
        assert facts.document_id == "rbi-fair-practices-code"
        assert facts.factor_values["EXT_SOURCE_2"] == pytest.approx(0.21)
        assert "principal_reasons" in facts.required_elements
