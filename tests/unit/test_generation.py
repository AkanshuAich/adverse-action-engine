"""The generation workflow: payload, graph, repair loop, and prose checking.

The repair tests are the point. A verifier that rejects bad notices is only
half a system; what makes it useful is that the rejection is fed back in terms
specific enough for the next attempt to fix, and that running out of attempts
is a recorded outcome rather than a silent fallback.

No test here touches a network. The provider is scripted, which is the only
way to make the first attempt fail in a chosen way and the second succeed.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Final

import pytest

from aae.domain.errors import GenerationError, ProviderError
from aae.domain.models import (
    CreditDecision,
    Decision,
    Factor,
    FactorDirection,
    ViolationCode,
)
from aae.generation.graph import NoticeGenerator
from aae.generation.payload import (
    ALLOWED_FEATURE_FIELDS,
    build_payload,
    round_for_presentation,
    sanitise_text_value,
)
from aae.generation.providers.stub import ScriptedProvider
from aae.generation.schemas import (
    RenderedBody,
    SelectedCitation,
    SelectedClaim,
    SelectedNotice,
    SelectedReason,
)
from aae.jurisdiction.india_rbi import INDIA_RBI
from aae.ml.features import PROTECTED_ATTRIBUTES
from aae.retrieval.corpus import (
    INDIA_RBI_PROVISIONS,
    RBI_FAIR_PRACTICES_CODE,
    india_rbi_corpus,
)
from aae.verification.prose import check_prose
from aae.verification.rules import VerificationPolicy
from aae.verification.verifier import NoticeVerifier

APPLICATION_ID: Final[str] = "APP-2002"
REAL_QUOTE: Final[str] = "convey in writing to the applicant the reasons"

ELEMENTS: Final[list[str]] = [
    "principal_reasons",
    "regulatory_basis",
    "decision_statement",
    "grievance_contact",
]

CLEAN_BODY: Final[str] = (
    "Dear applicant, we are unable to approve your application at this time. "
    "Your repayment burden relative to income was higher than we can accept, "
    "and your credit bureau score was below our current requirement. "
    "If you would like clarification, please contact our grievance officer."
)


def _factor(
    factor_id: str, rank: int, direction: FactorDirection, value: float | str | None
) -> Factor:
    return Factor(
        factor_id=factor_id,
        display_name=factor_id.replace("_", " ").title(),
        value=value,
        shap_value=0.5 if direction is FactorDirection.ADVERSE else -0.3,
        direction=direction,
        rank=rank,
    )


def make_decision(*, outcome: Decision = Decision.DECLINE) -> CreditDecision:
    return CreditDecision(
        application_id=APPLICATION_ID,
        probability_default=0.81,
        decision=outcome,
        threshold=0.15,
        model_version="xgb-test",
        feature_values={
            "ANNUITY_INCOME_RATIO": 0.3064,
            "EXT_SOURCE_2": 0.2189,
            "AMT_INCOME_TOTAL": 149_900.0,
            "OCCUPATION_TYPE": "Laborers",
        },
        factors=(
            _factor("ANNUITY_INCOME_RATIO", 1, FactorDirection.ADVERSE, 0.3064),
            _factor("EXT_SOURCE_2", 2, FactorDirection.ADVERSE, 0.2189),
            _factor("AMT_INCOME_TOTAL", 3, FactorDirection.FAVOURABLE, 149_900.0),
        ),
        scored_at=datetime.now(UTC),
    )


def make_payload(decision: CreditDecision | None = None):
    return build_payload(decision or make_decision(), INDIA_RBI, INDIA_RBI_PROVISIONS)


def good_selection() -> SelectedNotice:
    return SelectedNotice(
        principal_reasons=[
            SelectedReason(
                factor_id="ANNUITY_INCOME_RATIO",
                text="Your repayment burden relative to income was high.",
            ),
            SelectedReason(
                factor_id="EXT_SOURCE_2",
                text="Your credit bureau score was below our requirement.",
            ),
        ],
        factual_claims=[],
        citations=[
            SelectedCitation(
                document_id=RBI_FAIR_PRACTICES_CODE, section="2.3", quoted_span=REAL_QUOTE
            )
        ],
        included_elements=list(ELEMENTS),
    )


def bad_selection(factor_id: str = "SOCIAL_MEDIA_ACTIVITY") -> SelectedNotice:
    return SelectedNotice(
        principal_reasons=[
            SelectedReason(factor_id=factor_id, text="Your online activity concerned us.")
        ],
        citations=[
            SelectedCitation(
                document_id=RBI_FAIR_PRACTICES_CODE, section="2.3", quoted_span=REAL_QUOTE
            )
        ],
        included_elements=list(ELEMENTS),
    )


def build_generator(
    responses, *, max_attempts: int = 3
) -> tuple[NoticeGenerator, ScriptedProvider]:
    provider = ScriptedProvider(responses)
    generator = NoticeGenerator(
        provider=provider,
        verifier=NoticeVerifier(INDIA_RBI, india_rbi_corpus()),
        max_attempts=max_attempts,
    )
    return generator, provider


class TestPayload:
    def test_carries_no_applicant_identifier(self):
        """A model that never sees an id cannot misattribute a notice."""
        payload = make_payload()
        rendered = repr(payload)
        assert APPLICATION_ID not in rendered

    def test_includes_only_adverse_factors(self):
        """Favourable factors are not offered as temptation."""
        payload = make_payload()
        assert payload.factor_ids() == {"ANNUITY_INCOME_RATIO", "EXT_SOURCE_2"}
        assert "AMT_INCOME_TOTAL" not in payload.factor_ids()

    def test_no_protected_attribute_is_allowlisted(self):
        assert not ALLOWED_FEATURE_FIELDS & PROTECTED_ATTRIBUTES

    def test_provisions_are_carried_verbatim(self):
        """A paraphrase would make every citation unverifiable."""
        payload = make_payload()
        section = next(p for p in payload.provisions if p.section == "2.3")
        original = next(p for p in INDIA_RBI_PROVISIONS if p.section == "2.3")
        assert section.text == original.text

    def test_rejects_a_decision_with_nothing_adverse(self):
        decision = make_decision().model_copy(
            update={
                "factors": (_factor("AMT_INCOME_TOTAL", 1, FactorDirection.FAVOURABLE, 149_900.0),)
            }
        )
        with pytest.raises(GenerationError, match="no adverse factors"):
            build_payload(decision, INDIA_RBI, INDIA_RBI_PROVISIONS)

    def test_sanitises_control_characters(self):
        assert sanitise_text_value("Lab\x00or\x1fers") == "Laborers"

    def test_caps_a_long_value(self):
        assert len(sanitise_text_value("x" * 500)) < 200


class TestHappyPath:
    def test_a_valid_selection_is_issued(self):
        generator, provider = build_generator([good_selection(), RenderedBody(body=CLEAN_BODY)])
        outcome = generator.generate(make_decision(), make_payload())

        assert outcome.issued
        assert not outcome.escalated
        assert outcome.attempts == 1
        assert outcome.body == CLEAN_BODY
        assert provider.call_count == 2

    def test_the_notice_carries_the_identity_the_model_never_saw(self):
        generator, _ = build_generator([good_selection(), RenderedBody(body=CLEAN_BODY)])
        outcome = generator.generate(make_decision(), make_payload())

        assert outcome.notice is not None
        assert outcome.notice.application_id == APPLICATION_ID
        assert outcome.notice.jurisdiction == INDIA_RBI.code

    def test_the_renderer_is_shown_only_verified_sentences(self):
        """It cannot misstate a record it was never given."""
        generator, provider = build_generator([good_selection(), RenderedBody(body=CLEAN_BODY)])
        generator.generate(make_decision(), make_payload())

        _, render_user = provider.calls[1]
        assert "ANNUITY_INCOME_RATIO" not in render_user
        assert "149900" not in render_user.replace(",", "")
        assert "repayment burden" in render_user

    def test_the_trace_records_prompt_and_response_hashes(self):
        generator, _ = build_generator([good_selection(), RenderedBody(body=CLEAN_BODY)])
        outcome = generator.generate(make_decision(), make_payload())

        select = next(step for step in outcome.trace if step.node == "select")
        assert select.prompt_hash and len(select.prompt_hash) == 64
        assert select.response_hash and len(select.response_hash) == 64

    def test_the_audit_payload_is_json_ready(self):
        generator, _ = build_generator([good_selection(), RenderedBody(body=CLEAN_BODY)])
        payload = generator.generate(make_decision(), make_payload()).audit_payload()

        assert payload["issued"] is True
        assert payload["attempts"] == 1
        assert payload["provider"] == "scripted"
        assert [step["node"] for step in payload["trace"]] == [
            "select",
            "verify",
            "render",
            "check_prose",
        ]


class TestRepairLoop:
    def test_a_rejected_selection_is_regenerated(self):
        generator, provider = build_generator(
            [bad_selection(), good_selection(), RenderedBody(body=CLEAN_BODY)]
        )
        outcome = generator.generate(make_decision(), make_payload())

        assert outcome.issued
        assert outcome.attempts == 2
        assert provider.call_count == 3

    def test_the_repair_prompt_carries_the_verifier_violations(self):
        """Paraphrasing them would lose the locator saying which reason failed."""
        generator, provider = build_generator(
            [bad_selection(), good_selection(), RenderedBody(body=CLEAN_BODY)]
        )
        generator.generate(make_decision(), make_payload())

        _, repair_user = provider.calls[1]
        assert "REJECTED" in repair_user
        assert "SOCIAL_MEDIA_ACTIVITY" in repair_user
        assert "factor_grounding" in repair_user

    def test_the_trace_shows_every_attempt(self):
        generator, _ = build_generator(
            [bad_selection(), good_selection(), RenderedBody(body=CLEAN_BODY)]
        )
        outcome = generator.generate(make_decision(), make_payload())

        # Each step carries the attempt it belonged to, so a reviewer can see
        # that selection ran twice and why the first pass did not survive.
        assert [(step.node, step.attempt) for step in outcome.trace] == [
            ("select", 1),
            ("verify", 1),
            ("select", 2),
            ("verify", 2),
            ("render", 2),
            ("check_prose", 2),
        ]

    def test_the_trace_records_the_violations_that_forced_the_repair(self):
        generator, _ = build_generator(
            [bad_selection(), good_selection(), RenderedBody(body=CLEAN_BODY)]
        )
        outcome = generator.generate(make_decision(), make_payload())

        first_verify = next(step for step in outcome.trace if step.node == "verify")
        assert first_verify.violations
        assert "SOCIAL_MEDIA_ACTIVITY" in first_verify.violations[0]

    def test_repair_is_bounded(self):
        generator, provider = build_generator([bad_selection()], max_attempts=3)
        outcome = generator.generate(make_decision(), make_payload())

        assert outcome.escalated
        assert outcome.attempts == 3
        assert provider.call_count == 3

    def test_escalation_records_why(self):
        generator, _ = build_generator([bad_selection()])
        outcome = generator.generate(make_decision(), make_payload())

        assert outcome.escalation_reason is not None
        assert "SOCIAL_MEDIA_ACTIVITY" in outcome.escalation_reason
        assert not outcome.issued

    def test_an_escalated_notice_has_no_body(self):
        """Nothing may be sent when the content could not be verified."""
        generator, _ = build_generator([bad_selection()])
        outcome = generator.generate(make_decision(), make_payload())
        assert outcome.body is None

    def test_a_single_attempt_configuration_escalates_immediately(self):
        generator, provider = build_generator([bad_selection()], max_attempts=1)
        outcome = generator.generate(make_decision(), make_payload())

        assert outcome.escalated
        assert outcome.attempts == 1
        assert provider.call_count == 1


class TestProviderFailure:
    def test_a_provider_error_is_not_an_escalation(self):
        """An unreachable backend must not inflate the escalation rate.

        The escalation metric is how anyone notices the model getting worse.
        Folding transport failures into it would make a network outage look
        like a quality problem.
        """
        generator, _ = build_generator([ProviderError("connection reset")])
        with pytest.raises(GenerationError, match="failed at the provider"):
            generator.generate(make_decision(), make_payload())

    def test_a_malformed_response_fails_loudly(self):
        generator, _ = build_generator([RenderedBody(body="wrong schema for this call")])
        with pytest.raises(GenerationError):
            generator.generate(make_decision(), make_payload())


class TestProseChecking:
    def test_a_clean_letter_passes(self):
        notice = good_selection().to_domain(APPLICATION_ID, INDIA_RBI.code)
        assert check_prose(CLEAN_BODY, notice, make_payload(), INDIA_RBI) == ()

    def test_an_invented_figure_is_caught(self):
        """The dangerous prose failure: a number nobody supplied."""
        notice = good_selection().to_domain(APPLICATION_ID, INDIA_RBI.code)
        body = f"{CLEAN_BODY} Your recorded income of 250,000 was insufficient."
        violations = check_prose(body, notice, make_payload(), INDIA_RBI)

        assert violations
        assert violations[0].code is ViolationCode.VALUE_ACCURACY
        assert "250,000" in violations[0].detail

    def test_a_rounded_figure_is_accepted(self):
        notice = good_selection().to_domain(APPLICATION_ID, INDIA_RBI.code)
        payload = make_payload()
        body = f"{CLEAN_BODY} A ratio of 0.31 exceeds our limit."
        assert check_prose(body, notice, payload, INDIA_RBI) == ()

    def test_a_section_reference_is_accepted(self):
        notice = good_selection().to_domain(APPLICATION_ID, INDIA_RBI.code)
        body = f"{CLEAN_BODY} This notice is issued under section 2.3."
        assert check_prose(body, notice, make_payload(), INDIA_RBI) == ()

    def test_a_protected_characteristic_in_the_letter_is_caught(self):
        notice = good_selection().to_domain(APPLICATION_ID, INDIA_RBI.code)
        body = f"{CLEAN_BODY} As a married applicant this was weighed carefully."
        violations = check_prose(body, notice, make_payload(), INDIA_RBI)

        assert any(v.code is ViolationCode.PROHIBITED_CONTENT for v in violations)

    def test_a_letter_that_fails_prose_checks_escalates(self):
        """The structure was verified; the letter added something anyway."""
        dirty = f"{CLEAN_BODY} Your income of 999,999 was the issue."
        generator, _ = build_generator([good_selection(), RenderedBody(body=dirty)])
        outcome = generator.generate(make_decision(), make_payload())

        assert outcome.escalated
        assert outcome.body is None
        assert outcome.escalation_reason is not None
        assert "does not support" in outcome.escalation_reason


class TestScriptedProvider:
    def test_rejects_an_empty_script(self):
        with pytest.raises(ValueError, match="at least one response"):
            ScriptedProvider([])

    def test_repeats_the_last_response(self):
        provider = ScriptedProvider([good_selection()])
        for _ in range(3):
            assert provider.complete(system="s", user="u", schema=SelectedNotice)
        assert provider.call_count == 3

    def test_validates_against_the_requested_schema(self):
        provider = ScriptedProvider([RenderedBody(body="text")])
        with pytest.raises(ProviderError, match="did not match"):
            provider.complete(system="s", user="u", schema=SelectedNotice)


class TestPreconditionsAreNotRepaired:
    def test_an_approved_application_raises_rather_than_looping(self):
        """No amount of rewriting makes an approval into an adverse action."""
        generator, _ = build_generator([good_selection(), RenderedBody(body=CLEAN_BODY)])
        approved = make_decision(outcome=Decision.APPROVE)
        with pytest.raises(Exception, match="no adverse action"):
            generator.generate(approved, make_payload())


class TestClaimsFlowThrough:
    def test_an_accurate_claim_is_accepted(self):
        selection = good_selection().model_copy(
            update={
                "factual_claims": [SelectedClaim(field_name="EXT_SOURCE_2", stated_value=0.2189)]
            }
        )
        generator, _ = build_generator([selection, RenderedBody(body=CLEAN_BODY)])
        outcome = generator.generate(make_decision(), make_payload())
        assert outcome.issued

    def test_a_misstated_claim_is_repaired(self):
        wrong = good_selection().model_copy(
            update={"factual_claims": [SelectedClaim(field_name="EXT_SOURCE_2", stated_value=0.95)]}
        )
        generator, provider = build_generator(
            [wrong, good_selection(), RenderedBody(body=CLEAN_BODY)]
        )
        outcome = generator.generate(make_decision(), make_payload())

        assert outcome.issued
        assert outcome.attempts == 2
        _, repair = provider.calls[1]
        assert "value_accuracy" in repair


class TestProseNumericEdges:
    def test_small_integers_are_permitted(self):
        """List numbering and counts would otherwise fail every letter.

        This is the documented blind spot: a fabricated small integer passes.
        Acceptable only because this payload holds no small-integer facts
        about the applicant.
        """
        notice = good_selection().to_domain(APPLICATION_ID, INDIA_RBI.code)
        body = f"{CLEAN_BODY} There are 2 reasons, listed as 1 and 2 above."
        assert check_prose(body, notice, make_payload(), INDIA_RBI) == ()

    def test_a_ratio_stated_as_a_percentage_is_accepted(self):
        """0.3064 expressed as 30.64% is the same fact, presented for a reader."""
        notice = good_selection().to_domain(APPLICATION_ID, INDIA_RBI.code)
        body = f"{CLEAN_BODY} Your repayments come to 30.64% of income."
        assert check_prose(body, notice, make_payload(), INDIA_RBI) == ()

    def test_a_claimed_value_becomes_quotable(self):
        """A figure verified in the structured stage may appear in the prose."""
        selection = good_selection().model_copy(
            update={
                "factual_claims": [
                    SelectedClaim(field_name="AMT_INCOME_TOTAL", stated_value=149_900.0)
                ]
            }
        )
        notice = selection.to_domain(APPLICATION_ID, INDIA_RBI.code)
        body = f"{CLEAN_BODY} Your recorded income was 149,900."
        assert check_prose(body, notice, make_payload(), INDIA_RBI) == ()


class TestPresentationRounding:
    """A figure the model never sees at full precision cannot be copied at it.

    The prompt used to say "round for readability if you wish", which left the
    decision to the model. A live notice then quoted a bureau score as
    0.0935070975944776 - accurate, verifiable, and not something anyone would
    send a customer. Rounding now happens before the payload is built.
    """

    @pytest.mark.parametrize(
        ("exact", "expected"),
        [
            (0.0935070975944776, 0.09351),
            (0.169165769080108, 0.1692),
            (1.8347953216374269, 1.835),
            (0.0749003984063745, 0.0749),
            (136_800.0, 136_800.0),
            (0.0, 0.0),
            (-0.0935070975944776, -0.09351),
        ],
    )
    def test_keeps_four_significant_figures(self, exact: float, expected: float):
        assert round_for_presentation(exact) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "exact",
        [0.0935070975944776, 1.8347953216374269, 136_800.0, 0.0001234567, 9.999999],
    )
    def test_stays_within_the_verifier_tolerance(self, exact: float):
        """Rounding must stay invisible to value accuracy.

        Otherwise it trades one defect for a worse one: a letter that reads
        well and fails verification.
        """
        rounded = round_for_presentation(exact)

        assert abs(rounded - exact) <= abs(exact) * VerificationPolicy().value_relative_tolerance

    def test_leaves_non_finite_values_alone(self):
        """No magnitude to round to, and log10 would raise."""
        assert math.isnan(round_for_presentation(math.nan))
        assert math.isinf(round_for_presentation(math.inf))

    def test_the_payload_carries_the_rounded_figure(self):
        decision = make_decision().model_copy(
            update={
                "factors": (
                    _factor("EXT_SOURCE_2", 1, FactorDirection.ADVERSE, 0.0935070975944776),
                )
            }
        )
        payload = make_payload(decision)

        assert payload.factors[0].value == pytest.approx(0.09351)

    def test_a_categorical_value_is_untouched(self):
        """Rounding applies to figures; a job title has no significant digits."""
        decision = make_decision().model_copy(
            update={
                "factors": (_factor("OCCUPATION_TYPE", 1, FactorDirection.ADVERSE, "Laborers"),)
            }
        )
        payload = make_payload(decision)

        assert payload.factors[0].value == "Laborers"
