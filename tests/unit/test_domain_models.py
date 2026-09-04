"""Tests for the pure domain types.

These types are the contract the verifier checks against, so their invariants
matter: a decision must be able to report which factors were adverse, and a
violation must render into something both a human and a repair prompt can read.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from aae.domain.errors import AAEError, VerificationFailedError
from aae.domain.models import (
    AdverseActionNotice,
    Citation,
    CreditDecision,
    Decision,
    Factor,
    FactorDirection,
    FactualClaim,
    ReasonStatement,
    RenderedNotice,
    VerificationResult,
    Violation,
    ViolationCode,
)


def _factor(
    factor_id: str,
    rank: int,
    direction: FactorDirection = FactorDirection.ADVERSE,
    shap_value: float = 0.4,
) -> Factor:
    return Factor(
        factor_id=factor_id,
        display_name=factor_id.replace("_", " ").title(),
        value=1.0,
        shap_value=shap_value,
        direction=direction,
        rank=rank,
    )


def _decision(*factors: Factor) -> CreditDecision:
    return CreditDecision(
        application_id="APP-1",
        probability_default=0.72,
        decision=Decision.DECLINE,
        threshold=0.5,
        model_version="xgb-1.0.0",
        feature_values={"EXT_SOURCE_2": 0.21, "AMT_INCOME_TOTAL": 180000.0},
        factors=factors,
        scored_at=datetime.now(UTC),
    )


class TestFactor:
    def test_rank_must_be_positive(self):
        with pytest.raises(ValidationError):
            _factor("x", rank=0)

    def test_is_immutable(self):
        factor = _factor("EXT_SOURCE_2", rank=1)
        with pytest.raises(ValidationError):
            factor.rank = 2  # type: ignore[misc]

    def test_rejects_unknown_fields(self):
        with pytest.raises(ValidationError):
            Factor(
                factor_id="a",
                display_name="A",
                value=1.0,
                shap_value=0.1,
                direction=FactorDirection.ADVERSE,
                rank=1,
                unexpected="boom",  # type: ignore[call-arg]
            )


class TestCreditDecision:
    def test_adverse_factors_filters_favourable_ones(self):
        decision = _decision(
            _factor("EXT_SOURCE_2", 1, FactorDirection.ADVERSE),
            _factor("AMT_INCOME_TOTAL", 2, FactorDirection.FAVOURABLE),
            _factor("DAYS_EMPLOYED", 3, FactorDirection.ADVERSE),
        )
        assert [f.factor_id for f in decision.adverse_factors()] == [
            "EXT_SOURCE_2",
            "DAYS_EMPLOYED",
        ]

    def test_adverse_factors_preserves_rank_order(self):
        decision = _decision(_factor("a", 1), _factor("b", 2), _factor("c", 3))
        assert [f.rank for f in decision.adverse_factors()] == [1, 2, 3]

    def test_factor_by_id_finds_a_known_factor(self):
        decision = _decision(_factor("EXT_SOURCE_2", 1))
        found = decision.factor_by_id("EXT_SOURCE_2")
        assert found is not None
        assert found.factor_id == "EXT_SOURCE_2"

    def test_factor_by_id_returns_none_for_a_fabricated_factor(self):
        """This is the lookup the verifier relies on to catch invented reasons."""
        decision = _decision(_factor("EXT_SOURCE_2", 1))
        assert decision.factor_by_id("INVENTED_FACTOR") is None

    def test_probability_must_be_a_probability(self):
        with pytest.raises(ValidationError):
            CreditDecision(
                application_id="APP-1",
                probability_default=1.5,
                decision=Decision.DECLINE,
                threshold=0.5,
                model_version="v1",
                feature_values={},
                factors=(),
                scored_at=datetime.now(UTC),
            )


class TestNotice:
    def test_requires_at_least_one_principal_reason(self):
        with pytest.raises(ValidationError):
            AdverseActionNotice(
                application_id="APP-1",
                jurisdiction="india_rbi",
                principal_reasons=(),
            )

    def test_reason_text_cannot_be_empty(self):
        with pytest.raises(ValidationError):
            ReasonStatement(factor_id="a", text="")

    def test_citation_span_cannot_be_empty(self):
        with pytest.raises(ValidationError):
            Citation(document_id="rbi-fpc", section="2.1", quoted_span="")

    def test_a_complete_notice_round_trips(self):
        notice = AdverseActionNotice(
            application_id="APP-1",
            jurisdiction="india_rbi",
            principal_reasons=(ReasonStatement(factor_id="EXT_SOURCE_2", text="Credit history."),),
            factual_claims=(FactualClaim(field_name="AMT_INCOME_TOTAL", stated_value=180000.0),),
            citations=(
                Citation(document_id="rbi-fpc", section="2.1", quoted_span="reasons for rejection"),
            ),
            declared_elements=frozenset({"reasons", "contact"}),
        )
        rendered = RenderedNotice(notice=notice, body="Dear applicant, ...")
        assert rendered.notice.application_id == "APP-1"
        assert "reasons" in notice.declared_elements


class TestViolationRendering:
    def test_renders_code_and_detail(self):
        violation = Violation(code=ViolationCode.FACTOR_GROUNDING, detail="cites an unknown factor")
        assert violation.render() == "factor_grounding: cites an unknown factor"

    def test_includes_the_locator_when_present(self):
        violation = Violation(
            code=ViolationCode.VALUE_ACCURACY,
            detail="stated 999 but actual is 180000.0",
            locator="AMT_INCOME_TOTAL",
        )
        assert violation.render() == (
            "value_accuracy [AMT_INCOME_TOTAL]: stated 999 but actual is 180000.0"
        )

    def test_detail_cannot_be_empty(self):
        with pytest.raises(ValidationError):
            Violation(code=ViolationCode.REASON_COUNT, detail="")


class TestVerificationResult:
    def test_a_pass_has_no_violations(self):
        result = VerificationResult(passed=True)
        assert result.violations == ()
        assert result.rendered_violations() == []

    def test_renders_every_violation_for_the_repair_prompt(self):
        result = VerificationResult(
            passed=False,
            violations=(
                Violation(code=ViolationCode.FACTOR_GROUNDING, detail="unknown factor"),
                Violation(code=ViolationCode.REASON_COUNT, detail="5 reasons, limit is 4"),
            ),
            attempt=2,
        )
        rendered = result.rendered_violations()
        assert len(rendered) == 2
        assert rendered[0].startswith("factor_grounding")
        assert result.attempt == 2

    def test_attempt_must_be_at_least_one(self):
        with pytest.raises(ValidationError):
            VerificationResult(passed=True, attempt=0)


class TestErrors:
    def test_every_error_derives_from_the_package_base(self):
        error = VerificationFailedError("gave up", ["factor_grounding: unknown"])
        assert isinstance(error, AAEError)

    def test_verification_failure_carries_its_violations(self):
        """The escalation path must record why, not just that it failed."""
        violations = ["factor_grounding: unknown factor", "reason_count: too many"]
        error = VerificationFailedError("exhausted repair attempts", violations)
        assert error.violations == violations
        assert "exhausted" in str(error)
