"""The review queue, against a real database.

The claim under test is the one the project makes to a regulator: that a
decision can be reconstructed from the audit chain alone. The console
reconstructs every case that way, so these tests exercise the claim rather
than leaving it to be discovered untrue during an audit.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine

from aae.audit.models import AuditEventType
from aae.audit.repository import AuditRepository, decision_payload
from aae.audit.session import create_session_factory
from aae.console.review import ReviewQueue, build_case
from aae.domain.models import CreditDecision, Decision, Factor, FactorDirection

pytestmark = pytest.mark.integration

DECISION_ID = "DEC-REVIEW-1"
APPLICATION_ID = "APP-7007"


@pytest.fixture
def queue(migrated_db, owner_connection) -> ReviewQueue:
    owner_connection.execute("TRUNCATE audit_record")
    url = (
        f"postgresql+psycopg://aae_app:app_test_password"
        f"@{migrated_db.get_container_host_ip()}:{migrated_db.get_exposed_port(5432)}/aae"
    )
    factory = create_session_factory(create_engine(url, pool_pre_ping=True))
    return ReviewQueue(AuditRepository(factory))


def _decision() -> CreditDecision:
    return CreditDecision(
        application_id=APPLICATION_ID,
        probability_default=0.78,
        decision=Decision.DECLINE,
        threshold=0.15,
        model_version="xgb-test",
        feature_values={"EXT_SOURCE_2": 0.21},
        factors=(
            Factor(
                factor_id="EXT_SOURCE_2",
                display_name="Credit bureau score (second source)",
                value=0.21,
                shap_value=0.83,
                direction=FactorDirection.ADVERSE,
                rank=1,
            ),
        ),
        scored_at=datetime.now(UTC),
    )


def _generation_payload(*, escalated: bool = False, body: str | None = "Dear applicant...") -> dict:
    return {
        "issued": not escalated and body is not None,
        "escalated": escalated,
        "escalation_reason": "could not be verified" if escalated else None,
        "attempts": 3 if escalated else 1,
        "provider": "scripted",
        "model": "scripted-1",
        "passed_verification": not escalated,
        "violations": ["factor_grounding [X]: invented"] if escalated else [],
        "reasons": [{"factor_id": "EXT_SOURCE_2", "text": "Your bureau score was low."}],
        "citations": [
            {
                "document_id": "rbi-fair-practices-code",
                "section": "2.3",
                "quoted_span": "convey in writing",
            }
        ],
        "body": None if escalated else body,
        "trace": [],
    }


def _seed(
    queue: ReviewQueue,
    *,
    decision_id: str = DECISION_ID,
    escalated: bool = False,
) -> None:
    queue.repository.append_many(
        [
            (
                AuditEventType.DECISION_SCORED,
                APPLICATION_ID,
                decision_id,
                decision_payload(_decision()),
                None,
            ),
            (
                AuditEventType.ESCALATED_TO_HUMAN if escalated else AuditEventType.NOTICE_VERIFIED,
                APPLICATION_ID,
                decision_id,
                _generation_payload(escalated=escalated),
                None,
            ),
        ]
    )


class TestQueue:
    def test_an_empty_chain_has_nothing_to_review(self, queue: ReviewQueue):
        assert queue.pending() == ()

    def test_a_generated_notice_appears(self, queue: ReviewQueue):
        _seed(queue)
        pending = queue.pending()
        assert len(pending) == 1
        assert pending[0].application_id == APPLICATION_ID

    def test_an_escalated_case_appears_and_is_marked(self, queue: ReviewQueue):
        _seed(queue, escalated=True)
        case = queue.pending()[0]
        assert case.escalated
        assert case.needs_attention
        assert case.body is None
        assert "ESCALATED" in case.headline

    def test_a_reviewed_case_leaves_the_queue(self, queue: ReviewQueue):
        _seed(queue)
        queue.approve(queue.pending()[0], reviewer="a.reviewer")
        assert queue.pending() == ()

    def test_cases_come_back_oldest_first(self, queue: ReviewQueue):
        _seed(queue, decision_id="DEC-1")
        _seed(queue, decision_id="DEC-2")
        assert [case.decision_id for case in queue.pending()] == ["DEC-1", "DEC-2"]

    def test_the_limit_is_respected(self, queue: ReviewQueue):
        for index in range(5):
            _seed(queue, decision_id=f"DEC-{index}")
        assert len(queue.pending(limit=2)) == 2


class TestReconstruction:
    def test_a_case_is_rebuilt_from_the_chain_alone(self, queue: ReviewQueue):
        """No side table, no cache: the chain is the only source."""
        _seed(queue)
        case = queue.pending()[0]

        assert case.probability_default == pytest.approx(0.78)
        assert case.threshold == pytest.approx(0.15)
        assert case.model_version == "xgb-test"
        assert case.record_count == 2

    def test_the_factors_behind_the_decision_survive(self, queue: ReviewQueue):
        _seed(queue)
        case = queue.pending()[0]

        assert len(case.factors) == 1
        assert case.factors[0].factor_id == "EXT_SOURCE_2"
        assert case.factors[0].display_name == "Credit bureau score (second source)"
        assert case.factors[0].direction == "adverse"

    def test_the_notice_content_survives(self, queue: ReviewQueue):
        """The letter itself is in the chain, not merely a hash of it."""
        _seed(queue)
        case = queue.pending()[0]

        assert case.body == "Dear applicant..."
        assert case.reasons[0].text == "Your bureau score was low."
        assert case.reasons[0].factor_id == "EXT_SOURCE_2"
        assert case.citations[0][0] == "2.3"

    def test_records_without_a_decision_yield_no_case(self, queue: ReviewQueue):
        assert build_case([], "DEC-none") is None


class TestSignOff:
    def test_approval_is_appended_to_the_chain(self, queue: ReviewQueue):
        _seed(queue)
        before = queue.repository.count()
        queue.approve(queue.pending()[0], reviewer="a.reviewer", note="looks right")

        assert queue.repository.count() == before + 1
        assert queue.repository.verify().intact

    def test_the_reviewer_and_action_are_recorded(self, queue: ReviewQueue):
        _seed(queue)
        queue.approve(queue.pending()[0], reviewer="a.reviewer")

        records = queue.repository.records_for_decision(DECISION_ID)
        payload = records[-1].payload
        assert payload["reviewer"] == "a.reviewer"
        assert payload["action"] == "approved"
        assert payload["edited"] is False

    def test_an_edit_keeps_both_versions(self, queue: ReviewQueue):
        """What the system produced and what was sent are different facts."""
        _seed(queue)
        queue.edit(queue.pending()[0], reviewer="a.reviewer", body="Rewritten letter.")

        records = queue.repository.records_for_decision(DECISION_ID)
        assert records[1].payload["body"] == "Dear applicant..."
        assert records[-1].payload["edited_body"] == "Rewritten letter."
        assert records[-1].payload["edited"] is True

    def test_a_rejection_records_its_reason(self, queue: ReviewQueue):
        _seed(queue, escalated=True)
        queue.reject(queue.pending()[0], reviewer="a.reviewer", note="reason is not supported")

        records = queue.repository.records_for_decision(DECISION_ID)
        assert records[-1].payload["action"] == "rejected"

    def test_sign_off_cannot_overwrite_history(self, queue: ReviewQueue):
        """The review is appended; the original records are untouched."""
        _seed(queue)
        original = queue.repository.records_for_decision(DECISION_ID)
        queue.approve(queue.pending()[0], reviewer="a.reviewer")

        after = queue.repository.records_for_decision(DECISION_ID)
        assert after[: len(original)] == original
        assert queue.repository.verify().intact
