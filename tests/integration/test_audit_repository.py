"""The audit repository against a real Postgres.

The concurrency test is the reason this file exists. Appending to a hash chain
means reading the tail and extending it, so two simultaneous writers will both
read tail N and both try to write N+1. Without serialisation that either
collides on the unique constraint or forks the chain, and a forked chain is not
evidence of anything. Threads are used rather than mocks because the guarantee
is a property of Postgres advisory locks, and mocking the lock would test
nothing.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine

from aae.audit.chain import GENESIS_HASH, verify_chain
from aae.audit.models import AuditEventType
from aae.audit.repository import AuditRepository, decision_payload
from aae.audit.session import create_session_factory
from aae.domain.models import CreditDecision, Decision, Factor, FactorDirection

pytestmark = pytest.mark.integration


@pytest.fixture
def repository(migrated_db, owner_connection) -> AuditRepository:
    owner_connection.execute("TRUNCATE audit_record")
    url = (
        f"postgresql+psycopg://aae_app:app_test_password"
        f"@{migrated_db.get_container_host_ip()}:{migrated_db.get_exposed_port(5432)}/aae"
    )
    engine = create_engine(url, pool_size=10, max_overflow=10, pool_pre_ping=True)
    return AuditRepository(create_session_factory(engine))


def _decision(application_id: str = "APP-1") -> CreditDecision:
    return CreditDecision(
        application_id=application_id,
        probability_default=0.72,
        decision=Decision.DECLINE,
        threshold=0.5,
        model_version="xgb-test",
        feature_values={"EXT_SOURCE_2": 0.21, "AMT_INCOME_TOTAL": 180000.0},
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


class TestAppending:
    def test_first_record_starts_the_chain(self, repository: AuditRepository):
        record = repository.append(
            event_type=AuditEventType.APPLICATION_RECEIVED,
            application_id="APP-1",
            decision_id="DEC-1",
            payload={"step": "received"},
        )
        assert record.sequence == 0
        assert record.prev_hash == GENESIS_HASH

    def test_records_link_to_their_predecessor(self, repository: AuditRepository):
        first = repository.append(
            event_type=AuditEventType.APPLICATION_RECEIVED,
            application_id="APP-1",
            decision_id="DEC-1",
            payload={"n": 1},
        )
        second = repository.append(
            event_type=AuditEventType.DECISION_SCORED,
            application_id="APP-1",
            decision_id="DEC-1",
            payload={"n": 2},
        )
        assert second.prev_hash == first.record_hash
        assert repository.verify().intact

    def test_batch_shares_one_transaction(self, repository: AuditRepository):
        written = repository.append_many(
            [
                (AuditEventType.APPLICATION_RECEIVED, "APP-1", "DEC-1", {"n": i}, None)
                for i in range(5)
            ]
        )
        assert [r.sequence for r in written] == [0, 1, 2, 3, 4]
        assert repository.verify().intact

    def test_empty_batch_writes_nothing(self, repository: AuditRepository):
        assert repository.append_many([]) == ()
        assert repository.count() == 0

    def test_a_scored_decision_round_trips(self, repository: AuditRepository):
        decision = _decision()
        repository.record_decision(decision, "DEC-42")

        records = repository.records_for_decision("DEC-42")
        assert len(records) == 1

        payload = records[0].payload
        assert payload["model_version"] == "xgb-test"
        assert payload["decision"] == "decline"
        assert payload["probability_default"] == pytest.approx(0.72)
        assert payload["factors"][0]["factor_id"] == "EXT_SOURCE_2"

    def test_decision_payload_captures_everything_needed_to_reconstruct(self):
        payload = decision_payload(_decision())
        assert set(payload) >= {
            "application_id",
            "decision",
            "probability_default",
            "threshold",
            "model_version",
            "scored_at",
            "feature_values",
            "factors",
        }


class TestReconstruction:
    def test_records_for_a_decision_come_back_in_order(self, repository: AuditRepository):
        repository.append_many(
            [
                (AuditEventType.APPLICATION_RECEIVED, "APP-1", "DEC-A", {"n": 0}, None),
                (AuditEventType.DECISION_SCORED, "APP-1", "DEC-A", {"n": 1}, None),
                (AuditEventType.DECISION_SCORED, "APP-2", "DEC-B", {"n": 2}, None),
                (AuditEventType.FACTORS_EXPLAINED, "APP-1", "DEC-A", {"n": 3}, None),
            ]
        )
        sequences = [r.sequence for r in repository.records_for_decision("DEC-A")]
        assert sequences == [0, 1, 3]

    def test_unknown_decision_returns_nothing(self, repository: AuditRepository):
        assert repository.records_for_decision("DEC-nope") == ()

    def test_tail_tracks_the_latest_record(self, repository: AuditRepository):
        assert repository.tail() is None
        repository.append(
            event_type=AuditEventType.DECISION_SCORED,
            application_id="APP-1",
            decision_id="DEC-1",
            payload={"n": 1},
        )
        tail = repository.tail()
        assert tail is not None
        assert tail.sequence == 0


class TestConcurrentAppends:
    def test_parallel_writers_produce_one_unbroken_chain(self, repository: AuditRepository):
        """Twenty threads appending at once must not fork or collide.

        This is the property the advisory lock exists for. Without it the
        writers race on reading the tail, and the result is either a unique
        constraint violation or two records claiming the same predecessor.
        """
        writers = 20

        def write(index: int) -> int:
            record = repository.append(
                event_type=AuditEventType.DECISION_SCORED,
                application_id=f"APP-{index}",
                decision_id=f"DEC-{index}",
                payload={"writer": index},
            )
            return record.sequence

        with ThreadPoolExecutor(max_workers=writers) as pool:
            sequences = sorted(pool.map(write, range(writers)))

        # Every writer got a distinct, contiguous slot.
        assert sequences == list(range(writers))

        records = repository.all_records()
        assert len(records) == writers

        result = verify_chain(records)
        assert result.intact, result.reason

    def test_parallel_batches_stay_contiguous(self, repository: AuditRepository):
        """Batches must not interleave: a decision's records belong together."""

        def write_batch(index: int) -> tuple[int, ...]:
            written = repository.append_many(
                [
                    (
                        AuditEventType.APPLICATION_RECEIVED,
                        f"APP-{index}",
                        f"DEC-{index}",
                        {"s": 0},
                        None,
                    ),
                    (
                        AuditEventType.DECISION_SCORED,
                        f"APP-{index}",
                        f"DEC-{index}",
                        {"s": 1},
                        None,
                    ),
                    (
                        AuditEventType.FACTORS_EXPLAINED,
                        f"APP-{index}",
                        f"DEC-{index}",
                        {"s": 2},
                        None,
                    ),
                ]
            )
            return tuple(r.sequence for r in written)

        with ThreadPoolExecutor(max_workers=8) as pool:
            batches = list(pool.map(write_batch, range(8)))

        for sequences in batches:
            assert list(sequences) == list(range(sequences[0], sequences[0] + 3)), (
                "a batch was interleaved with another writer"
            )

        assert repository.verify().intact
        assert repository.count() == 24
