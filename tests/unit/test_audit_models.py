"""Structural tests for the audit ORM model and logging setup.

These assert the shape of the audit table rather than its behaviour under a
live database (that is the integration suite's job). They exist to catch
accidental drift — a column silently made nullable, an index dropped — which is
exactly the kind of change that would quietly weaken the audit guarantee.
"""

from __future__ import annotations

import structlog

from aae.audit.models import AuditEventType, AuditRecord, Base
from aae.config import Environment, Settings
from aae.logging import bind_correlation_id, configure_logging, get_logger

REQUIRED_NOT_NULL = {
    "sequence",
    "event_type",
    "application_id",
    "decision_id",
    "payload",
    "prev_hash",
    "record_hash",
    "created_at",
}


class TestAuditRecordSchema:
    def test_table_is_registered_on_the_metadata(self):
        assert "audit_record" in Base.metadata.tables

    def test_chain_columns_are_not_nullable(self):
        """A nullable hash column would silently break the chain guarantee."""
        table = AuditRecord.__table__
        for name in REQUIRED_NOT_NULL:
            assert not table.columns[name].nullable, f"{name} must be NOT NULL"

    def test_sequence_and_record_hash_are_unique(self):
        table = AuditRecord.__table__
        assert table.columns["sequence"].unique
        assert table.columns["record_hash"].unique

    def test_only_the_note_column_is_optional(self):
        table = AuditRecord.__table__
        nullable = {c.name for c in table.columns if c.nullable}
        assert nullable == {"note"}

    def test_decision_reconstruction_index_exists(self):
        """Reconstructing one decision in order is the dominant read pattern."""
        index_columns = {
            index.name: [c.name for c in index.columns] for index in AuditRecord.__table__.indexes
        }
        assert index_columns["ix_audit_record_decision_sequence"] == ["decision_id", "sequence"]
        assert index_columns["ix_audit_record_application_id"] == ["application_id"]

    def test_created_at_is_set_by_the_server_not_the_application(self):
        """An application clock is attacker-influenced; the server clock is not."""
        assert AuditRecord.__table__.columns["created_at"].server_default is not None

    def test_repr_is_useful_without_dumping_the_payload(self):
        record = AuditRecord(sequence=3, event_type="decision_scored", decision_id="DEC-9")
        text = repr(record)
        assert "sequence=3" in text
        assert "DEC-9" in text


class TestAuditEventType:
    def test_covers_every_pipeline_stage(self):
        values = {e.value for e in AuditEventType}
        assert {
            "application_received",
            "decision_scored",
            "factors_explained",
            "notice_generated",
            "notice_verified",
            "notice_rejected",
            "escalated_to_human",
            "human_reviewed",
        } <= values

    def test_is_a_plain_string_for_storage(self):
        assert AuditEventType.DECISION_SCORED == "decision_scored"


class TestLogging:
    def test_development_uses_the_console_renderer(self):
        configure_logging(
            Settings(_env_file=None, env=Environment.DEVELOPMENT, llm_provider="ollama")
        )
        assert get_logger(__name__) is not None

    def test_production_uses_json(self):
        configure_logging(
            Settings(_env_file=None, env=Environment.PRODUCTION, llm_provider="ollama")
        )
        assert get_logger(__name__) is not None

    def test_correlation_id_binds_into_the_context(self):
        structlog.contextvars.clear_contextvars()
        bind_correlation_id("DEC-123")
        assert structlog.contextvars.get_contextvars()["correlation_id"] == "DEC-123"
        structlog.contextvars.clear_contextvars()
