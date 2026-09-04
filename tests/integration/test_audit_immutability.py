"""The audit log is append-only because Postgres says so.

These tests are the evidence behind the claim made to an auditor. They assert
that the application role can add records and read them back, and that the
database itself rejects any attempt to alter or remove one. Application code
never gets the chance to be careless.
"""

from __future__ import annotations

import psycopg
import pytest

from aae.audit.chain import ChainedRecord, link, verify_chain

pytestmark = pytest.mark.integration


def _insert(conn: psycopg.Connection[tuple[str, ...]], record: ChainedRecord, event: str) -> None:
    conn.execute(
        """
        INSERT INTO audit_record
            (sequence, event_type, application_id, decision_id,
             payload, prev_hash, record_hash)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (
            record.sequence,
            event,
            "APP-001",
            "DEC-001",
            psycopg.types.json.Jsonb(record.payload),
            record.prev_hash,
            record.record_hash,
        ),
    )


class TestApplicationRolePrivileges:
    def test_can_insert(self, app_connection):
        record = link({"step": "scored", "probability": 0.71}, None)
        _insert(app_connection, record, "decision_scored")

        row = app_connection.execute(
            "SELECT record_hash FROM audit_record WHERE sequence = %s", (record.sequence,)
        ).fetchone()
        assert row is not None
        assert row[0] == record.record_hash

    def test_cannot_update(self, app_connection):
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            app_connection.execute("UPDATE audit_record SET note = 'tampered'")

    def test_cannot_delete(self, app_connection):
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            app_connection.execute("DELETE FROM audit_record")

    def test_cannot_truncate(self, app_connection):
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            app_connection.execute("TRUNCATE audit_record")


class TestChainSurvivesTheRoundTrip:
    def test_chain_written_and_read_back_verifies(self, app_connection, owner_connection):
        owner_connection.execute("TRUNCATE audit_record")

        records: list[ChainedRecord] = []
        previous: ChainedRecord | None = None
        for step, payload in enumerate(
            [
                {"step": "received"},
                {"step": "scored", "probability": 0.83},
                {"step": "explained", "top_factor": "EXT_SOURCE_2"},
                {"step": "verified", "passed": True},
            ]
        ):
            previous = link(payload, previous)
            records.append(previous)
            _insert(app_connection, previous, f"step_{step}")

        rows = app_connection.execute(
            "SELECT sequence, payload, prev_hash, record_hash FROM audit_record ORDER BY sequence"
        ).fetchall()

        rehydrated = [
            ChainedRecord(sequence=r[0], payload=r[1], prev_hash=r[2], record_hash=r[3])
            for r in rows
        ]
        result = verify_chain(rehydrated)
        assert result.intact
        assert result.checked == len(records)

    def test_owner_tampering_is_detected_by_the_chain(self, app_connection, owner_connection):
        """Even a privileged actor cannot alter history undetectably.

        The owner can bypass the grants, so the cryptographic chain is the
        second, independent line of defence. This is why both mechanisms exist.
        """
        owner_connection.execute("TRUNCATE audit_record")

        previous: ChainedRecord | None = None
        for payload in [{"n": 1}, {"n": 2}, {"n": 3}]:
            previous = link(payload, previous)
            _insert(app_connection, previous, "step")

        owner_connection.execute(
            "UPDATE audit_record SET payload = %s WHERE sequence = 1",
            (psycopg.types.json.Jsonb({"n": 999}),),
        )

        rows = app_connection.execute(
            "SELECT sequence, payload, prev_hash, record_hash FROM audit_record ORDER BY sequence"
        ).fetchall()
        rehydrated = [
            ChainedRecord(sequence=r[0], payload=r[1], prev_hash=r[2], record_hash=r[3])
            for r in rows
        ]

        result = verify_chain(rehydrated)
        assert not result.intact
        assert result.broken_at == 1
