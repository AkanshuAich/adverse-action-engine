"""Create the append-only, hash-chained audit log.

The privilege grants at the end are the point of this migration. The audit
table is append-only because Postgres refuses anything else to the application
role, not because application code is careful. That distinction is what makes
the guarantee worth stating to an auditor.

Revision ID: 0001_audit_log
Revises:
Create Date: 2026-09-04
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_audit_log"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# These are static SQL literals, not interpolated queries: role names cannot be
# bound as query parameters in GRANT/REVOKE, so the role is written inline
# rather than formatted in. Both statements are guarded on the role existing,
# so the same migration runs unchanged in CI (where there is no such role) and
# on Neon (where the role is created out of band).
_GRANTS = """
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'aae_app') THEN
        -- Start from nothing, then add back exactly what the application needs.
        REVOKE ALL ON TABLE audit_record FROM aae_app;
        GRANT SELECT, INSERT ON TABLE audit_record TO aae_app;
        GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO aae_app;
    END IF;
END
$$;
"""

_REVOKE = """
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'aae_app') THEN
        REVOKE ALL ON TABLE audit_record FROM aae_app;
    END IF;
END
$$;
"""


def upgrade() -> None:
    """Create the audit table, its indexes, and the append-only grants."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "audit_record",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column(
            "sequence",
            sa.BigInteger(),
            nullable=False,
            comment="Position in the hash chain, contiguous from zero.",
        ),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("application_id", sa.String(length=64), nullable=False),
        sa.Column(
            "decision_id",
            sa.String(length=64),
            nullable=False,
            comment="Correlation id shared by every record for one decision.",
        ),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            comment=(
                "Model version, feature values, SHAP output, prompt and response "
                "hashes, verifier result, human sign-off."
            ),
        ),
        sa.Column("prev_hash", sa.String(length=64), nullable=False),
        sa.Column("record_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("note", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_audit_record"),
        sa.UniqueConstraint("sequence", name="uq_audit_record_sequence"),
        sa.UniqueConstraint("record_hash", name="uq_audit_record_record_hash"),
    )

    # Reconstructing one decision in order is the dominant read pattern:
    # "show me every step that led to this outcome".
    op.create_index(
        "ix_audit_record_decision_sequence",
        "audit_record",
        ["decision_id", "sequence"],
    )
    # Supports "every decision ever made about this application".
    op.create_index("ix_audit_record_application_id", "audit_record", ["application_id"])

    op.execute(_GRANTS)


def downgrade() -> None:
    """Drop the audit table and its grants."""
    op.execute(_REVOKE)
    op.drop_index("ix_audit_record_application_id", table_name="audit_record")
    op.drop_index("ix_audit_record_decision_sequence", table_name="audit_record")
    op.drop_table("audit_record")
