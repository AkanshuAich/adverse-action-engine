"""Store the regulation corpus with embeddings.

The unique index on (document_id, section) is the one that matters. Citation
checking resolves that pair and compares the quoted span against the text it
returns, so two rows under one reference would let a fabricated quotation match
whichever copy happened to come back first.

Revision ID: 0002_regulation_corpus
Revises: 0001_audit_log
Create Date: 2026-09-04
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0002_regulation_corpus"
down_revision: str | None = "0001_audit_log"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDING_DIMENSIONS = 384

_GRANTS = """
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'aae_app') THEN
        -- Unlike the audit table, the corpus is legitimately mutable: a
        -- regulation is amended and the stored text must follow it. What must
        -- never change is a decision that cited the old wording, and that is
        -- protected by the audit log rather than by withholding grants here.
        GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE regulation_chunk TO aae_app;
        GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO aae_app;
    END IF;
END
$$;
"""

_REVOKE = """
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'aae_app') THEN
        REVOKE ALL ON TABLE regulation_chunk FROM aae_app;
    END IF;
END
$$;
"""


def upgrade() -> None:
    """Create the corpus table, its indexes, and the grants."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "regulation_chunk",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("jurisdiction", sa.String(length=64), nullable=False),
        sa.Column("document_id", sa.String(length=128), nullable=False),
        sa.Column("section", sa.String(length=32), nullable=False),
        sa.Column("title", sa.String(length=256), nullable=False, server_default=""),
        sa.Column(
            "text",
            sa.Text(),
            nullable=False,
            comment="Verbatim. A paraphrase would make every citation unverifiable.",
        ),
        sa.Column(
            "embedder",
            sa.String(length=128),
            nullable=False,
            comment="Which model produced the vector; vectors from different "
            "models are not comparable.",
        ),
        sa.Column("embedding", Vector(EMBEDDING_DIMENSIONS), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_regulation_chunk"),
    )

    op.create_index(
        "ix_regulation_chunk_reference",
        "regulation_chunk",
        ["document_id", "section"],
        unique=True,
    )
    op.create_index("ix_regulation_chunk_jurisdiction", "regulation_chunk", ["jurisdiction"])

    op.execute(_GRANTS)


def downgrade() -> None:
    """Drop the corpus table."""
    op.execute(_REVOKE)
    op.drop_index("ix_regulation_chunk_jurisdiction", table_name="regulation_chunk")
    op.drop_index("ix_regulation_chunk_reference", table_name="regulation_chunk")
    op.drop_table("regulation_chunk")
