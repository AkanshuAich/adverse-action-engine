"""Checking a deployment before trusting it.

Run against a freshly migrated database - Neon, or anything else - to confirm
the guarantees actually hold there rather than only in the test suite. Schema
migrations succeeding proves the tables exist; it proves nothing about whether
the privilege split survived, and the privilege split is what makes the audit
log evidence rather than a table.

The probe record is written through :func:`aae.audit.chain.link`, so it is
correctly chained and can simply stay. A hand-written row with a fabricated
hash would break verification for every reader afterwards, and could only be
removed by the owner role - which rather defeats the point of testing that the
application role cannot remove anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final

import psycopg
from sqlalchemy import text

from aae.audit.models import AuditEventType
from aae.audit.repository import AuditRepository
from aae.logging import get_logger

if TYPE_CHECKING:
    from sqlalchemy.orm import Session, sessionmaker

logger = get_logger(__name__)

PROBE_APPLICATION: Final[str] = "HEALTHCHECK"

MUTATIONS: Final[tuple[tuple[str, str], ...]] = (
    ("UPDATE", "UPDATE audit_record SET note = 'tampered'"),
    ("DELETE", "DELETE FROM audit_record"),
    ("TRUNCATE", "TRUNCATE audit_record"),
)


@dataclass(frozen=True)
class HealthReport:
    """What a deployment check found."""

    role: str
    database: str
    tables: tuple[str, ...]
    pgvector: bool
    can_insert: bool
    refused_mutations: tuple[str, ...]
    allowed_mutations: tuple[str, ...]
    chain_intact: bool
    records: int

    @property
    def healthy(self) -> bool:
        """Whether this deployment can be trusted to hold its guarantees."""
        return (
            self.pgvector
            and self.can_insert
            and not self.allowed_mutations
            and self.chain_intact
            and {"audit_record", "regulation_chunk"} <= set(self.tables)
        )

    def render(self) -> str:
        """Render the findings for a terminal.

        Returns:
            A printable report, ending in a verdict.
        """
        lines = [
            f"  role            {self.role}",
            f"  database        {self.database}",
            f"  tables          {', '.join(self.tables)}",
            f"  pgvector        {'installed' if self.pgvector else 'MISSING'}",
            f"  INSERT          {'allowed (correct)' if self.can_insert else 'REFUSED'}",
        ]
        lines.extend(
            f"  {name:<15} refused by Postgres (correct)" for name in self.refused_mutations
        )
        lines.extend(
            f"  {name:<15} *** ALLOWED - GUARANTEE BROKEN ***" for name in self.allowed_mutations
        )
        state = "intact" if self.chain_intact else "BROKEN"
        lines.append(f"  chain           {state} ({self.records} records)")
        lines.append("")
        lines.append("HEALTHY" if self.healthy else "NOT HEALTHY - do not route traffic here")
        return "\n".join(lines)


def check_deployment(session_factory: sessionmaker[Session]) -> HealthReport:
    """Verify a deployment holds the guarantees the design depends on.

    Args:
        session_factory: Sessions bound to the **application** role. Running
            this as the owner would report success while proving nothing: the
            owner is meant to be able to modify the table.

    Returns:
        What was found.
    """
    repository = AuditRepository(session_factory)

    with session_factory() as session:
        role, database = session.execute(text("SELECT current_user, current_database()")).one()
        tables = tuple(
            row[0]
            for row in session.execute(
                text(
                    "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
                )
            ).all()
        )
        pgvector = (
            session.execute(
                text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            ).scalar_one_or_none()
            is not None
        )

    # Chained properly, so the probe is a valid record and stays.
    can_insert = True
    try:
        repository.append(
            event_type=AuditEventType.APPLICATION_RECEIVED,
            application_id=PROBE_APPLICATION,
            decision_id=f"HEALTH-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}",
            payload={"check": "deployment", "role": str(role)},
            note="Deployment health check.",
        )
    except Exception:
        can_insert = False

    refused: list[str] = []
    allowed: list[str] = []
    for name, statement in MUTATIONS:
        with session_factory() as session:
            try:
                session.execute(text(statement))
                session.rollback()
                allowed.append(name)
            except Exception as exc:
                session.rollback()
                if isinstance(exc.__cause__, psycopg.errors.InsufficientPrivilege):
                    refused.append(name)
                else:
                    allowed.append(name)

    verification = repository.verify()

    return HealthReport(
        role=str(role),
        database=str(database),
        tables=tables,
        pgvector=pgvector,
        can_insert=can_insert,
        refused_mutations=tuple(refused),
        allowed_mutations=tuple(allowed),
        chain_intact=verification.intact,
        records=verification.checked,
    )


def _main() -> int:
    """Check the configured deployment.

    Returns:
        Zero if healthy, one otherwise.
    """
    from aae.audit.session import create_database_engine, create_session_factory
    from aae.config import get_settings

    settings = get_settings()
    report = check_deployment(create_session_factory(create_database_engine(settings)))

    print("\nDeployment check\n")  # noqa: T201
    print(report.render())  # noqa: T201
    print()  # noqa: T201
    return 0 if report.healthy else 1


if __name__ == "__main__":
    raise SystemExit(_main())
