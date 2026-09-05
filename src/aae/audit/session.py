"""Database engine and session management.

The application connects with a role that Postgres denies ``UPDATE`` and
``DELETE`` on the audit table, so the append-only guarantee holds even if this
code is wrong. Nothing here can weaken that; it only decides how connections
are pooled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from aae.database_url import normalise_database_url
from aae.logging import get_logger

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from aae.config import Settings

logger = get_logger(__name__)


def create_database_engine(settings: Settings, *, echo: bool = False) -> Engine:
    """Create the SQLAlchemy engine for the application role.

    Args:
        settings: Supplies the application connection URL. Never the migration
            URL - that role owns the schema and can drop tables.
        echo: Log every statement. Development only.

    Returns:
        A configured engine.
    """
    engine = create_engine(
        normalise_database_url(settings.database_url),
        echo=echo,
        # Recycle before typical managed-Postgres idle timeouts (Neon closes
        # idle connections), and check liveness rather than handing out a
        # dead connection.
        pool_pre_ping=True,
        pool_recycle=280,
        pool_size=5,
        max_overflow=5,
    )
    logger.info("database_engine_created", pool_size=5)
    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Build a session factory bound to an engine.

    Args:
        engine: The engine to bind.

    Returns:
        A session factory. Sessions do not expire attributes on commit, so a
        returned record stays readable after its transaction closes.
    """
    return sessionmaker(bind=engine, expire_on_commit=False, class_=Session)
