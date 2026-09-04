"""Alembic environment.

Migrations connect as the schema owner, never as the application role, and the
URL comes from ``AAE_MIGRATION_DATABASE_URL`` so no credential is committed.

This module deliberately reads the environment directly rather than going
through :class:`aae.config.Settings`. Migrations are infrastructure: they run in
CI, in deploy jobs, and against a fresh database, none of which have (or should
need) application runtime configuration such as an LLM API key. Coupling schema
management to those settings would make ``alembic upgrade head`` fail for
reasons that have nothing to do with the schema.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from aae.audit.models import Base

MIGRATION_URL_ENV = "AAE_MIGRATION_DATABASE_URL"
DEFAULT_MIGRATION_URL = "postgresql+psycopg://aae_owner:aae_dev_password@localhost:5433/aae"

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# A URL set programmatically (as the integration tests do) wins; otherwise fall
# back to the environment, then to the local development default.
if not config.get_main_option("sqlalchemy.url", default=None):
    config.set_main_option(
        "sqlalchemy.url",
        os.environ.get(MIGRATION_URL_ENV, DEFAULT_MIGRATION_URL),
    )

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting to a database."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Connect and run migrations against the live database."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
