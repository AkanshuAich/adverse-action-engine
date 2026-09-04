"""Fixtures for integration tests.

Spins up a real Postgres with pgvector via Testcontainers, creates the
restricted application role, and runs the migrations as the owner. Nothing is
mocked: the privilege behaviour under test is a property of Postgres, and
mocking it would test nothing at all.
"""

from __future__ import annotations

from collections.abc import Iterator

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from testcontainers.community.postgres import PostgresContainer

APP_ROLE = "aae_app"
APP_PASSWORD = "app_test_password"  # noqa: S105 - throwaway container credential
OWNER_ROLE = "aae_owner"
OWNER_PASSWORD = "owner_test_password"  # noqa: S105 - throwaway container credential


def _dsn(container: PostgresContainer, user: str, password: str) -> str:
    """Build a libpq DSN for a role against the running container.

    Args:
        container: The started Postgres container.
        user: Role to connect as.
        password: That role's password.

    Returns:
        A libpq connection string.
    """
    host = container.get_container_host_ip()
    port = container.get_exposed_port(5432)
    return f"host={host} port={port} dbname=aae user={user} password={password}"


def _sqlalchemy_url(container: PostgresContainer, user: str, password: str) -> str:
    """Build a SQLAlchemy URL for Alembic.

    Args:
        container: The started Postgres container.
        user: Role to connect as.
        password: That role's password.

    Returns:
        A SQLAlchemy connection URL using the psycopg driver.
    """
    host = container.get_container_host_ip()
    port = container.get_exposed_port(5432)
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/aae"


@pytest.fixture(scope="session")
def postgres() -> Iterator[PostgresContainer]:
    """Start Postgres with pgvector for the whole test session."""
    with PostgresContainer(
        "pgvector/pgvector:pg16",
        username=OWNER_ROLE,
        password=OWNER_PASSWORD,
        dbname="aae",
    ) as container:
        yield container


@pytest.fixture(scope="session")
def migrated_db(postgres: PostgresContainer) -> PostgresContainer:
    """Create the application role, then migrate as the owner.

    Order matters: the migration grants privileges only when the role already
    exists, which mirrors how Neon is provisioned in production.
    """
    with psycopg.connect(_dsn(postgres, OWNER_ROLE, OWNER_PASSWORD), autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(f"CREATE ROLE {APP_ROLE} LOGIN PASSWORD '{APP_PASSWORD}'")
        conn.execute(f"GRANT USAGE ON SCHEMA public TO {APP_ROLE}")

    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", _sqlalchemy_url(postgres, OWNER_ROLE, OWNER_PASSWORD))
    command.upgrade(config, "head")

    return postgres


@pytest.fixture
def app_connection(migrated_db: PostgresContainer) -> Iterator[psycopg.Connection[tuple[str, ...]]]:
    """Connect as the restricted application role."""
    with psycopg.connect(_dsn(migrated_db, APP_ROLE, APP_PASSWORD), autocommit=True) as conn:
        yield conn


@pytest.fixture
def owner_connection(
    migrated_db: PostgresContainer,
) -> Iterator[psycopg.Connection[tuple[str, ...]]]:
    """Connect as the schema owner, for setup and teardown only."""
    with psycopg.connect(_dsn(migrated_db, OWNER_ROLE, OWNER_PASSWORD), autocommit=True) as conn:
        yield conn
