"""Normalising Postgres connection strings.

Every managed Postgres - Neon, Supabase, Render - hands out a URL beginning
``postgresql://``. SQLAlchemy reads that bare scheme as "use psycopg2", which
this project does not ship, so the failure is a forty-line traceback ending in
``ModuleNotFoundError: No module named 'psycopg2'``. That names a library the
project never mentions and says nothing about the one-word difference that
actually caused it.

Since psycopg 3 is the only driver installed, a bare ``postgresql://`` URL is
always a mistake and always has the same fix. Normalising it is better than
documenting it: a connection string pasted straight from a provider's console
works, which is the only way it will ever be pasted.

An explicit ``postgresql+psycopg2://`` is a different matter and is rejected
rather than rewritten. That URL states a deliberate choice of driver, and
silently substituting another would be the kind of helpfulness that hides a
real disagreement about what is installed.
"""

from __future__ import annotations

from typing import Final

PSYCOPG_SCHEME: Final[str] = "postgresql+psycopg://"
BARE_SCHEMES: Final[tuple[str, ...]] = ("postgresql://", "postgres://")
REJECTED_SCHEME: Final[str] = "postgresql+psycopg2://"


def normalise_database_url(url: str) -> str:
    """Ensure a Postgres URL names the psycopg 3 driver.

    Args:
        url: A connection string, typically copied from a provider console.

    Returns:
        The URL with an explicit ``postgresql+psycopg`` scheme. Non-Postgres
        URLs and URLs already naming a driver are returned unchanged.

    Raises:
        ValueError: If the URL explicitly requests psycopg2, which is not
            installed.
    """
    if url.startswith(REJECTED_SCHEME):
        msg = (
            "Connection string requests psycopg2, which this project does not "
            f"install. Use {PSYCOPG_SCHEME!r} instead."
        )
        raise ValueError(msg)

    for scheme in BARE_SCHEMES:
        if url.startswith(scheme):
            return PSYCOPG_SCHEME + url[len(scheme) :]

    return url
