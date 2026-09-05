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
from urllib.parse import urlsplit

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


def describe_target(url: str) -> str:
    """Describe where a connection string points, without the credentials.

    A failed connection is unreadable without knowing what was dialled, and a
    deployment platform will redact any message that might carry a password -
    which means the useful part is withheld along with the dangerous part.
    Naming only the host and database keeps the diagnosis and drops the secret.

    Args:
        url: The connection string.

    Returns:
        A ``host/database`` description, or a note that it could not be read.
    """
    try:
        parsed = urlsplit(url)
    except ValueError:
        return "an unreadable connection string"

    host = parsed.hostname or "an unnamed host"
    port = f":{parsed.port}" if parsed.port else ""
    database = parsed.path.lstrip("/") or "an unnamed database"
    return f"{host}{port}/{database}"


def is_local_default(url: str) -> bool:
    """Report whether this is still the local development connection.

    Settings fall back to a localhost URL when nothing is configured, which is
    right for a checkout and wrong everywhere else: a deployment with no
    database secret does not fail saying so, it fails trying to reach a
    Postgres on its own container that was never there.

    Args:
        url: The connection string.

    Returns:
        Whether the host is a loopback address.
    """
    try:
        host = urlsplit(url).hostname
    except ValueError:
        return False
    return host in {"localhost", "127.0.0.1", "::1"}
