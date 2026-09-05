# Deploying the console to Streamlit Community Cloud

Free, no card, and it redeploys on every push to `main` — which is the reason
to prefer it over a second repository that has to be kept in step by hand.

## 1. Deploy

Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
**Create app** → **Deploy a public app from GitHub**, then:

| Field | Value |
|---|---|
| Repository | `AkanshuAich/adverse-action-engine` |
| Branch | `main` |
| Main file path | `deploy/streamlit/app.py` |

Open **Advanced settings** before deploying.

## 2. Python version

Set **Python 3.12** or newer. This is not optional: the package uses a PEP 695
`type` statement, which is a syntax error on 3.11, and `pyproject.toml`
declares `requires-python = ">=3.12"`.

## 3. Secrets

Paste this into the **Secrets** box, substituting your Neon connection string:

```toml
AAE_DATABASE_URL = "postgresql://aae_app:PASSWORD@HOST.neon.tech/aae?sslmode=require"
```

Use the **`aae_app`** role, never the owner. Postgres denies `aae_app` any
`UPDATE` or `DELETE` on the audit table, so a publicly reachable console cannot
rewrite history no matter what the application code does. Deploying with the
owner role would silently discard the guarantee the project exists to
demonstrate.

Streamlit exposes every root-level secret as an environment variable at
runtime, which is how `pydantic-settings` picks it up. No code change is needed
and `st.secrets` is never read.

A bare `postgresql://` scheme is fine — `aae.database_url` rewrites it to
`postgresql+psycopg://`, because SQLAlchemy reads the bare form as a request
for psycopg2 and only psycopg 3 is installed.

No LLM key is needed. The console calls no model; generation happens in the
API.

## Why the files live here

Community Cloud searches the entrypoint's directory for a dependency file
*before* the repository root. That is what keeps `requirements.txt` out of the
root, where the project's standards say it should not be.

It is also what keeps the build working. Community Cloud recognises a root
`pyproject.toml` and hands it to **poetry**; this project builds with
hatchling, so root-level resolution would fail. The requirements file beside
the entrypoint is found first, and poetry is never reached.

## Free-tier limits

The app sleeps after a period of inactivity and wakes on the next visit, so the
first request after a quiet spell is slow. Neon also suspends an idle database;
the engine is configured with `pool_pre_ping` and a 280-second recycle for
exactly that, so a woken app reconnects rather than serving a dead connection.
