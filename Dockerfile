# Multi-stage build. The `api` and `console` targets share one dependency layer
# so the two images differ only by entrypoint and extras.

# --------------------------------------------------------------- base
FROM python:3.12-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# curl is needed by the healthcheck; nothing else is installed at runtime.
RUN apt-get update \
    && apt-get install --no-install-recommends -y curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

RUN uv venv "$VIRTUAL_ENV"

WORKDIR /app

# --------------------------------------------------------------- builder
FROM base AS builder

# Copy only what the build backend needs, so dependency layers stay cached
# across source edits.
COPY pyproject.toml README.md ./
COPY src/aae/__init__.py src/aae/__init__.py

ARG EXTRAS=""
RUN uv pip install --python "$VIRTUAL_ENV/bin/python" ".${EXTRAS}"

# --------------------------------------------------------------- runtime
FROM base AS runtime

# Run unprivileged. Created before the copy so ownership is right from the start.
RUN groupadd --system --gid 1001 aae \
    && useradd --system --uid 1001 --gid aae --create-home aae

COPY --from=builder --chown=aae:aae /opt/venv /opt/venv
COPY --chown=aae:aae pyproject.toml README.md ./
COPY --chown=aae:aae alembic.ini ./
COPY --chown=aae:aae alembic ./alembic
COPY --chown=aae:aae src ./src

ENV PYTHONPATH=/app/src
USER aae

# --------------------------------------------------------------- api
FROM runtime AS api

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl --fail --silent http://localhost:8000/health || exit 1

CMD ["uvicorn", "aae.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# --------------------------------------------------------------- console
FROM runtime AS console

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail --silent http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/aae/console/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
