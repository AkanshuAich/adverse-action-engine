-- Local development bootstrap. Runs once, on first container start.
--
-- Production (Neon) equivalent: create the aae_app role through the Neon
-- console, then run `alembic upgrade head` as the owner. The migrations grant
-- privileges; this file only creates the role and the extension, because
-- neither is a schema concern and Neon manages roles outside of SQL.

-- pgvector, for regulation corpus search.
CREATE EXTENSION IF NOT EXISTS vector;

-- The application role. It is intentionally NOT the schema owner: the audit
-- table grants it INSERT and SELECT only, so the append-only guarantee is
-- enforced by Postgres rather than by application discipline.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'aae_app') THEN
        CREATE ROLE aae_app LOGIN PASSWORD 'aae_dev_password';
    END IF;
END
$$;

GRANT CONNECT ON DATABASE aae TO aae_app;
GRANT USAGE ON SCHEMA public TO aae_app;
