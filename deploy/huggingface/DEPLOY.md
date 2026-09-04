# Deploying the console

Two free services, no credit card: a Neon Postgres project for state and a
Hugging Face Space for the UI. Free Spaces have **ephemeral** storage, which is
why nothing is kept on the Space itself.

## 1. Database

Create a free project at [neon.tech](https://neon.tech) and enable pgvector:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE ROLE aae_app LOGIN PASSWORD '<choose-one>';
GRANT USAGE ON SCHEMA public TO aae_app;
```

The `aae_app` role must exist *before* migrating: the migration grants
privileges only to a role that is already there, which is what keeps the audit
table append-only.

Then, from a clone of the main repository:

```bash
export AAE_MIGRATION_DATABASE_URL='postgresql+psycopg://<owner>:<pw>@<host>.neon.tech/<db>?sslmode=require'
alembic upgrade head
```

## 2. Space

Create a Streamlit Space at [huggingface.co/new-space](https://huggingface.co/new-space),
then push the three files in this directory to it:

```bash
git clone https://huggingface.co/spaces/<you>/adverse-action-engine hf-space
cp deploy/huggingface/{README.md,app.py,requirements.txt} hf-space/
cd hf-space && git add -A && git commit -m "Deploy review console" && git push
```

## 3. Secret

In the Space settings, add a secret:

```
AAE_DATABASE_URL = postgresql+psycopg://aae_app:<pw>@<host>.neon.tech/<db>?sslmode=require
```

Use the **application** role here, not the owner. The whole point of the two
roles is that the thing serving traffic cannot rewrite history.

## 4. Seed something to review

The console shows cases awaiting sign-off, so an empty chain shows an empty
queue. Point the API at the same database and post an application that
declines:

```bash
AAE_DATABASE_URL='<same as above>' uvicorn aae.api.main:app
curl -X POST localhost:8000/v1/notices -H 'content-type: application/json' -d @samples/decline.json
```

## Verifying the deployment

The console sidebar reports the audit chain state. If it shows anything other
than "Audit chain intact", stop: recorded history cannot be trusted and nothing
on the screen should be acted on until it is explained.
