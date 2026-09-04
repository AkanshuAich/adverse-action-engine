---
title: Adverse Action Engine
emoji: ⚖️
colorFrom: indigo
colorTo: gray
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
license: mit
short_description: Verifiable adverse action notices for credit decisions
---

# Adverse Action Engine — review console

The underwriter review console for
[adverse-action-engine](https://github.com/AkanshuAich/adverse-action-engine).

When a lender declines a credit application, regulation requires it to state
the actual reasons. Generating those notices with a language model is
attractive and dangerous: a fabricated reason in a denial notice is a
compliance failure, not a cosmetic bug.

This system generates the notice **and mechanically proves every claim in it
is true** before a human reviewer ever sees it. Six deterministic checks
compare each assertion against the model's own SHAP attributions, the exact
feature values that were scored, and the source regulation text. Nothing here
asks a language model whether another language model was truthful.

Every case on this screen is reconstructed from an append-only, hash-chained
audit log. There is no queue table — a queue that could disagree with the log
would be a second source of truth about what happened.

## Configuration

Set `AAE_DATABASE_URL` as a Space secret, pointing at a Postgres with pgvector
(a Neon free-tier project works). The schema is created by
`alembic upgrade head` from the main repository.

Storage on the free CPU tier is ephemeral, so all state lives in the database
rather than on the Space.
