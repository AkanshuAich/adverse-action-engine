# Adverse Action Engine

Verifiable adverse action notices for credit decisions.

When a lender declines a credit application, regulation requires it to state the actual
reasons. Generating those notices with a language model is attractive and dangerous: a
fabricated reason in a denial notice is a compliance failure, not a cosmetic bug.

This system generates the notice **and mechanically proves every claim in it is true**
before a human reviewer ever sees it.

## The design decision

Do not verify prose. Verify a typed object, then render prose from it.

Generation runs in two stages:

1. **Select** — the model emits a structured `AdverseActionNotice`: which factors were
   principal reasons, what factual claims are made, which regulatory provisions are cited.
2. **Render** — the model writes customer-facing prose constrained to that verified object,
   and the verifier re-checks the prose introduces no claim absent from it.

Because stage-one output is typed, every field is checkable against ground truth the system
already holds — the real feature values, the real SHAP attributions, and the real regulation
corpus. Verification is deterministic. There is no model grading another model, which is why
the resulting groundedness figure means something.

## Pipeline

```
application
   -> feature prep            (pandera-validated)
   -> credit model            (XGBoost, isotonic-calibrated, ONNX Runtime)
   -> explainer               (SHAP, top-K adverse factors)
   -> decision                (versioned threshold policy)
        |
        +-- decline --> regulation retrieval   (pgvector over the RBI corpus)
                     -> generation             (LangGraph, select then render)
                     -> verifier               (six checks; repair loop; else escalate)
                     -> human review           (Streamlit console)
   -> audit log               (append-only, hash-chained, at every stage)
```

## The verifier

Six independent checks, in `src/aae/verification/`:

| Check | What it catches |
|---|---|
| Factor grounding | Reasons citing a factor that is not in the SHAP top-K, or whose direction is inverted |
| Value accuracy | Any stated figure that disagrees with the real feature value |
| Citation validity | Fabricated regulation — the quoted span must appear verbatim in the cited chunk |
| Element coverage | Legally required elements missing from the notice |
| Prohibited content | Any reference to a protected attribute. Must be zero |
| Reason count | More principal reasons than the jurisdiction permits |

Failures feed back into a repair prompt for up to three attempts, then escalate to a human.
The escalation rate is a reported metric, not a hidden fallback.

## Fair lending

`CODE_GENDER` and age are present in the dataset. They are used **exclusively as protected
attributes for fairness measurement and are never model features** — using them as inputs
would be unlawful in real lending. A unit test enforces their absence from the trained
feature set.

## Quickstart

```bash
python -m uv venv
python -m uv pip install --python .venv/bin/python -e ".[dev]"
cp .env.example .env
docker compose --profile core up -d
.venv/bin/alembic upgrade head
.venv/bin/python -m pytest
```

On Windows the venv scripts live in `.venv/Scripts/` rather than `.venv/bin/`.

Postgres is published on host port **5433**, not 5432. A locally installed
PostgreSQL service commonly already owns 5432 and would silently win for
`localhost` connections, producing authentication failures that look like a
credential problem rather than a port collision.

Compose profiles keep the footprint small enough for a 12 GB machine — bring up
only what you are working on:

```bash
docker compose --profile core up -d      # Postgres only
docker compose --profile api up -d       # + the API
docker compose --profile console up -d   # + the Streamlit console
```

### Verifying the audit guarantee

The append-only property is enforced by Postgres, so it can be checked directly:

```bash
.venv/bin/python -m pytest tests/integration -m integration
```

Those tests assert that the application role may `INSERT` and `SELECT`, that the
database rejects `UPDATE`, `DELETE`, and `TRUNCATE`, and that the hash chain
detects tampering even by a privileged role that can bypass the grants.

## Data

The real Home Credit extract is used when `data/application_train.csv` is
present. Otherwise a synthetic generator produces a frame with the same
columns, dtypes, missingness, and statistical structure, so tests, CI, and a
fresh clone all work without a Kaggle account. Provenance travels with the
data and is recorded on the model, because "trained on synthetic data" is a
material fact about a credit model.

## Model

XGBoost with native categorical support, so SHAP attributions name real
category levels rather than one-hot indices — a denial reason has to be
readable by the person receiving it.

Calibration is **guarded**: isotonic regression is fitted on part of the
calibration split, judged on the rest, and kept only if it measurably improves
expected calibration error. Otherwise the model ships with an identity mapping.
Fitting isotonic unconditionally degrades calibration as often as it helps —
measured here, it hurt at 2,400 calibration rows and helped at 4,800. The
guarantee is that calibration never makes a quoted probability worse.

`scale_pos_weight` is deliberately unused. It lifts ranking metrics on an 8%
positive rate and destroys calibration by construction, and a denial rests on a
probability threshold, not a ranking.

## Explaining a decision

SHAP attributes each decision to its individual features, ranked by how far
they moved the score. Those factors are the ground truth the verifier will
check generated reasons against: if a notice names a factor, it must appear
here with a matching direction.

One subtlety, because a reviewer will ask. SHAP explains the booster's raw
log-odds, not the calibrated probability the threshold is applied to. That is
sound because calibration is *monotone*: it can move where the threshold sits
but cannot reorder two applicants or flip the sign of a contribution, so the
ranking and direction of factors survive it exactly. A test asserts this rather
than assuming it.

## API

```
GET  /health                            model version, threshold, chain state
POST /v1/decisions                      score, explain, and record
GET  /v1/decisions/{id}/audit           reconstruct a decision from the chain
GET  /v1/audit/verify                   recompute every hash in the chain
```

A decision and its audit record are inseparable. If the audit write fails the
request fails, because returning a decision that was not recorded is exactly
the unauditable outcome this system exists to prevent.

The API will not accept sex, age, or marital status at all. They cannot
lawfully influence the outcome, so there is no reason to collect them in order
to score one application.

Appends are serialised with a Postgres transaction-level advisory lock.
Extending a hash chain means reading the tail, so concurrent writers would
otherwise fork it, and a forked chain is not evidence of anything. A test runs
twenty threads at once and verifies the result is a single unbroken chain.

## Status

Week 3 of 8 complete. Foundation, audit log, data layer, calibrated model,
per-decision explanations, and the decision API.
Next: the verifier.

## Licence

MIT
