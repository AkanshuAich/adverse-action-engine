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
python -m uv pip install -e ".[dev]"
cp .env.example .env
docker compose --profile core up -d
python -m uv run alembic upgrade head
python -m uv run pytest
```

## Status

Under construction. Week 1 of 8: foundation, tooling, CI, audit schema.

## Licence

MIT
