# Design decisions

What was chosen, what was rejected, and why. The rejections matter more than
the choices: anyone can list the libraries in a project, and the interesting
question is always what was considered and turned down.

---

## 1. Verify a typed object, not prose

**Chosen.** The model emits a structured `AdverseActionNotice` — which factors
are the principal reasons, what is claimed, which provisions are cited. Every
field is checked against evidence held independently of the generator. Prose is
rendered afterwards from the verified object.

**Rejected: fact-checking the letter directly.** Extracting claims from free
text is itself a language task, so a prose-first design ends up using a model
to check a model. The result measures agreement between two fallible
generators, not correctness.

**Rejected: LLM-as-judge.** Same objection, stated more plainly. A groundedness
score produced by asking a model whether another model was truthful has no
ground truth in it. This project exists to argue against exactly that, so
adopting it anywhere would be self-defeating.

**Cost.** Two model calls per notice instead of one, and a schema that has to
be kept in step with the verifier.

---

## 2. Build the verifier before the generator

**Chosen.** Week 4 built the fact-checker and tested it against hand-written
notices containing deliberate fabrications. Week 5 built the generator against
that working oracle.

**Rejected: generator first.** The usual order. It produces a system where the
checker is shaped around the mistakes the generator happens to make, and the
mistakes that matter are the rare ones. Hand-written adversarial notices cover
failures no live model produced during development.

---

## 3. The model never sees the applicant identifier

**Chosen.** The model returns reasons and citations. Identity is attached
afterwards from the decision.

**Consequence.** A model that never sees an id cannot attribute a notice to the
wrong person. The verifier's precondition check went from a live defence to a
guard against programming errors — the failure became structurally impossible
to cause by generation rather than merely detectable.

---

## 4. An allowlist, not a redactor

**Chosen.** The prompt payload is *built* from a fixed set of permitted fields.

**Rejected: Presidio.** A redactor inspects a payload and removes what it
recognises, so anything it fails to recognise is disclosed. For structured
data an allowlist is strictly stronger: a value can only reach a provider by
being named. Presidio also carries a spaCy model of several hundred megabytes.

**When the rejection expires.** The moment an applicant-supplied free-text
field enters the payload, detection becomes the only option and Presidio comes
back. Recorded in `pyproject.toml` so the reasoning is where the decision is.

---

## 5. Enforce append-only in the database

**Chosen.** The application role is granted `INSERT` and `SELECT` on the audit
table and denied `UPDATE` and `DELETE`, in a migration. Separately, records are
hash-chained so tampering by a privileged role is detectable.

**Rejected: enforcing it in application code.** A guarantee that depends on
every code path being careful is not a guarantee. There is an integration test
in which the *owner* role — which can bypass the grants — alters a historical
row, and the chain catches it.

**Cost.** Two roles to manage, and migrations that must run as the owner.

---

## 6. Serialise audit appends with an advisory lock

**Chosen.** A Postgres transaction-level advisory lock around every append.

**Why.** Extending a hash chain means reading the tail. Two concurrent writers
both read tail *N* and both try to write *N+1*: one collides on the unique
constraint, or worse, the chain forks. A forked chain is not evidence of
anything.

**Cost.** One writer at a time. For credit decisions arriving at human speed
that is the right trade; for a high-throughput event log it would not be.

---

## 7. XGBoost native UBJSON, not ONNX

**Chosen.** Booster saved as UBJSON, calibrator as a JSON knot table.

**Rejected: ONNX.** The requirement was never ONNX itself — it was *never
unpickle*, because unpickling executes arbitrary code and bank security teams
block the format. UBJSON satisfies that. ONNX's actual benefit is cross-runtime
portability, which this system does not need, and its XGBoost converter
requires generic `f0`-style feature names and cannot represent native
categorical splits. Degrading the model to fit the serialiser would have cost
the category names that denial reasons are written from.

**Rejected: pickle.** Arbitrary code execution on load.

---

## 8. Guarded calibration

**Chosen.** Isotonic regression is fitted on part of the calibration split,
judged on the rest, and kept only if it measurably improves expected
calibration error. Otherwise the model ships with an identity mapping.

**Rejected: fitting isotonic unconditionally.** Measured on this pipeline it
degraded ECE at 2,400 calibration rows and improved it at 4,800. Isotonic is
non-parametric and overfits small samples, and a boosted model trained with a
proper scoring rule is often already close to calibrated.

**Rejected: a row-count threshold.** Guesswork. The guarded version adapts to
the data and makes "calibration never makes a quoted probability worse" true by
construction.

**Also rejected: `scale_pos_weight`.** It lifts ranking metrics on an 8%
positive rate and destroys calibration by construction. A decline rests on a
probability threshold, not a ranking.

---

## 9. Implement the statistics rather than import them

**Chosen.** Flesch reading ease, PSI, and KS are written directly.

**Rejected: `textstat` and `evidently`.** Both pull `nltk`, which carries
PYSEC-2026-3740 with no fix released. Flesch is three terms and a syllable
count; PSI is a sum over bins. Taking a vulnerable transitive dependency for
forty lines of arithmetic is a poor trade.

**What it caught.** Writing Flesch surfaced a bug: the vowel-group pass already
counts the `e` in "table", so the conventional `-le` adjustment double-counts
unless the silent `e` was stripped first.

---

## 10. A simulated provider for the CI gate

**Chosen.** The evaluation gate runs against a provider that fails at
configured rates, seeded deterministically on the prompt.

**Rejected: calling a live model in CI.** No credential, rate limits, and a
number that moves for reasons unrelated to the change under review.

**Rejected: a provider scripted always to succeed.** Measures nothing.

**Stated limitation.** These figures measure how the *system* handles a fixed
distribution of model mistakes, not how good a real model is. The module says
so in its first paragraph, and live figures come from the same harness run
manually against a real backend.

**Design detail.** The simulator parses the prompt rather than being handed the
payload, so a prompt that omits something a model would need fails loudly
instead of showing up as poor metrics.

---

## 11. Gate on a baseline, not fixed thresholds

**Chosen.** CI fails when a gated metric falls more than three points below a
committed baseline, and unconditionally when prohibited content reaches an
issued notice.

**Rejected: absolute thresholds.** They either sit so low they never fire or
get raised until somebody turns them off. A baseline answers the question a
reviewer actually has: did this change make it worse?

**Definitions chosen not to flatter the system.** Groundedness is measured on
the *first* attempt, before repair — reporting the post-repair figure would
credit the verifier's work to the model. Prohibited content is measured over
*issued* notices only; a model proposing a prohibited reason is the case the
check exists for. Escalation excludes provider failures, because folding an
outage into that rate would make a network problem look like the model
degrading, and that rate is precisely what someone watches to notice exactly
that.

---

## 12. Document disparate impact; do not correct by group

**Chosen.** Measure adverse impact and equalized odds, report with group sizes,
and record a written position.

**Rejected: per-group threshold adjustment.** The obvious response to a low
ratio, and unlawful. Setting a different decision threshold for applicants of
one sex is disparate treatment; it does not stop being so because the intent
was to improve a fairness metric. It would also be plainly visible in an audit
log that records the threshold applied to every decision.

**Also.** Group sizes are reported alongside every ratio because a ratio
computed over eighteen applicants is noise, and a report that hides that invites
someone to act on it.

---

## 13. Streamlit, and no queue table

**Chosen.** An internal review console in Streamlit, reconstructing every case
from the audit chain.

**Rejected: React.** Two weeks of work for a tool a handful of underwriters
use, to do the same thing.

**Rejected: a queue table.** It could disagree with the audit log, and the log
exists so there is only one source of truth about what happened. Reconstructing
from the chain also means the console exercises the claim made to a regulator
daily, rather than leaving it to be discovered untrue during an audit — which
is how the missing notice content in the generation record was found.

---

## 14. Deliberately out of scope

- **Kafka, Flink, Spark.** Credit decisions arrive at human speed. Streaming
  infrastructure would add operational weight and demonstrate nothing the
  problem requires.
- **A dedicated vector database.** pgvector in the same Postgres means one
  datastore, transactional consistency with the audit log, and one fewer
  vendor. The corpus is four provisions; even the vector store is arguably
  premature, and it exists for when the corpus is a full Master Direction.
- **Fine-tuning.** The task is constrained generation into a schema that is
  then verified. Fine-tuning would trade a checkable system for an opaque one.
- **Java.** Python end to end. A second language would have to earn its
  operational cost.

---

## 15. Constrain the decoder, do not ask the model nicely

The provider originally sent `response_format: {"type": "json_object"}`. That
guarantees the response parses as JSON. It guarantees nothing about which keys
the JSON has.

Against a live backend this failed exactly as you would expect once stated
plainly. The model returned a well-formed, entirely reasonable object with
invented field names — `reason` and `citation` on each principal reason, plus
a top-level `decision_statement` and `grievance_contact` — where
`SelectedNotice` declares `text`. Fifteen validation errors, a provider error,
and a repair attempt burned on a problem no amount of reprompting fixes
reliably, because the schema was never actually communicated.

The adapter now sends the Pydantic JSON schema with `strict: true`, so the
shape is a decoding constraint rather than an instruction the model is free to
reinterpret. Pydantic's generated schema needs one transformation first: strict
mode requires every property to appear in `required`, and Pydantic omits fields
that have defaults. Listing them all is sound here because every optional field
defaults to an empty collection — the model is being asked to supply the key,
not to invent content for it.

Validation still runs on arrival. The constraint makes a wrong shape unlikely;
it does not make it impossible, and a backend that ignores the constraint must
still fail as a provider error rather than flow onward half-populated.

`ProviderConfig.supports_json_schema` is declared per backend rather than
discovered by catching a 400. Ollama's OpenAI-compatible surface has carried
this feature unevenly across versions, so it is declared unsupported and falls
back to `json_object` — a known, recorded limitation rather than a silent
downgrade.

## 16. One provider vanished, and the abstraction paid for itself

Cerebras was the primary backend: 1M tokens/day, no credit card. Partway
through, it began answering `402 Payment required` for every model, and the
model this project had pinned — `llama-3.3-70b` — no longer appeared in its
catalogue at all.

Two failure modes worth separating, because they look identical from the
outside:

- **A renamed model returns 404**, which reads as a configuration error.
- **A withdrawn free tier returns 402**, which reads as a billing error.

Neither is a bad credential, and both are indistinguishable from one if you
only look at "the LLM call failed". The key authenticated fine throughout.

The migration to Groq was two environment variables. Nothing in the generation
graph, the verifier, the audit chain, or the evaluation harness changed,
because none of them knows which backend answered. This is the case for the
abstraction, and it is worth more as a thing that actually happened than as a
paragraph about hypothetical vendor risk.

The practical lesson is narrower than "abstract your providers": **do not pin a
model name from memory.** Query the backend's `/models` endpoint and pin
against what it returns today. `.env.example` says so, with the command.

## 17. Round the figures before the model sees them

Rule 3 of the selection prompt used to end "Round for readability if you wish;
do not change the figure." A live notice then told an applicant their credit
bureau score was 0.0935070975944776.

That output is correct. It passes value accuracy, because it *is* the scored
value. It is also not something any lender would post. The failure is invisible
to five of the six checks, because none of them asks whether a true statement
is a sayable one — only `readability` notices, and only in aggregate.

Rounding is now done in `build_payload`, before the payload is assembled, so
full precision never reaches the model. An instruction the model may decline to
follow became a value it cannot obtain.

Four significant figures carries a relative error below 5e-4, an order of
magnitude inside the verifier's 0.005 tolerance for presentation rounding, so
the rounding cannot cause a value-accuracy violation. A test asserts that
relationship against both constants rather than stating it in a comment: if
either moves, the test fails rather than the pipeline.

Whole numbers are serialised as integers on the way into the prompt. JSON
cannot express "a float that happens to be integral", so `136800.0` reached the
model and was faithfully copied into a sentence about someone's annual income.

## 18. A gate that only checks metrics can pass a run that did not happen

`check_gate` verified that no metric had regressed and that `cases != 0`. The
first live run against Groq lost four of its five cases to a token-per-minute
limit and printed **GATE PASSED**, on the strength of the one case that
completed.

Every metric is computed over the cases that finished. That makes an incomplete
run quietly self-flattering, and the survivors are not a random sample of the
golden set: a token-per-minute limit falls hardest on the longest prompts,
which are the hard cases. Their scores were then compared against a baseline
measured over the whole set — two different populations, one number.

The gate now fails when more than 10% of attempted cases were abandoned, and
the message says the run is *unmeasured* rather than *worse*, because those
require different responses: one is re-run it, the other is investigate it.

The general form is worth stating, because it is not specific to rate limits:
**an evaluation harness must gate on whether the evaluation happened before it
gates on what the evaluation said.**

## 19. Two rate limits, and only one of them is in the headers

Measuring the pipeline against a live backend took three attempts, and the
first two diagnoses were wrong in instructive ways.

**Attempt one.** 71 of 100 cases abandoned, all within fourteen seconds. A 429
returns instantly, so the runner's throttle — which spaces out work that takes
time — never applied to the failures. Concluded: honour the rate limit rather
than treat it as fatal. Correct, and insufficient.

**Attempt two.** 88 of 100 abandoned, with the backoff engaging only twice. The
waits being requested were four to six minutes, past the ceiling. Concluded:
the ceiling is too low, because a token bucket refilling continuously can
legitimately report a long reset when deeply in debt. **Wrong.**

**What was actually happening.** Groq enforces a limit on tokens *per day* —
200,000 — alongside the per-minute one. It had been exhausted. And a daily
allowance appears in **no response header**: the request that finally revealed
it reported `x-ratelimit-remaining-tokens: 8000` out of a limit of 8000, a
completely full bucket, and failed anyway. The body said so in a sentence:

> Rate limit reached ... on tokens per day (TPD): Limit 200000, Used 197204

That sentence was in every one of the 159 failed responses across both runs.
The code discarded it and substituted a message of its own, so two rounds of
diagnosis were spent inferring a cause from headers that structurally could not
express it.

Three things changed:

- The backend's own explanation is quoted in the error. A message written at
  the call site can only describe what the author imagined; the body describes
  what happened.
- The per-minute limit is waited out, which is what it is for.
- The README states the ceiling: roughly 35 cases per day, so a live run is a
  labelled subset and the simulated run remains the gate.

The general lesson is not about rate limits. **When a service explains itself
in the response body, do not replace that explanation with your own.** The
inference was reasonable both times and wrong both times, and the answer was
sitting in a field the code was throwing away.

## 20. Known weaknesses

Listed because a design document that records only successes is a sales
brochure.

- **The prose numeric check permits small integers unconditionally.** List
  numbering and counts would otherwise fail every letter. A fabricated small
  integer therefore passes. Acceptable only because this payload holds no
  small-integer facts about an applicant, and it stops being acceptable the
  moment one is added.
- **Prohibited-content detection is pattern-based.** It catches the phrasings
  it knows. A model describing a protected characteristic in wording nobody
  anticipated would get through the text scan, though not the factor check.
- **The golden set is synthetic.** It measures the system, not real applicants.
- **Retrieval is barely exercised.** With four provisions, ranking is
  pointless; the API passes all of them and uses the vector store for nothing.
- **Latency figures from the simulator are meaningless.** They measure local
  arithmetic, not a model call.
