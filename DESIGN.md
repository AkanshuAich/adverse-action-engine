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

## 15. Known weaknesses

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
