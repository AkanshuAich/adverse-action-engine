"""The generation workflow.

Select, verify, repair, render, check. Expressed as an explicit state graph
rather than a loop because the execution path is part of the audit record: a
reviewer asking why a particular notice took three attempts gets the sequence
of nodes and the violations that drove each transition, not a stack trace.

The two stages exist for one reason. Stage one produces a *typed* selection
whose every field is checkable against evidence the system already holds.
Stage two turns the verified selection into prose and is shown nothing it
could get wrong about the record - not the identifier, not the raw values,
only the sentences that already passed. Prose is where a model invents; the
less it is given, the less there is to invent about.

Repair is bounded and its exhaustion is a reported outcome, not a swallowed
one. A notice that cannot be made truthful in three attempts goes to a human
with the violations attached, and the escalation rate is a metric the eval
harness reports rather than a fallback nobody sees.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, TypedDict

from langgraph.graph import END, START, StateGraph

from aae.domain.errors import GenerationError, ProviderError

# Imported at runtime, not under TYPE_CHECKING, because LangGraph resolves the
# state TypedDict's annotations when the graph is compiled. `from __future__
# import annotations` defers them to strings, and a string that names a type
# only imported for type checking cannot be evaluated. SQLAlchemy needed the
# same thing for its mapped columns; any library that reads annotations at
# runtime does.
from aae.domain.models import (
    AdverseActionNotice,
    CreditDecision,
    VerificationResult,
    Violation,
)
from aae.generation.payload import GenerationPayload
from aae.generation.prompts import (
    RENDER_SYSTEM,
    SELECT_SYSTEM,
    build_render_user_message,
    build_repair_user_message,
    build_select_user_message,
)
from aae.generation.schemas import RenderedBody, SelectedNotice
from aae.logging import get_logger
from aae.verification.prose import check_prose

if TYPE_CHECKING:
    from aae.generation.providers.base import StructuredProvider
    from aae.verification.verifier import NoticeVerifier

logger = get_logger(__name__)

DEFAULT_MAX_ATTEMPTS: Final[int] = 3

# StateGraph is generic over state, context, input and output. Only the
# state type is ours to choose; the rest default to it.
type GraphBuilder = StateGraph[GenerationState, Any, GenerationState, GenerationState]


def _digest(text: str) -> str:
    """Hash a prompt or response for the audit record.

    The text itself is not stored. A hash is enough to prove later that a
    given prompt produced a given response, without retaining the applicant's
    circumstances in a second place.

    Args:
        text: The material to hash.

    Returns:
        Lowercase hex SHA-256.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExecutionStep:
    """One node execution, for the audit trail.

    Attributes:
        node: Which node ran.
        attempt: Which generation attempt it belonged to.
        prompt_hash: Hash of the prompt sent, if any.
        response_hash: Hash of the response received, if any.
        violations: Violations produced by this step, rendered for a prompt.
        violation_codes: The same violations as codes, so metrics can be
            counted without parsing rendered text.
        reasons: How many principal reasons this attempt proposed.
        citations: How many citations this attempt proposed.
    """

    node: str
    attempt: int
    prompt_hash: str | None = None
    response_hash: str | None = None
    violations: tuple[str, ...] = ()
    violation_codes: tuple[str, ...] = ()
    reasons: int = 0
    citations: int = 0


@dataclass(frozen=True)
class GenerationOutcome:
    """The result of running the workflow.

    Attributes:
        notice: The verified structured notice, if one was reached.
        body: The customer-facing letter, if rendering succeeded.
        result: The final verification result.
        attempts: How many generation attempts were made.
        escalated: Whether this went to a human instead of being issued.
        escalation_reason: Why, if it did.
        provider: Which backend answered.
        model: Which model answered.
        trace: Every node execution, in order.
    """

    notice: AdverseActionNotice | None
    body: str | None
    result: VerificationResult | None
    attempts: int
    escalated: bool
    escalation_reason: str | None
    provider: str
    model: str
    trace: tuple[ExecutionStep, ...] = ()

    @property
    def issued(self) -> bool:
        """Whether a notice may be sent without human intervention."""
        return not self.escalated and self.body is not None

    def audit_payload(self) -> dict[str, Any]:
        """Render the outcome for the audit chain.

        Returns:
            A JSON-compatible summary, including the prompt and response
            hashes that make the generation reproducible in principle.
        """
        return {
            "issued": self.issued,
            "escalated": self.escalated,
            "escalation_reason": self.escalation_reason,
            "attempts": self.attempts,
            "provider": self.provider,
            "model": self.model,
            "passed_verification": self.result.passed if self.result else False,
            "violations": list(self.result.rendered_violations()) if self.result else [],
            "trace": [
                {
                    "node": step.node,
                    "attempt": step.attempt,
                    "prompt_hash": step.prompt_hash,
                    "response_hash": step.response_hash,
                    "violations": list(step.violations),
                    "violation_codes": list(step.violation_codes),
                    "reasons": step.reasons,
                    "citations": step.citations,
                }
                for step in self.trace
            ],
        }


class GenerationState(TypedDict, total=False):
    """State threaded through the graph."""

    payload: GenerationPayload
    decision: CreditDecision
    attempt: int
    max_attempts: int
    selected: SelectedNotice | None
    notice: AdverseActionNotice | None
    result: VerificationResult | None
    body: str | None
    prose_violations: tuple[Violation, ...]
    escalated: bool
    escalation_reason: str | None
    trace: list[ExecutionStep]


@dataclass
class NoticeGenerator:
    """Runs the select-verify-repair-render workflow.

    Attributes:
        provider: The language model backend.
        verifier: Checks each attempt against the decision.
        max_attempts: How many times to try before escalating.
    """

    provider: StructuredProvider
    verifier: NoticeVerifier
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    _graph: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Compile the graph once, at construction."""
        self._graph = self._build().compile()

    # ---------------------------------------------------------------- nodes

    def _select(self, state: GenerationState) -> GenerationState:
        """Ask the model for a structured selection, or a repair of one."""
        attempt = state["attempt"]
        payload = state["payload"]
        previous = state.get("result")

        user = (
            build_select_user_message(payload)
            if attempt == 1 or previous is None
            else build_repair_user_message(payload, previous.rendered_violations(), attempt)
        )

        selected = self.provider.complete(system=SELECT_SYSTEM, user=user, schema=SelectedNotice)
        notice = selected.to_domain(
            application_id=state["decision"].application_id,
            jurisdiction=self.verifier.jurisdiction.code,
        )

        step = ExecutionStep(
            node="select",
            attempt=attempt,
            prompt_hash=_digest(user),
            response_hash=_digest(selected.model_dump_json()),
            reasons=len(selected.principal_reasons),
            citations=len(selected.citations),
        )
        return {
            "selected": selected,
            "notice": notice,
            "trace": [*state.get("trace", []), step],
        }

    def _verify(self, state: GenerationState) -> GenerationState:
        """Check the selection against the decision."""
        notice = state["notice"]
        if notice is None:  # pragma: no cover - the graph cannot reach this
            msg = "Verification ran with no notice to verify."
            raise GenerationError(msg)

        result = self.verifier.verify(notice, state["decision"], attempt=state["attempt"])
        step = ExecutionStep(
            node="verify",
            attempt=state["attempt"],
            violations=tuple(result.rendered_violations()),
            violation_codes=tuple(v.code.value for v in result.violations),
        )
        return {"result": result, "trace": [*state.get("trace", []), step]}

    def _retry(self, state: GenerationState) -> GenerationState:
        """Advance the attempt counter before regenerating."""
        return {"attempt": state["attempt"] + 1}

    def _render(self, state: GenerationState) -> GenerationState:
        """Turn the verified selection into a letter.

        The renderer is given the verified sentences and nothing else - no
        identifier, no feature values, no scores. It cannot misstate a record
        it was never shown.
        """
        notice = state["notice"]
        if notice is None:  # pragma: no cover - unreachable via the graph
            msg = "Rendering ran with no verified notice."
            raise GenerationError(msg)

        user = build_render_user_message(
            state["payload"], [reason.text for reason in notice.principal_reasons]
        )
        rendered = self.provider.complete(system=RENDER_SYSTEM, user=user, schema=RenderedBody)

        step = ExecutionStep(
            node="render",
            attempt=state["attempt"],
            prompt_hash=_digest(user),
            response_hash=_digest(rendered.model_dump_json()),
        )
        return {"body": rendered.body, "trace": [*state.get("trace", []), step]}

    def _check_prose(self, state: GenerationState) -> GenerationState:
        """Check the letter introduced nothing the record does not support."""
        body = state["body"]
        notice = state["notice"]
        if body is None or notice is None:  # pragma: no cover - unreachable
            msg = "Prose checking ran with no rendered letter."
            raise GenerationError(msg)

        violations = check_prose(body, notice, state["payload"], self.verifier.jurisdiction)
        step = ExecutionStep(
            node="check_prose",
            attempt=state["attempt"],
            violations=tuple(v.render() for v in violations),
            violation_codes=tuple(v.code.value for v in violations),
        )
        return {"prose_violations": violations, "trace": [*state.get("trace", []), step]}

    def _escalate(self, state: GenerationState) -> GenerationState:
        """Hand the case to a human, with the violations attached."""
        prose = state.get("prose_violations") or ()
        result = state.get("result")

        if prose:
            reason = "The rendered letter contained statements the record does not support."
        elif result is not None and not result.passed:
            reason = (
                f"Verification still failed after {state['attempt']} attempts: "
                f"{'; '.join(result.rendered_violations())}"
            )
        else:  # pragma: no cover - escalation is only routed to on failure
            reason = "Escalated without a recorded cause."

        step = ExecutionStep(node="escalate", attempt=state["attempt"])
        logger.warning(
            "notice_escalated",
            application_id=state["decision"].application_id,
            attempts=state["attempt"],
            reason=reason,
        )
        return {
            "escalated": True,
            "escalation_reason": reason,
            "trace": [*state.get("trace", []), step],
        }

    # ------------------------------------------------------------- routing

    def _route_after_verify(self, state: GenerationState) -> str:
        result = state.get("result")
        if result is not None and result.passed:
            return "render"
        if state["attempt"] < state["max_attempts"]:
            return "retry"
        return "escalate"

    def _route_after_prose(self, state: GenerationState) -> str:
        return "escalate" if state.get("prose_violations") else "done"

    def _build(self) -> GraphBuilder:
        builder: GraphBuilder = StateGraph(GenerationState)
        builder.add_node("select", self._select)
        builder.add_node("verify", self._verify)
        builder.add_node("retry", self._retry)
        builder.add_node("render", self._render)
        builder.add_node("check_prose", self._check_prose)
        builder.add_node("escalate", self._escalate)

        builder.add_edge(START, "select")
        builder.add_edge("select", "verify")
        builder.add_conditional_edges(
            "verify",
            self._route_after_verify,
            {"render": "render", "retry": "retry", "escalate": "escalate"},
        )
        builder.add_edge("retry", "select")
        builder.add_edge("render", "check_prose")
        builder.add_conditional_edges(
            "check_prose", self._route_after_prose, {"done": END, "escalate": "escalate"}
        )
        builder.add_edge("escalate", END)
        return builder

    # -------------------------------------------------------------- public

    def generate(self, decision: CreditDecision, payload: GenerationPayload) -> GenerationOutcome:
        """Produce a verified notice for a declined application.

        Args:
            decision: The decision to explain.
            payload: The permitted view of it.

        Returns:
            The outcome, whether issued or escalated.

        Raises:
            GenerationError: If the provider fails outright. A backend that
                cannot be reached is an operational failure, not an
                unverifiable notice, and must not be reported as an
                escalation - that would quietly inflate a metric people rely
                on to spot the model getting worse.
        """
        initial: GenerationState = {
            "payload": payload,
            "decision": decision,
            "attempt": 1,
            "max_attempts": self.max_attempts,
            "selected": None,
            "notice": None,
            "result": None,
            "body": None,
            "prose_violations": (),
            "escalated": False,
            "escalation_reason": None,
            "trace": [],
        }

        try:
            final: GenerationState = self._graph.invoke(initial)
        except ProviderError as exc:
            msg = f"Notice generation failed at the provider: {exc}"
            raise GenerationError(msg) from exc

        outcome = GenerationOutcome(
            notice=final.get("notice"),
            body=None if final.get("escalated") else final.get("body"),
            result=final.get("result"),
            attempts=final.get("attempt", 1),
            escalated=bool(final.get("escalated")),
            escalation_reason=final.get("escalation_reason"),
            provider=self.provider.name,
            model=self.provider.model,
            trace=tuple(final.get("trace", [])),
        )

        logger.info(
            "notice_generation_complete",
            application_id=decision.application_id,
            issued=outcome.issued,
            attempts=outcome.attempts,
            provider=outcome.provider,
        )
        return outcome
