"""The decision API.

Scores an application, explains the outcome, and records both in the
append-only audit chain before responding. The audit write is not a background
task or a best-effort side effect: if it fails, the request fails. A decision
that was made but not recorded is exactly the thing this system exists to
prevent, so it must never be possible to return one.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Annotated

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from aae.api.deps import (
    get_audit_repository,
    get_corpus,
    get_decision_engine,
    get_notice_generator,
)
from aae.api.schemas import (
    ApplicationRequest,
    AuditRecordResponse,
    AuditTrailResponse,
    ChainVerificationResponse,
    DecisionResponse,
    HealthResponse,
    NoticeCitationResponse,
    NoticeReasonResponse,
    NoticeResponse,
)
from aae.audit.models import AuditEventType
from aae.audit.repository import decision_payload
from aae.config import get_settings
from aae.data.schema import validate_for_scoring
from aae.domain.errors import (
    AuditIntegrityError,
    ConfigurationError,
    DataValidationError,
    GenerationError,
)
from aae.domain.models import Decision
from aae.generation.payload import build_payload
from aae.jurisdiction.india_rbi import INDIA_RBI
from aae.logging import bind_correlation_id, configure_logging, get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from aae.audit.repository import AuditRepository
    from aae.generation.graph import NoticeGenerator
    from aae.ml.decision import DecisionEngine

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Warm the model and explainer before serving traffic.

    Building the SHAP explainer walks the whole ensemble. Doing that lazily
    would make the first request after every deploy dramatically slower than
    the rest, which is the sort of thing that shows up as a latency spike
    nobody can explain.
    """
    configure_logging(get_settings())
    engine = get_decision_engine()
    logger.info("api_ready", model_version=engine.model_version, threshold=engine.threshold)
    yield


app = FastAPI(
    title="Adverse Action Engine",
    description="Credit decisions with per-decision explanations and a tamper-evident audit log.",
    version="0.3.0",
    lifespan=lifespan,
)

EngineDep = Annotated["DecisionEngine", Depends(get_decision_engine)]
AuditDep = Annotated["AuditRepository", Depends(get_audit_repository)]
GeneratorDep = Annotated["NoticeGenerator", Depends(get_notice_generator)]


@app.get("/health", response_model=HealthResponse, tags=["operations"])
def health(engine: EngineDep, audit: AuditDep) -> HealthResponse:
    """Report liveness, the model in force, and audit chain integrity.

    Args:
        engine: The decision engine.
        audit: The audit repository.

    Returns:
        Current service state.
    """
    verification = audit.verify()
    return HealthResponse(
        status="ok" if verification.intact else "degraded",
        model_version=engine.model_version,
        threshold=engine.threshold,
        audit_records=verification.checked,
        chain_intact=verification.intact,
    )


@app.post(
    "/v1/decisions",
    response_model=DecisionResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["decisions"],
)
def create_decision(
    application: ApplicationRequest,
    engine: EngineDep,
    audit: AuditDep,
) -> DecisionResponse:
    """Score an application, explain it, and record it.

    Args:
        application: The application to decide.
        engine: The decision engine.
        audit: The audit repository.

    Returns:
        The decision with its ranked factors and audit position.

    Raises:
        HTTPException: 422 if the application fails validation, 500 if the
            audit record could not be written.
    """
    decision_id = audit.new_decision_id()
    bind_correlation_id(decision_id)

    frame = pd.DataFrame([application.model_dump()])
    try:
        validated = validate_for_scoring(frame)
    except DataValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    decision = engine.decide(validated)

    try:
        record = audit.record_decision(decision, decision_id)
    except AuditIntegrityError as exc:
        # Deliberately fatal. Returning a decision we failed to record would
        # produce exactly the unauditable outcome this system exists to prevent.
        logger.error("audit_write_failed", decision_id=decision_id, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Decision was not recorded and has therefore not been issued.",
        ) from exc

    return DecisionResponse.from_decision(decision, decision_id, record.sequence)


@app.post(
    "/v1/notices",
    response_model=NoticeResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["notices"],
)
def create_notice(
    application: ApplicationRequest,
    engine: EngineDep,
    audit: AuditDep,
    generator: GeneratorDep,
) -> NoticeResponse:
    """Score an application and, if declined, produce a verified notice.

    The decision and the notice are written to the audit chain in one
    transaction. A decision recorded without its notice, or a notice without
    the decision it explains, is a partial chain: it looks like evidence and
    is not.

    Args:
        application: The application to decide.
        engine: The decision engine.
        audit: The audit repository.
        generator: The notice generator.

    Returns:
        The notice, or the reason it was escalated to a human.

    Raises:
        HTTPException: 422 on invalid input, 409 if the application was
            approved and there is no adverse action to explain, 502 if the
            language model could not be reached, 500 if the audit write fails.
    """
    decision_id = audit.new_decision_id()
    bind_correlation_id(decision_id)

    frame = pd.DataFrame([application.model_dump()])
    try:
        validated = validate_for_scoring(frame)
    except DataValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    decision = engine.decide(validated)

    if decision.decision is not Decision.DECLINE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "This application was approved. An adverse action notice explains "
                "a decline; there is nothing here to explain."
            ),
        )

    try:
        payload = build_payload(decision, INDIA_RBI, list(get_corpus()))
        outcome = generator.generate(decision, payload)
    except GenerationError as exc:
        # An unreachable backend is an operational failure, not an unverifiable
        # notice. Reporting it as an escalation would inflate the metric people
        # rely on to notice the model getting worse.
        logger.error("notice_generation_failed", decision_id=decision_id, error=str(exc))
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    event = (
        AuditEventType.ESCALATED_TO_HUMAN if outcome.escalated else AuditEventType.NOTICE_VERIFIED
    )
    try:
        records = audit.append_many(
            [
                (
                    AuditEventType.DECISION_SCORED,
                    decision.application_id,
                    decision_id,
                    decision_payload(decision),
                    None,
                ),
                (
                    event,
                    decision.application_id,
                    decision_id,
                    outcome.audit_payload(),
                    None,
                ),
            ]
        )
    except AuditIntegrityError as exc:
        logger.error("audit_write_failed", decision_id=decision_id, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The notice was not recorded and has therefore not been issued.",
        ) from exc

    notice = outcome.notice
    return NoticeResponse(
        decision_id=decision_id,
        application_id=decision.application_id,
        probability_default=decision.probability_default,
        issued=outcome.issued,
        escalated=outcome.escalated,
        escalation_reason=outcome.escalation_reason,
        attempts=outcome.attempts,
        provider=outcome.provider,
        model=outcome.model,
        body=outcome.body,
        reasons=tuple(
            NoticeReasonResponse(factor_id=reason.factor_id, text=reason.text)
            for reason in (notice.principal_reasons if notice else ())
        ),
        citations=tuple(
            NoticeCitationResponse(
                document_id=citation.document_id,
                section=citation.section,
                quoted_span=citation.quoted_span,
            )
            for citation in (notice.citations if notice else ())
        ),
        violations=tuple(outcome.result.rendered_violations()) if outcome.result else (),
        audit_sequences=tuple(record.sequence for record in records),
    )


@app.exception_handler(ConfigurationError)
def configuration_error_handler(_: object, exc: ConfigurationError) -> JSONResponse:
    """Report a missing credential as unavailable, not as a server fault.

    The scoring endpoints work without a language model. Only notice
    generation needs one, and a deployment that has not configured it is
    misconfigured rather than broken.

    Args:
        _: The request, unused.
        exc: The configuration failure.

    Returns:
        A 503 naming what is missing.
    """
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": f"Notice generation is not configured: {exc}"},
    )


@app.get(
    "/v1/decisions/{decision_id}/audit",
    response_model=AuditTrailResponse,
    tags=["audit"],
)
def decision_audit_trail(decision_id: str, audit: AuditDep) -> AuditTrailResponse:
    """Reconstruct every recorded step of one decision.

    This is the "why was this person declined" endpoint, answerable years
    later from the chain alone.

    Args:
        decision_id: Correlation id returned when the decision was made.
        audit: The audit repository.

    Returns:
        The records for that decision and the chain's integrity verdict.

    Raises:
        HTTPException: 404 if no records exist for that id.
    """
    records = audit.records_for_decision(decision_id)
    if not records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No audit records for decision {decision_id!r}.",
        )

    return AuditTrailResponse(
        decision_id=decision_id,
        records=tuple(
            AuditRecordResponse(
                sequence=record.sequence,
                prev_hash=record.prev_hash,
                record_hash=record.record_hash,
                payload=dict(record.payload),
            )
            for record in records
        ),
        chain_intact=audit.verify().intact,
    )


@app.get("/v1/audit/verify", response_model=ChainVerificationResponse, tags=["audit"])
def verify_audit_chain(audit: AuditDep) -> ChainVerificationResponse:
    """Recompute every hash in the chain and report whether it holds.

    Args:
        audit: The audit repository.

    Returns:
        The integrity verdict, naming the first broken record if any.
    """
    result = audit.verify()
    return ChainVerificationResponse(
        intact=result.intact,
        records_checked=result.checked,
        broken_at=result.broken_at,
        reason=result.reason,
    )


@app.get("/v1/audit/events", tags=["audit"])
def audit_event_types() -> dict[str, list[str]]:
    """List the event types the chain can record.

    Returns:
        Every recordable pipeline stage.
    """
    return {"event_types": [event.value for event in AuditEventType]}
