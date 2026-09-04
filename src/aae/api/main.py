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

from aae.api.deps import get_audit_repository, get_decision_engine
from aae.api.schemas import (
    ApplicationRequest,
    AuditRecordResponse,
    AuditTrailResponse,
    ChainVerificationResponse,
    DecisionResponse,
    HealthResponse,
)
from aae.audit.models import AuditEventType
from aae.config import get_settings
from aae.data.schema import validate_for_scoring
from aae.domain.errors import AuditIntegrityError, DataValidationError
from aae.logging import bind_correlation_id, configure_logging, get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from aae.audit.repository import AuditRepository
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
