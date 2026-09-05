"""The underwriter review console.

Presentation only. Everything it shows is reconstructed by
:mod:`aae.console.review` from the audit chain, and everything a reviewer does
is appended to that same chain. There is no state here and no second store: a
queue that could disagree with the audit log would be a second source of truth
about what happened, and the log exists so there is only one.

Streamlit rather than a JavaScript front end because this is an internal tool
for a handful of underwriters, and the alternative is two weeks of work that
would look better and do the same thing.

Run with:

    streamlit run src/aae/console/app.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from aae.audit.repository import AuditRepository
from aae.audit.session import create_database_engine, create_session_factory
from aae.config import get_settings
from aae.console.review import ReviewQueue
from aae.database_url import describe_target, is_local_default
from aae.logging import configure_logging, get_logger

if TYPE_CHECKING:
    from aae.console.review import ReviewCase

logger = get_logger(__name__)

NOT_CONFIGURED: Final[str] = """
The database address points at **localhost**. Locally that is correct and the
database is simply not running. On a deployed console it means
`AAE_DATABASE_URL` never reached the process, so settings fell back to the
development default - and a deployed console has no Postgres of its own.

Set it in the platform's secrets as a **root-level** key, which is what makes
it arrive as an environment variable:

```toml
AAE_DATABASE_URL = "postgresql://aae_app:PASSWORD@HOST/aae?sslmode=require"
```

Use the `aae_app` role, never the owner. Postgres denies `aae_app` `UPDATE`
and `DELETE` on the audit table, which is what stops a public console from
rewriting history.
"""
"""Shown when the configured address is a loopback one."""


PAGE_TITLE = "Adverse Action Review"


@st.cache_resource
def _queue() -> ReviewQueue:
    """Build the review queue once per process."""
    settings = get_settings()
    configure_logging(settings)
    engine = create_database_engine(settings)
    return ReviewQueue(AuditRepository(create_session_factory(engine)))


def _render_sidebar(queue: ReviewQueue, pending: tuple[ReviewCase, ...]) -> str:
    """Draw the sidebar and return the reviewer's name.

    Args:
        queue: The review queue.
        pending: Cases awaiting review.

    Returns:
        The reviewer identifier to record against any action.
    """
    st.sidebar.header("Review queue")
    reviewer = st.sidebar.text_input("Reviewer", value="", placeholder="your name")

    verification = queue.repository.verify()
    st.sidebar.metric("Awaiting review", len(pending))
    st.sidebar.metric("Escalated", sum(1 for case in pending if case.escalated))
    st.sidebar.metric("Audit records", verification.checked)

    if verification.intact:
        st.sidebar.success("Audit chain intact")
    else:
        # Not a warning. A broken chain means recorded history cannot be
        # trusted, and nothing on this screen should be acted on until it is
        # explained.
        st.sidebar.error(
            f"AUDIT CHAIN BROKEN at record {verification.broken_at}: {verification.reason}"
        )

    return reviewer.strip()


def _render_case(case: ReviewCase) -> None:
    """Draw one case for review.

    Args:
        case: The case to display.
    """
    st.subheader(f"Application {case.application_id}")

    left, middle, right = st.columns(3)
    left.metric("Probability of default", f"{case.probability_default:.1%}")
    middle.metric("Threshold", f"{case.threshold:.0%}")
    right.metric("Attempts", case.attempts)

    st.caption(
        f"Model {case.model_version} · provider {case.provider} · "
        f"{case.record_count} audit records · decision {case.decision_id}"
    )

    if case.escalated:
        st.error(f"Escalated: {case.escalation_reason}")
        if case.violations:
            st.write("Verification failed on:")
            for violation in case.violations:
                st.write(f"- {violation}")

    st.markdown("#### Factors behind the decision")
    st.dataframe(
        [
            {
                "Rank": factor.rank,
                "Factor": factor.display_name,
                "Applicant value": factor.value,
                "Direction": factor.direction,
            }
            for factor in case.factors
        ],
        hide_index=True,
        width="stretch",
    )

    if case.reasons:
        st.markdown("#### Reasons given")
        for reason in case.reasons:
            st.write(f"- {reason.text}")
            st.caption(f"  grounded in {reason.factor_id}")

    if case.citations:
        st.markdown("#### Regulatory basis")
        for section, span in case.citations:
            st.write(f"- Section {section}: “{span}”")

    st.markdown("#### Notice as generated")
    if case.body:
        st.text_area("Body", value=case.body, height=260, key=f"body-{case.decision_id}")
    else:
        st.info("No letter was produced. This case could not be verified and needs writing.")


def _render_actions(queue: ReviewQueue, case: ReviewCase, reviewer: str) -> None:
    """Draw the sign-off controls.

    Args:
        queue: The review queue.
        case: The case under review.
        reviewer: Who is reviewing.
    """
    st.markdown("#### Sign-off")

    if not reviewer:
        st.warning("Enter your name in the sidebar before recording a decision.")
        return

    note = st.text_area("Note", key=f"note-{case.decision_id}", height=80)
    approve, edit, reject = st.columns(3)

    if approve.button("Approve as written", key=f"approve-{case.decision_id}"):
        if case.body is None:
            st.error("There is no letter to approve.")
        else:
            queue.approve(case, reviewer, note or None)
            st.success("Approved and recorded in the audit chain.")
            st.rerun()

    if edit.button("Save edit and issue", key=f"edit-{case.decision_id}"):
        edited = st.session_state.get(f"body-{case.decision_id}", "")
        if not edited.strip():
            st.error("The letter cannot be empty.")
        elif edited == case.body:
            st.warning("The letter is unchanged. Use approve instead.")
        else:
            queue.edit(case, reviewer, edited, note or None)
            st.success("Edit recorded. Both versions remain in the chain.")
            st.rerun()

    if reject.button("Reject", key=f"reject-{case.decision_id}"):
        if not note.strip():
            # Required, not merely encouraged: a rejection without a reason
            # teaches nobody anything and cannot become evaluation data.
            st.error("A rejection needs a reason.")
        else:
            queue.reject(case, reviewer, note)
            st.success("Rejected and recorded.")
            st.rerun()


def _render_connection_failure(exc: Exception) -> None:
    """Explain a failed database connection without leaking the credential.

    Streamlit redacts any exception that might carry a password, so a deployed
    app shows a stack trace ending in ``OperationalError`` and nothing about
    what went wrong. The two likely causes look identical there and need
    opposite fixes, so they are named here instead.

    Args:
        exc: The connection failure.
    """
    settings = get_settings()
    target = describe_target(settings.database_url)

    st.error(f"Cannot reach the audit database at {target}.")

    if is_local_default(settings.database_url):
        st.markdown(
            NOT_CONFIGURED,
        )
    else:
        st.markdown(
            "The address is configured, so the credential, the network path or "
            "TLS is at fault. Check the password, and that the connection string "
            "ends in `?sslmode=require` - most managed Postgres refuses a "
            "plaintext connection and reports it as a connection failure rather "
            "than a TLS one."
        )

    logger.error("console_database_unreachable", target=target, error=str(exc))


def main() -> None:
    """Draw the console."""
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title(PAGE_TITLE)

    try:
        queue = _queue()
        pending = queue.pending()
    except SQLAlchemyError as exc:
        _queue.clear()
        _render_connection_failure(exc)
        return
    reviewer = _render_sidebar(queue, pending)

    if not pending:
        st.success("Nothing awaiting review.")
        return

    labels = [case.headline for case in pending]
    chosen = st.selectbox(
        "Select a case", options=range(len(pending)), format_func=labels.__getitem__
    )

    case = pending[chosen]
    _render_case(case)
    _render_actions(queue, case, reviewer)


if __name__ == "__main__":
    main()
