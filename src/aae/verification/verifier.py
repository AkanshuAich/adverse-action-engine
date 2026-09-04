"""The verifier.

Runs every check against a generated notice and reports what is wrong with it.
This is the component the project exists to demonstrate, so it is worth being
precise about what it does and does not claim.

It claims that each assertion in a notice agrees with evidence the system
already holds: the feature values that were scored, the SHAP attributions that
explain the score, the corpus text a citation quotes, and the regulator's
requirements. Because that evidence is held independently of the generator,
the check is deterministic and repeatable, and its result means something.

It does not claim the notice is well written, kind, or persuasive. Those are
real qualities and are not verifiable this way; a language model judging them
is judging taste, and dressing that up as a groundedness score would be worse
than not measuring it.

Failures come in two kinds, deliberately separated. A **violation** is a way
the notice is wrong that a regeneration could fix, and is fed back into the
repair prompt. A **precondition failure** means the notice is not about this
decision at all - wrong applicant, wrong jurisdiction, an approved application
- and raises, because rewriting cannot fix a bug and retrying would only burn
attempts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aae.domain.errors import VerificationPreconditionError
from aae.domain.models import Decision, VerificationResult
from aae.logging import get_logger
from aae.verification.rules import (
    DEFAULT_POLICY,
    VerificationPolicy,
    check_citation_validity,
    check_element_coverage,
    check_factor_grounding,
    check_prohibited_content,
    check_reason_count,
    check_value_accuracy,
)

if TYPE_CHECKING:
    from aae.domain.models import AdverseActionNotice, CreditDecision, Violation
    from aae.jurisdiction.base import CorpusLookup, Jurisdiction

logger = get_logger(__name__)


class NoticeVerifier:
    """Checks a generated notice against the decision it claims to explain."""

    def __init__(
        self,
        jurisdiction: Jurisdiction,
        corpus: CorpusLookup,
        policy: VerificationPolicy = DEFAULT_POLICY,
    ) -> None:
        """Build the verifier.

        Args:
            jurisdiction: The governing rules.
            corpus: Resolves citations to the text they quote.
            policy: Tolerances applied to numeric claims.
        """
        self._jurisdiction = jurisdiction
        self._corpus = corpus
        self._policy = policy

    @property
    def jurisdiction(self) -> Jurisdiction:
        """The jurisdiction this verifier enforces."""
        return self._jurisdiction

    def _check_preconditions(self, notice: AdverseActionNotice, decision: CreditDecision) -> None:
        """Reject notices that do not belong to this decision.

        Args:
            notice: The generated notice.
            decision: The decision it claims to explain.

        Raises:
            VerificationPreconditionError: On any mismatch.
        """
        if notice.application_id != decision.application_id:
            msg = (
                f"Notice is for application {notice.application_id!r} but the decision "
                f"is for {decision.application_id!r}."
            )
            raise VerificationPreconditionError(msg)

        if notice.jurisdiction != self._jurisdiction.code:
            msg = (
                f"Notice declares jurisdiction {notice.jurisdiction!r} but this "
                f"verifier enforces {self._jurisdiction.code!r}."
            )
            raise VerificationPreconditionError(msg)

        if decision.decision is not Decision.DECLINE:
            msg = (
                "An adverse action notice was generated for an application that was "
                f"{decision.decision.value}d. There is no adverse action to explain."
            )
            raise VerificationPreconditionError(msg)

    def verify(
        self,
        notice: AdverseActionNotice,
        decision: CreditDecision,
        *,
        attempt: int = 1,
    ) -> VerificationResult:
        """Run every check and report the outcome.

        All checks run even after the first failure. A repair prompt that is
        told about one problem at a time takes as many attempts as there are
        problems, and each attempt is a model call that can introduce a new
        one.

        Args:
            notice: The generated notice.
            decision: The decision it claims to explain.
            attempt: Which generation attempt this is, recorded in the result.

        Returns:
            Whether the notice passed, and every way it did not.

        Raises:
            VerificationPreconditionError: If the notice does not belong to
                this decision.
        """
        self._check_preconditions(notice, decision)

        violations: tuple[Violation, ...] = (
            *check_factor_grounding(notice, decision),
            *check_value_accuracy(notice, decision, self._policy),
            *check_citation_validity(notice, self._corpus),
            *check_element_coverage(notice, self._jurisdiction),
            *check_prohibited_content(notice, self._jurisdiction),
            *check_reason_count(notice, self._jurisdiction),
        )

        result = VerificationResult(passed=not violations, violations=violations, attempt=attempt)

        if result.passed:
            logger.info(
                "notice_verified",
                application_id=notice.application_id,
                attempt=attempt,
                reasons=len(notice.principal_reasons),
            )
        else:
            logger.warning(
                "notice_rejected",
                application_id=notice.application_id,
                attempt=attempt,
                violations=result.rendered_violations(),
            )

        return result
