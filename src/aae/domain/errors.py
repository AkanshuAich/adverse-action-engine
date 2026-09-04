"""Exception hierarchy for the Adverse Action Engine.

Every exception raised by this package derives from :class:`AAEError`, so
callers can distinguish our failures from those of third-party libraries.
"""


class AAEError(Exception):
    """Base class for every error raised by this package."""


class ConfigurationError(AAEError):
    """Configuration is missing or internally inconsistent."""


class DataValidationError(AAEError):
    """Input data failed schema validation at a module boundary."""


class ModelError(AAEError):
    """A model artifact could not be loaded, or scoring failed."""


class FairLendingViolationError(AAEError):
    """A protected attribute was about to influence a credit decision.

    Raised at feature-specification time rather than at scoring time, so the
    failure happens while building the model rather than while denying someone
    credit. Covers both direct use of a protected column and proxy use through
    a derived feature that depends on one.
    """


class RetrievalError(AAEError):
    """The regulation corpus could not be searched."""


class GenerationError(AAEError):
    """The language model failed to produce usable structured output."""


class ProviderError(GenerationError):
    """An LLM provider call failed or was rate limited."""


class VerificationFailedError(AAEError):
    """A notice failed verification after the maximum repair attempts.

    This is not a bug: it is the designed escalation path to a human
    reviewer. It carries the violations so they can be recorded.
    """

    def __init__(self, message: str, violations: list[str]) -> None:
        """Initialise the error.

        Args:
            message: Human-readable summary.
            violations: Rendered violation descriptions, for the audit record.
        """
        super().__init__(message)
        self.violations = violations


class AuditIntegrityError(AAEError):
    """The audit log hash chain is broken or a record could not be written."""
