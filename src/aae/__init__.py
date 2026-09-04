"""Adverse Action Engine: verifiable credit denial notices.

Generates regulator-compliant adverse action notices for declined credit
applications, and mechanically verifies every claim in them against the
model's real feature values, SHAP attributions, and the source regulation
corpus before a human ever sees the output.
"""

__version__ = "0.1.0"
