"""The generated governance documents.

A model card that silently renders a broken table, or drops the section saying
what was *not* validated, is worse than none: it is read as a statement of
diligence. These assert the parts a reviewer would look for are actually there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aae.data.loaders import load_applications
from aae.ml.decision import DecisionEngine
from aae.ml.features import PROTECTED_ATTRIBUTES
from aae.ml.model_card import render_model_card, render_validation_report, write_documents
from aae.ml.reports import (
    build_drift_report,
    build_fairness_report,
    render_drift,
    render_fairness,
    write_report,
)
from aae.ml.train import train_model


@pytest.fixture(scope="module")
def artefacts():
    loaded = load_applications(force_synthetic=True, n_synthetic=8_000)
    model = train_model(loaded)
    engine = DecisionEngine(model, threshold=0.15)

    reference = loaded.frame.head(600)
    fairness = build_fairness_report(engine, reference)
    drift = build_drift_report(engine, reference, loaded.frame.tail(600))
    return model, fairness, drift


class TestReportBuilders:
    def test_fairness_covers_every_protected_attribute(self, artefacts):
        _, fairness, _ = artefacts
        assert {group.attribute for group in fairness.groups} == PROTECTED_ATTRIBUTES

    def test_drift_compares_the_two_populations(self, artefacts):
        _, _, drift = artefacts
        assert drift.reference_rows == 600
        assert drift.current_rows == 600

    def test_fairness_renders_with_group_sizes(self, artefacts):
        """A ratio without its group size invites acting on noise."""
        _, fairness, _ = artefacts
        rendered = render_fairness(fairness)
        assert "adverse impact ratio" in rendered
        assert "n=" in rendered

    def test_drift_renders_a_verdict(self, artefacts):
        _, _, drift = artefacts
        rendered = render_drift(drift)
        assert "Score PSI" in rendered
        assert "stable band" in rendered or "Requires attention" in rendered

    def test_a_report_is_written_as_json(self, artefacts, tmp_path):
        import json

        _, fairness, _ = artefacts
        path = write_report("fairness", fairness.to_dict(), directory=tmp_path)
        payload = json.loads(path.read_text(encoding="utf-8"))

        assert "generated_at" in payload
        assert payload["model_version"] == fairness.model_version


class TestModelCard:
    def test_names_the_model_version(self, artefacts):
        model, fairness, _ = artefacts
        assert model.model_version in render_model_card(model, fairness)

    def test_states_the_training_data_is_synthetic(self, artefacts):
        """The most material fact about this model, and easy to omit."""
        model, fairness, _ = artefacts
        card = render_model_card(model, fairness)
        assert "synthetic" in card
        assert "Any real lending decision" in card

    def test_names_every_protected_attribute_as_excluded(self, artefacts):
        model, fairness, _ = artefacts
        card = render_model_card(model, fairness)
        assert "No protected attribute is a feature" in card
        for attribute in PROTECTED_ATTRIBUTES:
            assert attribute in card

    def test_carries_the_mitigation_position(self, artefacts):
        model, fairness, _ = artefacts
        assert "unlawful" in render_model_card(model, fairness)

    def test_reports_calibration_honestly(self, artefacts):
        """Whether isotonic was accepted or rejected, the card says which."""
        model, fairness, _ = artefacts
        card = render_model_card(model, fairness)
        expected = "accepted" if model.metrics.calibration_applied else "identity mapping"
        assert expected in card

    def test_tables_are_well_formed(self, artefacts):
        model, fairness, _ = artefacts
        for line in render_model_card(model, fairness).splitlines():
            if line.startswith("|") and not set(line) <= set("|- "):
                assert line.rstrip().endswith("|"), line


class TestValidationReport:
    def test_states_what_was_not_validated(self, artefacts):
        """A report listing only successful checks is a sales brochure."""
        model, fairness, drift = artefacts
        report = render_validation_report(model, fairness, drift)

        assert "## What is not validated" in report
        assert "Real-portfolio behaviour" in report
        assert "Live model quality" in report

    def test_distinguishes_system_figures_from_model_quality(self, artefacts):
        model, fairness, drift = artefacts
        report = render_validation_report(
            model, fairness, drift, {"groundedness_rate": 0.54, "prohibited_content_rate": 0.0}
        )
        assert "not** a claim" in report
        assert "0.5400" in report

    def test_notes_when_evaluation_figures_are_missing(self, artefacts):
        model, fairness, drift = artefacts
        assert "unavailable" in render_validation_report(model, fairness, drift)

    def test_lists_conditions_for_deployment(self, artefacts):
        model, fairness, drift = artefacts
        report = render_validation_report(model, fairness, drift)
        assert "Conditions for deployment" in report
        assert "Retrain and revalidate on real portfolio data" in report

    def test_records_the_small_group_caveat(self, artefacts):
        """The age ratio is driven by a band of a few dozen applicants."""
        model, fairness, drift = artefacts
        assert "noise rather than" in render_validation_report(model, fairness, drift)


class TestWritingBothDocuments:
    def test_writes_both(self, artefacts, tmp_path, monkeypatch):
        model, fairness, drift = artefacts
        monkeypatch.chdir(tmp_path)
        card, report = write_documents(model, fairness, drift)

        assert card.read_text(encoding="utf-8").startswith("# Model card")
        assert report.read_text(encoding="utf-8").startswith("# Validation report")


class TestFairnessOnConstructedData:
    def test_a_deliberate_disparity_reaches_the_documents(self, artefacts):
        """The card must show a finding when there is one to show."""
        model, _, drift = artefacts

        frame = pd.DataFrame(
            {
                "CODE_GENDER": ["F"] * 100 + ["M"] * 100,
                "DAYS_BIRTH": [-14_000] * 200,
                "NAME_FAMILY_STATUS": ["Married"] * 200,
            }
        )
        declined = np.array([1] * 60 + [0] * 40 + [1] * 5 + [0] * 95)

        from aae.ml.fairness import analyse_fairness

        skewed = analyse_fairness(frame, declined, model_version=model.model_version)
        assert skewed.findings

        card = render_model_card(model, skewed)
        assert "below screen" in card

        report = render_validation_report(model, skewed, drift)
        assert "fell below the four-fifths screen" in report
