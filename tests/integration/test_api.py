"""The decision API end to end, against a real database.

The load-bearing assertion is that a decision and its audit record are
inseparable: the endpoint must not be able to return a decision it failed to
record. Everything else here is ordinary contract testing.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from aae.api.deps import get_audit_repository, get_decision_engine
from aae.api.main import app
from aae.audit.repository import AuditRepository
from aae.audit.session import create_session_factory
from aae.data.loaders import load_applications
from aae.data.schema import SCORING_SCHEMA
from aae.ml.decision import DecisionEngine
from aae.ml.train import train_model

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def engine():
    loaded = load_applications(force_synthetic=True, n_synthetic=8_000)
    return DecisionEngine(train_model(loaded), threshold=0.15), loaded.frame


@pytest.fixture
def client(engine, migrated_db, owner_connection):
    owner_connection.execute("TRUNCATE audit_record")
    decision_engine, _ = engine

    url = (
        f"postgresql+psycopg://aae_app:app_test_password"
        f"@{migrated_db.get_container_host_ip()}:{migrated_db.get_exposed_port(5432)}/aae"
    )
    repository = AuditRepository(create_session_factory(create_engine(url, pool_pre_ping=True)))

    app.dependency_overrides[get_decision_engine] = lambda: decision_engine
    app.dependency_overrides[get_audit_repository] = lambda: repository
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def application(engine) -> dict[str, object]:
    _, frame = engine
    row = frame.head(1)[list(SCORING_SCHEMA.columns)].iloc[0]
    payload: dict[str, object] = {}
    for name, value in row.items():
        # value != value is the NaN test; pandas NaN must become JSON null.
        payload[str(name)] = None if value != value else value
    payload["SK_ID_CURR"] = int(payload["SK_ID_CURR"])  # type: ignore[arg-type]
    payload["CNT_CHILDREN"] = int(payload["CNT_CHILDREN"])  # type: ignore[arg-type]
    payload["DAYS_EMPLOYED"] = int(payload["DAYS_EMPLOYED"])  # type: ignore[arg-type]
    payload["REGION_RATING_CLIENT"] = int(payload["REGION_RATING_CLIENT"])  # type: ignore[arg-type]
    return payload


class TestHealth:
    def test_reports_the_model_and_chain_state(self, client: TestClient):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["model_version"].startswith("xgb-")
        assert body["chain_intact"] is True
        assert body["audit_records"] == 0


class TestCreateDecision:
    def test_returns_a_decision_with_ranked_factors(self, client, application):
        response = client.post("/v1/decisions", json=application)
        assert response.status_code == 201, response.text

        body = response.json()
        assert body["decision"] in {"approve", "decline"}
        assert 0.0 < body["probability_default"] < 1.0
        assert body["factors"]
        assert [f["rank"] for f in body["factors"]] == list(range(1, len(body["factors"]) + 1))

    def test_factors_carry_plain_language_names(self, client, application):
        """A customer notice cannot say "EXT_SOURCE_2"."""
        body = client.post("/v1/decisions", json=application).json()
        for factor in body["factors"]:
            assert factor["display_name"] != factor["factor_id"]
            assert "_" not in factor["display_name"]

    def test_the_decision_is_recorded_before_it_is_returned(self, client, application):
        """A returned decision must already exist in the chain.

        This is the guarantee the endpoint exists to keep: an unrecorded
        decision is precisely the unauditable outcome the system prevents.
        """
        body = client.post("/v1/decisions", json=application).json()

        trail = client.get(f"/v1/decisions/{body['decision_id']}/audit")
        assert trail.status_code == 200

        records = trail.json()["records"]
        assert len(records) == 1
        assert records[0]["sequence"] == body["audit_sequence"]
        assert records[0]["payload"]["model_version"] == body["model_version"]

    def test_audit_payload_can_reconstruct_the_decision(self, client, application):
        body = client.post("/v1/decisions", json=application).json()
        payload = client.get(f"/v1/decisions/{body['decision_id']}/audit").json()["records"][0][
            "payload"
        ]

        assert payload["decision"] == body["decision"]
        assert payload["threshold"] == body["threshold"]
        assert payload["feature_values"]
        assert len(payload["factors"]) == len(body["factors"])

    def test_successive_decisions_extend_one_chain(self, client, application):
        first = client.post("/v1/decisions", json=application).json()
        second_application = {**application, "SK_ID_CURR": application["SK_ID_CURR"] + 1}  # type: ignore[operator]
        second = client.post("/v1/decisions", json=second_application).json()

        assert second["audit_sequence"] == first["audit_sequence"] + 1
        assert client.get("/v1/audit/verify").json()["intact"] is True


class TestValidation:
    def test_rejects_a_protected_attribute(self, client, application):
        """The API will not accept sex, age, or marital status at all.

        They cannot lawfully influence the outcome, so there is no reason to
        collect them to score one application.
        """
        for protected in ("CODE_GENDER", "DAYS_BIRTH", "NAME_FAMILY_STATUS"):
            response = client.post("/v1/decisions", json={**application, protected: "X"})
            assert response.status_code == 422, protected

    def test_rejects_a_negative_income(self, client, application):
        response = client.post("/v1/decisions", json={**application, "AMT_INCOME_TOTAL": -5.0})
        assert response.status_code == 422

    def test_rejects_an_out_of_range_bureau_score(self, client, application):
        response = client.post("/v1/decisions", json={**application, "EXT_SOURCE_2": 1.5})
        assert response.status_code == 422

    def test_rejects_a_missing_required_field(self, client, application):
        incomplete = {k: v for k, v in application.items() if k != "AMT_CREDIT"}
        assert client.post("/v1/decisions", json=incomplete).status_code == 422

    def test_nothing_is_recorded_for_a_rejected_application(self, client, application):
        client.post("/v1/decisions", json={**application, "AMT_INCOME_TOTAL": -5.0})
        assert client.get("/v1/audit/verify").json()["records_checked"] == 0


class TestAuditEndpoints:
    def test_unknown_decision_is_not_found(self, client: TestClient):
        assert client.get("/v1/decisions/DEC-nope/audit").status_code == 404

    def test_verify_reports_an_intact_empty_chain(self, client: TestClient):
        body = client.get("/v1/audit/verify").json()
        assert body["intact"] is True
        assert body["records_checked"] == 0
        assert body["broken_at"] is None

    def test_event_types_are_listed(self, client: TestClient):
        events = client.get("/v1/audit/events").json()["event_types"]
        assert "decision_scored" in events
        assert "human_reviewed" in events


class TestCreateNotice:
    """The full pipeline: score, explain, generate, verify, record."""

    @pytest.fixture
    def notice_client(self, engine, migrated_db, owner_connection):
        from aae.api.deps import get_notice_generator
        from aae.generation.graph import NoticeGenerator
        from aae.generation.providers.stub import ScriptedProvider
        from aae.generation.schemas import (
            RenderedBody,
            SelectedCitation,
            SelectedNotice,
            SelectedReason,
        )
        from aae.jurisdiction.india_rbi import INDIA_RBI
        from aae.retrieval.corpus import RBI_FAIR_PRACTICES_CODE, india_rbi_corpus
        from aae.verification.verifier import NoticeVerifier

        owner_connection.execute("TRUNCATE audit_record")
        decision_engine, frame = engine

        # A declined application, so there is an adverse action to explain.
        declined = None
        for index in range(len(frame)):
            candidate = decision_engine.decide(frame, row=index)
            if candidate.decision.value == "decline":
                declined = (index, candidate)
                break
        assert declined is not None, "expected a decline in the sample"
        row_index, decision = declined

        adverse = decision.adverse_factors()
        selection = SelectedNotice(
            principal_reasons=[
                SelectedReason(
                    factor_id=factor.factor_id,
                    text=f"Your {factor.display_name.lower()} did not meet our requirement.",
                )
                for factor in adverse[:2]
            ],
            citations=[
                SelectedCitation(
                    document_id=RBI_FAIR_PRACTICES_CODE,
                    section="2.3",
                    quoted_span="convey in writing to the applicant the reasons",
                )
            ],
            included_elements=[
                "principal_reasons",
                "regulatory_basis",
                "decision_statement",
                "grievance_contact",
            ],
        )
        body = RenderedBody(
            body=(
                "Dear applicant, we are unable to approve your application. "
                "Please contact our grievance officer for clarification."
            )
        )

        url = (
            f"postgresql+psycopg://aae_app:app_test_password"
            f"@{migrated_db.get_container_host_ip()}:{migrated_db.get_exposed_port(5432)}/aae"
        )
        repository = AuditRepository(create_session_factory(create_engine(url, pool_pre_ping=True)))
        generator = NoticeGenerator(
            provider=ScriptedProvider([selection, body]),
            verifier=NoticeVerifier(INDIA_RBI, india_rbi_corpus()),
        )

        app.dependency_overrides[get_decision_engine] = lambda: decision_engine
        app.dependency_overrides[get_audit_repository] = lambda: repository
        app.dependency_overrides[get_notice_generator] = lambda: generator
        with TestClient(app) as test_client:
            yield test_client, frame.iloc[[row_index]]
        app.dependency_overrides.clear()

    @staticmethod
    def _payload(row) -> dict[str, object]:
        payload = {}
        for name, value in row[list(SCORING_SCHEMA.columns)].iloc[0].items():
            payload[str(name)] = None if value != value else value
        for integer_field in (
            "SK_ID_CURR",
            "CNT_CHILDREN",
            "DAYS_EMPLOYED",
            "REGION_RATING_CLIENT",
        ):
            payload[integer_field] = int(payload[integer_field])
        return payload

    def test_a_declined_application_gets_a_verified_notice(self, notice_client):
        client, row = notice_client
        response = client.post("/v1/notices", json=self._payload(row))
        assert response.status_code == 201, response.text

        body = response.json()
        assert body["issued"] is True
        assert body["escalated"] is False
        assert body["body"]
        assert body["reasons"]
        assert body["citations"]

    def test_the_decision_and_the_notice_are_recorded_together(self, notice_client):
        """A partial chain looks like evidence and is not."""
        client, row = notice_client
        body = client.post("/v1/notices", json=self._payload(row)).json()

        assert len(body["audit_sequences"]) == 2
        trail = client.get(f"/v1/decisions/{body['decision_id']}/audit").json()
        assert len(trail["records"]) == 2
        assert trail["chain_intact"] is True

    def test_the_audit_records_the_generation_trace(self, notice_client):
        client, row = notice_client
        body = client.post("/v1/notices", json=self._payload(row)).json()
        records = client.get(f"/v1/decisions/{body['decision_id']}/audit").json()["records"]

        generation = records[1]["payload"]
        assert generation["issued"] is True
        assert generation["attempts"] == 1
        assert [step["node"] for step in generation["trace"]] == [
            "select",
            "verify",
            "render",
            "check_prose",
        ]
        assert generation["trace"][0]["prompt_hash"]

    def test_an_approved_application_is_refused(self, engine, migrated_db, owner_connection):
        """There is no adverse action to explain."""
        from aae.api.deps import get_notice_generator

        owner_connection.execute("TRUNCATE audit_record")
        decision_engine, frame = engine

        approved = None
        for index in range(len(frame)):
            candidate = decision_engine.decide(frame, row=index)
            if candidate.decision.value == "approve":
                approved = frame.iloc[[index]]
                break
        assert approved is not None

        url = (
            f"postgresql+psycopg://aae_app:app_test_password"
            f"@{migrated_db.get_container_host_ip()}:{migrated_db.get_exposed_port(5432)}/aae"
        )
        repository = AuditRepository(create_session_factory(create_engine(url, pool_pre_ping=True)))
        app.dependency_overrides[get_decision_engine] = lambda: decision_engine
        app.dependency_overrides[get_audit_repository] = lambda: repository
        app.dependency_overrides[get_notice_generator] = lambda: None

        with TestClient(app) as test_client:
            response = test_client.post("/v1/notices", json=self._payload(approved))
        app.dependency_overrides.clear()

        assert response.status_code == 409
        assert "approved" in response.json()["detail"]
