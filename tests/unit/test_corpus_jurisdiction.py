"""The regulation corpus and the jurisdiction definitions.

Citation checking is only as good as the corpus behind it. If a provision can
be registered twice, or its text quietly paraphrased, a fabricated quotation
becomes indistinguishable from a real one.
"""

from __future__ import annotations

import pytest

from aae.jurisdiction.base import (
    COMMON_PROHIBITED_PATTERNS,
    Jurisdiction,
    RequiredElement,
    compile_prohibited,
)
from aae.jurisdiction.india_rbi import INDIA_RBI
from aae.retrieval.corpus import (
    INDIA_RBI_PROVISIONS,
    RBI_FAIR_PRACTICES_CODE,
    InMemoryCorpus,
    Provision,
    india_rbi_corpus,
)


class TestInMemoryCorpus:
    def test_resolves_a_known_provision(self):
        corpus = india_rbi_corpus()
        passage = corpus.passage(RBI_FAIR_PRACTICES_CODE, "2.3")
        assert passage is not None
        assert "reasons" in passage

    def test_returns_none_for_an_unknown_section(self):
        assert india_rbi_corpus().passage(RBI_FAIR_PRACTICES_CODE, "99.9") is None

    def test_returns_none_for_an_unknown_document(self):
        assert india_rbi_corpus().passage("some-other-act", "1") is None

    def test_exposes_the_whole_provision(self):
        provision = india_rbi_corpus().provision(RBI_FAIR_PRACTICES_CODE, "6.1")
        assert provision is not None
        assert provision.title == "Grievance redressal mechanism"
        assert provision.key == (RBI_FAIR_PRACTICES_CODE, "6.1")

    def test_provision_lookup_returns_none_when_absent(self):
        assert india_rbi_corpus().provision(RBI_FAIR_PRACTICES_CODE, "99.9") is None

    def test_is_sized_and_iterable(self):
        corpus = india_rbi_corpus()
        assert len(corpus) == len(INDIA_RBI_PROVISIONS)
        assert {p.section for p in corpus} == {p.section for p in INDIA_RBI_PROVISIONS}

    def test_rejects_a_duplicate_provision(self):
        """Two texts under one reference would make a citation ambiguous.

        A fabricated quotation could then match whichever copy happened to be
        stored, which defeats the check entirely.
        """
        duplicate = Provision(document_id="doc", section="1", text="first")
        clashing = Provision(document_id="doc", section="1", text="second")
        with pytest.raises(ValueError, match="Duplicate provision"):
            InMemoryCorpus([duplicate, clashing])

    def test_an_empty_corpus_resolves_nothing(self):
        assert InMemoryCorpus().passage("doc", "1") is None
        assert len(InMemoryCorpus()) == 0


class TestIndiaRbiJurisdiction:
    def test_caps_principal_reasons(self):
        assert INDIA_RBI.max_principal_reasons == 4

    def test_requires_reasons_and_a_regulatory_basis(self):
        assert {"principal_reasons", "regulatory_basis"} <= INDIA_RBI.required_keys

    def test_element_lookup(self):
        element = INDIA_RBI.element("principal_reasons")
        assert element is not None
        assert element.checkable_structurally

    def test_element_lookup_returns_none_for_an_unknown_key(self):
        assert INDIA_RBI.element("not_a_real_element") is None

    def test_prose_elements_are_marked_as_such(self):
        """They cannot be checked structurally, and must not pretend to be."""
        grievance = INDIA_RBI.element("grievance_contact")
        assert grievance is not None
        assert not grievance.checkable_structurally

    def test_prohibited_patterns_are_compiled_case_insensitively(self):
        patterns = compile_prohibited((r"\bmarried\b",))
        assert patterns[0].search("You are MARRIED.") is not None

    def test_every_common_pattern_compiles(self):
        assert len(compile_prohibited(COMMON_PROHIBITED_PATTERNS)) == len(
            COMMON_PROHIBITED_PATTERNS
        )


class TestRequiredElement:
    def test_a_prose_element_falls_back_to_the_declaration(self):
        element = RequiredElement(key="contact", description="Contact details.")
        assert not element.checkable_structurally

    def test_a_structural_element_ignores_the_declaration(self):
        """Declaring an element must not substitute for providing it."""
        element = RequiredElement(
            key="reasons",
            description="Reasons.",
            predicate=lambda notice: len(notice.principal_reasons) > 2,
        )
        assert element.checkable_structurally


class TestJurisdictionShape:
    def test_a_jurisdiction_can_be_defined_from_scratch(self):
        """The interface must support a second regulator without changes."""
        other = Jurisdiction(
            code="test_regime",
            name="Test Regime",
            max_principal_reasons=2,
            required_elements=(RequiredElement(key="reasons", description="Reasons."),),
            prohibited_patterns=compile_prohibited((r"\bforbidden\b",)),
            prohibited_description="a prohibited basis",
        )
        assert other.required_keys == {"reasons"}
        assert other.prohibited_patterns[0].search("this is Forbidden") is not None


class TestUsRegulationB:
    """The second jurisdiction, which is the point of having an interface.

    If adding it had required changing the base module, the abstraction would
    have been wishful thinking.
    """

    def test_uses_the_same_jurisdiction_type(self):
        from aae.jurisdiction.base import Jurisdiction
        from aae.jurisdiction.us_regb import US_REG_B

        assert isinstance(US_REG_B, Jurisdiction)

    def test_caps_reasons_at_four(self):
        """12 CFR 1002.9, Official Interpretation: four is enough."""
        from aae.jurisdiction.us_regb import US_REG_B

        assert US_REG_B.max_principal_reasons == 4

    def test_requires_elements_india_does_not(self):
        from aae.jurisdiction.india_rbi import INDIA_RBI
        from aae.jurisdiction.us_regb import US_REG_B

        assert US_REG_B.required_keys - INDIA_RBI.required_keys == {
            "ecoa_notice",
            "creditor_contact",
        }

    def test_the_ecoa_notice_is_a_prose_element(self):
        """It is required statutory text, checked against the letter."""
        from aae.jurisdiction.us_regb import US_REG_B

        element = US_REG_B.element("ecoa_notice")
        assert element is not None
        assert not element.checkable_structurally

    @pytest.mark.parametrize(
        "text",
        [
            "You receive public assistance.",
            "Applicants on welfare are higher risk.",
            "Your color was considered.",
            "You exercised rights under the Act.",
        ],
    )
    def test_catches_prohibited_bases_specific_to_ecoa(self, text: str):
        """Receipt of public assistance is the one people forget.

        It does not look like a demographic characteristic, and it is a
        prohibited basis under ECOA all the same.
        """
        from aae.jurisdiction.us_regb import US_REG_B

        assert any(pattern.search(text) for pattern in US_REG_B.prohibited_patterns)

    def test_still_catches_the_shared_prohibited_bases(self):
        from aae.jurisdiction.us_regb import US_REG_B

        assert any(
            pattern.search("Your marital status was considered.")
            for pattern in US_REG_B.prohibited_patterns
        )


class TestUsCorpus:
    def test_provisions_resolve(self):
        from aae.retrieval.corpus import REGULATION_B, us_reg_b_corpus

        corpus = us_reg_b_corpus()
        passage = corpus.passage(REGULATION_B, "1002.9(b)(2)")
        assert passage is not None
        assert "principal reason or reasons" in passage

    def test_carries_the_provision_that_makes_this_project_necessary(self):
        """1002.9(b)(2) says a score alone is not a sufficient reason.

        Which is precisely what the verifier enforces: a notice must name the
        factors, checked against the model's own attributions.
        """
        from aae.retrieval.corpus import REGULATION_B, us_reg_b_corpus

        passage = us_reg_b_corpus().passage(REGULATION_B, "1002.9(b)(2)")
        assert passage is not None
        assert "insufficient" in passage

    def test_the_two_corpora_are_separate(self):
        from aae.retrieval.corpus import (
            RBI_FAIR_PRACTICES_CODE,
            REGULATION_B,
            india_rbi_corpus,
            us_reg_b_corpus,
        )

        assert us_reg_b_corpus().passage(RBI_FAIR_PRACTICES_CODE, "2.3") is None
        assert india_rbi_corpus().passage(REGULATION_B, "1002.9(a)(2)") is None
