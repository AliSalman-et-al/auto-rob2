from rob2_pipeline.models import empty_paper_evidence
from rob2_pipeline.nodes.verification import (
    quote_is_supported,
    quote_verifier_node,
    verify_sq_evidence,
)


def test_quote_support_accepts_exact_source_quote():
    source = "Participants were randomly assigned using a central web system."

    assert quote_is_supported(
        '"Participants were randomly assigned"', source.casefold()
    )


def test_quote_support_accepts_traceable_ellipsis_fragments():
    source = (
        "Discontinued study drug (n = 135) Discontinued study drug (n = 242) "
        "Progressivedisease† (n = 65; 11.3%) "
        "Progressivedisease† (n = 171; 29.7%)"
    )

    assert quote_is_supported(
        "Progressivedisease† (n = 65; 11.3%) ... Progressivedisease† (n = 171; 29.7%)",
        source.casefold(),
    )


def test_quote_support_rejects_ellipsis_with_hallucinated_fragment():
    source = "Progressivedisease† (n = 65; 11.3%) was listed as a reason."

    assert not quote_is_supported(
        "Progressivedisease† (n = 65; 11.3%) ... Sponsor confirmed no missing data",
        source.casefold(),
    )


def test_quote_support_rejects_hallucinated_quote():
    source = "Participants were randomly assigned using a central web system."

    assert not quote_is_supported(
        "Outcome assessors were blinded by an independent committee.", source.casefold()
    )


def test_verify_sq_evidence_flags_missing_d3_denominator():
    evidence = empty_paper_evidence()
    evidence["results"]["text"] = "Most participants had outcome data."
    state = {
        "full_text": "Most participants had outcome data.",
        "evidence": evidence,
        "rag_contexts": {},
        "sq_answers": {
            "3.1": {
                "answer": "Y",
                "quote": "Most participants had outcome data.",
                "justification": "Outcome data were nearly complete.",
            }
        },
    }

    flags = verify_sq_evidence(state)

    assert any(
        flag["sq_id"] == "3.1" and "denominator" in flag["issue"] for flag in flags
    )


def test_verify_sq_evidence_accepts_d3_percentage_in_quote():
    evidence = empty_paper_evidence()
    state = {
        "full_text": "Missing data 0 1 (0.3%).",
        "evidence": evidence,
        "rag_contexts": {},
        "sq_answers": {
            "3.1": {
                "answer": "Y",
                "quote": "Missing data 0 1 (0.3%).",
                "justification": "Outcome data were nearly complete.",
            }
        },
    }

    flags = verify_sq_evidence(state)

    assert not any(
        flag["sq_id"] == "3.1" and "denominator" in flag["issue"]
        for flag in flags
    )


def test_verify_sq_evidence_flags_unsupported_selective_reporting_quote():
    evidence = empty_paper_evidence()
    evidence["results"]["text"] = "The registered primary outcome was reported."
    state = {
        "full_text": "The registered primary outcome was reported.",
        "evidence": evidence,
        "rag_contexts": {},
        "sq_answers": {
            "5.2": {
                "answer": "PY",
                "quote": "Several unreported outcome scales were selectively omitted.",
                "justification": "The result appears selective.",
            }
        },
    }

    flags = verify_sq_evidence(state)

    assert any(flag["issue"] == "quote_not_found_in_source_context" for flag in flags)
    assert any("multiple eligible" in flag["issue"] for flag in flags)


def test_verify_sq_evidence_accepts_ctgov_registry_quote():
    state = {
        "full_text": "",
        "evidence": empty_paper_evidence(),
        "rag_contexts": {},
        "ctgov_design": (
            "Authoritative ClinicalTrials.gov registry design metadata:\n"
            "  Masking: NONE (masked parties: not specified)"
        ),
        "sq_answers": {
            "2.1": {
                "answer": "Y",
                "quote": "Masking: NONE (masked parties: not specified)",
                "justification": "Registry design metadata says masking was none.",
            }
        },
        "evidence_packets": {},
    }

    flags = verify_sq_evidence(state)

    assert not any(flag["issue"] == "quote_not_found_in_source_context" for flag in flags)


def test_verify_sq_evidence_accepts_packet_source_quote():
    state = {
        "full_text": "",
        "evidence": empty_paper_evidence(),
        "rag_contexts": {},
        "sq_answers": {
            "1.1": {
                "answer": "Y",
                "quote": "Randomization was stratified according to disease volume.",
                "justification": "The selected packet source directly states the method.",
            }
        },
        "evidence_packets": {
            "1.1": {
                "sources": [
                    {
                        "text": (
                            "Patients were randomly assigned. Randomization was "
                            "stratified according to disease volume."
                        )
                    }
                ]
            }
        },
    }

    flags = verify_sq_evidence(state)

    assert not any(flag["issue"] == "quote_not_found_in_source_context" for flag in flags)


def test_verify_sq_evidence_marks_raw_stream_only_quote():
    state = {
        "full_text": "The layout text split random ization across lines.",
        "evidence": empty_paper_evidence(),
        "rag_contexts": {},
        "sq_answers": {
            "1.1": {
                "answer": "Y",
                "quote": "Randomization was performed centrally.",
                "justification": "The report describes central randomization.",
            }
        },
        "evidence_packets": {},
        "parse_artifacts": [
            {
                "source_identity": {"document_id": "primary", "document_role": "primary"},
                "raw_character_stream": "Randomization was performed centrally.",
            }
        ],
    }

    flags = verify_sq_evidence(state)

    assert flags == [
        {
            "sq_id": "1.1",
            "issue": "quote_found_only_in_raw_character_stream",
            "quote": "Randomization was performed centrally.",
        }
    ]


def test_quote_verifier_surfaces_raw_stream_only_constraint_and_action():
    state = {
        "full_text": "The layout text split random ization across lines.",
        "evidence": empty_paper_evidence(),
        "rag_contexts": {},
        "sq_answers": {
            "1.1": {
                "answer": "Y",
                "quote": "Randomization was performed centrally.",
                "justification": "The report describes central randomization.",
                "support_level": "strong",
            }
        },
        "evidence_packets": {},
        "parse_artifacts": [
            {"raw_character_stream": "Randomization was performed centrally."}
        ],
        "verifier_trace": [],
    }

    result = quote_verifier_node(state)

    assert result["support_constraints"] == [
        {
            "constraint_type": "quote_raw_pdf_only",
            "sq_id": "1.1",
            "claim": {
                "answer": "Y",
                "quote": "Randomization was performed centrally.",
                "justification": "The report describes central randomization.",
                "support_level": "strong",
            },
            "evidence_label": "quote",
            "evidence": "Randomization was performed centrally.",
            "reason": "quote_found_only_in_raw_character_stream",
            "provenance": {
                "source": "quote_verifier",
                "fallback": "raw_character_stream",
            },
        }
    ]
    assert result["verification_actions"] == [
        {
            "sq_id": "1.1",
            "action": "review_raw_pdf_only_traceability",
            "reason": "quote_found_only_in_raw_character_stream",
        }
    ]


def test_quote_verifier_surfaces_packet_support_constraints():
    state = {
        "full_text": "Progression-free survival improved.",
        "evidence": empty_paper_evidence(),
        "rag_contexts": {},
        "sq_answers": {},
        "evidence_packets": {
            "5.1": {
                "sq_id": "5.1",
                "packet_grade": {
                    "relevance": 0.1,
                    "coverage": 0.0,
                    "missing_evidence": ["protocol_or_registration"],
                    "retry_recommended": True,
                },
                "negative_flags": ["results_without_prespecification"],
            }
        },
        "verifier_trace": [],
    }

    result = quote_verifier_node(state)

    assert result["support_constraints"]
    assert result["support_constraints"][0]["sq_id"] == "5.1"
    assert result["support_constraints"][0]["constraint_type"] == (
        "missing_required_evidence"
    )
    assert any(
        flag["sq_id"] == "5.1" and "packet" in flag["issue"]
        for flag in result["evidence_validation_flags"]
    )


def test_quote_verifier_ignores_not_applicable_packet_failures():
    state = {
        "full_text": "",
        "evidence": empty_paper_evidence(),
        "rag_contexts": {},
        "sq_answers": {
            "2.4": {
                "answer": "NA",
                "quote": "Not applicable",
                "justification": "Not applicable",
            }
        },
        "evidence_packets": {
            "2.4": {
                "sq_id": "2.4",
                "packet_grade": {
                    "missing_evidence": ["deviation_outcome_impact"],
                    "retry_recommended": True,
                },
            }
        },
        "verifier_trace": [],
    }

    result = quote_verifier_node(state)

    assert result["evidence_validation_flags"] == []
    assert result["support_constraints"] == []


def test_quote_verifier_surfaces_typed_support_constraints():
    state = {
        "full_text": "Progression-free survival improved.",
        "evidence": empty_paper_evidence(),
        "rag_contexts": {},
        "sq_answers": {
            "5.1": {
                "answer": "Y",
                "quote": "The protocol prespecified progression-free survival.",
                "justification": "Protocol evidence supports prespecification.",
                "support_level": "strong",
                "support_rationale": "The selected evidence directly supports it.",
            }
        },
        "evidence_packets": {
            "5.1": {
                "sq_id": "5.1",
                "packet_grade": {
                    "relevance": 0.2,
                    "coverage": 0.0,
                    "missing_evidence": ["protocol_or_registration"],
                    "retry_recommended": True,
                },
                "negative_flags": ["results_without_prespecification"],
            }
        },
        "verifier_trace": [],
    }

    result = quote_verifier_node(state)

    constraints = result["support_constraints"]
    assert {
        constraint["constraint_type"]
        for constraint in constraints
        if constraint["sq_id"] == "5.1"
    } == {
        "quote_untraceable",
        "missing_required_evidence",
        "semantic_support_conflict",
    }
    quote_constraint = next(
        constraint
        for constraint in constraints
        if constraint["constraint_type"] == "quote_untraceable"
    )
    assert quote_constraint["claim"]["answer"] == "Y"
    assert quote_constraint["claim"]["support_level"] == "strong"
    assert quote_constraint["evidence_label"] == "quote"


def test_quote_verifier_maps_wrong_outcome_packet_flags_to_constraints():
    state = {
        "full_text": "Overall survival was measured.",
        "evidence": empty_paper_evidence(),
        "rag_contexts": {},
        "sq_answers": {
            "4.2": {
                "answer": "N",
                "quote": "Overall survival was measured.",
                "justification": "The same method was used.",
                "support_level": "moderate",
                "support_rationale": "Outcome measurement evidence is indirect.",
            }
        },
        "evidence_packets": {
            "4.2": {
                "sq_id": "4.2",
                "packet_grade": {
                    "relevance": 0.8,
                    "coverage": 0.7,
                    "missing_evidence": [],
                    "retry_recommended": False,
                },
                "negative_flags": ["possible_wrong_outcome_context"],
            }
        },
        "verifier_trace": [],
    }

    result = quote_verifier_node(state)

    assert any(
        constraint["constraint_type"] == "wrong_outcome_context"
        and constraint["evidence_label"] == "possible_wrong_outcome_context"
        for constraint in result["support_constraints"]
    )
