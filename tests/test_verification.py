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
