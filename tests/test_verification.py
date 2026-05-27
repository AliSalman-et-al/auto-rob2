from rob2_pipeline.models import empty_paper_evidence
from rob2_pipeline.nodes.verification import (
    classify_evidence_support,
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


def test_classifies_exact_quote_support():
    result = classify_evidence_support(
        "Participants were randomly assigned using a central web system.",
        source_text="Participants were randomly assigned using a central web system.",
    )

    assert result["status"] == "supported"


def test_classifies_paraphrase_support_separately_from_exact_support():
    result = classify_evidence_support(
        "Participants were centrally randomized with a web-based system.",
        source_text="Participants were randomly assigned using a central web system.",
    )

    assert result["status"] == "paraphrase-supported"


def test_classifies_unsupported_evidence():
    result = classify_evidence_support(
        "Outcome assessors were blinded by an independent committee.",
        source_text="Participants were randomly assigned using a central web system.",
    )

    assert result["status"] == "unsupported"


def test_classifies_source_mismatched_evidence():
    result = classify_evidence_support(
        "Progression-free survival was assessed by investigators.",
        source_text=(
            "Progression-free survival was assessed by investigators. "
            "Participants were centrally randomized."
        ),
        provenance_text="Participants were centrally randomized.",
    )

    assert result["status"] == "source-mismatched"


def test_classifies_not_applicable_by_control_separately():
    result = classify_evidence_support("Not applicable", source_text="")

    assert result["status"] == "not-applicable-by-control"


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


def test_verify_sq_evidence_emits_support_status_with_provenance():
    evidence = empty_paper_evidence()
    state = {
        "full_text": "The trial used a central web randomization system.",
        "evidence": evidence,
        "rag_contexts": {},
        "sq_answers": {
            "1.1": {
                "answer": "Y",
                "quote": "The trial used a central web randomization system.",
                "justification": "Randomization was adequate.",
            }
        },
    }

    result = quote_verifier_node(state)

    support = result["evidence_support_statuses"]
    assert support
    assert support[0]["support_status"] == "supported"
    assert support[0]["provenance"]["source_scope"] == "assessment_context"
    assert result["evidence_validation_flags"] == []


def test_quote_verifier_surfaces_packet_retry_actions():
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

    assert result["verification_actions"]
    assert result["verification_actions"][0]["sq_id"] == "5.1"
    assert result["verification_actions"][0]["action"] == "retry_packet_or_escalate"
    assert any(
        flag["sq_id"] == "5.1" and "packet" in flag["issue"]
        for flag in result["evidence_validation_flags"]
    )
