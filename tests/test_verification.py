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


def test_quote_verifier_downgrades_unsupported_domain_critical_answer_before_reporting():
    state = {
        "full_text": "The report describes randomization and treatment allocation.",
        "evidence": empty_paper_evidence(),
        "rag_contexts": {},
        "sq_answers": {
            "3.1": {
                "answer": "Y",
                "quote": "Complete follow-up was available for every participant.",
                "justification": "Missing outcome data were negligible.",
            },
            "3.2": {"answer": "NA", "quote": "Not applicable"},
            "3.3": {"answer": "NA", "quote": "Not applicable"},
            "3.4": {"answer": "NA", "quote": "Not applicable"},
        },
        "domain_judgments": {"D3": "Low"},
        "domain_rationales": {"D3": "3.1=Y/PY (nearly complete data) -> Low"},
        "verifier_trace": [],
    }

    result = quote_verifier_node(state)

    assert result["sq_answers"]["3.1"]["answer"] == "NI"
    assert result["sq_answers"]["3.1"]["uncertainty_flag"] == "HIGH"
    assert result["domain_judgments"]["D3"] == "High"
    assert result["verification_actions"][0]["action"] == "downgrade_unsupported_sq"


def test_quote_verifier_retries_unsupported_answer_with_verified_packet_context():
    state = {
        "full_text": "The report describes randomization and treatment allocation.",
        "evidence": empty_paper_evidence(),
        "rag_contexts": {},
        "sq_answers": {
            "3.1": {
                "answer": "Y",
                "quote": "Complete follow-up was available for every participant.",
                "justification": "Missing outcome data were negligible.",
            },
        },
        "evidence_facts": {
            "3.1": [
                {
                    "quote": "Vital status was complete for 199 of 200 randomized participants.",
                    "support_status": "supported",
                }
            ]
        },
        "domain_judgments": {"D3": "Low"},
        "domain_rationales": {"D3": "3.1=Y/PY (nearly complete data) -> Low"},
        "verifier_trace": [],
    }

    result = quote_verifier_node(state)

    assert result["sq_answers"]["3.1"]["answer"] == "Y"
    retry = result["verification_actions"][0]
    assert retry["action"] == "retry_sq_with_verified_packet"
    assert "Vital status was complete" in retry["verified_packet_context"]


def test_quote_verifier_preserves_unsupported_no_relevant_text_found_as_ni():
    state = {
        "full_text": "The report describes randomization and treatment allocation.",
        "evidence": empty_paper_evidence(),
        "rag_contexts": {},
        "sq_answers": {
            "3.1": {
                "answer": "NI",
                "quote": "No relevant text found",
                "justification": "Missing outcome data were not reported.",
            },
        },
        "domain_judgments": {"D3": "High"},
        "domain_rationales": {"D3": "3.3=NI and 3.4=NI -> High"},
        "verifier_trace": [],
    }

    result = quote_verifier_node(state)

    assert result["sq_answers"]["3.1"]["answer"] == "NI"
    assert result["verification_actions"] == []


def test_quote_verifier_downgrades_source_mismatched_domain_judgment_quote():
    state = {
        "full_text": (
            "Progression was assessed by investigators. "
            "Randomization used a central web system."
        ),
        "evidence": empty_paper_evidence(),
        "rag_contexts": {},
        "sq_answers": {
            "4.1": {"answer": "N", "quote": "Same measurement method"},
            "4.2": {"answer": "N", "quote": "Same schedule"},
            "4.3": {"answer": "Y", "quote": "Open-label assessment"},
            "4.4": {"answer": "NI", "quote": "No relevant text found"},
            "4.5": {
                "answer": "Y",
                "quote": "Progression was assessed by investigators.",
                "justification": "Assessment was likely influenced.",
            },
        },
        "evidence_packets": {
            "4.5": {
                "selected_sources": [
                    {"text": "Randomization used a central web system."}
                ]
            }
        },
        "domain_judgments": {"D4": "High"},
        "domain_rationales": {"D4": "4.5=Y/PY -> High"},
        "verifier_trace": [],
    }

    result = quote_verifier_node(state)

    assert result["sq_answers"]["4.5"]["answer"] == "NI"
    assert (
        result["sq_answers"]["4.5"]["verification_support_status"]
        == "source-mismatched"
    )
    assert result["domain_judgments"]["D4"] == "High"
    assert result["verification_actions"][0]["action"] == "downgrade_unsupported_sq"
