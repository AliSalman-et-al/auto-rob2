from rob2_pipeline.nodes.d5_selection_evidence import build_d5_selection_evidence
from rob2_pipeline.nodes.domain5 import domain5_judge_node
from rob2_pipeline.pipeline import _assessment_json


def test_d5_selection_evidence_classifies_prespecified_single_options():
    state = {
        "outcome": "Overall Survival",
        "numerical_result": "HR 0.80",
        "registration_number": "NCT123",
        "registered_endpoint": "Overall Survival",
        "registered_analysis": "Overall survival will be compared with a stratified log-rank test.",
        "ctgov_outcomes": "Primary Outcome: Overall Survival",
        "evidence_packets": {
            "5.1": {
                "sources": [
                    {
                        "text": "Protocol prespecified overall survival as the primary endpoint.",
                        "section": "Protocol",
                        "document_role": "protocol",
                        "document_name": "protocol.pdf",
                        "page_numbers": [8],
                    }
                ],
                "missing_evidence": [],
            },
            "5.2": {
                "sources": [
                    {
                        "text": "Overall survival was the primary endpoint.",
                        "section": "Endpoints",
                        "document_role": "protocol",
                        "document_name": "protocol.pdf",
                        "page_numbers": [9],
                    }
                ],
                "missing_evidence": [],
            },
            "5.3": {
                "sources": [
                    {
                        "text": "The primary analysis used a stratified log-rank test.",
                        "section": "Statistics",
                        "document_role": "sap",
                        "document_name": "sap.pdf",
                        "page_numbers": [3],
                    }
                ],
                "missing_evidence": [],
            },
        },
        "sq_answers": {
            "5.1": {"answer": "Y", "quote": "Protocol prespecified overall survival."},
            "5.2": {
                "answer": "N",
                "quote": "Overall survival was the primary endpoint.",
            },
            "5.3": {
                "answer": "N",
                "quote": "The primary analysis used a stratified log-rank test.",
            },
        },
    }

    evidence = build_d5_selection_evidence(state)

    assert evidence["plan_availability"]["classification"] == "available"
    assert evidence["outcome_measurement_options"]["classification"] == "single"
    assert evidence["analysis_options"]["classification"] == "single"
    assert evidence["result_based_selection_support"]["classification"] == "absent"
    assert evidence["assessed_result_binding"]["classification"] == "exact"
    assert (
        evidence["plan_availability"]["provenance"][0]["document_name"]
        == "protocol.pdf"
    )


def test_assessment_json_exposes_d5_selection_evidence():
    state = {
        "d5_selection_evidence": {
            "plan_availability": {"classification": "unavailable", "provenance": []}
        },
        "rag_chunk_metadata": {},
    }

    data = _assessment_json(state)

    assert (
        data["d5_selection_evidence"]["plan_availability"]["classification"]
        == "unavailable"
    )


def test_domain5_judge_node_adds_d5_selection_evidence_to_state_update():
    result = domain5_judge_node(
        {
            "registration_number": "NCT123",
            "registered_endpoint": "Overall Survival",
            "outcome": "Overall Survival",
            "sq_answers": {
                "5.1": {"answer": "Y"},
                "5.2": {"answer": "N"},
                "5.3": {"answer": "N"},
            },
        }
    )

    assert result["domain_judgments"]["D5"] == "Low"
    assert (
        result["d5_selection_evidence"]["plan_availability"]["classification"]
        == "available"
    )


def test_domain5_judge_node_gates_unsupported_measurement_selection_before_judging():
    result = domain5_judge_node(
        {
            "outcome": "Overall Survival",
            "registration_number": "NCT123",
            "registered_endpoint": "Overall Survival",
            "sq_answers": {
                "5.1": {"answer": "Y"},
                "5.2": {
                    "answer": "Y",
                    "quote": "Overall survival was the prespecified primary endpoint.",
                },
                "5.3": {"answer": "N"},
            },
            "evidence_packets": {
                "5.2": {
                    "sources": [
                        {
                            "text": "Overall survival was the prespecified primary endpoint.",
                            "document_role": "protocol",
                        }
                    ],
                }
            },
        }
    )

    assert result["sq_answers"]["5.2"]["answer"] == "NI"
    assert result["domain_judgments"]["D5"] == "Some concerns"


def test_domain5_judge_node_preserves_supported_measurement_selection_high():
    result = domain5_judge_node(
        {
            "outcome": "Overall Survival",
            "registration_number": "NCT123",
            "registered_endpoint": "Overall Survival",
            "sq_answers": {
                "5.1": {"answer": "Y"},
                "5.2": {
                    "answer": "Y",
                    "quote": "A post hoc selected subset of survival endpoints was reported.",
                },
                "5.3": {"answer": "N"},
            },
            "evidence_packets": {
                "5.2": {
                    "sources": [
                        {
                            "text": "A post hoc selected subset of survival endpoints was reported.",
                            "document_role": "primary",
                        }
                    ],
                }
            },
        }
    )

    assert result["sq_answers"]["5.2"]["answer"] == "Y"
    assert result["domain_judgments"]["D5"] == "High"


def test_domain5_judge_node_gates_unsupported_analysis_selection_before_judging():
    result = domain5_judge_node(
        {
            "outcome": "Overall Survival",
            "registration_number": "NCT123",
            "registered_endpoint": "Overall Survival",
            "registered_analysis": "Overall survival will use a stratified log-rank test.",
            "sq_answers": {
                "5.1": {"answer": "Y"},
                "5.2": {"answer": "N"},
                "5.3": {
                    "answer": "PY",
                    "quote": "The SAP planned adjusted and unadjusted sensitivity analyses.",
                },
            },
            "evidence_packets": {
                "5.3": {
                    "sources": [
                        {
                            "text": "The SAP planned adjusted and unadjusted sensitivity analyses.",
                            "document_role": "sap",
                        }
                    ],
                }
            },
        }
    )

    assert result["sq_answers"]["5.3"]["answer"] == "NI"
    assert result["domain_judgments"]["D5"] == "Some concerns"


def test_domain5_judge_node_preserves_supported_analysis_selection_high():
    result = domain5_judge_node(
        {
            "outcome": "Overall Survival",
            "registration_number": "NCT123",
            "registered_endpoint": "Overall Survival",
            "sq_answers": {
                "5.1": {"answer": "Y"},
                "5.2": {"answer": "N"},
                "5.3": {
                    "answer": "PY",
                    "quote": "A post hoc selected subgroup analysis was reported.",
                },
            },
            "evidence_packets": {
                "5.3": {
                    "sources": [
                        {
                            "text": "A post hoc selected subgroup analysis was reported.",
                            "document_role": "primary",
                        }
                    ],
                }
            },
        }
    )

    assert result["sq_answers"]["5.3"]["answer"] == "PY"
    assert result["domain_judgments"]["D5"] == "High"


def test_domain5_judge_node_does_not_treat_prespecified_coprimary_endpoints_as_selection():
    result = domain5_judge_node(
        {
            "outcome": "Overall Survival",
            "registration_number": "NCT123",
            "registered_endpoint": "Overall Survival and Progression-free Survival",
            "sq_answers": {
                "5.1": {"answer": "Y"},
                "5.2": {
                    "answer": "Y",
                    "quote": "Protocol prespecified co-primary endpoints.",
                },
                "5.3": {"answer": "N"},
            },
            "evidence_packets": {
                "5.2": {
                    "sources": [
                        {
                            "text": "Protocol prespecified co-primary endpoints.",
                            "document_role": "protocol",
                        }
                    ],
                }
            },
        }
    )

    assert result["sq_answers"]["5.2"]["answer"] == "NI"
    assert result["domain_judgments"]["D5"] == "Some concerns"


def test_domain5_judge_node_does_not_treat_prespecified_composite_components_as_selection():
    result = domain5_judge_node(
        {
            "outcome": "Composite progression-free survival",
            "registration_number": "NCT123",
            "registered_endpoint": "Composite progression-free survival",
            "sq_answers": {
                "5.1": {"answer": "Y"},
                "5.2": {
                    "answer": "Y",
                    "quote": "The protocol prespecified a composite endpoint containing progression or death.",
                },
                "5.3": {"answer": "N"},
            },
            "evidence_packets": {
                "5.2": {
                    "sources": [
                        {
                            "text": "The protocol prespecified a composite endpoint containing progression or death.",
                            "document_role": "protocol",
                        }
                    ],
                }
            },
        }
    )

    assert result["sq_answers"]["5.2"]["answer"] == "NI"
    assert result["domain_judgments"]["D5"] == "Some concerns"


def test_domain5_judge_node_missing_plan_does_not_invent_high_risk_selection():
    result = domain5_judge_node(
        {
            "outcome": "Overall Survival",
            "sq_answers": {
                "5.1": {"answer": "NI"},
                "5.2": {"answer": "Y", "quote": "No relevant text found"},
                "5.3": {"answer": "N"},
            },
            "evidence_packets": {
                "5.1": {"missing_evidence": ["protocol", "SAP"]},
                "5.2": {"sources": [{"text": "Overall survival was reported."}]},
            },
        }
    )

    assert result["sq_answers"]["5.2"]["answer"] == "NI"
    assert result["domain_judgments"]["D5"] == "Some concerns"


def test_d5_selection_evidence_distinguishes_plan_availability_labels():
    assert (
        build_d5_selection_evidence({"sq_answers": {"5.1": {"answer": "N"}}})[
            "plan_availability"
        ]["classification"]
        == "unavailable"
    )
    assert (
        build_d5_selection_evidence(
            {"evidence_packets": {"5.1": {"missing_evidence": ["protocol"]}}}
        )["plan_availability"]["classification"]
        == "partial"
    )
    assert (
        build_d5_selection_evidence(
            {
                "registration_number": "NCT123",
                "sq_answers": {"5.1": {"answer": "N"}},
            }
        )["plan_availability"]["classification"]
        == "conflicting"
    )


def test_d5_selection_evidence_distinguishes_outcome_measurement_option_labels():
    def classification(text: str) -> str:
        return build_d5_selection_evidence(
            {"evidence_packets": {"5.2": {"sources": [{"text": text}]}}}
        )["outcome_measurement_options"]["classification"]

    assert classification("Overall survival was the endpoint.") == "single"
    assert (
        classification(
            "Protocol prespecified co-primary endpoints and two time points."
        )
        == "multiple prespecified"
    )
    assert (
        classification("Several outcome definitions and time points were reported.")
        == "multiple unclear"
    )
    assert (
        classification("A selected subset of endpoints was reported.")
        == "selected-subset"
    )


def test_d5_selection_evidence_distinguishes_analysis_option_labels():
    def classification(text: str) -> str:
        return build_d5_selection_evidence(
            {"evidence_packets": {"5.3": {"sources": [{"text": text}]}}}
        )["analysis_options"]["classification"]

    assert classification("The log-rank test was used.") == "single"
    assert (
        classification("The SAP planned adjusted and unadjusted sensitivity analyses.")
        == "multiple prespecified"
    )
    assert (
        classification("Several subgroup and adjusted analyses were reported.")
        == "multiple unclear"
    )
    assert (
        classification("A post hoc selected subgroup analysis was reported.")
        == "selected-subset"
    )


def test_d5_selection_evidence_distinguishes_result_selection_and_binding_labels():
    supported = build_d5_selection_evidence(
        {
            "evidence_packets": {
                "5.2": {
                    "sources": [
                        {
                            "text": "Overall survival was reported as a post hoc selected subset."
                        }
                    ]
                }
            },
            "outcome": "Overall Survival",
        }
    )
    possible = build_d5_selection_evidence({"sq_answers": {"5.2": {"answer": "NI"}}})
    wrong_outcome = build_d5_selection_evidence(
        {
            "evidence_packets": {
                "5.2": {"sources": [{"text": "Different outcome was reported."}]}
            }
        }
    )

    assert supported["result_based_selection_support"]["classification"] == "supported"
    assert possible["result_based_selection_support"]["classification"] == "possible"
    assert (
        build_d5_selection_evidence({})["result_based_selection_support"][
            "classification"
        ]
        == "absent"
    )
    assert supported["assessed_result_binding"]["classification"] == "partial"
    assert wrong_outcome["assessed_result_binding"]["classification"] == "wrong-outcome"
    assert (
        build_d5_selection_evidence({})["assessed_result_binding"]["classification"]
        == "unclear"
    )
