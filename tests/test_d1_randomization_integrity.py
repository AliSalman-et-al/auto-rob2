from rob2_pipeline.nodes.d1_randomization_integrity import (
    apply_d1_randomization_integrity_gate,
    build_d1_randomization_integrity_evidence,
)
from rob2_pipeline.nodes.domain1 import domain1_judge_node
from rob2_pipeline.pipeline import _assessment_json


def test_d1_randomization_integrity_classifies_core_dimensions_with_provenance():
    state = {
        "evidence_packets": {
            "1.1": {
                "sources": [
                    {
                        "text": "A computer-generated randomization schedule was prepared.",
                        "section": "Methods",
                        "document_name": "paper.pdf",
                        "page_numbers": [4],
                        "source_kind": "rag_chunk",
                    }
                ]
            },
            "1.2": {
                "sources": [
                    {
                        "text": "Allocation was concealed using a central web-response system.",
                        "section": "Methods",
                        "document_name": "paper.pdf",
                        "page_numbers": [4],
                        "source_kind": "rag_chunk",
                    }
                ]
            },
            "1.3": {
                "sources": [
                    {
                        "text": "Baseline prognostic factors were well balanced between groups.",
                        "section": "Table 1",
                        "document_name": "paper.pdf",
                        "page_numbers": [6],
                        "source_kind": "rag_chunk",
                    }
                ]
            },
        },
        "sq_answers": {
            "1.1": {"answer": "Y", "quote": "computer-generated randomization"},
            "1.2": {"answer": "Y", "quote": "central web-response system"},
            "1.3": {"answer": "N", "quote": "well balanced between groups"},
        },
    }

    evidence = build_d1_randomization_integrity_evidence(state)

    assert evidence["sequence_generation"]["classification"] == "adequate"
    assert evidence["allocation_concealment"]["classification"] == "adequate"
    assert evidence["enrolment_timing"]["classification"] == "unclear"
    assert evidence["baseline_imbalance_severity"]["classification"] == "none"
    assert evidence["prognostic_relevance"]["classification"] == "not_applicable"
    assert evidence["randomization_failure_signal"]["classification"] == "not_supported"
    assert (
        evidence["sequence_generation"]["provenance"][0]["document_name"] == "paper.pdf"
    )


def test_d1_randomization_integrity_distinguishes_required_labels():
    def evidence_for(sq_answers, source_text=""):
        return build_d1_randomization_integrity_evidence(
            {
                "sq_answers": sq_answers,
                "evidence_packets": {
                    "1.1": {"sources": [{"text": source_text}]},
                    "1.2": {"sources": [{"text": source_text}]},
                    "1.3": {"sources": [{"text": source_text}]},
                },
            }
        )

    inadequate = evidence_for(
        {
            "1.1": {"answer": "N"},
            "1.2": {"answer": "PN"},
            "1.3": {"answer": "PY"},
        },
        "Participants were randomized after enrolment; important prognostic baseline differences suggested randomization failure.",
    )
    unclear = evidence_for(
        {
            "1.1": {"answer": "NI"},
            "1.2": {"answer": "NI"},
            "1.3": {"answer": "NI"},
        }
    )

    assert inadequate["sequence_generation"]["classification"] == "inadequate"
    assert inadequate["allocation_concealment"]["classification"] == "inadequate"
    assert inadequate["enrolment_timing"]["classification"] == "after_randomization"
    assert (
        inadequate["baseline_imbalance_severity"]["classification"]
        == "suggests_randomization_failure"
    )
    assert inadequate["prognostic_relevance"]["classification"] == "prognostic"
    assert inadequate["randomization_failure_signal"]["classification"] == "supported"
    assert unclear["sequence_generation"]["classification"] == "unclear"
    assert unclear["allocation_concealment"]["classification"] == "unclear"
    assert unclear["enrolment_timing"]["classification"] == "unclear"


def test_d1_randomization_integrity_distinguishes_middle_imbalance_and_timing_labels():
    def classification(source_text):
        return build_d1_randomization_integrity_evidence(
            {
                "sq_answers": {
                    "1.1": {"answer": "Y"},
                    "1.2": {"answer": "Y"},
                    "1.3": {"answer": "NI"},
                },
                "evidence_packets": {
                    "1.1": {
                        "sources": [
                            {
                                "text": "Allocation occurred before enrolment in a central system."
                            }
                        ]
                    },
                    "1.2": {"sources": [{"text": ""}]},
                    "1.3": {"sources": [{"text": source_text}]},
                },
            }
        )

    not_concerning = classification(
        "Baseline differences were present but small and not clinically relevant."
    )
    prognostic = classification(
        "Baseline imbalance in disease stage and performance status was observed."
    )

    assert (
        not_concerning["enrolment_timing"]["classification"] == "before_randomization"
    )
    assert (
        not_concerning["baseline_imbalance_severity"]["classification"]
        == "present_not_concerning"
    )
    assert (
        prognostic["baseline_imbalance_severity"]["classification"]
        == "prognostic_concerning"
    )


def test_assessment_json_exposes_d1_randomization_integrity_evidence():
    state = {
        "d1_randomization_integrity_evidence": {
            "sequence_generation": {"classification": "adequate", "provenance": []}
        },
        "rag_chunk_metadata": {},
    }

    data = _assessment_json(state)

    assert (
        data["d1_randomization_integrity_evidence"]["sequence_generation"][
            "classification"
        ]
        == "adequate"
    )


def test_domain1_judge_node_adds_d1_randomization_integrity_evidence_to_state_update():
    result = domain1_judge_node(
        {
            "sq_answers": {
                "1.1": {"answer": "Y"},
                "1.2": {"answer": "Y"},
                "1.3": {"answer": "N"},
            }
        }
    )

    assert result["domain_judgments"]["D1"] == "Low"
    assert (
        result["d1_randomization_integrity_evidence"]["sequence_generation"][
            "classification"
        ]
        == "adequate"
    )


def test_domain1_judge_node_does_not_let_baseline_differences_alone_force_concern():
    result = domain1_judge_node(
        {
            "sq_answers": {
                "1.1": {"answer": "Y"},
                "1.2": {"answer": "Y"},
                "1.3": {
                    "answer": "PY",
                    "quote": "Baseline characteristics differed between arms.",
                },
            },
            "evidence_packets": {
                "1.3": {
                    "sources": [
                        {
                            "text": "Baseline characteristics differed between arms.",
                        }
                    ]
                }
            },
        }
    )

    assert result["sq_answers"]["1.3"]["answer"] == "PN"
    assert result["domain_judgments"]["D1"] == "Low"


def test_d1_randomization_integrity_gate_keeps_concerning_baseline_signal():
    sq_answers = {
        "1.1": {"answer": "Y"},
        "1.2": {"answer": "Y"},
        "1.3": {"answer": "PY"},
    }
    evidence = build_d1_randomization_integrity_evidence(
        {
            "sq_answers": sq_answers,
            "evidence_packets": {
                "1.3": {
                    "sources": [
                        {
                            "text": "A large imbalance in prognostic disease stage was unlikely by chance and suggested randomization failure.",
                        }
                    ]
                }
            },
        }
    )

    gated = apply_d1_randomization_integrity_gate(sq_answers, evidence)

    assert gated["1.3"]["answer"] == "PY"


def test_domain1_judge_node_calibrates_adequate_sequence_and_concealment_evidence():
    result = domain1_judge_node(
        {
            "sq_answers": {
                "1.1": {"answer": "NI"},
                "1.2": {"answer": "NI"},
                "1.3": {"answer": "N"},
            },
            "evidence_packets": {
                "1.1": {
                    "sources": [
                        {"text": "Randomization used a computer-generated schedule."}
                    ]
                },
                "1.2": {
                    "sources": [
                        {
                            "text": "Treatment was assigned by central randomization with allocation concealed."
                        }
                    ]
                },
                "1.3": {"sources": [{"text": "Groups were well balanced."}]},
            },
        }
    )

    assert result["sq_answers"]["1.1"]["answer"] == "Y"
    assert result["sq_answers"]["1.2"]["answer"] == "Y"
    assert result["domain_judgments"]["D1"] == "Low"


def test_domain1_judge_node_calibrates_unclear_and_inadequate_randomization_scenarios():
    unclear = domain1_judge_node(
        {
            "sq_answers": {
                "1.1": {"answer": "Y"},
                "1.2": {"answer": "Y"},
                "1.3": {"answer": "N"},
            },
            "evidence_packets": {
                "1.1": {"sources": [{"text": "Patients were assigned to groups."}]},
                "1.2": {"sources": [{"text": "Treatment assignment was performed."}]},
                "1.3": {"sources": [{"text": "No baseline imbalance was reported."}]},
            },
        }
    )
    inadequate = domain1_judge_node(
        {
            "sq_answers": {
                "1.1": {"answer": "Y"},
                "1.2": {"answer": "Y"},
                "1.3": {"answer": "N"},
            },
            "evidence_packets": {
                "1.1": {"sources": [{"text": "Allocation alternated by clinic visit."}]},
                "1.2": {"sources": [{"text": "The next allocation was known."}]},
                "1.3": {"sources": [{"text": "No baseline imbalance was reported."}]},
            },
        }
    )

    assert unclear["sq_answers"]["1.1"]["answer"] == "NI"
    assert unclear["sq_answers"]["1.2"]["answer"] == "NI"
    assert unclear["domain_judgments"]["D1"] == "Some concerns"
    assert inadequate["sq_answers"]["1.1"]["answer"] == "N"
    assert inadequate["sq_answers"]["1.2"]["answer"] == "N"
    assert inadequate["domain_judgments"]["D1"] == "High"
