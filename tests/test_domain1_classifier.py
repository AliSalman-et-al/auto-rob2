from rob2_pipeline.models import empty_paper_evidence
from rob2_pipeline.nodes.domain1 import domain1_sq_node


def _ready_d1_packet(sq_id: str) -> dict:
    return {
        "artifact_id": f"evidence-packet:d1:{sq_id}",
        "schema_version": "1.0",
        "sq_id": sq_id,
        "domain": "d1",
        "required_evidence": ["sequence_generation"],
        "sources": [
            {
                "text": "Randomization was performed centrally using a computer-generated sequence.",
                "section": "Methods",
                "page_numbers": [2],
                "document_name": "trial.pdf",
                "document_role": "primary",
            }
        ],
        "candidate_facts": [
            {
                "artifact_id": f"evidence-fact:d1:{sq_id}:0",
                "claim": "The random allocation sequence was computer-generated.",
                "quote": "computer-generated sequence",
                "support_level": "strong",
            }
        ],
        "missing_evidence": [],
        "negative_flags": [],
        "decision_table": {
            "artifact_id": f"decision-table:d1:{sq_id}",
            "schema_version": "1.0",
            "sq_id": sq_id,
            "allowed_answers": ["Y", "PY", "PN", "N", "NI"],
            "rows": [
                {
                    "answer": "Y",
                    "rule": "Random component in sequence generation is described.",
                    "allowed_by_packet": True,
                    "supporting_facts": [
                        {
                            "artifact_id": f"evidence-fact:d1:{sq_id}:0",
                            "claim": "The random allocation sequence was computer-generated.",
                        }
                    ],
                    "evidence_gaps": [],
                }
            ],
            "default_insufficient_evidence_answer": "NI",
            "classifier_instruction": "Use only selected packet evidence.",
        },
        "packet_readiness": {"status": "ready"},
    }


def _base_state_with_ready_d1_packets() -> dict:
    return {
        "evidence": empty_paper_evidence("test"),
        "intervention": "Drug A",
        "comparator": "Placebo",
        "outcome": "Overall Survival",
        "ctgov_design": "",
        "rag_contexts": {"d1": "outside generic retrieved text"},
        "rag_chunk_metadata": {},
        "trial_facts": {},
        "sq_answers": {},
        "evidence_packets": {
            sq_id: _ready_d1_packet(sq_id) for sq_id in ("1.1", "1.2", "1.3")
        },
    }


def test_domain1_ready_packets_delegate_to_generic_classifier(monkeypatch):
    state = _base_state_with_ready_d1_packets()
    captured = {}

    def fake_run_json_sq_classifier(state, **kwargs):
        captured.update(kwargs)
        return {"sq_answers": {"1.1": {"answer": "Y"}}}

    monkeypatch.setattr(
        "rob2_pipeline.nodes.domain1.run_json_sq_classifier",
        fake_run_json_sq_classifier,
    )

    result = domain1_sq_node(state)

    assert result == {"sq_answers": {"1.1": {"answer": "Y"}}}
    assert captured == {
        "domain": "d1",
        "stage": "sq",
        "sq_ids": ("1.1", "1.2", "1.3"),
        "node_name": "domain1_sq_json",
    }


def test_domain1_sq_node_classifies_ready_packets_with_json_contract(monkeypatch):
    captured = {}
    evidence = empty_paper_evidence("test")
    packet = {
        "artifact_id": "evidence-packet:d1:1.1",
        "schema_version": "1.0",
        "sq_id": "1.1",
        "domain": "d1",
        "required_evidence": ["sequence_generation"],
        "sources": [
            {
                "text": "Randomization was performed centrally using a computer-generated sequence.",
                "section": "Methods",
                "page_numbers": [2],
                "document_name": "trial.pdf",
                "document_role": "primary",
            }
        ],
        "candidate_facts": [
            {
                "artifact_id": "evidence-fact:d1:1.1:0",
                "claim": "The random allocation sequence was computer-generated.",
                "quote": "computer-generated sequence",
                "support_level": "strong",
            }
        ],
        "missing_evidence": [],
        "negative_flags": [],
        "decision_table": {
            "artifact_id": "decision-table:d1:1.1",
            "schema_version": "1.0",
            "sq_id": "1.1",
            "allowed_answers": ["Y", "PY", "PN", "N", "NI"],
            "rows": [
                {
                    "answer": "Y",
                    "rule": "Random component in sequence generation is described.",
                    "allowed_by_packet": True,
                    "supporting_facts": [
                        {
                            "artifact_id": "evidence-fact:d1:1.1:0",
                            "claim": "The random allocation sequence was computer-generated.",
                        }
                    ],
                    "evidence_gaps": [],
                }
            ],
            "default_insufficient_evidence_answer": "NI",
            "classifier_instruction": "Use only selected packet evidence.",
        },
        "packet_readiness": {"status": "ready"},
    }
    state = {
        "evidence": evidence,
        "intervention": "Drug A",
        "comparator": "Placebo",
        "outcome": "Overall Survival",
        "ctgov_design": "",
        "rag_contexts": {"d1": "outside generic retrieved text"},
        "rag_chunk_metadata": {},
        "trial_facts": {},
        "sq_answers": {},
        "evidence_packets": {
            "1.1": packet,
            "1.2": {**packet, "artifact_id": "evidence-packet:d1:1.2", "sq_id": "1.2"},
            "1.3": {**packet, "artifact_id": "evidence-packet:d1:1.3", "sq_id": "1.3"},
        },
    }

    def fake_contract_call(state, prompt, node_name, **kwargs):
        captured["prompt"] = prompt
        captured["node_name"] = node_name
        captured["schema_model"] = kwargs["schema_model"]
        return {
            "artifact": {
                "schema_version": "d1-sq-classifier-v1",
                "domain": "d1",
                "answers": [
                    {
                        "sq_id": "1.1",
                        "answer": "Y",
                        "quote": "computer-generated sequence",
                        "justification": "The packet states the sequence was computer-generated.",
                        "support_level": "strong",
                        "support_rationale": "The selected packet fact directly supports the answer.",
                        "uncertainty": False,
                        "packet_artifact_id": "evidence-packet:d1:1.1",
                        "decision_table_artifact_id": "decision-table:d1:1.1",
                        "supporting_fact_artifact_ids": ["evidence-fact:d1:1.1:0"],
                    },
                    {
                        "sq_id": "1.2",
                        "answer": "NI",
                        "quote": "No relevant text found",
                        "justification": "The packet does not establish concealment.",
                        "support_level": "unsupported",
                        "support_rationale": "No selected fact supports allocation concealment.",
                        "uncertainty": True,
                        "packet_artifact_id": "evidence-packet:d1:1.2",
                        "decision_table_artifact_id": "decision-table:d1:1.2",
                        "supporting_fact_artifact_ids": [],
                    },
                    {
                        "sq_id": "1.3",
                        "answer": "N",
                        "quote": "Baseline factors were balanced",
                        "justification": "The packet indicates no baseline imbalance.",
                        "support_level": "moderate",
                        "support_rationale": "The answer is supported by selected baseline evidence.",
                        "uncertainty": False,
                        "packet_artifact_id": "evidence-packet:d1:1.3",
                        "decision_table_artifact_id": "decision-table:d1:1.3",
                        "supporting_fact_artifact_ids": ["evidence-fact:d1:1.1:0"],
                    },
                ],
            },
            "log": [{"node": node_name, "validation_status": "validated"}],
            "status": "validated",
        }

    monkeypatch.setattr(
        "rob2_pipeline.nodes.domain_classifier.call_json_contract_llm",
        fake_contract_call,
    )

    result = domain1_sq_node(state)

    assert captured["node_name"] == "domain1_sq_json"
    assert "outside generic retrieved text" not in captured["prompt"]
    assert "evidence-packet:d1:1.1" in captured["prompt"]
    assert result["sq_answers"]["1.1"]["answer"] == "Y"
    assert result["sq_answers"]["1.1"]["support_level"] == "strong"
    assert result["sq_answers"]["1.1"]["uncertainty"] is False
    assert result["sq_answers"]["1.1"]["packet_artifact_id"] == "evidence-packet:d1:1.1"
    assert (
        result["domain_sq_classifier_artifacts"]["d1"]["sq"]["answers"][0]["quote"]
        == "computer-generated sequence"
    )
    assert result["llm_call_log"] == [{"node": "domain1_sq_json", "validation_status": "validated"}]


def test_domain1_json_classifier_rejects_quotes_outside_packet(monkeypatch):
    evidence = empty_paper_evidence("test")
    packet = {
        "artifact_id": "evidence-packet:d1:1.1",
        "schema_version": "1.0",
        "sq_id": "1.1",
        "domain": "d1",
        "required_evidence": ["sequence_generation"],
        "sources": [{"text": "Participants were randomized in blocks.", "section": "Methods"}],
        "candidate_facts": [],
        "missing_evidence": [],
        "negative_flags": [],
        "decision_table": {
            "artifact_id": "decision-table:d1:1.1",
            "schema_version": "1.0",
            "sq_id": "1.1",
            "allowed_answers": ["Y", "PY", "PN", "N", "NI"],
            "rows": [],
            "default_insufficient_evidence_answer": "NI",
        },
        "packet_readiness": {"status": "ready"},
    }
    state = {
        "evidence": evidence,
        "intervention": "Drug A",
        "comparator": "Placebo",
        "outcome": "Overall Survival",
        "ctgov_design": "",
        "rag_contexts": {},
        "rag_chunk_metadata": {},
        "trial_facts": {},
        "sq_answers": {},
        "evidence_packets": {
            "1.1": packet,
            "1.2": {**packet, "artifact_id": "evidence-packet:d1:1.2", "sq_id": "1.2"},
            "1.3": {**packet, "artifact_id": "evidence-packet:d1:1.3", "sq_id": "1.3"},
        },
    }

    def fake_contract_call(state, prompt, node_name, **kwargs):
        return {
            "artifact": {
                "schema_version": "d1-sq-classifier-v1",
                "domain": "d1",
                "answers": [
                    {
                        "sq_id": sq_id,
                        "answer": "Y",
                        "quote": "central web response system",
                        "justification": "This quote is not in the packet.",
                        "support_level": "strong",
                        "support_rationale": "Unsupported outside-packet evidence.",
                        "uncertainty": False,
                        "packet_artifact_id": f"evidence-packet:d1:{sq_id}",
                        "decision_table_artifact_id": f"decision-table:d1:{sq_id}",
                        "supporting_fact_artifact_ids": [],
                    }
                    for sq_id in ("1.1", "1.2", "1.3")
                ],
            },
            "log": [{"node": node_name, "validation_status": "validated"}],
            "status": "validated",
        }

    monkeypatch.setattr(
        "rob2_pipeline.nodes.domain_classifier.call_json_contract_llm",
        fake_contract_call,
    )

    result = domain1_sq_node(state)

    assert result["sq_answers"]["1.1"]["answer"] == "NI"
    assert result["sq_answers"]["1.1"]["support_level"] == "unsupported"
    assert result["sq_answers"]["1.1"]["outside_packet_evidence_rejected"] is True
    assert (
        result["domain_sq_classifier_artifacts"]["d1"]["sq"]["answers"][0]["quote"]
        == "No relevant text found"
    )


def test_domain1_json_classifier_rejects_missing_duplicate_or_extra_sq_ids(monkeypatch):
    evidence = empty_paper_evidence("test")
    packet = {
        "artifact_id": "evidence-packet:d1:1.1",
        "schema_version": "1.0",
        "sq_id": "1.1",
        "domain": "d1",
        "required_evidence": ["sequence_generation"],
        "sources": [{"text": "Participants were randomized in blocks.", "section": "Methods"}],
        "candidate_facts": [],
        "missing_evidence": [],
        "negative_flags": [],
        "decision_table": {
            "artifact_id": "decision-table:d1:1.1",
            "schema_version": "1.0",
            "sq_id": "1.1",
            "allowed_answers": ["Y", "PY", "PN", "N", "NI"],
            "rows": [],
            "default_insufficient_evidence_answer": "NI",
        },
        "packet_readiness": {"status": "ready"},
    }
    state = {
        "evidence": evidence,
        "intervention": "Drug A",
        "comparator": "Placebo",
        "outcome": "Overall Survival",
        "ctgov_design": "",
        "rag_contexts": {},
        "rag_chunk_metadata": {},
        "trial_facts": {},
        "sq_answers": {},
        "evidence_packets": {
            "1.1": packet,
            "1.2": {**packet, "artifact_id": "evidence-packet:d1:1.2", "sq_id": "1.2"},
            "1.3": {**packet, "artifact_id": "evidence-packet:d1:1.3", "sq_id": "1.3"},
        },
    }

    def invalid_answer(sq_id):
        return {
            "sq_id": sq_id,
            "answer": "Y",
            "quote": "Participants were randomized in blocks.",
            "justification": "Uses the packet text.",
            "support_level": "strong",
            "support_rationale": "Packet evidence supports the answer.",
            "uncertainty": False,
            "packet_artifact_id": f"evidence-packet:d1:{sq_id}",
            "decision_table_artifact_id": f"decision-table:d1:{sq_id}",
            "supporting_fact_artifact_ids": [],
        }

    def fake_contract_call(state, prompt, node_name, **kwargs):
        return {
            "artifact": {
                "schema_version": "d1-sq-classifier-v1",
                "domain": "d1",
                "stage": "sq",
                "branching": {},
                "outcome_specific_concerns": [],
                "answers": [
                    invalid_answer("1.1"),
                    invalid_answer("1.2"),
                    invalid_answer("1.2"),
                    invalid_answer("1.4"),
                ],
            },
            "log": [{"node": node_name, "validation_status": "validated"}],
            "status": "validated",
        }

    monkeypatch.setattr(
        "rob2_pipeline.nodes.domain_classifier.call_json_contract_llm",
        fake_contract_call,
    )

    result = domain1_sq_node(state)

    artifact = result["domain_sq_classifier_artifacts"]["d1"]["sq"]
    assert [answer["sq_id"] for answer in artifact["answers"]] == ["1.1", "1.2", "1.3"]
    assert {answer["answer"] for answer in artifact["answers"]} == {"NI"}
    assert all(
        "did not include each stage SQ exactly once" in answer["support_rationale"]
        for answer in artifact["answers"]
    )
