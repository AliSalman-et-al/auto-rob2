from rob2_pipeline.models import empty_paper_evidence
from rob2_pipeline.nodes.domain2 import domain2_sq12_node
from rob2_pipeline.nodes.domain4 import domain4_sq_node


def _packet(domain: str, sq_id: str, text: str, *, status: str = "ready") -> dict:
    return {
        "artifact_id": f"evidence-packet:{domain}:{sq_id}",
        "schema_version": "1.0",
        "sq_id": sq_id,
        "domain": domain,
        "required_evidence": ["selected_packet_evidence"],
        "sources": [{"text": text, "section": "Methods", "document_role": "primary"}],
        "candidate_facts": [
            {
                "artifact_id": f"evidence-fact:{domain}:{sq_id}:0",
                "claim": text,
                "quote": text,
                "support_level": "moderate",
            }
        ],
        "missing_evidence": [],
        "negative_flags": [],
        "decision_table": {
            "artifact_id": f"decision-table:{domain}:{sq_id}",
            "schema_version": "1.0",
            "sq_id": sq_id,
            "allowed_answers": ["Y", "PY", "PN", "N", "NI", "NA"],
            "rows": [],
            "default_insufficient_evidence_answer": "NI",
            "classifier_instruction": "Use only selected packet evidence.",
        },
        "packet_readiness": {"status": status},
    }


def _base_state(domain: str, packets: dict[str, dict]) -> dict:
    return {
        "evidence": empty_paper_evidence("test"),
        "intervention": "Drug A",
        "comparator": "Placebo",
        "outcome": "Overall Survival",
        "outcome_type": "clinician-graded",
        "effect_of_interest": "ITT",
        "ctgov_design": "",
        "rag_contexts": {domain: "outside generic retrieved text"},
        "rag_chunk_metadata": {},
        "trial_facts": {},
        "sq_answers": {},
        "evidence_packets": packets,
    }


def test_domain2_sq12_json_classifier_records_branching_metadata(monkeypatch):
    captured = {}
    packets = {
        sq_id: _packet("d2", sq_id, "Participants and carers were not blinded.")
        for sq_id in ("2.1", "2.2")
    }
    state = _base_state("d2", packets)

    def fake_contract_call(state, prompt, node_name, **kwargs):
        captured["prompt"] = prompt
        captured["node_name"] = node_name
        return {
            "artifact": {
                "schema_version": "d2-sq-classifier-v1",
                "domain": "d2",
                "stage": "sq12",
                "branching": {"effect_of_interest": "ITT", "stage": "sq12"},
                "outcome_specific_concerns": [],
                "answers": [
                    {
                        "sq_id": "2.1",
                        "answer": "Y",
                        "quote": "Participants and carers were not blinded.",
                        "justification": "The packet says participants and carers were unblinded.",
                        "support_level": "moderate",
                        "support_rationale": "The selected packet directly supports awareness.",
                        "uncertainty": False,
                        "packet_artifact_id": "evidence-packet:d2:2.1",
                        "decision_table_artifact_id": "decision-table:d2:2.1",
                        "supporting_fact_artifact_ids": ["evidence-fact:d2:2.1:0"],
                    },
                    {
                        "sq_id": "2.2",
                        "answer": "PY",
                        "quote": "Participants and carers were not blinded.",
                        "justification": "Awareness could plausibly affect deviations.",
                        "support_level": "weak",
                        "support_rationale": "The packet supports awareness but not deviation impact strongly.",
                        "uncertainty": True,
                        "packet_artifact_id": "evidence-packet:d2:2.2",
                        "decision_table_artifact_id": "decision-table:d2:2.2",
                        "supporting_fact_artifact_ids": ["evidence-fact:d2:2.2:0"],
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

    result = domain2_sq12_node(state)

    assert captured["node_name"] == "domain2_sq12_json"
    assert "outside generic retrieved text" not in captured["prompt"]
    assert '"stage": "sq12"' in captured["prompt"]
    assert result["sq_answers"]["2.1"]["support_level"] == "moderate"
    assert result["d2_sq12_classifier_artifact"]["branching"]["effect_of_interest"] == "ITT"


def test_domain4_json_classifier_artifact_carries_outcome_specific_concerns(monkeypatch):
    packets = {
        sq_id: _packet("d4", sq_id, "Outcome assessors were blinded to allocation.")
        for sq_id in ("4.1", "4.2", "4.3", "4.4", "4.5")
    }
    state = _base_state("d4", packets)

    def fake_contract_call(state, prompt, node_name, **kwargs):
        return {
            "artifact": {
                "schema_version": "d4-sq-classifier-v1",
                "domain": "d4",
                "stage": "sq",
                "branching": {"outcome_type": "clinician-graded"},
                "outcome_specific_concerns": [
                    {
                        "concern": "clinician-graded assessor blinding",
                        "support_level": "moderate",
                        "rationale": "The packet names blinded outcome assessors.",
                    }
                ],
                "answers": [
                    {
                        "sq_id": sq_id,
                        "answer": "N",
                        "quote": "Outcome assessors were blinded to allocation.",
                        "justification": "Blinded assessment lowers measurement bias concern.",
                        "support_level": "moderate",
                        "support_rationale": "The answer is supported by the selected packet.",
                        "uncertainty": False,
                        "packet_artifact_id": f"evidence-packet:d4:{sq_id}",
                        "decision_table_artifact_id": f"decision-table:d4:{sq_id}",
                        "supporting_fact_artifact_ids": [f"evidence-fact:d4:{sq_id}:0"],
                    }
                    for sq_id in ("4.1", "4.2", "4.3", "4.4", "4.5")
                ],
            },
            "log": [{"node": node_name, "validation_status": "validated"}],
            "status": "validated",
        }

    monkeypatch.setattr(
        "rob2_pipeline.nodes.domain_classifier.call_json_contract_llm",
        fake_contract_call,
    )

    result = domain4_sq_node(state)

    assert result["sq_answers"]["4.1"]["support_level"] == "moderate"
    assert result["d4_sq_classifier_artifact"]["outcome_specific_concerns"] == [
        {
            "concern": "clinician-graded assessor blinding",
            "support_level": "moderate",
            "rationale": "The packet names blinded outcome assessors.",
        }
    ]
