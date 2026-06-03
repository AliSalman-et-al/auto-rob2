from pydantic import ValidationError

from rob2_pipeline.nodes.retrieval_repair import (
    RetrievalRepairArtifactRecord,
    retrieval_repair_node,
)


def _packet(*, status="ready", missing=None, confidence=0.8, sources=None):
    return {
        "artifact_id": "evidence-packet:d3:3.1",
        "schema_version": "1.0",
        "sq_id": "3.1",
        "domain": "d3",
        "outcome": "Overall survival",
        "required_evidence": ["randomized_n", "outcome_data_n"],
        "sources": sources or [],
        "retrieval_confidence": confidence,
        "missing_evidence": missing or [],
        "negative_flags": [],
        "packet_grade": {
            "relevance": confidence,
            "coverage": 0.0 if missing else 1.0,
            "missing_evidence": missing or [],
            "retry_recommended": bool(missing) or confidence < 0.35,
        },
        "packet_readiness": {
            "artifact_id": "packet-readiness:3.1",
            "schema_version": "1.0",
            "sq_id": "3.1",
            "status": status,
            "mechanical_completeness": {
                "status": "incomplete" if missing else "complete",
                "missing_evidence": missing or [],
                "negative_flags": [],
                "contradictions": [],
            },
            "semantic_adequacy": {
                "status": "adequate",
                "support_levels": ["moderate"],
                "confidence": confidence,
            },
            "blocking_reason": "",
        },
    }


def test_retrieval_repair_agent_is_not_triggered_for_ready_packets():
    result = retrieval_repair_node(
        {"outcome": "Overall survival", "evidence_packets": {"3.1": _packet()}}
    )

    assert result == {
        "retrieval_repair_artifacts": {},
        "evidence_packets": {},
        "packet_grades": {},
        "packet_readiness": {},
    }


def test_retrieval_repair_agent_emits_valid_three_query_artifact_for_incomplete_packet():
    result = retrieval_repair_node(
        {
            "outcome": "Overall survival",
            "evidence_packets": {
                "3.1": _packet(
                    status="needs_retrieval_repair",
                    missing=["denominator_or_percentage"],
                    confidence=0.2,
                )
            },
        }
    )

    artifact = result["retrieval_repair_artifacts"]["3.1"]
    validated = RetrievalRepairArtifactRecord.model_validate(artifact)

    assert validated.artifact_id == "retrieval-repair:d3:3.1"
    assert validated.trigger_conditions == [
        "low_packet_confidence",
        "missing_required_evidence",
    ]
    assert [query.purpose for query in validated.query_payload.queries] == [
        "required_evidence",
        "outcome_binding",
        "source_hierarchy",
    ]


def test_retrieval_repair_artifact_rejects_extra_query_payload_fields():
    artifact = retrieval_repair_node(
        {
            "outcome": "Overall survival",
            "evidence_packets": {
                "3.1": _packet(
                    status="needs_retrieval_repair",
                    missing=["denominator_or_percentage"],
                    confidence=0.2,
                )
            },
        }
    )["retrieval_repair_artifacts"]["3.1"]
    artifact["query_payload"]["queries"][0]["temperature"] = 0.2

    try:
        RetrievalRepairArtifactRecord.model_validate(artifact)
    except ValidationError:
        return

    raise AssertionError("extra query payload fields should be rejected")


def test_retrieval_repair_updates_one_packet_and_records_source_changes():
    repaired_source = {
        "text": (
            "Overall survival outcome data were available for 798/800 "
            "randomized participants at final follow-up."
        ),
        "section": "Results",
        "page_numbers": [6],
        "score": 0.01,
        "source_kind": "rag_chunk",
        "document_id": "primary",
        "document_name": "Primary paper",
        "document_role": "primary",
        "source_path": "trial.pdf",
    }

    result = retrieval_repair_node(
        {
            "outcome": "Overall survival",
            "rag_chunk_metadata": {"d3": [repaired_source]},
            "evidence_packets": {
                "3.1": _packet(
                    status="needs_retrieval_repair",
                    missing=["denominator_or_percentage"],
                    confidence=0.2,
                    sources=[],
                )
            },
        }
    )

    repaired_packet = result["evidence_packets"]["3.1"]
    artifact = result["retrieval_repair_artifacts"]["3.1"]

    assert repaired_packet["packet_readiness"]["status"] == "ready"
    assert repaired_packet["missing_evidence"] == []
    assert result["packet_readiness"]["3.1"]["status"] == "ready"
    assert result["packet_grades"]["3.1"]["retry_recommended"] is False
    assert artifact["before_packet_status"]["status"] == "needs_retrieval_repair"
    assert artifact["after_packet_status"]["status"] == "ready"
    assert artifact["source_changes"]["added_source_ids"] == ["primary:Results:6"]
