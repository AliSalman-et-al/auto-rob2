from rob2_pipeline.pipeline import _assessment_json


def test_assessment_json_includes_supplement_fields():
    state = {
        "pdf_path": "paper.pdf",
        "supplementary_paths": ["protocol.pdf"],
        "source_documents": [
            {
                "document_id": "supplement:001",
                "document_name": "protocol.pdf",
                "document_role": "protocol",
                "status": "parsed",
            }
        ],
        "parse_artifacts": [
            {
                "source_identity": {
                    "document_id": "primary",
                    "document_name": "paper.pdf",
                    "document_role": "primary",
                },
                "pages": [{"page_number": 1, "text": "Primary text"}],
                "diagnostics": [],
                "provenance": {
                    "parser_name": "liteparse",
                    "parser_version": "2.0.0",
                    "adapter_name": "liteparse",
                    "artifact_schema_version": "parse-artifact-v1",
                    "config": {},
                },
            }
        ],
        "supplement_warnings": [],
        "rag_chunk_metadata": {},
    }

    data = _assessment_json(state)

    assert data["supplementary_paths"] == ["protocol.pdf"]
    assert data["source_documents"][0]["document_name"] == "protocol.pdf"
    assert data["parse_artifacts"][0]["provenance"]["parser_name"] == "liteparse"
    assert data["supplement_warnings"] == []


def test_assessment_json_preserves_sq_support_metadata():
    data = _assessment_json(
        {
            "sq_answers": {
                "1.1": {
                    "answer": "Y",
                    "quote": "Randomized",
                    "justification": "Stated randomized.",
                    "uncertainty_flag": "NORMAL",
                    "support_level": "strong",
                    "support_rationale": "Direct quote supports the answer.",
                }
            },
            "rag_chunk_metadata": {},
        }
    )

    assert data["sq_answers"]["1.1"]["support_level"] == "strong"
    assert (
        data["sq_answers"]["1.1"]["support_rationale"]
        == "Direct quote supports the answer."
    )


def test_assessment_json_preserves_support_constraints():
    state = {
        "support_constraints": [
            {
                "constraint_type": "quote_untraceable",
                "sq_id": "1.1",
                "claim": {"answer": "Y", "support_level": "strong"},
                "evidence_label": "quote",
                "evidence": "Randomized centrally.",
                "reason": "quote_not_found_in_source_context",
            }
        ],
        "rag_chunk_metadata": {},
    }

    data = _assessment_json(state)

    assert data["support_constraints"] == state["support_constraints"]


def test_assessment_json_preserves_outcome_classification_support():
    state = {
        "outcome_classification_support": {
            "support_level": "strong",
            "support_rationale": "Direct outcome-bound quote.",
            "quotes": [{"quote": "Overall survival was death from any cause."}],
            "constraints": [],
        },
        "rag_chunk_metadata": {},
    }

    data = _assessment_json(state)

    assert (
        data["outcome_classification_support"]
        == state["outcome_classification_support"]
    )
