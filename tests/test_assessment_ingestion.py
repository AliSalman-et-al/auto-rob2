from rob2_pipeline.ingestion.assessment import AssessmentIngestionResult
from rob2_pipeline.ingestion.parse_artifacts import (
    ParserProvenance,
    SourceParseArtifact,
)
from rob2_pipeline.models import empty_paper_evidence


LLM_LOG = {
    "node": "paper_evidence_extraction",
    "prompt_length_chars": 120,
    "response_length_chars": 80,
    "latency_ms": 5,
    "cache_hit": False,
}


def _source_parse_artifact(source, pages, *, status="parsed", error=""):
    source_identity = {**source, "status": status}
    if error:
        source_identity["error"] = error
    return SourceParseArtifact(
        source_identity=source_identity,
        pages=pages,
        diagnostics=[],
        provenance=ParserProvenance(
            parser_name="fake-liteparse",
            parser_version="1.0.0",
            adapter_name="fake",
            artifact_schema_version="parse-artifact-v1",
            config={},
        ),
    )


def test_assessment_ingestion_result_to_state_update_omits_empty_llm_log():
    evidence = empty_paper_evidence("parse_artifact")
    result = AssessmentIngestionResult(
        full_text="Primary text",
        evidence=evidence,
        docling_chunks=[],
        source_documents=[],
        supplement_warnings=[],
    )

    assert result.to_state_update() == {
        "full_text": "Primary text",
        "evidence": evidence,
        "docling_chunks": [],
        "source_documents": [],
        "parse_artifacts": [],
        "supplement_warnings": [],
    }


def test_assessment_ingestion_result_to_state_update_includes_llm_log_when_present():
    evidence = empty_paper_evidence("parse_artifact")
    result = AssessmentIngestionResult(
        full_text="Primary text",
        evidence=evidence,
        docling_chunks=[],
        source_documents=[],
        supplement_warnings=[],
        llm_call_log=[LLM_LOG],
    )

    assert result.to_state_update()["llm_call_log"] == [LLM_LOG]


def test_ingest_assessment_documents_prefers_parser_neutral_artifacts(
    monkeypatch,
    tmp_path,
):
    import rob2_pipeline.ingestion.assessment as assessment

    primary = tmp_path / "trial.pdf"
    protocol = tmp_path / "trial_protocol.pdf"
    primary.write_bytes(b"%PDF-1.4")
    protocol.write_bytes(b"%PDF-1.4")

    def parse_sources(sources):
        artifacts = []
        for source in sources:
            pages = (
                [
                    {
                        "page_number": 1,
                        "text": "Methods\nParticipants were randomly assigned.",
                    },
                    {
                        "page_number": 2,
                        "text": "Results\nThe primary outcome was reported.",
                    },
                ]
                if source["document_id"] == "primary"
                else [
                    {
                        "page_number": 3,
                        "text": "Protocol\nAllocation was concealed centrally.",
                    }
                ]
            )
            artifacts.append(_source_parse_artifact(source, pages))
        return artifacts

    monkeypatch.setattr(assessment, "parse_sources", parse_sources)
    monkeypatch.setattr(assessment, "allow_remote_evidence_extraction", lambda: False)

    result = assessment.ingest_assessment_documents(str(primary), [str(protocol)])

    assert "Participants were randomly assigned" in result.full_text
    assert result.evidence["extraction_method"] == "parse_artifact"
    assert [source["document_id"] for source in result.source_documents] == [
        "primary",
        "supplement:001",
    ]
    assert [source["status"] for source in result.source_documents] == [
        "parsed",
        "parsed",
    ]
    assert result.supplement_warnings == []
    assert [chunk.metadata["document_id"] for chunk in result.docling_chunks] == [
        "primary",
        "primary",
        "supplement:001",
    ]
    assert result.docling_chunks[0].metadata["section"] == "METHODS"
    assert result.docling_chunks[0].metadata["original_heading"] == "Methods"
    assert result.docling_chunks[0].metadata["page_numbers"] == [1]
    assert result.docling_chunks[2].metadata["document_role"] == "protocol"
    assert result.parse_artifacts[0]["pages"][0]["text"].startswith("Methods")


def test_ingest_assessment_documents_keeps_failed_supplements_best_effort(
    monkeypatch,
    tmp_path,
):
    import rob2_pipeline.ingestion.assessment as assessment

    primary = tmp_path / "trial.pdf"
    missing = tmp_path / "missing_protocol.pdf"
    primary.write_bytes(b"%PDF-1.4")

    def parse_sources(sources):
        artifacts = []
        for source in sources:
            if source["document_id"] == "primary":
                artifacts.append(
                    _source_parse_artifact(
                        source,
                        [
                            {
                                "page_number": 1,
                                "text": "Methods\nRandomized trial text with enough content to parse.",
                            }
                        ],
                    )
                )
            else:
                artifacts.append(
                    _source_parse_artifact(
                        source,
                        [],
                        status="failed",
                        error="file missing",
                    )
                )
        return artifacts

    monkeypatch.setattr(assessment, "parse_sources", parse_sources)
    monkeypatch.setattr(assessment, "allow_remote_evidence_extraction", lambda: False)

    result = assessment.ingest_assessment_documents(str(primary), [str(missing)])

    assert result.source_documents[1]["status"] == "failed"
    assert result.supplement_warnings == ["file missing"]
    assert [chunk.metadata["document_id"] for chunk in result.docling_chunks] == [
        "primary"
    ]


def test_ingest_assessment_documents_raises_when_primary_parse_fails(
    monkeypatch,
    tmp_path,
):
    import rob2_pipeline.ingestion.assessment as assessment

    primary = tmp_path / "trial.pdf"
    primary.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(
        assessment,
        "parse_sources",
        lambda sources: [
            _source_parse_artifact(
                sources[0],
                [],
                status="failed",
                error="parser failed",
            )
        ],
    )

    try:
        assessment.ingest_assessment_documents(str(primary), [])
    except RuntimeError as error:
        assert "Primary PDF parsing failed" in str(error)
    else:
        raise AssertionError("Expected primary parse failure to stop ingestion")
