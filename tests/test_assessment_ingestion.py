from langchain_core.documents import Document

from rob2_pipeline.ingestion.assessment import AssessmentIngestionResult
from rob2_pipeline.models import empty_paper_evidence


LLM_LOG = {
    "node": "paper_evidence_extraction",
    "prompt_length_chars": 120,
    "response_length_chars": 80,
    "latency_ms": 5,
    "cache_hit": False,
}


def _doc_repr(text: str):
    return type(
        "DocRepr",
        (),
        {
            "full_text": text,
            "to_prompt_repr": lambda self: text,
            "blocks": [],
        },
    )()


def _result():
    return type("Result", (), {"document": object()})()


def _parse_artifact(source):
    return type(
        "Artifact",
        (),
        {
            "to_dict": lambda self: {
                "source_identity": dict(source),
                "pages": [
                    {
                        "page_number": 1,
                        "text": f"{source['document_id']} parse text",
                    }
                ],
                "diagnostics": [],
                "provenance": {
                    "parser_name": "liteparse",
                    "parser_version": "2.0.0",
                    "adapter_name": "liteparse",
                    "artifact_schema_version": "parse-artifact-v1",
                    "config": {},
                },
            }
        },
    )()


def _patch_primary_success(
    monkeypatch,
    assessment,
    *,
    text: str = "Primary text",
    structural=None,
    chunks=None,
):
    monkeypatch.setattr(assessment, "extract_full_text", lambda path: text)
    monkeypatch.setattr(assessment, "_configure_docling_runtime", lambda: None)
    monkeypatch.setattr(
        assessment, "_build_docling_chunks", lambda conv_result: chunks or []
    )
    monkeypatch.setattr(assessment, "build_document_repr", lambda doc: _doc_repr(text))
    monkeypatch.setattr(
        assessment,
        "extract_structural_paper_evidence",
        lambda doc_repr: structural or empty_paper_evidence("docling_struct"),
    )

    class Converter:
        def convert(self, path):
            return _result()

    monkeypatch.setattr(
        assessment, "_get_docling_converter", lambda use_ocr=False: Converter()
    )


def test_assessment_ingestion_result_to_state_update_omits_empty_llm_log():
    evidence = empty_paper_evidence("docling_struct")
    result = AssessmentIngestionResult(
        full_text="Primary text",
        evidence=evidence,
        docling_doc=None,
        docling_chunks=[],
        source_documents=[],
        supplement_warnings=[],
    )

    assert result.to_state_update() == {
        "full_text": "Primary text",
        "evidence": evidence,
        "docling_doc": None,
        "docling_chunks": [],
        "source_documents": [],
        "parse_artifacts": [],
        "supplement_warnings": [],
    }


def test_assessment_ingestion_result_to_state_update_includes_llm_log_when_present():
    evidence = empty_paper_evidence("docling_llm")
    result = AssessmentIngestionResult(
        full_text="Primary text",
        evidence=evidence,
        docling_doc=None,
        docling_chunks=[],
        source_documents=[],
        supplement_warnings=[],
        llm_call_log=[LLM_LOG],
    )

    assert result.to_state_update()["llm_call_log"] == [LLM_LOG]


def test_ingest_assessment_documents_returns_primary_structural_result_when_remote_disabled(
    monkeypatch,
):
    import rob2_pipeline.ingestion.assessment as assessment

    evidence = empty_paper_evidence("docling_struct")
    primary_chunk = Document(
        page_content="Primary chunk",
        metadata={"section": "Methods", "page_numbers": [1]},
    )
    _patch_primary_success(
        monkeypatch, assessment, structural=evidence, chunks=[primary_chunk]
    )
    monkeypatch.setattr(assessment, "allow_remote_evidence_extraction", lambda: False)
    monkeypatch.setattr(assessment, "ingest_supplements", lambda paths: ([], [], []))
    monkeypatch.setattr(
        assessment, "parse_sources", lambda sources: [_parse_artifact(sources[0])]
    )

    result = assessment.ingest_assessment_documents("primary.pdf", [])

    assert result.full_text == "Primary text"
    assert result.evidence is evidence
    assert len(result.docling_chunks) == 1
    assert result.docling_chunks[0].metadata["document_id"] == "primary"
    assert result.source_documents == [
        {
            "document_id": "primary",
            "document_name": "primary.pdf",
            "document_role": "primary",
            "source_kind": "rag_chunk",
            "path": "primary.pdf",
            "is_primary": True,
            "status": "parsed",
        }
    ]
    assert result.supplement_warnings == []
    assert result.llm_call_log == []
    assert result.parse_artifacts[0]["source_identity"]["document_id"] == "primary"
    assert result.parse_artifacts[0]["pages"][0]["text"] == "primary parse text"


def test_ingest_assessment_documents_preserves_primary_when_supplement_ingestion_escapes(
    monkeypatch,
):
    import rob2_pipeline.ingestion.assessment as assessment

    primary_chunk = Document(
        page_content="Primary chunk",
        metadata={"section": "Methods", "page_numbers": [1]},
    )
    _patch_primary_success(monkeypatch, assessment, chunks=[primary_chunk])
    monkeypatch.setattr(assessment, "allow_remote_evidence_extraction", lambda: False)
    monkeypatch.setattr(
        assessment,
        "ingest_supplements",
        lambda paths: (_ for _ in ()).throw(
            RuntimeError("unexpected supplement error")
        ),
    )

    result = assessment.ingest_assessment_documents("primary.pdf", ["protocol.pdf"])

    assert len(result.docling_chunks) == 1
    assert result.docling_chunks[0].metadata["document_id"] == "primary"
    assert result.source_documents[0]["document_role"] == "primary"
    assert result.supplement_warnings == [
        "Supplement ingestion failed: unexpected supplement error"
    ]


def test_ingest_assessment_documents_persists_primary_and_supplement_parse_artifacts(
    monkeypatch,
):
    import rob2_pipeline.ingestion.assessment as assessment

    _patch_primary_success(monkeypatch, assessment)
    monkeypatch.setattr(assessment, "allow_remote_evidence_extraction", lambda: False)
    monkeypatch.setattr(
        assessment,
        "ingest_supplements",
        lambda paths: (
            [],
            [
                {
                    "document_id": "supplement:001",
                    "document_name": "protocol.pdf",
                    "document_role": "protocol",
                    "source_kind": "rag_chunk",
                    "path": "protocol.pdf",
                    "is_primary": False,
                    "status": "parsed",
                }
            ],
            [],
        ),
    )
    monkeypatch.setattr(
        assessment,
        "parse_sources",
        lambda sources: [_parse_artifact(source) for source in sources],
    )

    result = assessment.ingest_assessment_documents("primary.pdf", ["protocol.pdf"])

    assert [
        artifact["source_identity"]["document_id"]
        for artifact in result.parse_artifacts
    ] == ["primary", "supplement:001"]
    assert result.to_state_update()["parse_artifacts"] == result.parse_artifacts


def test_ingest_assessment_documents_skips_remote_extraction_for_apparent_non_rct(
    monkeypatch,
):
    import rob2_pipeline.ingestion.assessment as assessment

    _patch_primary_success(monkeypatch, assessment, text="Editorial commentary.")
    monkeypatch.setattr(assessment, "allow_remote_evidence_extraction", lambda: True)
    monkeypatch.setattr(assessment, "appears_rct_candidate", lambda text: False)
    monkeypatch.setattr(assessment, "ingest_supplements", lambda paths: ([], [], []))

    def fail_if_called(doc_repr):
        raise AssertionError("remote extraction should be skipped")

    monkeypatch.setattr(assessment, "extract_paper_evidence", fail_if_called)

    result = assessment.ingest_assessment_documents("primary.pdf", [])

    assert result.evidence["extraction_method"] == "docling_struct"
    assert (
        "Remote evidence extraction skipped for apparent non-RCT document."
        in result.evidence["warnings"]
    )


def test_ingest_assessment_documents_returns_llm_evidence_and_log(monkeypatch):
    import rob2_pipeline.ingestion.assessment as assessment

    remote = empty_paper_evidence("docling_llm")
    _patch_primary_success(monkeypatch, assessment, text="Randomized trial.")
    monkeypatch.setattr(assessment, "allow_remote_evidence_extraction", lambda: True)
    monkeypatch.setattr(assessment, "appears_rct_candidate", lambda text: True)
    monkeypatch.setattr(
        assessment,
        "extract_paper_evidence",
        lambda doc_repr: (remote, [LLM_LOG]),
    )
    monkeypatch.setattr(assessment, "ingest_supplements", lambda paths: ([], [], []))

    result = assessment.ingest_assessment_documents("primary.pdf", [])

    assert result.evidence is remote
    assert result.llm_call_log == [LLM_LOG]


def test_ingest_assessment_documents_falls_back_when_remote_extraction_fails(
    monkeypatch,
):
    import rob2_pipeline.ingestion.assessment as assessment

    _patch_primary_success(monkeypatch, assessment, text="Randomized trial.")
    monkeypatch.setattr(assessment, "allow_remote_evidence_extraction", lambda: True)
    monkeypatch.setattr(assessment, "appears_rct_candidate", lambda text: True)
    monkeypatch.setattr(
        assessment,
        "extract_paper_evidence",
        lambda doc_repr: (_ for _ in ()).throw(RuntimeError("bad xml")),
    )
    monkeypatch.setattr(assessment, "ingest_supplements", lambda paths: ([], [], []))

    result = assessment.ingest_assessment_documents("primary.pdf", [])

    assert result.evidence["extraction_method"] == "docling_struct"
    assert "LLM evidence extraction failed: bad xml" in result.evidence["warnings"]


def test_ingest_assessment_documents_falls_back_to_keyword_parse_when_docling_structure_fails(
    monkeypatch,
):
    import rob2_pipeline.ingestion.assessment as assessment

    known_text = (
        "Methods\nParticipants were randomly assigned in a 1:1 ratio.\nResults\nDone."
    )
    monkeypatch.setattr(assessment, "extract_full_text", lambda path: known_text)

    class BrokenConverter:
        def convert(self, path):
            raise RuntimeError("docling structured parse failed")

    monkeypatch.setattr(
        assessment, "_get_docling_converter", lambda use_ocr=False: BrokenConverter()
    )

    result = assessment.ingest_assessment_documents(
        "trial.pdf",
        ["inputs/benchmark/supplement/TITAN/protocol.pdf"],
    )

    assert result.evidence["extraction_method"] == "fallback"
    assert result.docling_doc is None
    assert result.docling_chunks == []
    assert "randomly assigned" in result.evidence["methods"]["text"]
    assert result.source_documents[0]["document_role"] == "primary"
    assert result.source_documents[1]["document_name"] == "protocol.pdf"
    assert result.source_documents[1]["status"] == "failed"
    assert "Supplement not ingested" in result.supplement_warnings[0]
