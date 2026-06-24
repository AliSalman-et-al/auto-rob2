from rob2_pipeline.ingestion.assessment import AssessmentIngestionResult
from rob2_pipeline.ingestion import supplement_segments
from rob2_pipeline.ingestion.parse_artifacts import (
    PARSE_ARTIFACT_SCHEMA_VERSION,
    ParserDiagnostic,
    ParserProvenance,
    SourceParseArtifact,
)
from rob2_pipeline.models import empty_paper_evidence
from rob2_pipeline.supplement_retrieval import SupplementSegment


LLM_LOG = {
    "node": "paper_evidence_extraction",
    "prompt_length_chars": 120,
    "response_length_chars": 80,
    "latency_ms": 5,
    "cache_hit": False,
}


def _source_parse_artifact(
    source, pages, *, status="parsed", error="", diagnostics=None
):
    source_identity = {**source, "status": status}
    if error:
        source_identity["error"] = error
    return SourceParseArtifact(
        source_identity=source_identity,
        pages=pages,
        diagnostics=list(diagnostics or []),
        provenance=ParserProvenance(
            parser_name="fake-pymupdf",
            parser_version="1.0.0",
            adapter_name="fake-sectionmap",
            artifact_schema_version=PARSE_ARTIFACT_SCHEMA_VERSION,
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
        "source_documents": [],
        "parse_artifacts": [],
        "supplement_warnings": [],
        "supplement_segments": [],
        "supplement_indexes": {},
        "supplement_retrieval_grades": {},
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


def test_assessment_ingestion_result_rebuilds_supplement_indexes_from_segments():
    evidence = empty_paper_evidence("parse_artifact")
    segment = SupplementSegment(
        segment_id="supplement:001:segment:0001",
        document_id="supplement:001",
        document_name="protocol.pdf",
        document_role="protocol",
        source_path="protocol.pdf",
        heading="Protocol",
        page_numbers=[1],
        domain_tags=["D1"],
        annotation="Central allocation evidence.",
        text="Central allocation concealment was used.",
    )
    result = AssessmentIngestionResult(
        full_text="Primary text",
        evidence=evidence,
        docling_chunks=[],
        source_documents=[],
        supplement_warnings=[],
        supplement_segments=[segment.to_dict()],
        supplement_indexes={},
    )

    update = result.to_state_update()

    assert update["supplement_segments"] == [segment.to_dict()]
    assert set(update["supplement_indexes"]) == {"supplement:001"}
    retrieved = update["supplement_indexes"]["supplement:001"].retrieve(
        "allocation concealment", domain="d1"
    )
    assert retrieved["segments"][0]["segment_id"] == segment.segment_id


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


def test_ingest_assessment_documents_produces_content_detected_supplement_segments(
    monkeypatch,
    tmp_path,
):
    import rob2_pipeline.ingestion.assessment as assessment

    primary = tmp_path / "trial.pdf"
    supplement = tmp_path / "generic-file.pdf"
    primary.write_bytes(b"%PDF-1.4")
    supplement.write_bytes(b"%PDF-1.4")

    def parse_sources(sources):
        return [
            _source_parse_artifact(
                sources[0],
                [
                    {
                        "page_number": 1,
                        "text": "Methods\nRandomized trial text with enough content to parse.",
                    }
                ],
            ),
            _source_parse_artifact(
                sources[1],
                [
                    {
                        "page_number": 1,
                        "text": (
                            "Clinical Trial Protocol\n"
                            "Randomisation and allocation were concealed centrally.\n"
                            "Blinding\n"
                            "Participants and clinicians were masked.\n"
                            "Statistical Analysis\n"
                            "The primary outcome analysis used the intention-to-treat population."
                        ),
                        "section_header_boxes": [
                            {
                                "text": "Clinical Trial Protocol",
                                "bbox": [36, 40, 220, 52],
                            },
                            {"text": "Blinding", "bbox": [36, 96, 92, 108]},
                            {
                                "text": "Statistical Analysis",
                                "bbox": [36, 144, 170, 156],
                            },
                        ],
                    }
                ],
            ),
        ]

    monkeypatch.setattr(assessment, "parse_sources", parse_sources)
    monkeypatch.setattr(assessment, "allow_remote_evidence_extraction", lambda: False)

    result = assessment.ingest_assessment_documents(str(primary), [str(supplement)])

    assert result.source_documents[1]["document_role"] == "protocol"
    assert [segment["heading"] for segment in result.supplement_segments] == [
        "Clinical Trial Protocol",
        "Blinding",
        "Statistical Analysis",
    ]
    assert result.supplement_segments[0]["domain_tags"] == ["D1"]
    assert result.supplement_segments[1]["domain_tags"] == ["D2", "D4"]
    assert {"D2", "D5"}.issubset(result.supplement_segments[2]["domain_tags"])
    assert all(segment["annotation"] for segment in result.supplement_segments)
    assert result.supplement_warnings == []


def test_ingest_assessment_documents_uses_all_domain_fallback_for_sparse_supplement(
    monkeypatch,
    tmp_path,
):
    import rob2_pipeline.ingestion.assessment as assessment

    primary = tmp_path / "trial.pdf"
    supplement = tmp_path / "appendix.pdf"
    primary.write_bytes(b"%PDF-1.4")
    supplement.write_bytes(b"%PDF-1.4")

    def parse_sources(sources):
        return [
            _source_parse_artifact(
                sources[0],
                [
                    {
                        "page_number": 1,
                        "text": "Methods\nRandomized trial text with enough content to parse.",
                    }
                ],
            ),
            _source_parse_artifact(
                sources[1],
                [
                    {
                        "page_number": 4,
                        "text": "Supplementary Appendix\nCentral allocation was used.",
                        "section_header_boxes": [
                            {
                                "text": "Supplementary Appendix",
                                "bbox": [36, 40, 220, 52],
                            },
                        ],
                    }
                ],
            ),
        ]

    monkeypatch.setattr(assessment, "parse_sources", parse_sources)
    monkeypatch.setattr(assessment, "allow_remote_evidence_extraction", lambda: False)

    result = assessment.ingest_assessment_documents(str(primary), [str(supplement)])

    assert len(result.supplement_segments) == 1
    segment = result.supplement_segments[0]
    # Per ADR-0006 change 1, the sparse-supplement fallback now emits one
    # all-domain-tagged segment per page (heading "Page N") instead of a single
    # uncapped "Full document" segment, so BM25S can retrieve selectively.
    assert segment["heading"] == "Page 4"
    assert segment["page_numbers"] == [4]
    assert segment["domain_tags"] == ["D1", "D2", "D3", "D4", "D5"]
    assert "Supplementary Appendix" in segment["text"]


def test_ingest_assessment_documents_warns_and_uses_fallback_annotations_over_cap(
    monkeypatch,
    tmp_path,
):
    import rob2_pipeline.ingestion.assessment as assessment
    import rob2_pipeline.ingestion.supplement_segments as supplement_segments

    primary = tmp_path / "trial.pdf"
    supplement = tmp_path / "sap.pdf"
    primary.write_bytes(b"%PDF-1.4")
    supplement.write_bytes(b"%PDF-1.4")
    headings = [f"Section {index}" for index in range(1, 6)]
    text = "\n".join(
        f"{heading}\nAdministrative content without methodological detail."
        for heading in headings
    )

    def parse_sources(sources):
        return [
            _source_parse_artifact(
                sources[0],
                [
                    {
                        "page_number": 1,
                        "text": "Methods\nRandomized trial text with enough content to parse.",
                    }
                ],
            ),
            _source_parse_artifact(
                sources[1],
                [
                    {
                        "page_number": 1,
                        "text": "Statistical Analysis Plan\n" + text,
                        "section_header_boxes": [
                            {
                                "text": "Statistical Analysis Plan",
                                "bbox": [36, 20, 220, 32],
                            },
                            *[
                                {
                                    "text": heading,
                                    "bbox": [36, 40 + index * 20, 120, 52 + index * 20],
                                }
                                for index, heading in enumerate(headings, start=1)
                            ],
                        ],
                    }
                ],
            ),
        ]

    monkeypatch.setattr(assessment, "parse_sources", parse_sources)
    monkeypatch.setattr(assessment, "allow_remote_evidence_extraction", lambda: False)
    monkeypatch.setattr(supplement_segments, "ANNOTATION_CAP_PER_SUPPLEMENT", 2)

    result = assessment.ingest_assessment_documents(str(primary), [str(supplement)])

    assert len(result.supplement_segments) == 5
    assert any(
        "exceeded annotation cap" in warning for warning in result.supplement_warnings
    )
    assert any(
        segment["annotation"] == supplement_segments.FALLBACK_ANNOTATION
        for segment in result.supplement_segments
    )


def test_supplement_annotation_prompt_blocks_include_cache_control_preamble():
    segment = SupplementSegment(
        segment_id="protocol:segment:0001",
        document_id="protocol",
        document_name="Protocol.pdf",
        document_role="protocol",
        source_path="Protocol.pdf",
        heading="Randomisation",
        page_numbers=[3],
        domain_tags=["D1"],
        annotation="",
        text="Participants were randomized centrally.",
    )

    blocks = supplement_segments.supplement_annotation_user_blocks(
        segment,
        document_preamble="Protocol PDF parsed into page-aware segments.",
    )

    assert blocks[0] == {
        "type": "text",
        "text": "Protocol PDF parsed into page-aware segments.",
        "cache_control": {"type": "ephemeral"},
    }
    assert blocks[1]["type"] == "text"
    assert "Participants were randomized centrally." in blocks[1]["text"]


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


def test_ingest_assessment_documents_keeps_degraded_supplements_best_effort(
    monkeypatch,
    tmp_path,
):
    import rob2_pipeline.ingestion.assessment as assessment

    primary = tmp_path / "trial.pdf"
    supplement = tmp_path / "trial_supplement.pdf"
    primary.write_bytes(b"%PDF-1.4")
    supplement.write_bytes(b"%PDF-1.4")

    def parse_sources(sources):
        return [
            _source_parse_artifact(
                sources[0],
                [
                    {
                        "page_number": 1,
                        "text": "Methods\nRandomized trial text with enough content to parse.",
                    }
                ],
            ),
            _source_parse_artifact(
                sources[1],
                [
                    {
                        "page_number": 4,
                        "text": "Appendix\nAllocation concealment details were partially recovered.",
                    }
                ],
                status="degraded",
                error="layout recovery degraded",
                diagnostics=[
                    ParserDiagnostic(
                        level="warning",
                        message="page 4 has overlapping text",
                        page_number=4,
                    )
                ],
            ),
        ]

    monkeypatch.setattr(assessment, "parse_sources", parse_sources)
    monkeypatch.setattr(assessment, "allow_remote_evidence_extraction", lambda: False)

    result = assessment.ingest_assessment_documents(str(primary), [str(supplement)])

    assert result.source_documents[1]["status"] == "degraded"
    assert result.supplement_warnings == ["layout recovery degraded"]
    assert result.parse_artifacts[1]["provenance"]["artifact_schema_version"] == (
        PARSE_ARTIFACT_SCHEMA_VERSION
    )
    assert result.parse_artifacts[1]["diagnostics"] == [
        {
            "level": "warning",
            "message": "page 4 has overlapping text",
            "page_number": 4,
        }
    ]
    assert [chunk.metadata["document_id"] for chunk in result.docling_chunks] == [
        "primary",
        "supplement:001",
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


def test_ingest_assessment_documents_raises_when_primary_parse_is_degraded(
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
                [
                    {
                        "page_number": 1,
                        "text": "Methods\nRandomized trial text with enough content to parse.",
                    }
                ],
                status="degraded",
                error="layout recovery degraded",
            )
        ],
    )

    try:
        assessment.ingest_assessment_documents(str(primary), [])
    except RuntimeError as error:
        assert "Primary PDF parsing failed" in str(error)
    else:
        raise AssertionError("Expected degraded primary parse to stop ingestion")
