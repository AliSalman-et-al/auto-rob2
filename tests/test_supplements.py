from pathlib import Path

from langchain_core.documents import Document

from rob2_pipeline.ingestion.supplements import (
    DEFAULT_SUPPLEMENT_MAX_SCAN_PAGES,
    DEFAULT_SUPPLEMENT_PAGE_WINDOW,
    apply_source_metadata,
    build_source_document,
    classify_supplement,
    ingest_supplements,
    skipped_source_documents,
)


def test_classify_supplement_from_filename():
    cases = {
        "nejmoa1903307_protocol.pdf": "protocol",
        "trial_statistical_analysis_plan.pdf": "sap",
        "nejmoa1903307_appendix.pdf": "appendix",
        "mmc1.pdf": "appendix",
        "nejmoa1903307_disclosures.pdf": "disclosure",
        "nejmoa1903307_data-sharing.pdf": "data_sharing",
        "ds_jco.19.00799.pdf": "data_sharing",
        "dss_jco.21.02517.pdf": "data_sharing",
        "unlabeled-file.pdf": "unknown_supplement",
    }

    for filename, expected in cases.items():
        assert classify_supplement(Path(filename)) == expected


def test_build_source_document_uses_stable_supplement_id():
    source = build_source_document(
        Path("inputs/benchmark/supplement/TITAN/nejmoa1903307_protocol.pdf"),
        role="protocol",
        index=1,
    )

    assert source["document_id"] == "supplement:001"
    assert source["document_name"] == "nejmoa1903307_protocol.pdf"
    assert source["document_role"] == "protocol"
    assert source["source_kind"] == "rag_chunk"
    assert source["is_primary"] is False
    assert source["status"] == "pending"


def test_apply_source_metadata_preserves_existing_chunk_metadata():
    chunks = [
        Document(
            page_content="Protocol text.",
            metadata={"section": "Methods", "page_numbers": [3]},
        )
    ]
    source = build_source_document(Path("protocol.pdf"), role="protocol", index=1)

    result = apply_source_metadata(chunks, source)

    assert result[0].metadata["section"] == "Methods"
    assert result[0].metadata["page_numbers"] == [3]
    assert result[0].metadata["document_id"] == "supplement:001"
    assert result[0].metadata["document_name"] == "protocol.pdf"
    assert result[0].metadata["document_role"] == "protocol"
    assert result[0].metadata["source_kind"] == "rag_chunk"


def test_ingest_supplements_records_missing_file_warning(tmp_path):
    missing = tmp_path / "missing_protocol.pdf"

    chunks, documents, warnings = ingest_supplements([str(missing)])

    assert chunks == []
    assert documents[0]["document_name"] == "missing_protocol.pdf"
    assert documents[0]["status"] == "missing"
    assert "Supplement not found" in warnings[0]


def test_ingest_supplements_records_converter_setup_failure(tmp_path, monkeypatch):
    import rob2_pipeline.ingestion.supplements as supplements

    supplement = tmp_path / "protocol.pdf"
    supplement.write_bytes(b"%PDF-1.4")

    def fail_runtime():
        raise RuntimeError("converter setup failed")

    monkeypatch.setattr(
        "rob2_pipeline.pdf_ingestion._configure_docling_runtime", fail_runtime
    )

    chunks, documents, warnings = supplements.ingest_supplements([str(supplement)])

    assert chunks == []
    assert documents[0]["status"] == "failed"
    assert "converter setup failed" in documents[0]["error"]
    assert "Supplement parse failed" in warnings[0]


def test_ingest_supplements_uses_lightweight_converter_and_windowed_pages(
    tmp_path, monkeypatch
):
    import rob2_pipeline.ingestion.supplements as supplements

    supplement = tmp_path / "protocol.pdf"
    supplement.write_bytes(b"%PDF-1.4")
    calls = []

    class Converter:
        def convert(self, path, **kwargs):
            calls.append((path, kwargs))
            return type("Result", (), {"document": object()})()

    monkeypatch.setattr(supplements, "_get_supplement_converter", lambda: Converter())
    monkeypatch.setattr(
        "rob2_pipeline.pdf_ingestion._configure_docling_runtime", lambda: None
    )
    monkeypatch.setattr(
        "rob2_pipeline.pdf_ingestion._build_docling_chunks",
        lambda conv_result: [
            Document(
                page_content="Protocol methods.",
                metadata={"section": "Protocol", "page_numbers": [1]},
            )
        ],
    )

    chunks, documents, warnings = supplements.ingest_supplements([str(supplement)])

    assert chunks[0].metadata["document_role"] == "protocol"
    assert documents[0]["status"] == "parsed"
    assert warnings == []
    assert calls[0] == (
        str(supplement),
        {"page_range": (1, DEFAULT_SUPPLEMENT_PAGE_WINDOW)},
    )


def test_ingest_supplements_skips_failed_page_window_and_continues(
    tmp_path, monkeypatch
):
    import rob2_pipeline.ingestion.supplements as supplements

    supplement = tmp_path / "protocol.pdf"
    supplement.write_bytes(b"%PDF-1.4")
    calls = []

    class Converter:
        def convert(self, path, **kwargs):
            page_range = kwargs["page_range"]
            calls.append(page_range)
            if page_range == (3, 4):
                raise RuntimeError("std::bad_alloc")
            if page_range == (7, 8):
                raise RuntimeError("page range outside document")
            return type(
                "Result", (), {"document": object(), "page_range": page_range}
            )()

    def fake_chunks(conv_result):
        start, end = conv_result.page_range
        return [
            Document(
                page_content=f"Window {start}-{end}",
                metadata={"section": "Protocol", "page_numbers": [start]},
            )
        ]

    monkeypatch.setenv("ROB2_SUPPLEMENT_PAGE_WINDOW", "2")
    monkeypatch.setenv("ROB2_SUPPLEMENT_MAX_SCAN_PAGES", "8")
    monkeypatch.setattr(supplements, "_get_supplement_converter", lambda: Converter())
    monkeypatch.setattr(
        "rob2_pipeline.pdf_ingestion._configure_docling_runtime", lambda: None
    )
    monkeypatch.setattr(
        "rob2_pipeline.pdf_ingestion._build_docling_chunks", fake_chunks
    )

    chunks, documents, warnings = supplements.ingest_supplements([str(supplement)])

    assert calls == [(1, 2), (3, 4), (5, 6), (7, 8)]
    assert [chunk.page_content for chunk in chunks] == ["Window 1-2", "Window 5-6"]
    assert documents[0]["status"] == "partial"
    assert "std::bad_alloc" in documents[0]["error"]
    assert "std::bad_alloc" in warnings[0]


def test_ingest_supplements_continues_after_empty_window(tmp_path, monkeypatch):
    import rob2_pipeline.ingestion.supplements as supplements

    supplement = tmp_path / "protocol.pdf"
    supplement.write_bytes(b"%PDF-1.4")
    calls = []

    class Converter:
        def convert(self, path, **kwargs):
            page_range = kwargs["page_range"]
            calls.append(page_range)
            if page_range == (7, 8):
                raise RuntimeError("page range outside document")
            return type(
                "Result", (), {"document": object(), "page_range": page_range}
            )()

    def fake_chunks(conv_result):
        if conv_result.page_range == (1, 2):
            return []
        start, end = conv_result.page_range
        return [
            Document(
                page_content=f"Window {start}-{end}",
                metadata={"section": "Protocol", "page_numbers": [start]},
            )
        ]

    monkeypatch.setenv("ROB2_SUPPLEMENT_PAGE_WINDOW", "2")
    monkeypatch.setenv("ROB2_SUPPLEMENT_MAX_SCAN_PAGES", "8")
    monkeypatch.setattr(supplements, "_get_supplement_converter", lambda: Converter())
    monkeypatch.setattr(
        "rob2_pipeline.pdf_ingestion._configure_docling_runtime", lambda: None
    )
    monkeypatch.setattr(
        "rob2_pipeline.pdf_ingestion._build_docling_chunks", fake_chunks
    )

    chunks, documents, warnings = supplements.ingest_supplements([str(supplement)])

    assert calls == [(1, 2), (3, 4), (5, 6), (7, 8)]
    assert [chunk.page_content for chunk in chunks] == ["Window 3-4", "Window 5-6"]
    assert documents[0]["status"] == "parsed"
    assert warnings == []


def test_page_range_error_with_bad_alloc_is_not_exhausted():
    import rob2_pipeline.ingestion.supplements as supplements

    error = RuntimeError("failed converting page range 41-60: std::bad_alloc")

    assert supplements._is_page_range_exhausted(error) is False


def test_supplement_page_window_reads_positive_env_value(monkeypatch):
    import rob2_pipeline.ingestion.supplements as supplements

    monkeypatch.setenv("ROB2_SUPPLEMENT_PAGE_WINDOW", "12")

    assert supplements._supplement_page_window() == 12


def test_supplement_max_scan_pages_defaults_to_large_defensive_limit(monkeypatch):
    import rob2_pipeline.ingestion.supplements as supplements

    monkeypatch.delenv("ROB2_SUPPLEMENT_MAX_SCAN_PAGES", raising=False)
    monkeypatch.delenv("ROB2_SUPPLEMENT_MAX_PAGES", raising=False)

    assert supplements._supplement_max_scan_pages() == DEFAULT_SUPPLEMENT_MAX_SCAN_PAGES


def test_skipped_source_documents_records_requested_supplements():
    documents, warnings = skipped_source_documents(
        ["inputs/benchmark/supplement/TITAN/protocol.pdf"],
        "primary Docling structural extraction failed",
    )

    assert documents[0]["document_name"] == "protocol.pdf"
    assert documents[0]["document_role"] == "protocol"
    assert documents[0]["status"] == "failed"
    assert "primary Docling structural extraction failed" in documents[0]["error"]
    assert warnings == [documents[0]["error"]]


def test_assessment_ingestion_appends_supplement_chunks(monkeypatch):
    import rob2_pipeline.ingestion.assessment as node

    from rob2_pipeline.models import empty_paper_evidence

    monkeypatch.setattr(node, "extract_full_text", lambda path: "Primary text")
    monkeypatch.setattr(node, "_configure_docling_runtime", lambda: None)
    monkeypatch.setattr(
        node,
        "_build_docling_chunks",
        lambda conv_result: [
            Document(
                page_content="Primary chunk",
                metadata={"section": "Methods", "page_numbers": [1]},
            )
        ],
    )
    monkeypatch.setattr(
        node,
        "build_document_repr",
        lambda doc: type(
            "DocRepr",
            (),
            {
                "full_text": "Primary text",
                "to_prompt_repr": lambda self: "Primary text",
                "blocks": [],
            },
        )(),
    )
    monkeypatch.setattr(
        node,
        "extract_structural_paper_evidence",
        lambda doc_repr: empty_paper_evidence("docling_struct"),
    )
    monkeypatch.setattr(node, "allow_remote_evidence_extraction", lambda: False)
    monkeypatch.setattr(
        node,
        "ingest_supplements",
        lambda paths: (
            [
                Document(
                    page_content="Protocol chunk",
                    metadata={
                        "section": "Protocol",
                        "page_numbers": [2],
                        "document_id": "supplement:001",
                        "document_name": "protocol.pdf",
                        "document_role": "protocol",
                        "source_kind": "rag_chunk",
                        "source_path": "protocol.pdf",
                    },
                )
            ],
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

    class Result:
        document = object()

    class Converter:
        def convert(self, path):
            return Result()

    monkeypatch.setattr(
        node, "_get_docling_converter", lambda use_ocr=False: Converter()
    )

    result = node.ingest_assessment_documents(
        "primary.pdf", ["protocol.pdf"]
    ).to_state_update()

    assert len(result["docling_chunks"]) == 2
    assert result["docling_chunks"][0].metadata["document_id"] == "primary"
    assert result["docling_chunks"][1].metadata["document_id"] == "supplement:001"
    assert result["source_documents"][0]["document_role"] == "primary"
    assert result["source_documents"][1]["document_role"] == "protocol"


def test_assessment_ingestion_preserves_primary_chunks_when_supplement_ingestion_escapes(
    monkeypatch,
):
    import rob2_pipeline.ingestion.assessment as node

    from rob2_pipeline.models import empty_paper_evidence

    monkeypatch.setattr(node, "extract_full_text", lambda path: "Primary text")
    monkeypatch.setattr(node, "_configure_docling_runtime", lambda: None)
    monkeypatch.setattr(
        node,
        "_build_docling_chunks",
        lambda conv_result: [
            Document(
                page_content="Primary chunk",
                metadata={"section": "Methods", "page_numbers": [1]},
            )
        ],
    )
    monkeypatch.setattr(
        node,
        "build_document_repr",
        lambda doc: type(
            "DocRepr",
            (),
            {
                "full_text": "Primary text",
                "to_prompt_repr": lambda self: "Primary text",
                "blocks": [],
            },
        )(),
    )
    monkeypatch.setattr(
        node,
        "extract_structural_paper_evidence",
        lambda doc_repr: empty_paper_evidence("docling_struct"),
    )
    monkeypatch.setattr(node, "allow_remote_evidence_extraction", lambda: False)
    monkeypatch.setattr(
        node,
        "ingest_supplements",
        lambda paths: (_ for _ in ()).throw(
            RuntimeError("unexpected supplement error")
        ),
    )

    class Result:
        document = object()

    class Converter:
        def convert(self, path):
            return Result()

    monkeypatch.setattr(
        node, "_get_docling_converter", lambda use_ocr=False: Converter()
    )

    result = node.ingest_assessment_documents(
        "primary.pdf", ["protocol.pdf"]
    ).to_state_update()

    assert len(result["docling_chunks"]) == 1
    assert result["docling_chunks"][0].metadata["document_id"] == "primary"
    assert "unexpected supplement error" in result["supplement_warnings"][0]
