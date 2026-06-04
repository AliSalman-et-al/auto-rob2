from pathlib import Path

from langchain_core.documents import Document

from rob2_pipeline.ingestion.source_catalog import (
    apply_source_metadata,
    classify_document_role,
    mark_failed,
    mark_missing,
    mark_parsed,
    mark_partial,
    primary_source_document,
    skipped_source_documents,
    supplement_source_document,
)


def test_classify_document_role_from_filename():
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
        assert classify_document_role(Path(filename)) == expected


def test_primary_source_document_uses_primary_invariants():
    source = primary_source_document(Path("primary.pdf"))

    assert source == {
        "document_id": "primary",
        "document_name": "primary.pdf",
        "document_role": "primary",
        "source_kind": "rag_chunk",
        "path": "primary.pdf",
        "is_primary": True,
        "status": "parsed",
    }


def test_supplement_source_document_uses_stable_pending_id():
    source = supplement_source_document(Path("protocol.pdf"), index=2)

    assert source["document_id"] == "supplement:002"
    assert source["document_name"] == "protocol.pdf"
    assert source["document_role"] == "protocol"
    assert source["source_kind"] == "rag_chunk"
    assert source["is_primary"] is False
    assert source["status"] == "pending"


def test_status_markers_preserve_source_identity_and_set_error_shape():
    source = supplement_source_document(Path("protocol.pdf"), index=1)

    missing = mark_missing(source, Path("protocol.pdf"))
    failed = mark_failed(source, "Supplement parse failed: protocol.pdf: bad pdf")
    parsed = mark_parsed(source)
    partial = mark_partial(source, ["page window skipped"])

    assert missing["document_id"] == "supplement:001"
    assert missing["status"] == "missing"
    assert "Supplement not found: protocol.pdf" == missing["error"]
    assert failed["status"] == "failed"
    assert failed["error"] == "Supplement parse failed: protocol.pdf: bad pdf"
    assert parsed["status"] == "parsed"
    assert "error" not in parsed
    assert partial["status"] == "partial"
    assert partial["error"] == "page window skipped"


def test_skipped_source_documents_records_failed_supplements_and_warnings():
    documents, warnings = skipped_source_documents(
        ["inputs/benchmark/supplement/TITAN/protocol.pdf"],
        "primary parser artifact extraction failed",
    )

    assert documents[0]["document_id"] == "supplement:001"
    assert documents[0]["document_role"] == "protocol"
    assert documents[0]["status"] == "failed"
    assert "Supplement not ingested" in documents[0]["error"]
    assert warnings == [documents[0]["error"]]


def test_apply_source_metadata_preserves_existing_chunk_metadata():
    chunks = [
        Document(
            page_content="Protocol text.",
            metadata={"section": "Methods", "page_numbers": [3]},
        )
    ]
    source = supplement_source_document(Path("protocol.pdf"), index=1)

    result = apply_source_metadata(chunks, source)

    assert result[0].metadata["section"] == "Methods"
    assert result[0].metadata["page_numbers"] == [3]
    assert result[0].metadata["document_id"] == "supplement:001"
    assert result[0].metadata["document_name"] == "protocol.pdf"
    assert result[0].metadata["document_role"] == "protocol"
    assert result[0].metadata["source_kind"] == "rag_chunk"
    assert result[0].metadata["source_path"] == "protocol.pdf"
