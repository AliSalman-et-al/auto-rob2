from __future__ import annotations

import json

from rob2_pipeline.ingestion.parse_artifacts import (
    ParserDiagnostic,
    ParserProvenance,
    SourceParseArtifact,
    build_page_aware_artifacts,
    parse_source_with_adapter,
    write_page_aware_artifacts,
)


class FakeParser:
    producer = "fake-parser"
    producer_version = "1.2.3"

    def parse(self, path):
        return SourceParseArtifact(
            source_identity={
                "document_id": "primary",
                "document_name": "trial.pdf",
                "document_role": "primary",
                "source_kind": "rag_chunk",
                "path": str(path),
                "is_primary": True,
                "status": "parsed",
            },
            pages=[
                {
                    "page_number": 1,
                    "text": "Participants were randomized.",
                    "width": 612.0,
                    "height": 792.0,
                }
            ],
            diagnostics=[],
            provenance=ParserProvenance(
                parser_name=self.producer,
                parser_version=self.producer_version,
                adapter_name="fake",
                artifact_schema_version="parse-artifact-v1",
                config={},
            ),
        )


def test_parser_adapter_contract_returns_parser_neutral_page_text():
    source = {
        "document_id": "primary",
        "document_name": "trial.pdf",
        "document_role": "primary",
        "source_kind": "rag_chunk",
        "path": "trial.pdf",
        "is_primary": True,
        "status": "parsed",
    }

    artifact = parse_source_with_adapter(source, FakeParser())

    assert artifact.source_identity == source
    assert artifact.pages == [
        {
            "page_number": 1,
            "text": "Participants were randomized.",
            "width": 612.0,
            "height": 792.0,
        }
    ]
    assert artifact.provenance.parser_name == "fake-parser"
    assert artifact.provenance.artifact_schema_version == "parse-artifact-v1"


def test_parser_adapter_contract_captures_diagnostics_without_native_result():
    class BrokenParser(FakeParser):
        def parse(self, path):
            raise RuntimeError("bad pdf")

    source = {
        "document_id": "supplement:001",
        "document_name": "protocol.pdf",
        "document_role": "protocol",
        "source_kind": "rag_chunk",
        "path": "protocol.pdf",
        "is_primary": False,
        "status": "pending",
    }

    artifact = parse_source_with_adapter(source, BrokenParser())

    assert artifact.pages == []
    assert artifact.diagnostics == [
        ParserDiagnostic(level="error", message="bad pdf", page_number=None)
    ]
    assert artifact.source_identity["status"] == "failed"


def test_page_aware_artifacts_persist_sections_and_multi_page_chunks(tmp_path):
    source = {
        "document_id": "primary",
        "document_name": "trial.pdf",
        "document_role": "primary",
        "source_kind": "rag_chunk",
        "path": "trial.pdf",
        "is_primary": True,
        "status": "parsed",
    }
    parse_artifact = SourceParseArtifact(
        source_identity=source,
        pages=[
            {
                "page_number": 1,
                "text": "Methods\nParticipants were randomized",
                "width": 612.0,
                "height": 792.0,
            },
            {
                "page_number": 2,
                "text": "with allocation concealment.\nResults\nOverall survival improved.",
                "width": 612.0,
                "height": 792.0,
            },
        ],
        diagnostics=[],
        provenance=ParserProvenance(
            parser_name="fake-parser",
            parser_version="1.2.3",
            adapter_name="fake",
            artifact_schema_version="parse-artifact-v1",
            config={},
        ),
    )

    artifacts = build_page_aware_artifacts(parse_artifact)

    assert [section.heading for section in artifacts.sections] == [
        "Methods",
        "Results",
    ]
    assert artifacts.sections[0].page_numbers == [1, 2]
    assert artifacts.chunks[0].chunk_id == "primary:chunk:0001"
    assert artifacts.chunks[0].source_id == "primary"
    assert artifacts.chunks[0].document_role == "primary"
    assert artifacts.chunks[0].section_id == "primary:section:0001"
    assert artifacts.chunks[0].page_numbers == [1, 2]
    assert "allocation concealment" in artifacts.chunks[0].text

    output_path = tmp_path / "page-aware-artifacts.json"
    write_page_aware_artifacts(artifacts, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["source_id"] == "primary"
    assert payload["sections"][0]["section_id"] == "primary:section:0001"
    assert payload["chunks"][0]["chunk_id"] == "primary:chunk:0001"


def test_page_aware_artifacts_keep_missing_headings_as_unsectioned_chunk():
    parse_artifact = SourceParseArtifact(
        source_identity={
            "document_id": "supplement:001",
            "document_name": "appendix.pdf",
            "document_role": "appendix",
            "source_kind": "rag_chunk",
            "path": "appendix.pdf",
            "is_primary": False,
            "status": "parsed",
        },
        pages=[
            {
                "page_number": 7,
                "text": "Participants were stratified by disease volume.",
                "width": 612.0,
                "height": 792.0,
            }
        ],
        diagnostics=[],
        provenance=ParserProvenance(
            parser_name="fake-parser",
            parser_version="1.2.3",
            adapter_name="fake",
            artifact_schema_version="parse-artifact-v1",
            config={},
        ),
    )

    artifacts = build_page_aware_artifacts(parse_artifact)

    assert len(artifacts.sections) == 1
    assert artifacts.sections[0].heading == "Unsectioned"
    assert artifacts.chunks[0].chunk_id == "supplement:001:chunk:0001"
    assert artifacts.chunks[0].document_role == "appendix"
    assert artifacts.chunks[0].page_numbers == [7]


def test_page_aware_artifacts_ignore_empty_and_low_text_pages():
    parse_artifact = SourceParseArtifact(
        source_identity={
            "document_id": "primary",
            "document_name": "trial.pdf",
            "document_role": "primary",
            "source_kind": "rag_chunk",
            "path": "trial.pdf",
            "is_primary": True,
            "status": "parsed",
        },
        pages=[
            {"page_number": 1, "text": "   ", "width": 612.0, "height": 792.0},
            {"page_number": 61, "text": "x\n\n2", "width": 612.0, "height": 792.0},
        ],
        diagnostics=[],
        provenance=ParserProvenance(
            parser_name="fake-parser",
            parser_version="1.2.3",
            adapter_name="fake",
            artifact_schema_version="parse-artifact-v1",
            config={},
        ),
    )

    artifacts = build_page_aware_artifacts(parse_artifact)

    assert artifacts.sections == []
    assert artifacts.chunks == []
