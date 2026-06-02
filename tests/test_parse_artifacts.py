from __future__ import annotations

from rob2_pipeline.ingestion.parse_artifacts import (
    ParserDiagnostic,
    ParserProvenance,
    SourceParseArtifact,
    parse_source_with_adapter,
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
