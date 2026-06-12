from __future__ import annotations

import json
import tomllib
from pathlib import Path

from rob2_pipeline.ingestion.parse_artifacts import (
    PARSE_ARTIFACT_SCHEMA_VERSION,
    PyMuPDFSectionMapSourceParser,
    ParserDiagnostic,
    ParserProvenance,
    SourceParseArtifact,
    build_page_aware_artifacts,
    documents_from_page_aware_artifacts,
    full_text_from_parse_artifact,
    parse_sources,
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

    assert [section.canonical_label for section in artifacts.sections] == [
        "METHODS",
        "RESULTS",
    ]
    assert [section.original_heading for section in artifacts.sections] == [
        "Methods",
        "Results",
    ]
    assert artifacts.sections[0].page_numbers == [1, 2]
    assert artifacts.chunks[0].chunk_id == "primary:chunk:0001"
    assert artifacts.chunks[0].source_id == "primary"
    assert artifacts.chunks[0].document_role == "primary"
    assert artifacts.chunks[0].section_id == "primary:section:0001"
    assert artifacts.chunks[0].section_heading == "METHODS"
    assert artifacts.chunks[0].original_heading == "Methods"
    assert artifacts.chunks[0].page_numbers == [1, 2]
    assert "allocation concealment" in artifacts.chunks[0].text

    documents = documents_from_page_aware_artifacts(artifacts, source)
    assert documents[0].metadata["section"] == "METHODS"
    assert documents[0].metadata["original_heading"] == "Methods"

    output_path = tmp_path / "page-aware-artifacts.json"
    write_page_aware_artifacts(artifacts, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["source_id"] == "primary"
    assert payload["sections"][0]["section_id"] == "primary:section:0001"
    assert payload["sections"][0]["canonical_label"] == "METHODS"
    assert payload["sections"][0]["original_heading"] == "Methods"
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
    assert artifacts.sections[0].canonical_label == "UNSECTIONED"
    assert artifacts.sections[0].original_heading == "UNSECTIONED"
    assert artifacts.chunks[0].chunk_id == "supplement:001:chunk:0001"
    assert artifacts.chunks[0].document_role == "appendix"
    assert artifacts.chunks[0].section_heading == "UNSECTIONED"
    assert artifacts.chunks[0].original_heading == "UNSECTIONED"
    assert artifacts.chunks[0].page_numbers == [7]


def test_page_aware_artifacts_keep_repeated_canonical_labels_ordered():
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
            {
                "page_number": 3,
                "text": (
                    "Results\n"
                    "Overall survival improved in the intervention group.\n"
                    "Safety\n"
                    "Grade 3 adverse events were balanced between groups."
                ),
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

    assert [section.canonical_label for section in artifacts.sections] == [
        "RESULTS",
        "RESULTS",
    ]
    assert [section.original_heading for section in artifacts.sections] == [
        "Results",
        "Safety",
    ]
    assert [section.section_id for section in artifacts.sections] == [
        "primary:section:0001",
        "primary:section:0002",
    ]
    assert "Overall survival improved" in artifacts.sections[0].text
    assert "Grade 3 adverse events" in artifacts.sections[1].text


def test_page_aware_artifacts_map_standard_rct_subheadings_to_methods_and_results():
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
            {
                "page_number": 4,
                "text": (
                    "Study Design\n"
                    "This was a phase 3 randomized controlled trial.\n"
                    "Statistical Analysis\n"
                    "Analyses used the intention-to-treat population.\n"
                    "Baseline Characteristics\n"
                    "Baseline disease volume was balanced across groups.\n"
                    "Protocol Deviations\n"
                    "Major deviations were uncommon in both groups."
                ),
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

    assert [section.canonical_label for section in artifacts.sections] == [
        "METHODS",
        "METHODS",
        "RESULTS",
        "RESULTS",
    ]
    assert [section.original_heading for section in artifacts.sections] == [
        "Study Design",
        "Statistical Analysis",
        "Baseline Characteristics",
        "Protocol Deviations",
    ]


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


def test_default_parser_produces_v2_layout_text_raw_stream_and_provenance(tmp_path):
    import pymupdf

    pdf_path = tmp_path / "trial.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=300)
    page.insert_text((36, 48), "Methods")
    page.insert_text((36, 72), "Participants were randomized centrally.")
    doc.save(pdf_path)
    doc.close()

    source = {
        "document_id": "primary",
        "document_name": "trial.pdf",
        "document_role": "primary",
        "source_kind": "rag_chunk",
        "path": str(pdf_path),
        "is_primary": True,
        "status": "pending",
    }

    artifact = parse_sources([source])[0]

    assert artifact.source_identity["status"] == "parsed"
    assert artifact.provenance.parser_name == "pymupdf+pymupdf4llm"
    assert artifact.provenance.adapter_name == "pymupdf-sectionmap"
    assert artifact.provenance.artifact_schema_version == PARSE_ARTIFACT_SCHEMA_VERSION
    assert artifact.provenance.config["layout_text_engine"] == "pymupdf4llm"
    assert artifact.provenance.config["raw_character_stream_engine"] == "pymupdf"
    assert artifact.parse_time_ms >= 0
    assert artifact.diagnostics
    assert artifact.pages
    assert "Participants were randomized centrally." in full_text_from_parse_artifact(
        artifact
    )
    assert "Participants were randomized centrally." in artifact.raw_character_stream
    assert artifact.to_dict()["raw_character_stream"] == artifact.raw_character_stream


def test_default_parser_returns_degraded_artifact_for_corrupt_pdf(tmp_path):
    pdf_path = tmp_path / "corrupt.pdf"
    pdf_path.write_bytes(b"not a pdf")
    source = {
        "document_id": "primary",
        "document_name": "corrupt.pdf",
        "document_role": "primary",
        "source_kind": "rag_chunk",
        "path": str(pdf_path),
        "is_primary": True,
        "status": "pending",
    }

    artifact = parse_sources([source])[0]

    assert artifact.source_identity["status"] == "failed"
    assert artifact.pages == []
    assert artifact.raw_character_stream == ""
    assert artifact.diagnostics
    assert artifact.diagnostics[0].level == "error"
    assert artifact.provenance.parser_name == "pymupdf+pymupdf4llm"
    assert artifact.provenance.adapter_name == "pymupdf-sectionmap"


def test_default_parser_path_does_not_use_liteparse():
    parser = PyMuPDFSectionMapSourceParser()

    assert parser.producer == "pymupdf+pymupdf4llm"
    assert parser.producer_version != "unknown"


def test_runtime_dependencies_do_not_include_liteparse():
    payload = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )

    dependencies = payload["project"]["dependencies"]
    assert not any(dependency.lower().startswith("liteparse") for dependency in dependencies)
