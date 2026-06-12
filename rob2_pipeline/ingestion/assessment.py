from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document

from rob2_pipeline.ingestion.evidence import (
    extract_paper_evidence,
    paper_evidence_from_sections,
    parse_sections,
)
from rob2_pipeline.ingestion.parse_artifacts import parse_sources
from rob2_pipeline.ingestion.parse_artifacts import (
    build_page_aware_artifacts,
    documents_from_page_aware_artifacts,
    full_text_from_parse_artifact,
)
from rob2_pipeline.ingestion.settings import (
    MIN_EXTRACTED_CHARS,
    allow_remote_evidence_extraction,
    appears_rct_candidate,
)
from rob2_pipeline.ingestion.source_catalog import (
    primary_source_document,
    supplement_source_document,
)
from rob2_pipeline.ingestion.supplement_segments import (
    build_supplement_ingestion_artifacts,
    supplement_segment_artifacts,
)
from rob2_pipeline.models import PaperEvidence
from rob2_pipeline.types import (
    LLMCallLogEntry,
    SourceDocument,
    SupplementSegmentArtifact,
)


@dataclass(frozen=True)
class AssessmentIngestionResult:
    full_text: str
    evidence: PaperEvidence
    docling_chunks: list[Document]
    source_documents: list[SourceDocument]
    supplement_warnings: list[str]
    supplement_segments: list[SupplementSegmentArtifact] = field(default_factory=list)
    supplement_indexes: dict = field(default_factory=dict)
    supplement_retrieval_grades: dict = field(default_factory=dict)
    parse_artifacts: list[dict] = field(default_factory=list)
    llm_call_log: list[LLMCallLogEntry] = field(default_factory=list)

    def to_state_update(self, include_llm_call_log: bool = True) -> dict:
        update = {
            "full_text": self.full_text,
            "evidence": self.evidence,
            "docling_chunks": self.docling_chunks,
            "source_documents": self.source_documents,
            "parse_artifacts": self.parse_artifacts,
            "supplement_warnings": self.supplement_warnings,
            "supplement_segments": self.supplement_segments,
            "supplement_indexes": self.supplement_indexes,
            "supplement_retrieval_grades": self.supplement_retrieval_grades,
        }
        if include_llm_call_log and self.llm_call_log:
            update["llm_call_log"] = self.llm_call_log
        return update


def ingest_assessment_documents(
    pdf_path: str, supplementary_paths: list[str] | None = None
) -> AssessmentIngestionResult:
    # Full-text extraction is strict. If it fails, the Assessment cannot proceed.
    supplementary_paths = list(supplementary_paths or [])
    primary_source = primary_source_document(Path(pdf_path))
    parser_sources = [
        primary_source,
        *[
            supplement_source_document(Path(path), index)
            for index, path in enumerate(supplementary_paths, start=1)
        ],
    ]
    parsed_result = _ingest_from_parse_artifacts(parser_sources)
    if parsed_result is None:
        raise RuntimeError("Primary PDF parsing failed or returned too little text.")
    return parsed_result


def _ingest_from_parse_artifacts(
    sources: list[SourceDocument],
) -> AssessmentIngestionResult | None:
    artifacts = parse_sources(sources)
    if not artifacts:
        return None

    primary_artifact = artifacts[0]
    if not all(
        hasattr(artifact, "pages") and hasattr(artifact, "source_identity")
        for artifact in artifacts
    ):
        return None
    primary_text = full_text_from_parse_artifact(primary_artifact)
    if (
        primary_artifact.source_identity.get("status") != "parsed"
        or len(primary_text.strip()) < MIN_EXTRACTED_CHARS
    ):
        return None

    supplement_ingestion = [
        build_supplement_ingestion_artifacts(artifact)
        for artifact in artifacts[1:]
        if artifact.source_identity.get("status") == "parsed"
    ]
    supplement_sources_by_id = {
        result.source_document.get("document_id"): result.source_document
        for result in supplement_ingestion
    }
    source_documents = [
        artifacts[0].source_identity,
        *[
            supplement_sources_by_id.get(
                artifact.source_identity.get("document_id"), artifact.source_identity
            )
            for artifact in artifacts[1:]
        ],
    ]
    page_artifacts = [build_page_aware_artifacts(artifact) for artifact in artifacts]
    docling_chunks = [
        chunk
        for artifact, source in zip(page_artifacts, source_documents, strict=True)
        for chunk in documents_from_page_aware_artifacts(artifact, source)
    ]
    supplement_warnings = [
        source.get("error", "")
        for source in source_documents[1:]
        if source.get("status") in {"failed", "missing", "partial", "degraded"}
        and source.get("error")
    ]
    supplement_warnings.extend(
        warning for result in supplement_ingestion for warning in result.warnings
    )
    supplement_segments = [
        segment
        for result in supplement_ingestion
        for segment in supplement_segment_artifacts(result.segments)
    ]
    supplement_indexes = {
        result.source_document.get("document_id", ""): result.index
        for result in supplement_ingestion
        if result.index is not None
    }
    sections = parse_sections(primary_text)
    evidence = paper_evidence_from_sections(
        sections,
        extraction_method="parse_artifact",
        source="parser_neutral_pages",
        warnings=[],
    )
    parse_artifacts = [artifact.to_dict() for artifact in artifacts]

    if not allow_remote_evidence_extraction():
        evidence["warnings"].append(
            "Remote evidence extraction disabled by ROB2_REMOTE_EVIDENCE_EXTRACTION."
        )
        return AssessmentIngestionResult(
            full_text=primary_text,
            evidence=evidence,
            docling_chunks=docling_chunks,
            source_documents=source_documents,
            parse_artifacts=parse_artifacts,
            supplement_warnings=supplement_warnings,
            supplement_segments=supplement_segments,
            supplement_indexes=supplement_indexes,
        )

    if not appears_rct_candidate(primary_text):
        evidence["warnings"].append(
            "Remote evidence extraction skipped for apparent non-RCT document."
        )
        return AssessmentIngestionResult(
            full_text=primary_text,
            evidence=evidence,
            docling_chunks=docling_chunks,
            source_documents=source_documents,
            parse_artifacts=parse_artifacts,
            supplement_warnings=supplement_warnings,
            supplement_segments=supplement_segments,
            supplement_indexes=supplement_indexes,
        )

    doc_repr = _ParseArtifactDocumentRepr(primary_text)
    try:
        evidence, log = extract_paper_evidence(doc_repr)
        return AssessmentIngestionResult(
            full_text=primary_text,
            evidence=evidence,
            docling_chunks=docling_chunks,
            source_documents=source_documents,
            parse_artifacts=parse_artifacts,
            supplement_warnings=supplement_warnings,
            supplement_segments=supplement_segments,
            supplement_indexes=supplement_indexes,
            llm_call_log=log,
        )
    except Exception as error:  # noqa: BLE001
        evidence["warnings"].append(f"LLM evidence extraction failed: {error}")

    return AssessmentIngestionResult(
        full_text=primary_text,
        evidence=evidence,
        docling_chunks=docling_chunks,
        source_documents=source_documents,
        parse_artifacts=parse_artifacts,
        supplement_warnings=supplement_warnings,
        supplement_segments=supplement_segments,
        supplement_indexes=supplement_indexes,
    )


class _ParseArtifactDocumentRepr:
    def __init__(self, full_text: str):
        self.full_text = full_text
        self.blocks = []

    def to_prompt_repr(self) -> str:
        return self.full_text
