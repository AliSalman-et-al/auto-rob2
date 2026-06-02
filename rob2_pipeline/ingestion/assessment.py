from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from rob2_pipeline.ingestion.docling_extract import (
    _build_docling_chunks,
    _configure_docling_runtime,
    _get_docling_converter,
    _normalize_extracted_text,
    extract_full_text,
)
from rob2_pipeline.ingestion.document_repr import build_document_repr
from rob2_pipeline.ingestion.evidence import (
    extract_paper_evidence,
    extract_structural_paper_evidence,
    paper_evidence_from_sections,
    parse_sections,
)
from rob2_pipeline.ingestion.parse_artifacts import parse_sources
from rob2_pipeline.ingestion.settings import (
    MIN_EXTRACTED_CHARS,
    allow_remote_evidence_extraction,
    appears_rct_candidate,
)
from rob2_pipeline.ingestion.source_catalog import (
    apply_source_metadata,
    primary_source_document,
    skipped_source_documents,
)
from rob2_pipeline.ingestion.supplements import (
    ingest_supplements,
)
from rob2_pipeline.models import PaperEvidence
from rob2_pipeline.types import LLMCallLogEntry, SourceDocument


@dataclass(frozen=True)
class AssessmentIngestionResult:
    full_text: str
    evidence: PaperEvidence
    docling_doc: Any | None
    docling_chunks: list[Document]
    source_documents: list[SourceDocument]
    supplement_warnings: list[str]
    parse_artifacts: list[dict] = field(default_factory=list)
    llm_call_log: list[LLMCallLogEntry] = field(default_factory=list)

    def to_state_update(self, include_llm_call_log: bool = True) -> dict:
        update = {
            "full_text": self.full_text,
            "evidence": self.evidence,
            "docling_doc": self.docling_doc,
            "docling_chunks": self.docling_chunks,
            "source_documents": self.source_documents,
            "parse_artifacts": self.parse_artifacts,
            "supplement_warnings": self.supplement_warnings,
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

    try:
        full_text, conv_result, doc_repr = _convert_primary_pdf(pdf_path)
        docling_chunks = apply_source_metadata(
            _build_docling_chunks(conv_result), primary_source
        )
        try:
            supplement_chunks, supplement_documents, supplement_warnings = (
                ingest_supplements(supplementary_paths)
            )
        except Exception as error:  # noqa: BLE001
            supplement_chunks = []
            supplement_documents = []
            supplement_warnings = [f"Supplement ingestion failed: {error}"]
        docling_chunks = [*docling_chunks, *supplement_chunks]
        source_documents = [primary_source, *supplement_documents]
        parse_artifacts = [
            artifact.to_dict() for artifact in parse_sources(source_documents)
        ]

        if not doc_repr.full_text:
            doc_repr.full_text = full_text
        evidence = extract_structural_paper_evidence(doc_repr)

        if not allow_remote_evidence_extraction():
            evidence["warnings"].append(
                "Remote evidence extraction disabled by ROB2_REMOTE_EVIDENCE_EXTRACTION."
            )
            return AssessmentIngestionResult(
                full_text=full_text,
                evidence=evidence,
                docling_doc=conv_result,
                docling_chunks=docling_chunks,
                source_documents=source_documents,
                parse_artifacts=parse_artifacts,
                supplement_warnings=supplement_warnings,
            )

        if not appears_rct_candidate(doc_repr.to_prompt_repr() or doc_repr.full_text):
            evidence["warnings"].append(
                "Remote evidence extraction skipped for apparent non-RCT document."
            )
            return AssessmentIngestionResult(
                full_text=full_text,
                evidence=evidence,
                docling_doc=conv_result,
                docling_chunks=docling_chunks,
                source_documents=source_documents,
                parse_artifacts=parse_artifacts,
                supplement_warnings=supplement_warnings,
            )

        try:
            evidence, log = extract_paper_evidence(doc_repr)
            return AssessmentIngestionResult(
                full_text=full_text,
                evidence=evidence,
                docling_doc=conv_result,
                docling_chunks=docling_chunks,
                source_documents=source_documents,
                parse_artifacts=parse_artifacts,
                supplement_warnings=supplement_warnings,
                llm_call_log=log,
            )
        except Exception as error:  # noqa: BLE001
            evidence = extract_structural_paper_evidence(doc_repr)
            evidence["warnings"].append(f"LLM evidence extraction failed: {error}")
            return AssessmentIngestionResult(
                full_text=full_text,
                evidence=evidence,
                docling_doc=conv_result,
                docling_chunks=docling_chunks,
                source_documents=source_documents,
                parse_artifacts=parse_artifacts,
                supplement_warnings=supplement_warnings,
            )
    except Exception as error:
        full_text = extract_full_text(pdf_path)
        sections = parse_sections(full_text)
        evidence = paper_evidence_from_sections(
            sections,
            extraction_method="fallback",
            source="keyword_fallback",
            warnings=[
                "Docling structural extraction failed; used text keyword fallback."
            ],
        )
        supplement_documents, supplement_warnings = skipped_source_documents(
            supplementary_paths,
            f"primary Docling structural extraction failed: {error}",
        )
        source_documents = [primary_source, *supplement_documents]
        parse_artifacts = [
            artifact.to_dict() for artifact in parse_sources(source_documents)
        ]
        return AssessmentIngestionResult(
            full_text=full_text,
            evidence=evidence,
            docling_doc=None,
            docling_chunks=[],
            source_documents=source_documents,
            parse_artifacts=parse_artifacts,
            supplement_warnings=supplement_warnings,
        )


def _convert_primary_pdf(pdf_path: str) -> tuple[str, Any, Any]:
    errors = []
    for use_ocr in (False, True):
        try:
            _configure_docling_runtime()
            converter = _get_docling_converter(use_ocr=use_ocr)
            conv_result = converter.convert(pdf_path)
            doc_repr = build_document_repr(conv_result.document)
            full_text = _normalize_extracted_text(doc_repr.full_text)
            if not full_text:
                full_text = _normalize_extracted_text(doc_repr.to_prompt_repr())
            if len(full_text.strip()) < MIN_EXTRACTED_CHARS:
                full_text = _normalize_extracted_text(extract_full_text(pdf_path))
            return full_text, conv_result, doc_repr
        except Exception as error:  # noqa: BLE001
            errors.append(f"OCR={use_ocr}: {error}")
    raise RuntimeError("; ".join(errors))
