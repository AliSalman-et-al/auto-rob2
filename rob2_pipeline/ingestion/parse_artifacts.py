from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Protocol, TypedDict

from langchain_core.documents import Document

from rob2_pipeline.ingestion.source_catalog import mark_failed, mark_parsed
from rob2_pipeline.types import SourceDocument


PARSE_ARTIFACT_SCHEMA_VERSION = "parse-artifact-v2"
PAGE_AWARE_ARTIFACT_SCHEMA_VERSION = "page-aware-artifacts-v1"
MIN_SECTION_TEXT_CHARS = 20

METHODS_HEADINGS = {
    "methods",
    "method",
    "materials and methods",
    "patients and methods",
    "participants and methods",
    "study design",
    "trial design",
    "study oversight",
    "trial oversight",
    "participants",
    "patients",
    "eligibility",
    "setting",
    "interventions",
    "intervention",
    "treatment",
    "procedures",
    "randomization",
    "randomisation",
    "allocation",
    "allocation concealment",
    "masking",
    "blinding",
    "outcomes",
    "outcome",
    "endpoints",
    "endpoint",
    "assessments",
    "assessment",
    "sample size",
    "statistical analysis",
    "statistics",
    "analysis population",
    "protocol",
    "ethics",
}

RESULTS_HEADINGS = {
    "results",
    "result",
    "patient disposition",
    "participant flow",
    "trial profile",
    "consort flow",
    "baseline characteristics",
    "demographics",
    "efficacy",
    "primary outcome",
    "secondary outcomes",
    "secondary outcome",
    "safety",
    "adverse events",
    "adverse event",
    "harms",
    "follow-up",
    "follow up",
    "missing data",
    "protocol deviations",
    "protocol deviation",
}

OTHER_CANONICAL_HEADINGS = {
    "abstract": "ABSTRACT",
    "introduction": "INTRODUCTION",
    "background": "BACKGROUND",
    "discussion": "DISCUSSION",
    "conclusion": "CONCLUSION",
    "conclusions": "CONCLUSION",
}

SECTION_HEADING_ALIASES = {
    **{heading: "METHODS" for heading in METHODS_HEADINGS},
    **{heading: "RESULTS" for heading in RESULTS_HEADINGS},
    **OTHER_CANONICAL_HEADINGS,
}


class ParsedPageArtifact(TypedDict, total=False):
    page_number: int
    text: str
    width: float
    height: float


@dataclass(frozen=True)
class ParserDiagnostic:
    level: str
    message: str
    page_number: int | None = None


@dataclass(frozen=True)
class ParserProvenance:
    parser_name: str
    parser_version: str
    adapter_name: str
    artifact_schema_version: str
    config: dict


@dataclass(frozen=True)
class SourceParseArtifact:
    source_identity: SourceDocument
    pages: list[ParsedPageArtifact]
    diagnostics: list[ParserDiagnostic]
    provenance: ParserProvenance
    raw_character_stream: str = ""
    parse_time_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "source_identity": dict(self.source_identity),
            "pages": [dict(page) for page in self.pages],
            "raw_character_stream": self.raw_character_stream,
            "diagnostics": [asdict(diagnostic) for diagnostic in self.diagnostics],
            "parse_time_ms": self.parse_time_ms,
            "provenance": asdict(self.provenance),
        }


@dataclass(frozen=True)
class PageAwareSectionArtifact:
    section_id: str
    source_id: str
    canonical_label: str
    original_heading: str
    page_numbers: list[int]
    text: str


@dataclass(frozen=True)
class PageAwareChunkArtifact:
    chunk_id: str
    source_id: str
    document_role: str
    section_id: str
    section_heading: str
    original_heading: str
    page_numbers: list[int]
    text: str


@dataclass(frozen=True)
class PageAwareArtifacts:
    source_id: str
    document_role: str
    artifact_schema_version: str
    sections: list[PageAwareSectionArtifact]
    chunks: list[PageAwareChunkArtifact]

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "document_role": self.document_role,
            "artifact_schema_version": self.artifact_schema_version,
            "sections": [asdict(section) for section in self.sections],
            "chunks": [asdict(chunk) for chunk in self.chunks],
        }


class SourceParserAdapter(Protocol):
    producer: str
    producer_version: str

    def parse(self, path: str | Path) -> SourceParseArtifact:
        ...


class PyMuPDFSectionMapSourceParser:
    producer = "pymupdf+pymupdf4llm"
    adapter_name = "pymupdf-sectionmap"

    def __init__(self) -> None:
        self.config = {
            "layout_text_engine": "pymupdf4llm",
            "raw_character_stream_engine": "pymupdf",
            "page_chunks": True,
        }
        pymupdf_version = _package_version("pymupdf")
        pymupdf4llm_version = _package_version("pymupdf4llm")
        self.producer_version = (
            f"pymupdf={pymupdf_version}; pymupdf4llm={pymupdf4llm_version}"
        )

    def parse(self, path: str | Path) -> SourceParseArtifact:
        source = SourceDocument(
            document_id=Path(path).stem,
            document_name=Path(path).name,
            document_role="primary",
            source_kind="rag_chunk",
            path=str(path),
            is_primary=True,
            status="pending",
        )
        return self.parse_source(source)

    def parse_source(self, source: SourceDocument) -> SourceParseArtifact:
        import pymupdf
        import pymupdf4llm

        chunks = pymupdf4llm.to_markdown(source["path"], page_chunks=True)
        raw_pages: list[str] = []
        dimensions: dict[int, tuple[float, float]] = {}
        warnings = ""
        with pymupdf.open(source["path"]) as doc:
            for page_index, page in enumerate(doc, start=1):
                raw_pages.append(page.get_text())
                dimensions[page_index] = (float(page.rect.width), float(page.rect.height))
            warnings = pymupdf.TOOLS.mupdf_warnings()

        pages = []
        for index, chunk in enumerate(chunks, start=1):
            metadata = chunk.get("metadata", {})
            page_number = int(metadata.get("page_number") or index)
            width, height = dimensions.get(page_number, (0.0, 0.0))
            pages.append(
                ParsedPageArtifact(
                    page_number=page_number,
                    text=str(chunk.get("text", "")),
                    width=width,
                    height=height,
                )
            )
        diagnostics = [
            ParserDiagnostic(
                level="info",
                message=(
                    f"Parsed {len(pages)} page(s) with PyMuPDF4LLM layout text "
                    "and PyMuPDF raw character stream."
                ),
                page_number=None,
            )
        ]
        if warnings:
            diagnostics.append(
                ParserDiagnostic(
                    level="warning",
                    message=warnings,
                    page_number=None,
                )
            )
        return SourceParseArtifact(
            source_identity=mark_parsed(source),
            pages=pages,
            raw_character_stream="\n\n".join(
                page.strip() for page in raw_pages if page.strip()
            ).strip(),
            diagnostics=diagnostics,
            provenance=ParserProvenance(
                parser_name=self.producer,
                parser_version=self.producer_version,
                adapter_name=self.adapter_name,
                artifact_schema_version=PARSE_ARTIFACT_SCHEMA_VERSION,
                config=dict(self.config),
            ),
        )


def parse_source_with_adapter(
    source: SourceDocument, parser: SourceParserAdapter
) -> SourceParseArtifact:
    started = time.perf_counter()
    try:
        parse_source = getattr(parser, "parse_source", None)
        if parse_source is not None:
            artifact = parse_source(source)
        else:
            artifact = parser.parse(source["path"])
        parse_time_ms = max(0, round((time.perf_counter() - started) * 1000))
        return SourceParseArtifact(
            source_identity=_source_identity_from_parser_artifact(source, artifact),
            pages=artifact.pages,
            diagnostics=artifact.diagnostics,
            provenance=artifact.provenance,
            raw_character_stream=artifact.raw_character_stream,
            parse_time_ms=parse_time_ms,
        )
    except Exception as error:  # noqa: BLE001
        parse_time_ms = max(0, round((time.perf_counter() - started) * 1000))
        return SourceParseArtifact(
            source_identity=mark_failed(source, str(error)),
            pages=[],
            raw_character_stream="",
            diagnostics=[
                ParserDiagnostic(level="error", message=str(error), page_number=None)
            ],
            provenance=ParserProvenance(
                parser_name=getattr(parser, "producer", parser.__class__.__name__),
                parser_version=getattr(parser, "producer_version", "unknown"),
                adapter_name=getattr(parser, "adapter_name", parser.__class__.__name__),
                artifact_schema_version=PARSE_ARTIFACT_SCHEMA_VERSION,
                config=dict(getattr(parser, "config", {})),
            ),
            parse_time_ms=parse_time_ms,
        )


def _source_identity_from_parser_artifact(
    source: SourceDocument, artifact: SourceParseArtifact
) -> SourceDocument:
    artifact_identity = dict(artifact.source_identity)
    status = artifact_identity.get("status") or "parsed"
    if status == "parsed":
        return mark_parsed(source)

    updated = SourceDocument(**source)
    updated["status"] = status
    if artifact_identity.get("error"):
        updated["error"] = artifact_identity["error"]
    return updated


def parse_sources(
    sources: list[SourceDocument],
    parser: SourceParserAdapter | None = None,
) -> list[SourceParseArtifact]:
    parser = parser or PyMuPDFSectionMapSourceParser()
    return [parse_source_with_adapter(source, parser) for source in sources]


def build_page_aware_artifacts(
    artifact: SourceParseArtifact,
) -> PageAwareArtifacts:
    source_id = artifact.source_identity.get("document_id", "")
    document_role = artifact.source_identity.get("document_role", "unknown_supplement")
    sections = _build_page_aware_sections(artifact, source_id)
    chunks = [
        PageAwareChunkArtifact(
            chunk_id=f"{source_id}:chunk:{index:04d}",
            source_id=source_id,
            document_role=document_role,
            section_id=section.section_id,
            section_heading=section.canonical_label,
            original_heading=section.original_heading,
            page_numbers=section.page_numbers,
            text=section.text,
        )
        for index, section in enumerate(sections, start=1)
        if section.text.strip()
    ]
    return PageAwareArtifacts(
        source_id=source_id,
        document_role=document_role,
        artifact_schema_version=PAGE_AWARE_ARTIFACT_SCHEMA_VERSION,
        sections=sections,
        chunks=chunks,
    )


def write_page_aware_artifacts(
    artifacts: PageAwareArtifacts,
    path: str | Path,
) -> None:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(artifacts.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def full_text_from_parse_artifact(artifact: SourceParseArtifact) -> str:
    return "\n\n".join(
        page.get("text", "").strip()
        for page in artifact.pages
        if page.get("text", "").strip()
    ).strip()


def documents_from_page_aware_artifacts(
    artifacts: PageAwareArtifacts,
    source: SourceDocument,
) -> list[Document]:
    return [
        Document(
            page_content=chunk.text,
            metadata={
                "section": chunk.section_heading,
                "original_heading": chunk.original_heading,
                "page_numbers": list(chunk.page_numbers),
                "document_id": source.get("document_id", ""),
                "document_name": source.get("document_name", ""),
                "document_role": source.get("document_role", ""),
                "source_kind": source.get("source_kind", "rag_chunk"),
                "source_path": source.get("path", ""),
            },
        )
        for chunk in artifacts.chunks
        if chunk.text.strip()
    ]


def _build_page_aware_sections(
    artifact: SourceParseArtifact,
    source_id: str,
) -> list[PageAwareSectionArtifact]:
    sections: list[PageAwareSectionArtifact] = []
    current_label = "UNSECTIONED"
    current_original_heading = "UNSECTIONED"
    current_lines: list[str] = []
    current_pages: list[int] = []

    def flush() -> None:
        nonlocal current_lines, current_pages
        text = "\n".join(current_lines).strip()
        if len(text) < MIN_SECTION_TEXT_CHARS:
            current_lines = []
            current_pages = []
            return
        sections.append(
            PageAwareSectionArtifact(
                section_id=f"{source_id}:section:{len(sections) + 1:04d}",
                source_id=source_id,
                canonical_label=current_label,
                original_heading=current_original_heading,
                page_numbers=sorted(set(current_pages)),
                text=text,
            )
        )
        current_lines = []
        current_pages = []

    for page in artifact.pages:
        page_number = int(page.get("page_number", 0))
        for raw_line in page.get("text", "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            heading_label = _canonical_section_label(line)
            if heading_label is not None:
                flush()
                current_label = heading_label
                current_original_heading = _clean_section_heading(line)
                continue
            current_lines.append(line)
            if page_number:
                current_pages.append(page_number)
    flush()
    return sections


def _canonical_section_label(raw_heading: str) -> str | None:
    cleaned = _clean_section_heading(raw_heading)
    normalized = re.sub(r"\s+", " ", cleaned).casefold()
    return SECTION_HEADING_ALIASES.get(normalized)


def _clean_section_heading(raw_heading: str) -> str:
    return raw_heading.lstrip("#").strip().rstrip(":").strip()


def _package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "unknown"


__all__ = [
    "PARSE_ARTIFACT_SCHEMA_VERSION",
    "PAGE_AWARE_ARTIFACT_SCHEMA_VERSION",
    "PageAwareArtifacts",
    "PageAwareChunkArtifact",
    "PageAwareSectionArtifact",
    "ParsedPageArtifact",
    "ParserDiagnostic",
    "ParserProvenance",
    "PyMuPDFSectionMapSourceParser",
    "SourceParseArtifact",
    "SourceParserAdapter",
    "build_page_aware_artifacts",
    "documents_from_page_aware_artifacts",
    "full_text_from_parse_artifact",
    "parse_source_with_adapter",
    "parse_sources",
    "write_page_aware_artifacts",
]
