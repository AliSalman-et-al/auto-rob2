from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Protocol, TypedDict

from rob2_pipeline.ingestion.source_catalog import mark_failed, mark_parsed
from rob2_pipeline.types import SourceDocument


PARSE_ARTIFACT_SCHEMA_VERSION = "parse-artifact-v1"
PAGE_AWARE_ARTIFACT_SCHEMA_VERSION = "page-aware-artifacts-v1"
MIN_SECTION_TEXT_CHARS = 20

SECTION_HEADING_RE = re.compile(
    r"^\s*(?:#+\s*)?(abstract|introduction|background|methods?|results?|discussion|"
    r"conclusions?|randomi[sz]ation|masking|blinding|outcomes?|endpoints?|"
    r"statistical analysis|baseline characteristics)\s*:?\s*$",
    re.IGNORECASE,
)


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
    parse_time_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "source_identity": dict(self.source_identity),
            "pages": [dict(page) for page in self.pages],
            "diagnostics": [asdict(diagnostic) for diagnostic in self.diagnostics],
            "parse_time_ms": self.parse_time_ms,
            "provenance": asdict(self.provenance),
        }


@dataclass(frozen=True)
class PageAwareSectionArtifact:
    section_id: str
    source_id: str
    heading: str
    page_numbers: list[int]
    text: str


@dataclass(frozen=True)
class PageAwareChunkArtifact:
    chunk_id: str
    source_id: str
    document_role: str
    section_id: str
    section_heading: str
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


class LiteParseSourceParser:
    producer = "liteparse"

    def __init__(self, *, ocr_enabled: bool = False, quiet: bool = True):
        self.config = {"ocr_enabled": ocr_enabled, "quiet": quiet}
        self.producer_version = _package_version("liteparse")

    def parse(self, path: str | Path) -> SourceParseArtifact:
        raise NotImplementedError(
            "LiteParseSourceParser.parse requires parse_source_with_adapter."
        )

    def parse_source(self, source: SourceDocument) -> SourceParseArtifact:
        from liteparse import LiteParse

        parser = LiteParse(**self.config)
        native_result = parser.parse(source["path"])
        pages = [
            ParsedPageArtifact(
                page_number=page.page_num,
                text=page.text,
                width=page.width,
                height=page.height,
            )
            for page in native_result.pages
        ]
        return SourceParseArtifact(
            source_identity=mark_parsed(source),
            pages=pages,
            diagnostics=[],
            provenance=ParserProvenance(
                parser_name=self.producer,
                parser_version=self.producer_version,
                adapter_name="liteparse",
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
            source_identity=mark_parsed(source),
            pages=artifact.pages,
            diagnostics=artifact.diagnostics,
            provenance=artifact.provenance,
            parse_time_ms=parse_time_ms,
        )
    except Exception as error:  # noqa: BLE001
        parse_time_ms = max(0, round((time.perf_counter() - started) * 1000))
        return SourceParseArtifact(
            source_identity=mark_failed(source, str(error)),
            pages=[],
            diagnostics=[
                ParserDiagnostic(level="error", message=str(error), page_number=None)
            ],
            provenance=ParserProvenance(
                parser_name=getattr(parser, "producer", parser.__class__.__name__),
                parser_version=getattr(parser, "producer_version", "unknown"),
                adapter_name=parser.__class__.__name__,
                artifact_schema_version=PARSE_ARTIFACT_SCHEMA_VERSION,
                config=dict(getattr(parser, "config", {})),
            ),
            parse_time_ms=parse_time_ms,
        )


def parse_sources(
    sources: list[SourceDocument],
    parser: SourceParserAdapter | None = None,
) -> list[SourceParseArtifact]:
    parser = parser or LiteParseSourceParser()
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
            section_heading=section.heading,
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


def _build_page_aware_sections(
    artifact: SourceParseArtifact,
    source_id: str,
) -> list[PageAwareSectionArtifact]:
    sections: list[PageAwareSectionArtifact] = []
    current_heading = "Unsectioned"
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
                heading=current_heading,
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
            if SECTION_HEADING_RE.match(line):
                flush()
                current_heading = line.lstrip("#").strip().rstrip(":")
                continue
            current_lines.append(line)
            if page_number:
                current_pages.append(page_number)
    flush()
    return sections


def _package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "unknown"


__all__ = [
    "PARSE_ARTIFACT_SCHEMA_VERSION",
    "PAGE_AWARE_ARTIFACT_SCHEMA_VERSION",
    "LiteParseSourceParser",
    "PageAwareArtifacts",
    "PageAwareChunkArtifact",
    "PageAwareSectionArtifact",
    "ParsedPageArtifact",
    "ParserDiagnostic",
    "ParserProvenance",
    "SourceParseArtifact",
    "SourceParserAdapter",
    "build_page_aware_artifacts",
    "parse_source_with_adapter",
    "parse_sources",
    "write_page_aware_artifacts",
]
