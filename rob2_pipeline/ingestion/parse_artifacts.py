from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Protocol, TypedDict

from rob2_pipeline.ingestion.source_catalog import mark_failed, mark_parsed
from rob2_pipeline.types import SourceDocument


PARSE_ARTIFACT_SCHEMA_VERSION = "parse-artifact-v1"


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

    def to_dict(self) -> dict:
        return {
            "source_identity": dict(self.source_identity),
            "pages": [dict(page) for page in self.pages],
            "diagnostics": [asdict(diagnostic) for diagnostic in self.diagnostics],
            "provenance": asdict(self.provenance),
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
    try:
        parse_source = getattr(parser, "parse_source", None)
        if parse_source is not None:
            artifact = parse_source(source)
        else:
            artifact = parser.parse(source["path"])
        return SourceParseArtifact(
            source_identity=mark_parsed(source),
            pages=artifact.pages,
            diagnostics=artifact.diagnostics,
            provenance=artifact.provenance,
        )
    except Exception as error:  # noqa: BLE001
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
        )


def parse_sources(
    sources: list[SourceDocument],
    parser: SourceParserAdapter | None = None,
) -> list[SourceParseArtifact]:
    parser = parser or LiteParseSourceParser()
    return [parse_source_with_adapter(source, parser) for source in sources]


def _package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "unknown"


__all__ = [
    "PARSE_ARTIFACT_SCHEMA_VERSION",
    "LiteParseSourceParser",
    "ParsedPageArtifact",
    "ParserDiagnostic",
    "ParserProvenance",
    "SourceParseArtifact",
    "SourceParserAdapter",
    "parse_source_with_adapter",
    "parse_sources",
]
