from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from rob2_pipeline.types import SourceDocument


def classify_document_role(path: Path) -> str:
    name = path.name.casefold()
    compact = name.replace("_", "-")
    if (
        "statistical-analysis" in compact
        or "analysis-plan" in compact
        or "sap" in compact
    ):
        return "sap"
    if "protocol" in compact:
        return "protocol"
    if (
        "data-sharing" in compact
        or compact.startswith("ds-")
        or compact.startswith("ds.")
        or compact.startswith("dss-")
        or compact.startswith("dss.")
    ):
        return "data_sharing"
    if "disclosure" in compact or "coi" in compact or "conflict" in compact:
        return "disclosure"
    if "appendix" in compact or "supplement" in compact or compact.startswith("mmc"):
        return "appendix"
    return "unknown_supplement"


def primary_source_document(path: Path) -> SourceDocument:
    return SourceDocument(
        document_id="primary",
        document_name=path.name,
        document_role="primary",
        source_kind="rag_chunk",
        path=str(path),
        is_primary=True,
        status="parsed",
    )


def supplement_source_document(path: Path, index: int) -> SourceDocument:
    return SourceDocument(
        document_id=f"supplement:{index:03d}",
        document_name=path.name,
        document_role=classify_document_role(path),
        source_kind="rag_chunk",
        path=str(path),
        is_primary=False,
        status="pending",
    )


def mark_missing(source: SourceDocument, path: Path) -> SourceDocument:
    updated = SourceDocument(**source)
    updated["status"] = "missing"
    updated["error"] = f"Supplement not found: {path}"
    return updated


def mark_failed(source: SourceDocument, message: str) -> SourceDocument:
    updated = SourceDocument(**source)
    updated["status"] = "failed"
    updated["error"] = message
    return updated


def mark_parsed(source: SourceDocument) -> SourceDocument:
    updated = SourceDocument(**source)
    updated["status"] = "parsed"
    updated.pop("error", None)
    return updated


def mark_partial(source: SourceDocument, warnings: list[str]) -> SourceDocument:
    updated = SourceDocument(**source)
    updated["status"] = "partial"
    updated["error"] = "; ".join(warnings)
    return updated


def skipped_source_documents(
    paths: list[str], reason: str
) -> tuple[list[SourceDocument], list[str]]:
    documents: list[SourceDocument] = []
    warnings: list[str] = []
    for index, raw_path in enumerate(paths, start=1):
        path = Path(raw_path)
        source = supplement_source_document(path, index)
        source = mark_failed(source, f"Supplement not ingested: {path}: {reason}")
        documents.append(source)
        warnings.append(source["error"])
    return documents, warnings


def apply_source_metadata(chunks: list, source: SourceDocument) -> list:
    enriched = []
    for chunk in chunks:
        if not isinstance(chunk, Document):
            enriched.append(chunk)
            continue
        metadata = dict(chunk.metadata)
        metadata.update(
            {
                "document_id": source.get("document_id", ""),
                "document_name": source.get("document_name", ""),
                "document_role": source.get("document_role", ""),
                "source_kind": source.get("source_kind", "rag_chunk"),
                "source_path": source.get("path", ""),
            }
        )
        enriched.append(Document(page_content=chunk.page_content, metadata=metadata))
    return enriched
