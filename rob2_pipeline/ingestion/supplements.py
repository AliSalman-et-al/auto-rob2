from __future__ import annotations

import os
import re
from pathlib import Path

from langchain_core.documents import Document

from rob2_pipeline.ingestion.source_catalog import (
    apply_source_metadata,
    classify_document_role,
    mark_failed,
    mark_missing,
    mark_parsed,
    mark_partial,
    primary_source_document as _primary_source_document,
    skipped_source_documents as _skipped_source_documents,
    supplement_source_document,
)
from rob2_pipeline.types import SourceDocument


DEFAULT_SUPPLEMENT_PAGE_WINDOW = 20
DEFAULT_SUPPLEMENT_MAX_SCAN_PAGES = 300
SUPPLEMENT_EVIDENCE_TERMS = (
    "random",
    "allocation",
    "conceal",
    "mask",
    "blind",
    "protocol",
    "deviation",
    "adherence",
    "analysis",
    "intention",
    "per-protocol",
    "outcome",
    "endpoint",
    "missing",
    "censor",
    "withdraw",
    "registry",
    "registered",
)
_SUPPLEMENT_CONVERTER = None


classify_supplement = classify_document_role
primary_source_document = _primary_source_document
skipped_source_documents = _skipped_source_documents


def build_source_document(path: Path, role: str, index: int) -> SourceDocument:
    source = supplement_source_document(path, index)
    source["document_role"] = role
    return source


def ingest_supplements(
    paths: list[str],
) -> tuple[list[Document], list[SourceDocument], list[str]]:
    from rob2_pipeline.pdf_ingestion import (
        _configure_docling_runtime,
    )

    chunks: list[Document] = []
    documents: list[SourceDocument] = []
    warnings: list[str] = []
    if not paths:
        return chunks, documents, warnings

    for index, raw_path in enumerate(paths, start=1):
        path = Path(raw_path)
        source = supplement_source_document(path, index)
        if not path.exists():
            source = mark_missing(source, path)
            documents.append(source)
            warnings.append(source["error"])
            continue
        try:
            _configure_docling_runtime()
            converter = _get_supplement_converter()
            source_chunks, window_warnings = _convert_supplement_in_windows(
                converter, str(path), source
            )
            if window_warnings:
                source = mark_partial(source, window_warnings)
            else:
                source = mark_parsed(source)
            chunks.extend(source_chunks)
            warnings.extend(window_warnings)
        except Exception as error:  # noqa: BLE001
            source = mark_failed(source, f"Supplement parse failed: {path}: {error}")
            warnings.append(source["error"])
        documents.append(source)
    return chunks, documents, warnings


def _convert_supplement_in_windows(
    converter, path: str, source: SourceDocument
) -> tuple[list[Document], list[str]]:
    from rob2_pipeline.pdf_ingestion import _build_docling_chunks

    all_chunks: list[Document] = []
    warnings: list[str] = []
    window_size = _supplement_page_window()
    max_scan_pages = _effective_supplement_max_pages(path)

    for start, end in _ordered_supplement_page_ranges(path, window_size, max_scan_pages):
        try:
            conv_result = converter.convert(str(path), page_range=(start, end))
            window_chunks = apply_source_metadata(
                _build_docling_chunks(conv_result), source
            )
        except Exception as error:  # noqa: BLE001
            if _is_page_range_exhausted(error):
                if all_chunks:
                    break
                raise
            warnings.append(
                f"Supplement page window skipped: {path} pages {start}-{end}: {error}"
            )
            continue

        all_chunks.extend(window_chunks)

    return all_chunks, warnings


def _supplement_page_window() -> int:
    raw_value = os.getenv("ROB2_SUPPLEMENT_PAGE_WINDOW", "").strip()
    if not raw_value:
        return DEFAULT_SUPPLEMENT_PAGE_WINDOW
    try:
        window_size = int(raw_value)
    except ValueError:
        return DEFAULT_SUPPLEMENT_PAGE_WINDOW
    if window_size <= 0:
        return DEFAULT_SUPPLEMENT_PAGE_WINDOW
    return window_size


def _supplement_max_scan_pages() -> int:
    raw_value = os.getenv("ROB2_SUPPLEMENT_MAX_SCAN_PAGES", "").strip()
    if not raw_value:
        legacy_value = os.getenv("ROB2_SUPPLEMENT_MAX_PAGES", "").strip()
        raw_value = legacy_value
    if not raw_value:
        return DEFAULT_SUPPLEMENT_MAX_SCAN_PAGES
    try:
        max_pages = int(raw_value)
    except ValueError:
        return DEFAULT_SUPPLEMENT_MAX_SCAN_PAGES
    if max_pages <= 0:
        return DEFAULT_SUPPLEMENT_MAX_SCAN_PAGES
    return max_pages


def _effective_supplement_max_pages(path: str) -> int:
    configured = _supplement_max_scan_pages()
    page_count = _pdf_page_count(path)
    if page_count is None:
        return configured
    return max(0, min(configured, page_count))


def _ordered_supplement_page_ranges(
    path: str, window_size: int, max_pages: int
) -> list[tuple[int, int]]:
    ranges = [
        (start, min(start + window_size - 1, max_pages))
        for start in range(1, max_pages + 1, window_size)
    ]
    if not ranges:
        return []
    evidence_pages = _evidence_pages(path, max_pages)
    if not evidence_pages:
        return ranges
    evidence_ranges = []
    for page in evidence_pages:
        index = (page - 1) // window_size
        if 0 <= index < len(ranges):
            evidence_ranges.append(ranges[index])
    early_count = min(2, len(ranges))
    prioritized = [*ranges[:early_count], *evidence_ranges]
    ordered = []
    seen = set()
    for item in [*prioritized, *ranges]:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def _pdf_page_count(path: str) -> int | None:
    try:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(path)
        try:
            return len(pdf)
        finally:
            close = getattr(pdf, "close", None)
            if close is not None:
                close()
    except Exception:  # noqa: BLE001
        return None


def _evidence_pages(path: str, max_pages: int) -> list[int]:
    pattern = re.compile(
        "|".join(re.escape(term) for term in SUPPLEMENT_EVIDENCE_TERMS), re.I
    )
    pages = []
    try:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(path)
        try:
            for index in range(min(len(pdf), max_pages)):
                page = pdf[index]
                try:
                    textpage = page.get_textpage()
                    text = textpage.get_text_range() or ""
                    if pattern.search(text):
                        pages.append(index + 1)
                finally:
                    close_page = getattr(page, "close", None)
                    if close_page is not None:
                        close_page()
        finally:
            close = getattr(pdf, "close", None)
            if close is not None:
                close()
    except Exception:  # noqa: BLE001
        return []
    return pages


def _is_page_range_exhausted(error: Exception) -> bool:
    message = str(error).casefold()
    processing_error_markers = (
        "bad_alloc",
        "memory",
        "preprocess failed",
        "conversion failed",
        "failed converting",
        "runtimeerror",
    )
    if any(marker in message for marker in processing_error_markers):
        return False
    exhausted_markers = (
        "input document",
        "is not valid",
        "page range outside",
        "page range out of range",
        "outside document",
        "exceeds document",
        "no pages",
        "invalid page range",
    )
    return any(marker in message for marker in exhausted_markers)


def _get_supplement_converter():
    global _SUPPLEMENT_CONVERTER
    if _SUPPLEMENT_CONVERTER is not None:
        return _SUPPLEMENT_CONVERTER

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions()
    pipeline_options.allow_external_plugins = True
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = False
    pipeline_options.force_backend_text = True
    pipeline_options.layout_batch_size = 1
    pipeline_options.table_batch_size = 1
    pipeline_options.ocr_batch_size = 1
    pipeline_options.queue_max_size = 1
    _SUPPLEMENT_CONVERTER = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        },
    )
    return _SUPPLEMENT_CONVERTER
