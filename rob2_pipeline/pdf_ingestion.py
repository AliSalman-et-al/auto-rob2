from langchain_core.documents import Document

from rob2_pipeline.config import build_provider
from rob2_pipeline.ingestion import evidence as _evidence
from rob2_pipeline.ingestion.docling_extract import (
    _DOCLING_CONVERTERS,
    _chunk_page_numbers,
    _configure_docling_runtime,
    _extract_with_docling,
    _extract_with_docling_loader,
    _get_docling_converter,
    _normalize_extracted_text,
)
from rob2_pipeline.ingestion.document_repr import (
    DocBlock,
    DocumentRepr,
    _export_doc_markdown,
    _page_no,
    build_document_repr,
)
from rob2_pipeline.ingestion.evidence import (
    PAPER_EXTRACTION_SYSTEM_MESSAGE,
    PROMPT_PAPER_EXTRACTION,
    _augment_consort_from_results,
    _detect_heading,
    _extract_keyword_context,
    _normalize_heading,
    _parse_paper_evidence_response,
    _parse_sections_from_docling_document,
    _tables_from_xml,
    cap_section,
    extract_censoring_context,
    extract_structural_paper_evidence,
    paper_evidence_from_sections,
    parse_sections,
)
from rob2_pipeline.ingestion.settings import (
    CENSORING_PATTERNS,
    EMBED_MAX_TOKENS,
    EMBED_MODEL_ID,
    MAX_SECTION_CHARS,
    MIN_EXTRACTED_CHARS,
    SECTION_ORDER,
    SECTION_PATTERNS,
    TOKENIZER_COUNTING_MAX_LENGTH,
    allow_remote_evidence_extraction,
    appears_rct_candidate,
)
from rob2_pipeline.models import PaperEvidence
from rob2_pipeline.types import LLMCallLogEntry


_EMBED_MODEL_ID = EMBED_MODEL_ID
_EMBED_MAX_TOKENS = EMBED_MAX_TOKENS
_TOKENIZER_COUNTING_MAX_LENGTH = TOKENIZER_COUNTING_MAX_LENGTH
_CENSORING_PATTERNS = CENSORING_PATTERNS
HuggingFaceTokenizer = None
HybridChunker = None

__all__ = [
    "CENSORING_PATTERNS",
    "DocBlock",
    "Document",
    "DocumentRepr",
    "EMBED_MAX_TOKENS",
    "EMBED_MODEL_ID",
    "HuggingFaceTokenizer",
    "HybridChunker",
    "LLMCallLogEntry",
    "MAX_SECTION_CHARS",
    "MIN_EXTRACTED_CHARS",
    "PAPER_EXTRACTION_SYSTEM_MESSAGE",
    "PROMPT_PAPER_EXTRACTION",
    "PaperEvidence",
    "SECTION_ORDER",
    "SECTION_PATTERNS",
    "TOKENIZER_COUNTING_MAX_LENGTH",
    "_CENSORING_PATTERNS",
    "_DOCLING_CONVERTERS",
    "_EMBED_MAX_TOKENS",
    "_EMBED_MODEL_ID",
    "_TOKENIZER_COUNTING_MAX_LENGTH",
    "_augment_consort_from_results",
    "_build_docling_chunker",
    "_build_docling_chunks",
    "_chunk_page_numbers",
    "_configure_docling_runtime",
    "_detect_heading",
    "_export_doc_markdown",
    "_extract_keyword_context",
    "_extract_with_docling",
    "_extract_with_docling_loader",
    "_get_docling_converter",
    "_normalize_extracted_text",
    "_normalize_heading",
    "_page_no",
    "_parse_paper_evidence_response",
    "_parse_sections_from_docling_document",
    "_tables_from_xml",
    "allow_remote_evidence_extraction",
    "appears_rct_candidate",
    "build_document_repr",
    "build_provider",
    "cap_section",
    "extract_censoring_context",
    "extract_full_text",
    "extract_paper_evidence",
    "extract_structural_paper_evidence",
    "paper_evidence_from_sections",
    "parse_sections",
]


def extract_full_text(pdf_path: str) -> str:
    return _normalize_extracted_text(_extract_with_docling(pdf_path))


def _build_docling_chunker():
    global HuggingFaceTokenizer, HybridChunker
    if HybridChunker is not None and HuggingFaceTokenizer is None:
        tokenizer = None
        return HybridChunker(tokenizer=tokenizer)
    if HuggingFaceTokenizer is None:
        from docling_core.transforms.chunker.tokenizer.huggingface import (
            HuggingFaceTokenizer as _HuggingFaceTokenizer,
        )

        HuggingFaceTokenizer = _HuggingFaceTokenizer
    if HybridChunker is None:
        from docling.chunking import HybridChunker as _HybridChunker

        HybridChunker = _HybridChunker
    tokenizer = HuggingFaceTokenizer.from_pretrained(
        EMBED_MODEL_ID,
        max_tokens=EMBED_MAX_TOKENS,
        model_max_length=TOKENIZER_COUNTING_MAX_LENGTH,
    )
    return HybridChunker(tokenizer=tokenizer)


def _build_docling_chunks(conv_result) -> list[Document]:
    chunker = _build_docling_chunker()
    docs: list[Document] = []
    for chunk in chunker.chunk(conv_result.document):
        page_numbers = _chunk_page_numbers(chunk)
        docs.append(
            Document(
                page_content=chunk.text,
                metadata={
                    "section": chunk.meta.headings[0] if chunk.meta.headings else "",
                    "page_numbers": page_numbers,
                    "dl_meta": chunk.meta.export_json_dict(),
                },
            )
        )
    return docs


def extract_paper_evidence(
    doc_repr: DocumentRepr,
) -> tuple[PaperEvidence, list[LLMCallLogEntry]]:
    original_build_provider = _evidence.build_provider
    _evidence.build_provider = build_provider
    try:
        return _evidence.extract_paper_evidence(doc_repr)
    finally:
        _evidence.build_provider = original_build_provider
