import logging
import os
import re

from langchain_core.documents import Document

from rob2_pipeline.ingestion.settings import (
    EMBED_MAX_TOKENS,
    EMBED_MODEL_ID,
    MIN_EXTRACTED_CHARS,
    TOKENIZER_COUNTING_MAX_LENGTH,
)


_DOCLING_CONVERTERS: dict[bool, object] = {}


def extract_full_text(pdf_path: str) -> str:
    return _normalize_extracted_text(_extract_with_docling(pdf_path))


def _extract_with_docling(pdf_path: str) -> str:
    errors = []
    for use_ocr in (False, True):
        try:
            return _extract_with_docling_loader(pdf_path, use_ocr=use_ocr)
        except Exception as error:
            errors.append(f"OCR={use_ocr}: {error}")
    raise RuntimeError("; ".join(errors))


def _extract_with_docling_loader(pdf_path: str, use_ocr: bool) -> str:
    _configure_docling_runtime()

    from langchain_docling import DoclingLoader
    from langchain_docling.loader import ExportType

    loader = DoclingLoader(
        file_path=pdf_path,
        converter=_get_docling_converter(use_ocr=use_ocr),
        export_type=ExportType.MARKDOWN,
    )
    docs = list(loader.lazy_load())
    markdown = "\n\n".join(doc.page_content for doc in docs if doc.page_content)
    if len(markdown.strip()) < MIN_EXTRACTED_CHARS:
        raise RuntimeError(
            f"langchain-docling returned too little text to use with OCR={use_ocr}"
        )
    return markdown


def _get_docling_converter(use_ocr: bool):
    converter = _DOCLING_CONVERTERS.get(use_ocr)
    if converter is not None:
        return converter

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions()
    pipeline_options.allow_external_plugins = True
    pipeline_options.do_ocr = use_ocr
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        },
    )
    _DOCLING_CONVERTERS[use_ocr] = converter
    return converter


def _configure_docling_runtime() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")
    logging.getLogger("RapidOCR").setLevel(logging.WARNING)
    logging.getLogger("rapidocr").setLevel(logging.WARNING)


def _normalize_extracted_text(text: str) -> str:
    text = text.replace("\x00", "").replace("\xa0", " ")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def _build_docling_chunker():
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import (
        HuggingFaceTokenizer,
    )

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


def _chunk_page_numbers(chunk) -> list[int]:
    direct_pages = getattr(chunk.meta, "page_numbers", None)
    if direct_pages is not None:
        return list(direct_pages)
    pages = set()
    for item in getattr(chunk.meta, "doc_items", []) or []:
        for prov in getattr(item, "prov", []) or []:
            page_no = getattr(prov, "page_no", None)
            if page_no:
                pages.add(int(page_no))
    return sorted(pages)
