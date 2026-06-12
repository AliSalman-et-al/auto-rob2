"""LangChain FAISS-backed retrieval helpers."""

from __future__ import annotations

import re
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from rob2_pipeline.types import ChunkMeta

_EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"

_embeddings: HuggingFaceEmbeddings | None = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=_EMBED_MODEL_ID,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


DOMAIN_SECTION_FILTERS: dict[str, list[str]] = {
    "d1": [
        "method",
        "random",
        "allocat",
        "participant",
        "baseline",
        "consort",
        "enrol",
    ],
    "d2": [
        "method",
        "blind",
        "mask",
        "deviat",
        "protocol",
        "intervention",
        "treatment",
    ],
    "d3": ["missing", "censor", "loss", "follow", "withdraw", "dropout", "attrition"],
    "d4": ["outcome", "measur", "assess", "endpoint", "instrument", "adjudicat"],
    "d5": ["registr", "protocol", "trial design", "nct"],
}

DOMAIN_REQUIRED_TERMS: dict[str, list[str]] = {
    "d1": ["random", "allocat"],
    "d2": ["blind", "mask", "open-label", "deviation", "adherence", "analysis"],
    "d3": ["missing", "follow", "withdraw", "censor", "analysis population"],
    "d4": ["outcome", "measure", "assess", "blind", "adjudicat"],
    "d5": ["registr", "protocol", "outcome", "analysis"],
}


def build_index(chunks: list[Document]) -> FAISS:
    if not chunks:
        raise ValueError("Cannot build index from empty chunk list")
    return FAISS.from_documents(chunks, _get_embeddings())


def build_filtered_index(chunks: list[Document], keywords: list[str]) -> FAISS | None:
    lowered_keywords = [keyword.lower() for keyword in keywords]
    filtered = [
        chunk
        for chunk in chunks
        if any(
            keyword in (chunk.metadata.get("section") or "").lower()
            for keyword in lowered_keywords
        )
    ]
    if len(filtered) < 3:
        return None
    return FAISS.from_documents(filtered, _get_embeddings())


def build_trial_retrieval_indexes(chunks: list[Document]) -> dict[str, Any]:
    index = build_index(chunks)
    filtered = {
        domain: build_filtered_index(chunks, keywords)
        for domain, keywords in DOMAIN_SECTION_FILTERS.items()
    }
    return {"index": index, "filtered": filtered}


def retrieve_adaptive(
    index: FAISS,
    filtered_index: FAISS | None,
    queries: list[str],
    token_budget: int = 1200,
    candidate_k: int = 12,
) -> tuple[str, list[ChunkMeta]]:
    scores: dict[str, float] = {}
    docs_by_key: dict[str, Document] = {}

    def search(search_index: FAISS) -> None:
        for query in queries:
            for doc, score in search_index.similarity_search_with_score(
                query, k=candidate_k
            ):
                key = _doc_key(doc)
                if key not in scores or float(score) < scores[key]:
                    scores[key] = float(score)
                    docs_by_key[key] = doc

    search(filtered_index or index)
    if filtered_index is not None and len(docs_by_key) < 3:
        search(index)

    texts: list[str] = []
    metas: list[ChunkMeta] = []
    total_tokens = 0

    for key in sorted(scores, key=lambda item: scores[item]):
        doc = docs_by_key[key]
        chunk_tokens = max(1, len(doc.page_content) // 4)
        if total_tokens + chunk_tokens > token_budget:
            if texts:
                break
            allowed_chars = max(1, token_budget * 4)
            page_content = doc.page_content[:allowed_chars].rstrip()
            chunk_tokens = max(1, len(page_content) // 4)
        else:
            page_content = doc.page_content
        total_tokens += chunk_tokens
        texts.append(page_content)
        metas.append(
            ChunkMeta(
                text=page_content,
                section=doc.metadata.get("section", ""),
                original_heading=doc.metadata.get("original_heading", ""),
                page_numbers=list(doc.metadata.get("page_numbers") or []),
                score=scores[key],
                document_id=doc.metadata.get("document_id", ""),
                document_name=doc.metadata.get("document_name", ""),
                document_role=doc.metadata.get("document_role", ""),
                source_kind=doc.metadata.get("source_kind", "rag_chunk"),
                source_path=doc.metadata.get("source_path", ""),
            )
        )

    return "\n\n".join(texts), metas


def retrieve_lexical(
    chunks: list[Document],
    queries: list[str],
    *,
    token_budget: int = 1200,
    candidate_k: int = 12,
    section_keywords: list[str] | None = None,
) -> tuple[str, list[ChunkMeta]]:
    """Dependency-light retrieval fallback used when vector retrieval is unavailable."""
    if not chunks:
        return "", []
    query_terms = _query_terms(queries)
    lowered_section_keywords = [keyword.lower() for keyword in section_keywords or []]
    scored: list[tuple[float, int, Document]] = []
    for index, chunk in enumerate(chunks):
        text = chunk.page_content or ""
        lowered = text.casefold()
        section = str(chunk.metadata.get("section", "")).casefold()
        term_hits = sum(1 for term in query_terms if term in lowered)
        section_hits = sum(1 for term in lowered_section_keywords if term in section)
        role_boost = 0.0
        role = str(chunk.metadata.get("document_role", "")).casefold()
        if role in {"protocol", "sap", "registry"}:
            role_boost = 0.75
        score = float(term_hits) + (0.5 * section_hits) + role_boost
        if score > 0:
            scored.append((score, index, chunk))
    if not scored:
        scored = [(0.0, index, chunk) for index, chunk in enumerate(chunks[:candidate_k])]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return _materialize_ranked_documents(
        [(chunk, 1.0 / (score + 1.0)) for score, _, chunk in scored[:candidate_k]],
        token_budget=token_budget,
    )


def grade_retrieved_context(domain: str, text: str, metas: list[ChunkMeta]) -> dict:
    lowered = (text or "").lower()
    required = DOMAIN_REQUIRED_TERMS.get(domain, [])
    missing = [term for term in required if term not in lowered]
    if not text.strip():
        relevance = 0.0
    else:
        relevance = (len(required) - len(missing)) / len(required) if required else 1.0
    unique_sections = {meta.get("section", "") for meta in metas if meta.get("section")}
    coverage = min(1.0, (len(unique_sections) / 2) if unique_sections else 0.0)
    return {
        "relevance": round(relevance, 3),
        "coverage": round(coverage, 3),
        "missing_evidence": missing,
        "retry_recommended": relevance < 0.4 or coverage < 0.5,
    }


def _materialize_ranked_documents(
    ranked_docs: list[tuple[Document, float]], *, token_budget: int
) -> tuple[str, list[ChunkMeta]]:
    texts: list[str] = []
    metas: list[ChunkMeta] = []
    total_tokens = 0
    for doc, score in ranked_docs:
        chunk_tokens = max(1, len(doc.page_content) // 4)
        if total_tokens + chunk_tokens > token_budget:
            if texts:
                break
            allowed_chars = max(1, token_budget * 4)
            page_content = doc.page_content[:allowed_chars].rstrip()
            chunk_tokens = max(1, len(page_content) // 4)
        else:
            page_content = doc.page_content
        total_tokens += chunk_tokens
        texts.append(page_content)
        metas.append(
            ChunkMeta(
                text=page_content,
                section=doc.metadata.get("section", ""),
                original_heading=doc.metadata.get("original_heading", ""),
                page_numbers=list(doc.metadata.get("page_numbers") or []),
                score=score,
                document_id=doc.metadata.get("document_id", ""),
                document_name=doc.metadata.get("document_name", ""),
                document_role=doc.metadata.get("document_role", ""),
                source_kind=doc.metadata.get("source_kind", "rag_chunk"),
                source_path=doc.metadata.get("source_path", ""),
            )
        )
    return "\n\n".join(texts), metas


def _query_terms(queries: list[str]) -> tuple[str, ...]:
    terms: list[str] = []
    for query in queries:
        terms.extend(
            term
            for term in re.findall(r"[a-z0-9-]+", query.casefold())
            if len(term) >= 5
        )
    return tuple(dict.fromkeys(terms))


def _doc_key(doc: Document) -> str:
    pages = ",".join(str(page) for page in doc.metadata.get("page_numbers") or [])
    section = doc.metadata.get("section", "")
    document_id = doc.metadata.get("document_id", "")
    return f"{document_id}|{section}|{pages}|{doc.page_content[:160]}"
