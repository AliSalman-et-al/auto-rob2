from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_doc(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_readme_documents_bm25s_supplement_retrieval_without_legacy_rag() -> None:
    readme = read_doc("README.md")

    required_terms = [
        "document type detection",
        "SupplementSegments",
        "BM25S",
        "SupplementIndex",
        "evidence packets",
        "supplement_segments",
        "supplement_retrieval_grades",
        "supplement parsing",
        "annotation",
        "retrieval diagnostics",
    ]
    for term in required_terms:
        assert term in readme

    legacy_active_terms = [
        "LangChain FAISS",
        "BGE-small embeddings",
        "mixed RAG retrieval",
        "`rag_sources`",
        "`trial_retrieval_indexes`",
        "Empty RAG output",
    ]
    for term in legacy_active_terms:
        assert term not in readme


def test_architecture_documents_supplement_retrieval_boundaries() -> None:
    architecture = read_doc("ARCHITECTURE.md")
    adr = read_doc("docs/adr/0002-reuse-trial-level-ingestion-and-retrieval-indexes.md")

    required_architecture_terms = [
        "SupplementSegment",
        "SupplementIndex",
        "`supplement_segment`",
        "`supplement_segments`",
        "`supplement_indexes`",
        "`supplement_retrieval_grades`",
        "primary-paper evidence is not BM25S-indexed",
        "section_text fallbacks",
    ]
    for term in required_architecture_terms:
        assert term in architecture

    assert "standalone RAG retrieval node" not in architecture
    assert "same per-study RAG index" not in architecture
    assert "Common `source_kind` values are `rag_chunk`" not in architecture

    assert "ADR-0002" in architecture
    assert "vector-index reuse detail is superseded" in architecture
    assert "trial-level ingestion artifact reuse remains valid" in architecture
    assert "BM25S" in adr
    assert "superseded" in adr.casefold()
