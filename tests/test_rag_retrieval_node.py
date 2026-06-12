"""Tests for rag_retrieval_node with the LangChain FAISS API."""

from unittest.mock import MagicMock

from langchain_core.documents import Document

from rob2_pipeline.models import empty_paper_evidence
from rob2_pipeline.types import ChunkMeta


def _make_doc(text: str, section: str = "Methods") -> Document:
    return Document(
        page_content=text, metadata={"section": section, "page_numbers": [1]}
    )


def _evidence():
    evidence = empty_paper_evidence("fallback")
    evidence["d1_randomization"]["text"] = "Randomized centrally."
    evidence["d2_blinding"]["text"] = "Open-label trial."
    evidence["d3_missing_data"]["text"] = "Complete follow-up."
    evidence["d4_outcome_meas"]["text"] = "Overall survival endpoint."
    evidence["d5_registration"]["text"] = "NCT00000000."
    evidence["methods"]["text"] = "Intention-to-treat methods."
    evidence["results"]["text"] = "All randomized participants analysed."
    return evidence


def _base_state(docling_chunks):
    return {
        "docling_chunks": docling_chunks,
        "full_text": "sample",
        "evidence": _evidence(),
        "errors": [],
        "outcome": "Overall Survival",
    }


def test_rag_retrieval_node_populates_compatibility_rag_contexts(monkeypatch):
    import rob2_pipeline.nodes.rag_retrieval as node

    chunks = [_make_doc(f"Chunk {i} about randomization.") for i in range(10)]
    monkeypatch.setattr(node, "build_index", lambda _: MagicMock())
    monkeypatch.setattr(node, "build_filtered_index", lambda chunks, keywords: None)
    monkeypatch.setattr(
        node,
        "retrieve_adaptive",
        lambda index, filtered, queries: (
            "Retrieved text.",
            [
                ChunkMeta(
                    text="Retrieved text.",
                    section="Methods",
                    page_numbers=[2],
                    score=0.9,
                )
            ],
        ),
    )

    result = node.rag_retrieval_node(_base_state(chunks))

    assert set(result["rag_contexts"]) >= {
        "d1",
        "d2_blinding",
        "d2_deviations",
        "d2_analysis",
        "d3",
        "d4_measurement",
        "d4_assessor",
        "d5",
    }
    assert all(isinstance(text, str) for text in result["rag_contexts"].values())


def test_rag_retrieval_node_populates_rag_chunk_metadata(monkeypatch):
    import rob2_pipeline.nodes.rag_retrieval as node

    chunks = [_make_doc(f"Chunk {i}.") for i in range(10)]
    monkeypatch.setattr(node, "build_index", lambda _: MagicMock())
    monkeypatch.setattr(node, "build_filtered_index", lambda chunks, keywords: None)
    monkeypatch.setattr(
        node,
        "retrieve_adaptive",
        lambda index, filtered, queries: (
            "Some text.",
            [
                ChunkMeta(
                    text="Some text.", section="Methods", page_numbers=[3], score=0.85
                )
            ],
        ),
    )

    result = node.rag_retrieval_node(_base_state(chunks))

    assert set(result["rag_chunk_metadata"]) == {"d1", "d2", "d3", "d4", "d5"}
    for metas in result["rag_chunk_metadata"].values():
        assert isinstance(metas, list)
        assert len(metas) > 0
        assert "section" in metas[0]
        assert "page_numbers" in metas[0]
        assert "score" in metas[0]


def test_rag_retrieval_node_preserves_original_heading_metadata(monkeypatch):
    import rob2_pipeline.nodes.rag_retrieval as node

    chunks = [_make_doc("Participants were randomized.", section="METHODS")]
    monkeypatch.setattr(node, "build_index", lambda _: MagicMock())
    monkeypatch.setattr(node, "build_filtered_index", lambda chunks, keywords: None)
    monkeypatch.setattr(
        node,
        "retrieve_adaptive",
        lambda index, filtered, queries: (
            "Study design text.",
            [
                ChunkMeta(
                    text="Study design text.",
                    section="METHODS",
                    original_heading="Study Design",
                    page_numbers=[4],
                    score=0.8,
                )
            ],
        ),
    )

    result = node.rag_retrieval_node(_base_state(chunks))

    meta = result["rag_chunk_metadata"]["d1"][0]
    assert meta["section"] == "METHODS"
    assert meta["original_heading"] == "Study Design"
    assert meta["page_numbers"] == [4]


def test_rag_retrieval_node_falls_back_when_no_chunks():
    from rob2_pipeline.nodes.rag_retrieval import rag_retrieval_node

    result = rag_retrieval_node(_base_state([]))

    assert "Randomized centrally" in result["rag_contexts"]["d1"]
    assert result["rag_chunk_metadata"] == {
        "d1": [],
        "d2": [],
        "d3": [],
        "d4": [],
        "d5": [],
    }


def test_rag_retrieval_node_falls_back_when_rag_errors(monkeypatch):
    import rob2_pipeline.nodes.rag_retrieval as node

    def boom(_chunks):
        raise RuntimeError("embedding init failed")

    state = _base_state([_make_doc("Chunk.")])
    monkeypatch.setattr(node, "build_index", boom)
    monkeypatch.setattr(
        node, "extract_censoring_context", lambda full_text, outcome: ""
    )

    result = node.rag_retrieval_node(state)

    assert "Randomized centrally" in result["rag_contexts"]["d1"]
    assert result["rag_chunk_metadata"] == {
        "d1": [],
        "d2": [],
        "d3": [],
        "d4": [],
        "d5": [],
    }
    assert any(
        "RAG retrieval failed" in warning for warning in state["evidence"]["warnings"]
    )
