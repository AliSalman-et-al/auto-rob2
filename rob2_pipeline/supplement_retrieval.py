from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

try:
    import bm25s
except ImportError:  # pragma: no cover - exercised by monkeypatch in tests
    bm25s = None


@dataclass(frozen=True)
class SupplementSegment:
    segment_id: str
    document_id: str
    document_name: str
    document_role: str
    source_path: str
    heading: str
    page_numbers: list[int]
    domain_tags: list[str]
    annotation: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_source(self, *, score: float) -> dict[str, Any]:
        return {
            **self.to_dict(),
            "section": self.heading,
            "original_heading": self.heading,
            "score": score,
            "source_kind": "supplement_segment",
        }


@dataclass(frozen=True)
class _DomainIndex:
    segments: list[SupplementSegment]
    retriever: Any


class SupplementIndex:
    def __init__(
        self,
        segments: list[SupplementSegment],
        all_index: _DomainIndex | None,
        domain_indexes: dict[str, _DomainIndex],
    ) -> None:
        self._segments = list(segments)
        self._all_index = all_index
        self._domain_indexes = dict(domain_indexes)

    @classmethod
    def from_segments(cls, segments: list[SupplementSegment]) -> "SupplementIndex":
        segments = list(segments)
        if not segments:
            return cls([], None, {})
        if bm25s is None:
            raise RuntimeError(
                "BM25S is required to build SupplementIndex for non-empty supplements."
            )
        all_index = _build_index(segments)
        domain_indexes = {
            domain: _build_index(domain_segments)
            for domain, domain_segments in _segments_by_domain(segments).items()
            if len(domain_segments) >= 2
        }
        return cls(segments, all_index, domain_indexes)

    def retrieve(
        self,
        query: str,
        *,
        domain: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        if top_k <= 0 or self._all_index is None:
            return {"segments": [], "best_score": 0.0}
        selected_index = self._domain_indexes.get(domain.casefold(), self._all_index)
        ranked = _retrieve(selected_index, query, top_k=top_k)
        segments = [
            segment.to_source(score=round(score, 6)) for segment, score in ranked
        ]
        best_score = segments[0]["score"] if segments else 0.0
        return {"segments": segments, "best_score": best_score}

    def to_dict(self) -> dict[str, Any]:
        return {"segments": [segment.to_dict() for segment in self._segments]}


def _segments_by_domain(
    segments: list[SupplementSegment],
) -> dict[str, list[SupplementSegment]]:
    grouped: dict[str, list[SupplementSegment]] = {}
    for segment in segments:
        for tag in segment.domain_tags:
            grouped.setdefault(tag.casefold(), []).append(segment)
    return grouped


def _build_index(segments: list[SupplementSegment]) -> _DomainIndex:
    if bm25s is None:
        raise RuntimeError(
            "BM25S is required to build SupplementIndex for non-empty supplements."
        )
    retriever = bm25s.BM25(corpus=segments)
    retriever.index(
        bm25s.tokenize(
            [_indexed_text(segment) for segment in segments],
            show_progress=False,
        ),
        show_progress=False,
    )
    return _DomainIndex(segments=segments, retriever=retriever)


def _retrieve(
    domain_index: _DomainIndex,
    query: str,
    *,
    top_k: int,
) -> list[tuple[SupplementSegment, float]]:
    if bm25s is None:
        raise RuntimeError("BM25S is required to retrieve from SupplementIndex.")
    query_tokens = bm25s.tokenize(query or "", show_progress=False)
    results, scores = domain_index.retriever.retrieve(
        query_tokens,
        corpus=domain_index.segments,
        k=min(top_k, len(domain_index.segments)),
        show_progress=False,
    )
    return [
        (segment, float(score))
        for segment, score in zip(results[0], scores[0], strict=True)
    ]


def _indexed_text(segment: SupplementSegment) -> str:
    return "\n".join(
        part
        for part in [segment.heading, segment.annotation, segment.text]
        if part.strip()
    )
