import json

import pytest

from rob2_pipeline.supplement_retrieval import SupplementIndex, SupplementSegment


def _segment(
    segment_id: str,
    text: str,
    *,
    domains: list[str],
    annotation: str = "",
) -> SupplementSegment:
    return SupplementSegment(
        segment_id=segment_id,
        document_id="supplement:001",
        document_name="protocol.pdf",
        document_role="protocol",
        source_path="protocol.pdf",
        heading="Protocol",
        page_numbers=[1],
        domain_tags=domains,
        annotation=annotation or text,
        text=text,
    )


def test_empty_supplement_index_returns_empty_results_and_serializes_cleanly():
    index = SupplementIndex.from_segments([])

    result = index.retrieve("allocation concealment", domain="d1", top_k=3)

    assert result == {"segments": [], "best_score": 0.0}
    assert index.to_dict() == {"segments": []}
    assert json.dumps(index.to_dict()) == '{"segments": []}'


def test_non_empty_supplement_index_serializes_segments_without_bm25_internals():
    segment = _segment(
        "s1",
        "Central allocation concealment was used.",
        domains=["d1"],
    )

    index = SupplementIndex.from_segments([segment])

    serialized = index.to_dict()
    assert serialized == {"segments": [segment.to_dict()]}
    assert "bm25" not in json.dumps(serialized).lower()


def test_retrieve_limits_to_top_k_and_reports_best_score():
    index = SupplementIndex.from_segments(
        [
            _segment(
                "s1",
                "Allocation concealment with central randomization.",
                domains=["d1"],
            ),
            _segment(
                "s2", "Randomization sequence used permuted blocks.", domains=["d1"]
            ),
            _segment(
                "s3",
                "Outcome assessors measured radiographic progression.",
                domains=["d4"],
            ),
        ]
    )

    result = index.retrieve("randomization allocation", domain="d1", top_k=1)

    assert [segment["segment_id"] for segment in result["segments"]] == ["s1"]
    assert result["best_score"] == result["segments"][0]["score"]
    assert result["best_score"] > 0
    assert result["segments"][0]["source_kind"] == "supplement_segment"


def test_retrieve_filters_to_domain_when_two_or_more_segments_are_tagged():
    index = SupplementIndex.from_segments(
        [
            _segment("d1-a", "Central allocation was concealed.", domains=["d1"]),
            _segment("d1-b", "Randomization sequence used blocks.", domains=["d1"]),
            _segment(
                "d5-a",
                "The statistical analysis plan prespecified the progression-free survival analysis.",
                domains=["d5"],
            ),
        ]
    )

    result = index.retrieve("analysis plan prespecified", domain="d1", top_k=3)

    assert {segment["segment_id"] for segment in result["segments"]} == {"d1-a", "d1-b"}


def test_retrieve_falls_back_to_all_segments_when_fewer_than_two_domain_tagged():
    index = SupplementIndex.from_segments(
        [
            _segment("d1-a", "Central allocation was concealed.", domains=["d1"]),
            _segment(
                "d5-a",
                "The statistical analysis plan prespecified the progression-free survival analysis.",
                domains=["d5"],
            ),
        ]
    )

    result = index.retrieve("analysis plan prespecified", domain="d1", top_k=2)

    assert result["segments"][0]["segment_id"] == "d5-a"


def test_missing_bm25s_fails_clearly_for_non_empty_segments(monkeypatch):
    import rob2_pipeline.supplement_retrieval as supplement_retrieval

    monkeypatch.setattr(supplement_retrieval, "bm25s", None)

    with pytest.raises(RuntimeError, match="BM25S is required"):
        SupplementIndex.from_segments(
            [_segment("s1", "Central allocation concealment.", domains=["d1"])]
        )

    assert SupplementIndex.from_segments([]).retrieve("anything", domain="d1") == {
        "segments": [],
        "best_score": 0.0,
    }
