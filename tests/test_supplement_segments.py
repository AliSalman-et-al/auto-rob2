from rob2_pipeline.ingestion.supplement_segments import _page_segments, ALL_ROB2_DOMAINS


def _source():
    return {
        "document_id": "supp:001",
        "document_name": "protocol.pdf",
        "document_role": "protocol",
        "path": "protocol.pdf",
    }


def test_page_segments_one_per_nonempty_page():
    pages = [{"page_number": i, "text": f"page {i} content"} for i in range(1, 6)]
    segments = _page_segments(pages, _source())
    assert len(segments) == 5
    assert [s.page_numbers for s in segments] == [[1], [2], [3], [4], [5]]
    assert segments[0].text == "page 1 content"
    assert segments[2].heading == "Page 3"
    assert all(s.domain_tags == ALL_ROB2_DOMAINS for s in segments)
    assert all(s.annotation == "" for s in segments)
    assert segments[0].segment_id == "supp:001:segment:0001"
    assert len({s.segment_id for s in segments}) == 5


def test_page_segments_skips_empty_pages():
    pages = [
        {"page_number": 1, "text": "real content"},
        {"page_number": 2, "text": "   "},
        {"page_number": 3, "text": ""},
        {"page_number": 4, "text": "more content"},
    ]
    segments = _page_segments(pages, _source())
    assert [s.page_numbers for s in segments] == [[1], [4]]
