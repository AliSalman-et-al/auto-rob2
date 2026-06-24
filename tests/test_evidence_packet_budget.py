from rob2_pipeline.nodes.evidence_packets import _estimate_tokens, _cap_source_text


def test_estimate_tokens_is_conservative():
    # ~3 chars/token (over-counts tokens vs the observed ~3.6, so it is safe)
    assert _estimate_tokens("a" * 300) >= 100


def test_cap_source_text_truncates_and_marks_long_text():
    src = {"text": "x" * 50000, "score": 1.0}
    capped = _cap_source_text(src, max_chars=6000)
    assert len(capped["text"]) <= 6000 + 40  # body + truncation marker
    assert "truncated" in capped["text"].lower()
    # original dict is not mutated
    assert len(src["text"]) == 50000


def test_cap_source_text_leaves_short_text_unchanged():
    src = {"text": "short evidence", "score": 1.0}
    assert _cap_source_text(src, max_chars=6000)["text"] == "short evidence"
