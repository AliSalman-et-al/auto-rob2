"""Quote-binding guard must tolerate typographic dash noise but still reject
genuine outside-packet quotes.

The classifier's `_answer_quote_is_packet_bound` guard rejects an SQ answer
whose quote is not found in the selected packet, overwriting it to NI. Its job
is to catch hallucinated or paraphrased quotes. But it compared quote against
packet text by exact (casefolded, whitespace-collapsed) substring, so a model
quote that reproduced the source with a non-breaking hyphen (U+2011) instead of
a plain hyphen (U+002D) - e.g. "Kaplan-Meier" - was rejected even though it is
the same text. That discarded correct Domain 5 answers and dragged the domain to
"Some concerns". Dash variants must normalize to a plain hyphen before matching.
"""

from rob2_pipeline.nodes.domain_classifier import _answer_quote_is_packet_bound


def _packet(text: str) -> dict:
    return {"sources": [{"text": text}], "candidate_facts": []}


def test_quote_binding_accepts_unicode_dash_variant():
    # Packet has a plain hyphen; the model emitted the same sentence with a
    # non-breaking hyphen (U+2011) in "Kaplan-Meier" and "event-time". To a
    # human these are identical, so the quote must count as packet-bound.
    packet = _packet(
        "Kaplan-Meier estimates were used for event-time distributions."
    )
    answer = {
        "quote": "Kaplan‑Meier estimates were used for event‑time distributions."
    }

    assert _answer_quote_is_packet_bound(answer, packet) is True


def test_quote_binding_accepts_en_and_em_dash_variants():
    packet = _packet("Analyses used a pre-specified intention-to-treat population.")
    en_dash = {"quote": "Analyses used a pre–specified intention–to–treat population."}
    em_dash = {"quote": "Analyses used a pre—specified intention—to—treat population."}

    assert _answer_quote_is_packet_bound(en_dash, packet) is True
    assert _answer_quote_is_packet_bound(em_dash, packet) is True


def test_quote_binding_accepts_smart_quote_variants():
    # PDF extraction and LLM output frequently disagree on apostrophe/quote
    # glyphs (curly U+2018/U+2019/U+201C/U+201D vs straight). This is the same
    # false-rejection class as dashes and must also normalize.
    packet = _packet(
        "The patients' outcomes were assessed by the trial's committee."
    )
    curly = {
        "quote": "The patients’ outcomes were assessed by the trial’s committee."
    }

    assert _answer_quote_is_packet_bound(curly, packet) is True


def test_quote_binding_still_rejects_genuine_outside_packet_quote():
    # Dash normalization must not weaken the guard: a quote that is genuinely
    # not in the packet (beyond dash/whitespace noise) is still rejected.
    packet = _packet(
        "Kaplan-Meier estimates were used for event-time distributions."
    )
    answer = {"quote": "A central web randomization system concealed allocation."}

    assert _answer_quote_is_packet_bound(answer, packet) is False
