"""SQ 5.3 must surface the reported statistical-analysis-methods sentence.

Regression guard for the D5 SQ 5.3 retrieval-targeting gap (2026-06-05).

The reported-methods chunk (Cox proportional-hazards / Kaplan-Meier) is already
retrieved into the d5 candidate pool, but it matches none of SQ 5.3's
prespecification / selective-reporting vocabulary. Source selection ranks by
matched-term count, so the methods chunk loses the top-3 sort to interim-
monitoring, prespecification, and registered-outcome text. With no reported
analysis methods in its packet, the classifier honestly answers NI, and NI
cannot reach Low, so Low-reference survival trials park at "Some concerns".

The fixtures below use CHAARTED's real d5 chunk text. The fix is to add the
survival-analysis-methods lexicon to the 5.3 evidence contract so the methods
chunk wins a top-3 slot. This touches only source ranking, never the judges or
skip/NA logic.
"""

from rob2_pipeline.models import empty_paper_evidence
from rob2_pipeline.nodes.evidence_packets import build_evidence_packets
from tests.test_evidence_packets import _RecordingSupplementIndex


# Real CHAARTED d5 'Statistical Analysis' chunk: the load-bearing reported-methods
# sentence SQ 5.3 needs to confirm the analyses were reported as planned.
METHODS_CHUNK = {
    "text": (
        "Descriptive statistics were used to characterize patients at study "
        "entry. Kaplan-Meier estimates were used for event-time distributions. "
        "Cox proportional-hazard models, stratified according to the factors "
        "described above, were used to estimate hazard ratios for time-to-event "
        "end points."
    ),
    "section": "Statistical Analysis",
    "page_numbers": [2],
    "score": 0.30,
    "document_id": "primary:CHAARTED",
    "document_name": "CHAARTED primary report",
    "document_role": "primary",
    "source_kind": "rag_chunk",
    "source_path": "inputs/benchmark/CHAARTED.pdf",
}

# Distractor chunks that currently out-rank the methods chunk because each matches
# at least one of 5.3's prespecification / selective-reporting terms.
MONITORING_CHUNK = {
    "text": (
        "Interim analyses were to be performed before all semiannual meetings of "
        "the data and safety monitoring committee until full information was "
        "obtained."
    ),
    "section": "Statistical Analysis",
    "page_numbers": [2],
    "score": 0.31,
    "document_id": "primary:CHAARTED",
    "document_name": "CHAARTED primary report",
    "document_role": "primary",
    "source_kind": "rag_chunk",
    "source_path": "inputs/benchmark/CHAARTED.pdf",
}
PRESPEC_CHUNK = {
    "text": (
        "The statistical analysis plan (SAP) pre-specified overall survival as "
        "the primary analysis; all pre-specified outcomes were reported."
    ),
    "section": "Statistical Analysis",
    "page_numbers": [3],
    "score": 0.32,
    "document_id": "primary:CHAARTED",
    "document_name": "CHAARTED primary report",
    "document_role": "primary",
    "source_kind": "rag_chunk",
    "source_path": "inputs/benchmark/CHAARTED.pdf",
}
REGISTERED_OUTCOMES_CHUNK = {
    "text": (
        "Registered outcomes: primary overall survival; secondary time to "
        "clinical progression and PSA response outcomes."
    ),
    "section": "ClinicalTrials.gov",
    "page_numbers": [4],
    "score": 0.33,
    "document_id": "primary:CHAARTED",
    "document_name": "CHAARTED primary report",
    "document_role": "primary",
    "source_kind": "rag_chunk",
    "source_path": "inputs/benchmark/CHAARTED.pdf",
}


def _d5_state(chunks: list[dict]) -> dict:
    supplement_chunks = [
        {
            "source_kind": "supplement_segment",
            "document_id": chunk.get("document_id", "supplement:d5"),
            "document_name": chunk.get("document_name", "supplement.pdf"),
            "document_role": chunk.get("document_role", "protocol"),
            "source_path": chunk.get("source_path", "supplement.pdf"),
            **chunk,
        }
        for chunk in chunks
    ]
    return {
        "outcome": "Overall Survival",
        "evidence": empty_paper_evidence("test"),
        "supplement_indexes": {"supplement:d5": _RecordingSupplementIndex(supplement_chunks)},
    }


def test_sq53_packet_selects_reported_methods_sentence_over_prespecification_text():
    state = _d5_state(
        [MONITORING_CHUNK, PRESPEC_CHUNK, REGISTERED_OUTCOMES_CHUNK, METHODS_CHUNK]
    )

    packet = build_evidence_packets(state)["evidence_packets"]["5.3"]
    selected_text = " ".join(source.get("text", "") for source in packet["sources"])

    assert "Cox proportional" in selected_text, (
        "SQ 5.3 must select the reported statistical-analysis-methods sentence "
        "(Cox proportional-hazards / Kaplan-Meier) into its top-3 sources. With "
        "the current contract terms the methods chunk matches none of 5.3's "
        "prespecification vocabulary and is ranked out of the packet, so 5.3 "
        "answers NI and the trial cannot reach Low."
    )


# A genuine but deliberately THIN prespecification chunk: it matches only a
# few of 5.3's selective-reporting terms (outcomes / pre-specified), so an
# over-broad methods lexicon can evict it from the top-3 while the pruned
# lexicon keeps it. This is the vulnerable case the review flagged.
PRESPEC_PLAN_CHUNK = {
    "text": "Outcomes were pre-specified in the trial protocol.",
    "section": "Statistical Analysis",
    "page_numbers": [3],
    "score": 0.40,
    "document_id": "primary:TRIAL",
    "document_name": "Primary report",
    "document_role": "primary",
    "source_kind": "rag_chunk",
    "source_path": "inputs/benchmark/TRIAL.pdf",
}


def _efficacy_methods_chunk(score: float, page: int) -> dict:
    # Methods-flavored efficacy sentence: matches cox/proportional under the
    # pruned lexicon, but would also match "stratified" and "hazard ratio" under
    # an over-broad lexicon, letting three of these crowd out a prespec chunk.
    return {
        "text": (
            "Cox proportional-hazards models, stratified by risk group, "
            "estimated the hazard ratio for overall survival."
        ),
        "section": "Results",
        "page_numbers": [page],
        "score": score,
        "document_id": "primary:TRIAL",
        "document_name": "Primary report",
        "document_role": "primary",
        "source_kind": "rag_chunk",
        "source_path": "inputs/benchmark/TRIAL.pdf",
    }


def test_sq53_does_not_evict_prespecification_chunk_with_efficacy_chunks():
    # Guards against future lexicon over-broadening: three methods-flavored
    # efficacy chunks plus one thin prespecification chunk. Under the pruned
    # lexicon the prespec chunk already out-ranks the efficacy chunks, and the
    # coverage gate now also reserves a slot for it, so it is doubly protected;
    # this test fails only if someone broadens the methods lexicon to
    # results-collision terms ("hazard ratio"/"stratified") AND removes coverage.
    state = _d5_state(
        [
            _efficacy_methods_chunk(0.30, 8),
            _efficacy_methods_chunk(0.31, 9),
            _efficacy_methods_chunk(0.32, 10),
            PRESPEC_PLAN_CHUNK,
        ]
    )

    packet = build_evidence_packets(state)["evidence_packets"]["5.3"]
    selected_text = " ".join(source.get("text", "") for source in packet["sources"])

    assert "Outcomes were pre-specified" in selected_text, (
        "SQ 5.3 must keep the prespecification chunk in its top-3 sources even "
        "when methods-flavored efficacy chunks are present. An over-broad "
        "methods lexicon (e.g. including 'hazard ratio' / 'stratified') lets "
        "efficacy text crowd out the prespecification evidence 5.3 depends on."
    )


def _strong_methods_chunk(score: float, page: int) -> dict:
    # A clean reported-methods sentence that matches several methods terms
    # (cox / proportional / kaplan / log-rank) under the pruned lexicon, so a
    # few of these out-rank a thin prespecification chunk on matched-term count.
    return {
        "text": (
            "Cox proportional-hazards models, Kaplan-Meier estimates, and "
            "log-rank tests analysed overall survival."
        ),
        "section": "Statistical Analysis",
        "page_numbers": [page],
        "score": score,
        "document_id": "primary:TRIAL",
        "document_name": "Primary report",
        "document_role": "primary",
        "source_kind": "rag_chunk",
        "source_path": "inputs/benchmark/TRIAL.pdf",
    }


def test_sq53_packet_keeps_both_prespecification_and_reported_methods_halves():
    # SQ 5.3 ("was the result selected from multiple eligible analyses?") can
    # only be answered N/PN when the model sees BOTH the pre-specified analysis
    # plan AND the reported analysis methods. Here three strong reported-methods
    # chunks out-rank one thin prespecification chunk on matched-term count, so a
    # plain top-3-by-rank selection would keep only methods and evict the plan,
    # leaving the model unable to confirm "reported as planned" -> NI. The packet
    # must reserve a slot for each half.
    state = _d5_state(
        [
            _strong_methods_chunk(0.30, 8),
            _strong_methods_chunk(0.31, 9),
            _strong_methods_chunk(0.32, 10),
            PRESPEC_PLAN_CHUNK,
        ]
    )

    packet = build_evidence_packets(state)["evidence_packets"]["5.3"]
    selected_text = " ".join(source.get("text", "") for source in packet["sources"])

    assert "Outcomes were pre-specified" in selected_text, (
        "SQ 5.3 must keep the prespecification ('what was planned') source in "
        "its packet even when stronger-ranking reported-methods chunks are "
        "present; otherwise the model cannot confirm the analysis was reported "
        "as planned and falls back to NI."
    )
    assert "Cox proportional" in selected_text, (
        "SQ 5.3 must also keep a reported-methods ('what was done') source so "
        "the model can compare it against the plan."
    )


# Junk with no real prespecification content. "function" contains the substring
# "nct", which must NOT qualify it as a prespecification (plan) source.
JUNK_SUBSTRING_CHUNK = {
    "text": "Renal function was monitored throughout the study at each visit.",
    "section": "Safety",
    "page_numbers": [5],
    "score": 0.40,
    "document_id": "primary:TRIAL",
    "document_name": "Primary report",
    "document_role": "primary",
    "source_kind": "rag_chunk",
    "source_path": "inputs/benchmark/TRIAL.pdf",
}


def test_sq53_does_not_seat_junk_plan_source_from_substring_noise():
    # The coverage gate reserves a packet slot for the plan group, so its terms
    # must be precise. A chunk that matches a plan term only as an incidental
    # substring ("function" contains "nct") must not be seated as the
    # pre-specified-plan witness, or the model is handed irrelevant prose as the
    # plan half.
    state = _d5_state(
        [
            JUNK_SUBSTRING_CHUNK,
            _strong_methods_chunk(0.30, 8),
            _strong_methods_chunk(0.31, 9),
            _strong_methods_chunk(0.32, 10),
        ]
    )

    packet = build_evidence_packets(state)["evidence_packets"]["5.3"]
    selected_text = " ".join(source.get("text", "") for source in packet["sources"])

    assert "Renal function was monitored" not in selected_text, (
        "A chunk that matches a prespecification coverage term only via an "
        "incidental substring must not be seated as the plan witness."
    )
