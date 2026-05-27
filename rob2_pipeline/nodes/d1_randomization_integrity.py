from __future__ import annotations

import re

from rob2_pipeline.state import RoB2State
from rob2_pipeline.types import D1RandomizationIntegrityEvidence


BEFORE_RANDOMIZATION_TERMS = re.compile(
    r"\b(randomi[sz]|allocat).{0,80}\b(before consent|before enrol|before eligibility)|"
    r"\b(enrol(?:led|ment)?|consent(?:ed)?).{0,80}\b(after randomi[sz]|after allocat)",
    re.I,
)
AFTER_RANDOMIZATION_TERMS = re.compile(
    r"\b(before|prior to|beforehand|preceded).{0,80}\b(randomi[sz]|allocat)|"
    r"\b(randomi[sz]|allocat).{0,80}\b(after eligibility|after consent|after enrol)",
    re.I,
)
NO_IMBALANCE_TERMS = re.compile(
    r"\b(well balanced|balanced between|similar between|comparable between|"
    r"no (?:important|substantial|clinically relevant|meaningful)? ?(?:baseline )?"
    r"(?:imbalance|difference))\b",
    re.I,
)
IMBALANCE_TERMS = re.compile(
    r"\b(baseline (?:imbalance|difference|differences)|imbalanced|differed at baseline|"
    r"important imbalance|substantial imbalance|clinically relevant imbalance)\b",
    re.I,
)
PROGNOSTIC_TERMS = re.compile(
    r"\b(prognostic|severity|stage|performance status|age|risk factor|"
    r"disease burden|baseline score|important predictor)\b",
    re.I,
)
FAILURE_TERMS = re.compile(
    r"\b(randomi[sz]ation failure|selection bias|tamper|predictable|"
    r"allocation was known|subversion|systematic imbalance|chance alone unlikely)\b",
    re.I,
)


def _packet_sources(state: RoB2State, sq_id: str) -> list[dict]:
    packet = (state.get("evidence_packets") or {}).get(sq_id) or {}
    return list(packet.get("sources") or [])


def _source_text(sources: list[dict]) -> str:
    return "\n".join(str(source.get("text", "")) for source in sources)


def _provenance_from_sources(sources: list[dict]) -> list[dict]:
    provenance = []
    for source in sources:
        provenance.append(
            {
                "quote": source.get("text", ""),
                "section": source.get("section", ""),
                "page_numbers": source.get("page_numbers", []),
                "document_id": source.get("document_id", ""),
                "document_name": source.get("document_name", ""),
                "document_role": source.get("document_role", ""),
                "source_kind": source.get("source_kind", ""),
                "source_path": source.get("source_path", ""),
            }
        )
    return provenance


def _sq_provenance(state: RoB2State, sq_id: str) -> list[dict]:
    answer = (state.get("sq_answers") or {}).get(sq_id) or {}
    quote = answer.get("quote")
    if not quote:
        return []
    return [{"quote": quote, "source_kind": "sq_answer", "sq_id": sq_id}]


def _dimension(classification: str, rationale: str, provenance: list[dict]) -> dict:
    return {
        "classification": classification,
        "rationale": rationale,
        "provenance": provenance,
    }


def _answer(state: RoB2State, sq_id: str) -> str:
    return str(((state.get("sq_answers") or {}).get(sq_id) or {}).get("answer", "NI"))


def _classify_adequacy(answer: str) -> str:
    if answer in {"Y", "PY"}:
        return "adequate"
    if answer in {"N", "PN"}:
        return "inadequate"
    return "unclear"


def _classify_enrolment_timing(text: str) -> str:
    if AFTER_RANDOMIZATION_TERMS.search(text):
        return "after_randomization"
    if BEFORE_RANDOMIZATION_TERMS.search(text):
        return "before_randomization"
    return "unclear"


def _classify_baseline_imbalance(state: RoB2State, text: str) -> str:
    sq13 = _answer(state, "1.3")
    if FAILURE_TERMS.search(text) or sq13 in {"Y", "PY"}:
        return "suggests_randomization_failure"
    if IMBALANCE_TERMS.search(text):
        if PROGNOSTIC_TERMS.search(text):
            return "prognostic_concerning"
        return "present_not_concerning"
    if NO_IMBALANCE_TERMS.search(text) or sq13 in {"N", "PN"}:
        return "none"
    return "present_not_concerning" if sq13 == "NI" and text.strip() else "none"


def _classify_prognostic_relevance(imbalance: str, text: str) -> str:
    if imbalance == "none":
        return "not_applicable"
    if PROGNOSTIC_TERMS.search(text):
        return "prognostic"
    if imbalance == "suggests_randomization_failure":
        return "prognostic"
    return "not_prognostic_or_unclear"


def _classify_failure_signal(imbalance: str, text: str) -> str:
    if imbalance == "suggests_randomization_failure" or FAILURE_TERMS.search(text):
        return "supported"
    return "not_supported"


def build_d1_randomization_integrity_evidence(
    state: RoB2State,
) -> D1RandomizationIntegrityEvidence:
    sequence_sources = _packet_sources(state, "1.1")
    concealment_sources = _packet_sources(state, "1.2")
    baseline_sources = _packet_sources(state, "1.3")
    sequence_text = _source_text(sequence_sources)
    concealment_text = _source_text(concealment_sources)
    baseline_text = _source_text(baseline_sources)
    timing_text = "\n".join([sequence_text, concealment_text])
    all_text = "\n".join([sequence_text, concealment_text, baseline_text])
    imbalance = _classify_baseline_imbalance(state, baseline_text or all_text)
    prognostic_relevance = _classify_prognostic_relevance(imbalance, baseline_text)

    return D1RandomizationIntegrityEvidence(
        sequence_generation=_dimension(
            _classify_adequacy(_answer(state, "1.1")),
            "Sequence generation is classified from SQ 1.1 support.",
            _provenance_from_sources(sequence_sources) + _sq_provenance(state, "1.1"),
        ),
        allocation_concealment=_dimension(
            _classify_adequacy(_answer(state, "1.2")),
            "Allocation concealment is classified from SQ 1.2 support.",
            _provenance_from_sources(concealment_sources)
            + _sq_provenance(state, "1.2"),
        ),
        enrolment_timing=_dimension(
            _classify_enrolment_timing(timing_text),
            "Enrolment timing is classified from randomization and allocation evidence.",
            _provenance_from_sources(sequence_sources + concealment_sources),
        ),
        baseline_imbalance_severity=_dimension(
            imbalance,
            "Baseline imbalance severity is classified from SQ 1.3 and baseline evidence.",
            _provenance_from_sources(baseline_sources) + _sq_provenance(state, "1.3"),
        ),
        prognostic_relevance=_dimension(
            prognostic_relevance,
            "Prognostic relevance is classified from baseline factor descriptions.",
            _provenance_from_sources(baseline_sources),
        ),
        randomization_failure_signal=_dimension(
            _classify_failure_signal(imbalance, baseline_text),
            "Randomization-failure signal is classified from explicit failure language and severe prognostic imbalance.",
            _provenance_from_sources(baseline_sources) + _sq_provenance(state, "1.3"),
        ),
    )
