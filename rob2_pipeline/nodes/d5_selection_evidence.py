from __future__ import annotations

import re

from rob2_pipeline.state import RoB2State
from rob2_pipeline.types import D5SelectionEvidence


PLAN_TERMS = re.compile(
    r"\b(protocol|sap|statistical analysis plan|registr|nct|prespec)", re.I
)
RESULT_SELECTION_TERMS = re.compile(
    r"\b(post hoc|exploratory|not prespecified|data[- ]driven|selected|reported subset|"
    r"favorable|significant only|changed from)\b",
    re.I,
)
MULTIPLE_TERMS = re.compile(
    r"\b(multiple|several|co-primary|coprimary|secondary endpoints|time points|subgroups|"
    r"adjusted and unadjusted|sensitivity analyses)\b",
    re.I,
)
PRESPEC_TERMS = re.compile(
    r"\b(prespecified|pre-specified|protocol|sap|registr|planned)\b", re.I
)
SELECTED_SUBSET_TERMS = re.compile(
    r"\b(subset|selected|reported only|post hoc|exploratory)\b", re.I
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


def _classify_plan(state: RoB2State, text: str) -> str:
    packet = (state.get("evidence_packets") or {}).get("5.1") or {}
    sq_answer = ((state.get("sq_answers") or {}).get("5.1") or {}).get("answer")
    has_plan = bool(PLAN_TERMS.search(text)) or state.get(
        "registration_number"
    ) not in (
        None,
        "",
        "Not reported",
    )
    missing_plan = bool(packet.get("missing_evidence")) or sq_answer == "NI"
    no_plan = sq_answer in {"N", "PN"} and not has_plan
    if has_plan and missing_plan:
        return "partial"
    if has_plan and sq_answer in {"N", "PN"}:
        return "conflicting"
    if has_plan:
        return "available"
    if missing_plan and not no_plan:
        return "partial"
    return "unavailable"


def _classify_options(text: str) -> str:
    if SELECTED_SUBSET_TERMS.search(text):
        return "selected-subset"
    if MULTIPLE_TERMS.search(text):
        if PRESPEC_TERMS.search(text):
            return "multiple prespecified"
        return "multiple unclear"
    return "single"


def _classify_result_selection(state: RoB2State, text: str) -> str:
    sq_answers = state.get("sq_answers") or {}
    if RESULT_SELECTION_TERMS.search(text) or any(
        (sq_answers.get(sq_id) or {}).get("answer") in {"Y", "PY"}
        for sq_id in ("5.2", "5.3")
    ):
        return "supported"
    if any(
        (sq_answers.get(sq_id) or {}).get("answer") == "NI" for sq_id in ("5.2", "5.3")
    ):
        return "possible"
    return "absent"


def _classify_binding(state: RoB2State, text: str) -> str:
    outcome = str(state.get("outcome", "")).lower()
    numerical_result = str(state.get("numerical_result", "")).lower()
    lowered = text.lower()
    if (
        outcome
        and outcome in lowered
        and numerical_result
        and numerical_result in lowered
    ):
        return "exact"
    if outcome and outcome in lowered:
        return "partial"
    if "different outcome" in lowered or "wrong outcome" in lowered:
        return "wrong-outcome"
    return "unclear"


def build_d5_selection_evidence(state: RoB2State) -> D5SelectionEvidence:
    plan_sources = _packet_sources(state, "5.1")
    measurement_sources = _packet_sources(state, "5.2")
    analysis_sources = _packet_sources(state, "5.3")
    all_sources = plan_sources + measurement_sources + analysis_sources
    plan_text = "\n".join(
        [
            _source_text(plan_sources),
            str(state.get("registration_number", "")),
            str(state.get("registered_endpoint", "")),
            str(state.get("registered_analysis", "")),
            str(state.get("ctgov_outcomes", "")),
        ]
    )
    measurement_text = "\n".join(
        [
            _source_text(measurement_sources),
            str(state.get("registered_endpoint", "")),
            str(state.get("ctgov_outcomes", "")),
        ]
    )
    analysis_text = "\n".join(
        [
            _source_text(analysis_sources),
            str(state.get("registered_analysis", "")),
        ]
    )
    all_text = "\n".join([plan_text, measurement_text, analysis_text])
    binding_text = "\n".join([all_text, str(state.get("numerical_result", ""))])

    return D5SelectionEvidence(
        plan_availability=_dimension(
            _classify_plan(state, plan_text),
            "Plan evidence is classified from registration, protocol, SAP, and SQ 5.1 support.",
            _provenance_from_sources(plan_sources) + _sq_provenance(state, "5.1"),
        ),
        outcome_measurement_options=_dimension(
            _classify_options(measurement_text),
            "Outcome-measurement options are classified from endpoint and time-point evidence.",
            _provenance_from_sources(measurement_sources)
            + _sq_provenance(state, "5.2"),
        ),
        analysis_options=_dimension(
            _classify_options(analysis_text),
            "Analysis options are classified from prespecified and reported analysis evidence.",
            _provenance_from_sources(analysis_sources) + _sq_provenance(state, "5.3"),
        ),
        result_based_selection_support=_dimension(
            _classify_result_selection(state, all_text),
            "Result-based selection support is classified from direct selection language and SQ answers.",
            _provenance_from_sources(all_sources),
        ),
        assessed_result_binding=_dimension(
            _classify_binding(state, binding_text),
            "Assessed-result binding compares the assessed outcome and numerical result with D5 evidence.",
            _provenance_from_sources(all_sources),
        ),
    )
