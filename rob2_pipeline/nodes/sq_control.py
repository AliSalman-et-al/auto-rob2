import re
from typing import Literal

from rob2_pipeline.nodes.common import set_na
from rob2_pipeline.state import RoB2State


_TTE_DIRECT_MISSINGNESS_RE = re.compile(
    r"\b("
    r"missing\s+outcome\s+data|"
    r"outcome\s+data\s+(?:were\s+)?(?:available|complete)|"
    r"complete\s+(?:outcome\s+)?(?:data|follow-up|followup)|"
    r"all\s+(?:patients|participants).{0,40}(?:followed|outcome\s+data)|"
    r"(?:lost|loss)\s+to\s+follow[- ]?up|"
    r"no\s+(?:patients|participants).{0,40}(?:lost|missing)|"
    r"ascertain(?:ed|ment).{0,40}(?:complete|status|outcome|survival|event)|"
    r"(?:vital|survival|mortality)\s+status.{0,40}(?:available|ascertained|complete)"
    r")\b",
    re.IGNORECASE,
)


_TTE_ANALYSIS_ONLY_RE = re.compile(
    r"\b("
    r"intention[- ]to[- ]treat|itt|"
    r"kaplan[- ]meier|cox|hazard\s+ratio|"
    r"censor(?:ed|ing)?|last\s+follow[- ]?up|analysis\s+population"
    r")\b",
    re.IGNORECASE,
)


def apply_domain2_sq12_control(
    state: RoB2State, sq_answers: dict[str, dict]
) -> dict[str, dict]:
    s21 = sq_answers.get("2.1", {}).get("answer", "NI")
    s22 = sq_answers.get("2.2", {}).get("answer", "NI")
    if (
        state.get("effect_of_interest", "ITT").lower() != "per-protocol"
        and s21 in ("N", "PN")
        and s22 in ("N", "PN")
    ):
        return set_na(sq_answers, "2.3", "2.4", "2.5")
    return sq_answers


def next_domain2_stage(state: RoB2State) -> Literal["conditional", "analysis"]:
    if state.get("effect_of_interest", "ITT").lower() == "per-protocol":
        return "conditional"
    s21 = state["sq_answers"].get("2.1", {}).get("answer", "NI")
    s22 = state["sq_answers"].get("2.2", {}).get("answer", "NI")
    if s21 in ("N", "PN") and s22 in ("N", "PN"):
        return "analysis"
    return "conditional"


def apply_domain2_conditional_control(
    state: RoB2State, sq_answers: dict[str, dict]
) -> dict[str, dict]:
    if state.get("effect_of_interest", "ITT").lower() == "per-protocol":
        return sq_answers
    s23 = sq_answers.get("2.3", {}).get("answer", "NI")
    s24 = sq_answers.get("2.4", {}).get("answer", "NI")
    if s23 in ("N", "PN", "NI"):
        return set_na(sq_answers, "2.4", "2.5")
    if s24 in ("N", "PN", "NA"):
        return set_na(sq_answers, "2.5")
    return sq_answers


def apply_domain2_analysis_control(
    state: RoB2State, sq_answers: dict[str, dict]
) -> dict[str, dict]:
    if state.get(
        "effect_of_interest", "ITT"
    ).lower() == "per-protocol" or sq_answers.get("2.6", {}).get("answer", "NI") in (
        "Y",
        "PY",
    ):
        return set_na(sq_answers, "2.7")
    return sq_answers


def apply_domain3_control(
    state: RoB2State, sq_answers: dict[str, dict]
) -> dict[str, dict]:
    sq_answers = _calibrate_time_to_event_d3_1(state, sq_answers)
    s31 = sq_answers.get("3.1", {}).get("answer", "NI")
    s32 = sq_answers.get("3.2", {}).get("answer", "NA")
    s33 = sq_answers.get("3.3", {}).get("answer", "NA")
    if s31 in ("Y", "PY"):
        return set_na(sq_answers, "3.2", "3.3", "3.4")
    if s32 in ("Y", "PY"):
        return set_na(sq_answers, "3.3", "3.4")
    if s33 in ("N", "PN"):
        return set_na(sq_answers, "3.4")
    return sq_answers


def _calibrate_time_to_event_d3_1(
    state: RoB2State, sq_answers: dict[str, dict]
) -> dict[str, dict]:
    if not state.get("outcome_properties", {}).get("time_to_event"):
        return sq_answers
    sq31 = sq_answers.get("3.1", {})
    if sq31.get("answer") not in ("Y", "PY"):
        return sq_answers

    evidence_text = _domain3_missingness_text(state)
    quote_text = " ".join(
        str(sq31.get(field, ""))
        for field in ("quote", "completeness_calculation", "justification")
    )
    text = " ".join([evidence_text, quote_text])
    if _TTE_DIRECT_MISSINGNESS_RE.search(text):
        return sq_answers
    if not _TTE_ANALYSIS_ONLY_RE.search(text):
        return sq_answers

    updated = dict(sq_answers)
    updated["3.1"] = {
        **sq31,
        "answer": "NI",
        "justification": (
            f"{sq31.get('justification', '').strip()} "
            "For time-to-event outcomes, ITT, survival-model, Kaplan-Meier, or censoring language alone does not directly establish negligible missing outcome data."
        ).strip(),
        "uncertainty_flag": sq31.get("uncertainty_flag", "HIGH"),
    }
    for sq_id in ("3.2", "3.3", "3.4"):
        updated.setdefault(
            sq_id,
            {
                "answer": "NI",
                "quote": "No relevant text found",
                "justification": "D3 remains applicable after time-to-event missingness calibration.",
                "uncertainty_flag": "HIGH",
            },
        )
    return updated


def _domain3_missingness_text(state: RoB2State) -> str:
    evidence = state.get("evidence", {})
    parts = [
        _section_text(evidence.get("d3_missing_data", {})),
        _section_text(evidence.get("consort_flow", {})),
        _section_text(evidence.get("results", {})),
        state.get("rag_contexts", {}).get("d3", ""),
        state.get("ctgov_flow", ""),
    ]
    return "\n".join(str(part) for part in parts if part)


def _section_text(section: object) -> str:
    if isinstance(section, dict):
        return "\n".join(
            str(section.get(key, "")) for key in ("text", "tables") if section.get(key)
        )
    return str(section or "")


def apply_domain4_control(
    state: RoB2State, sq_answers: dict[str, dict]
) -> dict[str, dict]:
    updated = dict(sq_answers)
    outcome_type = _effective_domain4_outcome_type(state)
    sq_2_1 = state.get("sq_answers", {}).get("2.1", {}).get("answer", "NI")
    sq_2_2 = state.get("sq_answers", {}).get("2.2", {}).get("answer", "NI")
    trial_is_open_label = sq_2_1 in ("Y", "PY") or sq_2_2 in ("Y", "PY")

    has_blinded_adjudication = _has_blinded_adjudication(state)
    pfs_open_label_concern = (
        trial_is_open_label
        and _is_pfs_outcome(state)
        and not has_blinded_adjudication
        and _progression_uses_clinician_or_investigator_assessment(state)
    )

    if trial_is_open_label and outcome_type == "patient-reported":
        existing_quote = updated.get("4.3", {}).get("quote") or ""
        quote = (
            existing_quote
            if existing_quote and not existing_quote.startswith("Auto-set:")
            else (
                state.get("sq_answers", {}).get("2.1", {}).get("quote")
                or state.get("sq_answers", {}).get("2.2", {}).get("quote")
                or "No relevant text found"
            )
        )
        updated["4.3"] = {
            "answer": "Y",
            "quote": quote or "No relevant text found",
            "justification": "Participant is the assessor; cannot be blinded to own treatment.",
            "uncertainty_flag": "NORMAL",
        }
    elif (
        trial_is_open_label
        and not has_blinded_adjudication
        and outcome_type
        in (
            "clinician-graded",
            "clinician-composite",
        )
    ):
        existing_quote = updated.get("4.3", {}).get("quote") or ""
        quote = (
            existing_quote
            if existing_quote and not existing_quote.startswith("Auto-set:")
            else (
                state.get("sq_answers", {}).get("2.1", {}).get("quote")
                or state.get("sq_answers", {}).get("2.2", {}).get("quote")
                or "No relevant text found"
            )
        )
        updated["4.3"] = {
            "answer": "PY",
            "quote": quote or "No relevant text found",
            "justification": "In an open-label trial, the clinician grading or adjudicating the outcome is likely aware of treatment assignment.",
            "uncertainty_flag": "NORMAL",
        }
        if pfs_open_label_concern:
            updated["4.4"] = {
                "answer": "PY",
                "quote": _domain4_quote(state, updated),
                "justification": "Progression-free survival includes clinician or investigator assessment, so intervention knowledge could plausibly influence progression assessment.",
                "uncertainty_flag": "NORMAL",
            }
            updated["4.5"] = {
                "answer": "PN",
                "quote": _domain4_quote(state, updated),
                "justification": "No blinded independent adjudication is shown, but the evidence does not establish that assessment was likely influenced.",
                "uncertainty_flag": "NORMAL",
            }
    elif outcome_type in ("vital-status", "biomarker"):
        s41 = updated.get("4.1", {}).get("answer", "NI")
        s42 = updated.get("4.2", {}).get("answer", "NI")
        s43 = updated.get("4.3", {}).get("answer", "NI")
        s44 = updated.get("4.4", {}).get("answer", "NI")
        if s41 in ("N", "PN", "NI") and s42 in ("N", "PN") and s43 == "NA":
            updated["4.3"] = {
                "answer": "NI",
                "quote": updated.get("4.3", {}).get("quote")
                or "No relevant text found",
                "justification": "Assessor awareness is not reported; NA is not applicable when 4.1 and 4.2 do not indicate measurement problems.",
                "uncertainty_flag": "NORMAL",
            }
            s43 = "NI"
        if (
            s41 in ("N", "PN", "NI")
            and s42 in ("N", "PN")
            and s43 in ("Y", "PY", "NI")
            and s44 in ("NI", "NA")
        ):
            updated["4.4"] = {
                "answer": "N",
                "quote": updated.get("4.1", {}).get("quote")
                or updated.get("4.2", {}).get("quote")
                or "No relevant text found",
                "justification": "The outcome is inherently objective, so knowledge of intervention assignment is unlikely to influence assessment.",
                "uncertainty_flag": "NORMAL",
            }
            s44 = "N"
        if (
            s41 in ("N", "PN", "NI")
            and s42 in ("N", "PN")
            and s43 in ("Y", "PY", "NI")
            and s44 in ("N", "PN")
        ):
            updated["4.5"] = {
                "answer": "NA",
                "quote": "Not applicable",
                "justification": "Not applicable",
                "uncertainty_flag": "NORMAL",
            }

    s41 = updated.get("4.1", {}).get("answer", "NI")
    s42 = updated.get("4.2", {}).get("answer", "NI")
    s43 = updated.get("4.3", {}).get("answer", "NA")
    s44 = updated.get("4.4", {}).get("answer", "NA")
    if s41 in ("Y", "PY") or s42 in ("Y", "PY"):
        return set_na(updated, "4.3", "4.4", "4.5")
    if s43 in ("N", "PN"):
        return set_na(updated, "4.4", "4.5")
    if s44 in ("N", "PN"):
        return set_na(updated, "4.5")
    return updated


def _is_pfs_outcome(state: RoB2State) -> bool:
    if str(state.get("outcome_code", "")).casefold() == "pfs":
        return True
    outcome = str(state.get("outcome", "")).casefold()
    return (
        "progression-free survival" in outcome or "progression free survival" in outcome
    )


def _effective_domain4_outcome_type(state: RoB2State) -> str:
    if _is_objective_os_outcome(state):
        return "vital-status"
    return str(state.get("outcome_type", "clinician-composite"))


def _is_objective_os_outcome(state: RoB2State) -> bool:
    if str(state.get("outcome_code", "")).casefold() == "os":
        return True
    outcome = str(state.get("outcome", "")).casefold()
    if "overall survival" in outcome or "death from any cause" in outcome:
        return True
    props = state.get("outcome_properties") or {}
    return bool(
        props.get("objective_event")
        and not props.get("patient_reported")
        and not props.get("safety_harm")
    )


def _has_blinded_adjudication(state: RoB2State) -> bool:
    props = state.get("outcome_properties") or {}
    if props.get("blinded_adjudication"):
        return True
    text = _domain4_text(state)
    return bool(
        re.search(
            r"\b(blinded|masked|independent|central).{0,80}\b(adjudication|committee|review|assessor)\b",
            text,
            re.I,
        )
    )


def _progression_uses_clinician_or_investigator_assessment(state: RoB2State) -> bool:
    text = _domain4_text(state)
    if not re.search(r"\bprogression\b", text, re.I):
        return False
    return bool(
        re.search(
            r"\b(investigator|clinician|physician|radiographic|imaging|recist|assessment|assessed)\b",
            text,
            re.I,
        )
    )


def _domain4_text(state: RoB2State) -> str:
    parts = [
        str(state.get("outcome", "")),
        str((state.get("rag_contexts") or {}).get("d4_measurement", "")),
        str((state.get("rag_contexts") or {}).get("d4_assessor", "")),
    ]
    evidence = state.get("evidence") or {}
    d4_evidence = evidence.get("d4_outcome_meas") or {}
    if isinstance(d4_evidence, dict):
        parts.append(str(d4_evidence.get("text", "")))
    return "\n".join(part for part in parts if part)


def _domain4_quote(state: RoB2State, sq_answers: dict[str, dict]) -> str:
    return (
        sq_answers.get("4.1", {}).get("quote")
        or sq_answers.get("4.2", {}).get("quote")
        or (state.get("rag_contexts") or {}).get("d4_measurement")
        or "No relevant text found"
    )
