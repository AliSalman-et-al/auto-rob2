from typing import Literal

from rob2_pipeline.nodes.common import set_na
from rob2_pipeline.state import RoB2State


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


def apply_domain4_control(
    state: RoB2State, sq_answers: dict[str, dict]
) -> dict[str, dict]:
    updated = dict(sq_answers)
    outcome_type = state.get("outcome_type", "clinician-composite")
    sq_2_1 = state.get("sq_answers", {}).get("2.1", {}).get("answer", "NI")
    sq_2_2 = state.get("sq_answers", {}).get("2.2", {}).get("answer", "NI")
    trial_is_open_label = sq_2_1 in ("Y", "PY") or sq_2_2 in ("Y", "PY")

    if trial_is_open_label and outcome_type in (
        "patient-reported",
        "clinician-graded",
        "clinician-composite",
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
        if outcome_type == "patient-reported":
            answer = "Y"
            justification = (
                "Participant is the assessor; cannot be blinded to own treatment."
            )
        else:
            answer = "PY"
            justification = "In an open-label trial, the clinician grading or adjudicating the outcome is likely aware of treatment assignment."
        updated["4.3"] = {
            "answer": answer,
            "quote": quote or "No relevant text found",
            "justification": justification,
            "uncertainty_flag": "NORMAL",
            "support_level": "moderate",
            "support_rationale": "Derived from treatment-awareness evidence and outcome type.",
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
                "support_level": updated.get("4.3", {}).get("support_level", "weak"),
                "support_rationale": updated.get("4.3", {}).get(
                    "support_rationale",
                    "No direct assessor-awareness evidence was reported.",
                ),
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
                "support_level": "moderate",
                "support_rationale": "Derived from objective outcome classification.",
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
                "support_level": "unsupported",
                "support_rationale": "Not applicable",
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
