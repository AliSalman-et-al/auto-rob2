from rob2_pipeline.judges.domain4 import judge_domain4
from rob2_pipeline.nodes.common import (
    add_domain_judgment,
    call_node_llm,
    set_na,
)
from rob2_pipeline.nodes.domain_context import build_domain4_context
from rob2_pipeline.nodes.domain_helpers import DomainSqStage, run_domain_sq_stage
from rob2_pipeline.prompts import PROMPT_DOMAIN4
from rob2_pipeline.state import RoB2State


def build_domain4_prompt(state: RoB2State) -> str:
    context = build_domain4_context(state)
    return PROMPT_DOMAIN4.format(
        intervention=state["intervention"],
        comparator=state["comparator"],
        outcome=state["outcome"],
        outcome_type=context.outcome_type,
        sq_2_1=context.sq_2_1,
        outcome_measurement_text=context.outcome_measurement_text,
        blinding_text=context.blinding_text,
        rag_text=context.rag_text,
    )


def apply_domain4_control_flow(
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


DOMAIN4_STAGE = DomainSqStage(
    node_name="domain4_sq",
    sq_ids=("4.1", "4.2", "4.3", "4.4", "4.5"),
    source_domain="d4",
    build_prompt=build_domain4_prompt,
    postprocess=apply_domain4_control_flow,
)


def domain4_sq_node(state: RoB2State) -> RoB2State:
    return run_domain_sq_stage(state, DOMAIN4_STAGE, call_fn=call_node_llm)


def domain4_judge_node(state: RoB2State) -> RoB2State:
    judgment, rationale = judge_domain4(state["sq_answers"])
    return add_domain_judgment(state, "D4", judgment, rationale)
