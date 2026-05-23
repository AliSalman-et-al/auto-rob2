from rob2_pipeline.judges.domain2 import judge_domain2
from rob2_pipeline.nodes.common import (
    add_domain_judgment,
    call_node_llm,
    set_na,
)
from rob2_pipeline.nodes.domain_context import (
    build_domain2_analysis_context,
    build_domain2_conditional_context,
    build_domain2_sq12_context,
)
from rob2_pipeline.nodes.domain_helpers import DomainSqStage, run_domain_sq_stage
from rob2_pipeline.prompts import (
    PROMPT_DOMAIN2_ADHERING_ANALYSIS,
    PROMPT_DOMAIN2_ADHERING_CONDITIONAL,
    PROMPT_DOMAIN2_ANALYSIS,
    PROMPT_DOMAIN2_CONDITIONAL,
    PROMPT_DOMAIN2_SQ12,
)
from rob2_pipeline.state import RoB2State


def build_domain2_sq12_prompt(state: RoB2State) -> str:
    context = build_domain2_sq12_context(state)
    return PROMPT_DOMAIN2_SQ12.format(
        intervention=state["intervention"],
        comparator=state["comparator"],
        outcome=state["outcome"],
        blinding_text=context.blinding_text,
        methods_text=context.methods_text,
        rag_text=context.rag_text,
        ctgov_design=context.ctgov_design,
    )


def apply_domain2_sq12_control_flow(
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


DOMAIN2_SQ12_STAGE = DomainSqStage(
    node_name="domain2_sq12",
    sq_ids=("2.1", "2.2"),
    source_domain="d2",
    build_prompt=build_domain2_sq12_prompt,
    postprocess=apply_domain2_sq12_control_flow,
)


def domain2_sq12_node(state: RoB2State) -> RoB2State:
    return run_domain_sq_stage(state, DOMAIN2_SQ12_STAGE, call_fn=call_node_llm)


def d2_needs_conditional(state: RoB2State) -> str:
    if state.get("effect_of_interest", "ITT").lower() == "per-protocol":
        return "conditional"
    s21 = state["sq_answers"].get("2.1", {}).get("answer", "NI")
    s22 = state["sq_answers"].get("2.2", {}).get("answer", "NI")
    if s21 in ("N", "PN") and s22 in ("N", "PN"):
        return "analysis"
    return "conditional"


def build_domain2_conditional_prompt(state: RoB2State) -> str:
    context = build_domain2_conditional_context(state)
    prompt_template = (
        PROMPT_DOMAIN2_ADHERING_CONDITIONAL
        if state.get("effect_of_interest", "ITT").lower() == "per-protocol"
        else PROMPT_DOMAIN2_CONDITIONAL
    )
    return prompt_template.format(
        intervention=state["intervention"],
        comparator=state["comparator"],
        outcome=state["outcome"],
        sq_2_1=context.sq_2_1,
        sq_2_2=context.sq_2_2,
        deviations_text=context.deviations_text,
        concomitant_text=context.concomitant_text,
        rag_text=context.rag_text,
    )


def apply_domain2_conditional_control_flow(
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


DOMAIN2_CONDITIONAL_STAGE = DomainSqStage(
    node_name="domain2_conditional",
    sq_ids=("2.3", "2.4", "2.5"),
    source_domain="d2",
    build_prompt=build_domain2_conditional_prompt,
    postprocess=apply_domain2_conditional_control_flow,
)


def domain2_conditional_node(state: RoB2State) -> RoB2State:
    return run_domain_sq_stage(
        state, DOMAIN2_CONDITIONAL_STAGE, call_fn=call_node_llm
    )


def build_domain2_analysis_prompt(state: RoB2State) -> str:
    context = build_domain2_analysis_context(state)
    prompt_template = (
        PROMPT_DOMAIN2_ADHERING_ANALYSIS
        if state.get("effect_of_interest", "ITT").lower() == "per-protocol"
        else PROMPT_DOMAIN2_ANALYSIS
    )
    return prompt_template.format(
        intervention=state["intervention"],
        comparator=state["comparator"],
        outcome=state["outcome"],
        effect_of_interest=context.effect_of_interest,
        analysis_text=context.analysis_text,
        results_text=context.results_text,
        rag_text=context.rag_text,
    )


def apply_domain2_analysis_control_flow(
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


DOMAIN2_ANALYSIS_STAGE = DomainSqStage(
    node_name="domain2_analysis",
    sq_ids=("2.6", "2.7"),
    source_domain="d2",
    build_prompt=build_domain2_analysis_prompt,
    postprocess=apply_domain2_analysis_control_flow,
)


def domain2_analysis_node(state: RoB2State) -> RoB2State:
    return run_domain_sq_stage(state, DOMAIN2_ANALYSIS_STAGE, call_fn=call_node_llm)


def domain2_judge_node(state: RoB2State) -> RoB2State:
    judgment, rationale = judge_domain2(
        state["sq_answers"], state.get("effect_of_interest", "ITT")
    )
    return add_domain_judgment(state, "D2", judgment, rationale)
