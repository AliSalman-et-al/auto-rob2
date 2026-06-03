from rob2_pipeline.judges.domain4 import judge_domain4
from rob2_pipeline.nodes.common import (
    add_domain_judgment_with_pivotality_tests,
    call_node_llm,
)
from rob2_pipeline.nodes.domain_context import build_domain4_context
from rob2_pipeline.nodes.domain_classifier import has_ready_packets, run_json_sq_classifier
from rob2_pipeline.nodes.domain_helpers import DomainSqStage, run_domain_sq_stage
from rob2_pipeline.nodes.sq_control import apply_domain4_control
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


DOMAIN4_STAGE = DomainSqStage(
    node_name="domain4_sq",
    sq_ids=("4.1", "4.2", "4.3", "4.4", "4.5"),
    source_domain="d4",
    build_prompt=build_domain4_prompt,
    postprocess=apply_domain4_control,
)


def domain4_sq_node(state: RoB2State) -> RoB2State:
    if has_ready_packets(state, domain="d4", sq_ids=DOMAIN4_STAGE.sq_ids):
        outcome_type = state.get("outcome_type", "")
        return run_json_sq_classifier(
            state,
            domain="d4",
            stage="sq",
            sq_ids=DOMAIN4_STAGE.sq_ids,
            node_name="domain4_sq_json",
            artifact_key="d4_sq_classifier_artifact",
            branching={"stage": "sq", "outcome_type": outcome_type},
            outcome_specific_concerns=[
                {
                    "concern": f"{outcome_type or 'unknown'} outcome measurement bias",
                    "support_level": "unsupported",
                    "rationale": (
                        "Classifier must update this with packet-supported "
                        "outcome-specific measurement concerns."
                    ),
                }
            ],
            postprocess=apply_domain4_control,
        )
    return run_domain_sq_stage(state, DOMAIN4_STAGE, call_fn=call_node_llm)


def domain4_judge_node(state: RoB2State) -> RoB2State:
    judgment, rationale = judge_domain4(state["sq_answers"])
    state = _with_outcome_classification_constraints_for_d4(state)
    return add_domain_judgment_with_pivotality_tests(
        state, "D4", judgment, rationale, judge_domain4, DOMAIN4_STAGE.sq_ids
    )


def _with_outcome_classification_constraints_for_d4(state: RoB2State) -> RoB2State:
    constraints = list(state.get("support_constraints", []))
    mapped = []
    for constraint in constraints:
        if constraint.get("sq_id") != "outcome_classification":
            continue
        mapped.append(
            {
                **constraint,
                "sq_id": "4.4",
                "reason": (
                    constraint.get("reason", "")
                    or "Outcome classification support constrains derived SQ 4.4."
                ),
            }
        )
    if not mapped:
        return state
    return {**state, "support_constraints": constraints + mapped}
