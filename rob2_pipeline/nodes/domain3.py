from rob2_pipeline.judges.domain3 import judge_domain3, judge_domain3_artifact
from rob2_pipeline.nodes.common import (
    add_domain_judgment_with_pivotality_tests,
)
from rob2_pipeline.nodes.domain_context import build_domain3_context
from rob2_pipeline.nodes.domain_classifier import has_ready_packets, run_json_sq_classifier
from rob2_pipeline.nodes.domain_helpers import DomainSqStage, run_domain_sq_stage
from rob2_pipeline.nodes.sq_control import apply_domain3_control
from rob2_pipeline.prompts import PROMPT_DOMAIN3
from rob2_pipeline.state import RoB2State


def build_domain3_prompt(state: RoB2State) -> str:
    context = build_domain3_context(state)
    return PROMPT_DOMAIN3.format(
        intervention=state["intervention"],
        comparator=state["comparator"],
        outcome=state["outcome"],
        n_randomized=context.n_randomized,
        consort_text=context.consort_text,
        missing_data_text=context.missing_data_text,
        sensitivity_text=context.sensitivity_text,
        rag_text=context.rag_text,
        ctgov_flow=context.ctgov_flow,
    )


DOMAIN3_STAGE = DomainSqStage(
    node_name="domain3_sq",
    sq_ids=("3.1", "3.2", "3.3", "3.4"),
    source_domain="d3",
    build_prompt=build_domain3_prompt,
    postprocess=apply_domain3_control,
)


def domain3_sq_node(state: RoB2State) -> RoB2State:
    if has_ready_packets(state, domain="d3", sq_ids=DOMAIN3_STAGE.sq_ids):
        return run_json_sq_classifier(
            state,
            domain="d3",
            stage="sq",
            sq_ids=DOMAIN3_STAGE.sq_ids,
            node_name="domain3_sq_json",
            artifact_key="d3_sq_classifier_artifact",
            branching={"stage": "sq", "domain_focus": "missing outcome data"},
            postprocess=apply_domain3_control,
        )
    return run_domain_sq_stage(state, DOMAIN3_STAGE)


def domain3_judge_node(state: RoB2State) -> RoB2State:
    judgment_artifact = judge_domain3_artifact(state["sq_answers"])
    judgment = judgment_artifact["label"]
    rationale = judgment_artifact["rationale"]
    update = add_domain_judgment_with_pivotality_tests(
        state, "D3", judgment, rationale, judge_domain3, DOMAIN3_STAGE.sq_ids
    )
    final_sq_answers = update.get("sq_answers", state["sq_answers"])
    controlled_sq_answers = apply_domain3_control(state, final_sq_answers)
    if controlled_sq_answers != final_sq_answers:
        update["sq_answers"] = controlled_sq_answers
        final_sq_answers = controlled_sq_answers
        judgment, rationale = judge_domain3(final_sq_answers)
        update["domain_judgments"]["D3"] = judgment
        update["domain_rationales"]["D3"] = rationale
    final_judgment = update["domain_judgments"]["D3"]
    final_rationale = update["domain_rationales"]["D3"]
    if final_judgment != judgment or final_rationale != rationale:
        judgment_artifact = judge_domain3_artifact(final_sq_answers)
    update["d3_judgment_artifact"] = {
        **judgment_artifact,
        "artifact_id": f"d3-judgment:{state.get('outcome', '')}",
        "pivotality_tests": update.get("pivotality_tests", {}).get("D3", []),
        "sq_support_adjudications": update.get("sq_support_adjudications", {}).get(
            "D3", []
        ),
    }
    return update
