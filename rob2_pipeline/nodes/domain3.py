from rob2_pipeline.judges.domain3 import judge_domain3
from rob2_pipeline.nodes.common import (
    add_domain_judgment,
)
from rob2_pipeline.nodes.domain_context import build_domain3_context
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
    return run_domain_sq_stage(state, DOMAIN3_STAGE)


def domain3_judge_node(state: RoB2State) -> RoB2State:
    judgment, rationale = judge_domain3(state["sq_answers"])
    return add_domain_judgment(state, "D3", judgment, rationale)
