from rob2_pipeline.judges.domain1 import judge_domain1
from rob2_pipeline.nodes.common import (
    add_domain_judgment_with_pivotality_tests,
)
from rob2_pipeline.nodes.domain_context import build_domain1_context
from rob2_pipeline.nodes.domain_helpers import DomainSqStage, run_domain_sq_stage
from rob2_pipeline.prompts import PROMPT_DOMAIN1
from rob2_pipeline.state import RoB2State


def build_domain1_prompt(state: RoB2State) -> str:
    context = build_domain1_context(state)
    return PROMPT_DOMAIN1.format(
        intervention=state["intervention"],
        comparator=state["comparator"],
        outcome=state["outcome"],
        randomization_text=context.randomization_text,
        baseline_text=context.baseline_text,
        consort_text=context.consort_text,
        rag_text=context.rag_text,
        ctgov_design=context.ctgov_design,
    )


DOMAIN1_STAGE = DomainSqStage(
        node_name="domain1_sq",
        sq_ids=("1.1", "1.2", "1.3"),
        source_domain="d1",
        build_prompt=build_domain1_prompt,
)


def domain1_sq_node(state: RoB2State) -> RoB2State:
    return run_domain_sq_stage(state, DOMAIN1_STAGE)


def domain1_judge_node(state: RoB2State) -> RoB2State:
    judgment, rationale = judge_domain1(state["sq_answers"])
    return add_domain_judgment_with_pivotality_tests(
        state, "D1", judgment, rationale, judge_domain1, DOMAIN1_STAGE.sq_ids
    )
