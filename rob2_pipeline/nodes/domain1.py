from rob2_pipeline.judges.domain1 import judge_domain1
from rob2_pipeline.nodes.common import (
    add_domain_judgment,
)
from rob2_pipeline.nodes.domain_context import build_domain1_context
from rob2_pipeline.nodes.domain_helpers import call_domain_sq_prompt
from rob2_pipeline.prompts import PROMPT_DOMAIN1
from rob2_pipeline.state import RoB2State


def domain1_sq_node(state: RoB2State) -> RoB2State:
    context = build_domain1_context(state)
    prompt = PROMPT_DOMAIN1.format(
        intervention=state["intervention"],
        comparator=state["comparator"],
        outcome=state["outcome"],
        randomization_text=context.randomization_text,
        baseline_text=context.baseline_text,
        consort_text=context.consort_text,
        rag_text=context.rag_text,
        ctgov_design=context.ctgov_design,
    )
    sq_answers, log = call_domain_sq_prompt(
        state,
        prompt,
        node_name="domain1_sq",
        sq_ids=["1.1", "1.2", "1.3"],
        source_domain="d1",
    )
    return {"sq_answers": sq_answers, "llm_call_log": log}


def domain1_judge_node(state: RoB2State) -> RoB2State:
    judgment, rationale = judge_domain1(state["sq_answers"])
    return add_domain_judgment(state, "D1", judgment, rationale)
