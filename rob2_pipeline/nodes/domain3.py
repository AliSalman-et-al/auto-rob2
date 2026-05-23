from rob2_pipeline.judges.domain3 import judge_domain3
from rob2_pipeline.nodes.common import (
    add_domain_judgment,
    set_na,
)
from rob2_pipeline.nodes.domain_context import build_domain3_context
from rob2_pipeline.nodes.domain_helpers import call_domain_sq_prompt
from rob2_pipeline.prompts import PROMPT_DOMAIN3
from rob2_pipeline.state import RoB2State


def domain3_sq_node(state: RoB2State) -> RoB2State:
    context = build_domain3_context(state)
    prompt = PROMPT_DOMAIN3.format(
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
    sq_answers, log = call_domain_sq_prompt(
        state,
        prompt,
        node_name="domain3_sq",
        sq_ids=["3.1", "3.2", "3.3", "3.4"],
        source_domain="d3",
    )
    s31 = sq_answers.get("3.1", {}).get("answer", "NI")
    s32 = sq_answers.get("3.2", {}).get("answer", "NA")
    s33 = sq_answers.get("3.3", {}).get("answer", "NA")
    if s31 in ("Y", "PY"):
        sq_answers = set_na(sq_answers, "3.2", "3.3", "3.4")
    elif s32 in ("Y", "PY"):
        sq_answers = set_na(sq_answers, "3.3", "3.4")
    elif s33 in ("N", "PN"):
        sq_answers = set_na(sq_answers, "3.4")
    return {"sq_answers": sq_answers, "llm_call_log": log}


def domain3_judge_node(state: RoB2State) -> RoB2State:
    judgment, rationale = judge_domain3(state["sq_answers"])
    return add_domain_judgment(state, "D3", judgment, rationale)
