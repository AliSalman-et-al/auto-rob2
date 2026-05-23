from rob2_pipeline.judges.domain5 import judge_domain5
from rob2_pipeline.nodes.common import (
    add_domain_judgment,
)
from rob2_pipeline.nodes.domain_context import build_domain5_context
from rob2_pipeline.nodes.domain_helpers import call_domain_sq_prompt
from rob2_pipeline.prompts import PROMPT_DOMAIN5
from rob2_pipeline.state import RoB2State


def domain5_sq_node(state: RoB2State) -> RoB2State:
    context = build_domain5_context(state)
    errors = list(state.get("errors", []))
    human_review_priority = state.get("human_review_priority", "HIGH")
    if state.get("intervention") == "Not reported":
        errors.append(
            "Intervention not reported; manual review required for Domain 5 assessment."
        )
        human_review_priority = "HIGH"
    prompt = PROMPT_DOMAIN5.format(
        intervention=state["intervention"],
        comparator=state["comparator"],
        outcome=state["outcome"],
        outcome_type=context.outcome_type,
        numerical_result=context.numerical_result,
        registration_number=context.registration_number,
        registered_endpoint=context.registered_endpoint,
        registered_secondary_endpoints=context.registered_secondary_endpoints,
        reported_endpoint=context.reported_endpoint,
        ctgov_outcomes=context.ctgov_outcomes,
        ctgov_description=context.ctgov_description,
        registration_text=context.registration_text,
        sap_text=context.sap_text,
        results_text=context.results_text,
        rag_text=context.rag_text,
    )
    sq_answers, log = call_domain_sq_prompt(
        state,
        prompt,
        node_name="domain5_sq",
        sq_ids=["5.1", "5.2", "5.3"],
        source_domain="d5",
    )
    return {
        "sq_answers": sq_answers,
        "llm_call_log": log,
        "errors": errors,
        "human_review_priority": human_review_priority,
    }


def domain5_judge_node(state: RoB2State) -> RoB2State:
    judgment, rationale = judge_domain5(state["sq_answers"])
    return add_domain_judgment(state, "D5", judgment, rationale)
