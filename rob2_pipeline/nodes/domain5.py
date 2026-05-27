from rob2_pipeline.judges.domain5 import judge_domain5
from rob2_pipeline.nodes.common import (
    add_domain_judgment,
)
from rob2_pipeline.nodes.d5_selection_evidence import build_d5_selection_evidence
from rob2_pipeline.nodes.domain_context import build_domain5_context
from rob2_pipeline.nodes.domain_helpers import DomainSqStage, run_domain_sq_stage
from rob2_pipeline.prompts import PROMPT_DOMAIN5
from rob2_pipeline.state import RoB2State


def build_domain5_prompt(state: RoB2State) -> str:
    context = build_domain5_context(state)
    return PROMPT_DOMAIN5.format(
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


DOMAIN5_STAGE = DomainSqStage(
    node_name="domain5_sq",
    sq_ids=("5.1", "5.2", "5.3"),
    source_domain="d5",
    build_prompt=build_domain5_prompt,
)


def domain5_sq_node(state: RoB2State) -> RoB2State:
    errors = list(state.get("errors", []))
    human_review_priority = state.get("human_review_priority", "HIGH")
    if state.get("intervention") == "Not reported":
        errors.append(
            "Intervention not reported; manual review required for Domain 5 assessment."
        )
        human_review_priority = "HIGH"
    result = run_domain_sq_stage(state, DOMAIN5_STAGE)
    result["errors"] = errors
    result["human_review_priority"] = human_review_priority
    return result


def domain5_judge_node(state: RoB2State) -> RoB2State:
    judgment, rationale = judge_domain5(state["sq_answers"])
    result = add_domain_judgment(state, "D5", judgment, rationale)
    result["d5_selection_evidence"] = build_d5_selection_evidence(state)
    return result
