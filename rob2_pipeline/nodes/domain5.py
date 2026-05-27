from rob2_pipeline.judges.domain5 import judge_domain5
from rob2_pipeline.nodes.common import (
    add_domain_judgment,
)
from rob2_pipeline.nodes.d5_selection_evidence import build_d5_selection_evidence
from rob2_pipeline.nodes.domain_context import build_domain5_context
from rob2_pipeline.nodes.domain_helpers import DomainSqStage, run_domain_sq_stage
from rob2_pipeline.prompts import PROMPT_DOMAIN5
from rob2_pipeline.state import RoB2State


def _dimension_classification(selection_evidence: dict, dimension: str) -> str:
    return (
        (selection_evidence.get(dimension) or {}).get("classification") or "unclear"
    )


def _downgrade_answer(answer: dict, reason: str) -> dict:
    updated = dict(answer)
    updated["answer"] = "NI"
    existing = updated.get("justification")
    updated["justification"] = f"{existing} {reason}".strip() if existing else reason
    updated["uncertainty_flag"] = "HIGH"
    return updated


def apply_d5_selection_evidence_gate(
    sq_answers: dict[str, dict], selection_evidence: dict
) -> dict[str, dict]:
    gated = {sq_id: dict(answer) for sq_id, answer in sq_answers.items()}
    binding = _dimension_classification(selection_evidence, "assessed_result_binding")
    result_selection = _dimension_classification(
        selection_evidence, "result_based_selection_support"
    )

    s51 = gated.get("5.1", {})
    if s51.get("answer") in {"Y", "PY"}:
        plan = _dimension_classification(selection_evidence, "plan_availability")
        if plan not in {"available", "partial", "conflicting"} or binding == "wrong-outcome":
            gated["5.1"] = _downgrade_answer(
                s51,
                "D5 selection evidence does not show an assessed-outcome-relevant prespecified plan.",
            )

    s52 = gated.get("5.2", {})
    if s52.get("answer") in {"Y", "PY"}:
        measurement_options = _dimension_classification(
            selection_evidence, "outcome_measurement_options"
        )
        if (
            measurement_options not in {"multiple unclear", "selected-subset"}
            or result_selection not in {"supported", "possible"}
            or binding == "wrong-outcome"
        ):
            gated["5.2"] = _downgrade_answer(
                s52,
                "D5 selection evidence does not support result-based selection among multiple eligible outcome measurements.",
            )

    s53 = gated.get("5.3", {})
    if s53.get("answer") in {"Y", "PY"}:
        analysis_options = _dimension_classification(
            selection_evidence, "analysis_options"
        )
        if (
            analysis_options not in {"multiple unclear", "selected-subset"}
            or result_selection not in {"supported", "possible"}
            or binding == "wrong-outcome"
        ):
            gated["5.3"] = _downgrade_answer(
                s53,
                "D5 selection evidence does not support result-based selection among multiple eligible analyses.",
            )

    return gated


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
    selection_evidence = build_d5_selection_evidence(state)
    sq_answers = apply_d5_selection_evidence_gate(
        state.get("sq_answers", {}), selection_evidence
    )
    judgment, rationale = judge_domain5(sq_answers)
    result = add_domain_judgment(state, "D5", judgment, rationale)
    result["sq_answers"] = sq_answers
    result["d5_selection_evidence"] = selection_evidence
    return result
