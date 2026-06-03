from rob2_pipeline.judges.overall import judge_overall_artifact
from rob2_pipeline.state import RoB2State


WEAK_SUPPORT_LEVELS = {"weak", "unsupported"}


def overall_judge_node(state: RoB2State) -> RoB2State:
    domain_judgments = state.get("domain_judgments", {})
    overall_policy = state.get("overall_policy", "official_rob2")
    judgment_artifact = judge_overall_artifact(domain_judgments, overall_policy)
    judgment = judgment_artifact["label"]
    rationale = judgment_artifact["rationale"]
    some_concerns_count = sum(
        1 for value in domain_judgments.values() if value == "Some concerns"
    )
    sq_answers = state.get("sq_answers", {})
    ni_count = sum(1 for answer in sq_answers.values() if answer.get("answer") == "NI")
    high_uncertainty_sqs = [
        sq_id
        for sq_id, answer in sq_answers.items()
        if answer.get("uncertainty_flag") == "HIGH"
    ]
    support_audit_escalation = _support_audit_escalates_priority(state)
    if (
        judgment == "High"
        or some_concerns_count >= 3
        or high_uncertainty_sqs
        or ni_count >= 5
        or support_audit_escalation
    ):
        priority = "HIGH"
    elif judgment == "Some concerns" or ni_count >= 2:
        priority = "MEDIUM"
    else:
        priority = "LOW"
    return {
        "overall_judgment": judgment,
        "overall_rationale": rationale,
        "ni_count": ni_count,
        "high_uncertainty_sqs": high_uncertainty_sqs,
        "human_review_priority": priority,
        "overall_policy": overall_policy,
        "overall_judgment_artifact": {
            **judgment_artifact,
            "artifact_id": f"overall-judgment:{state.get('outcome', '')}",
            "policy": overall_policy,
            "label": judgment,
            "rationale": rationale,
        },
    }


def _support_audit_escalates_priority(state: RoB2State) -> bool:
    return (
        _has_unresolved_pivotal_weak_support(state)
        or _has_adjudication_conflict(state)
        or _has_repeated_weak_support_pattern(state)
    )


def _has_unresolved_pivotal_weak_support(state: RoB2State) -> bool:
    for attempts in state.get("sq_support_adjudications", {}).values():
        for attempt in attempts:
            adjudicated = attempt.get("adjudicated_answer", {})
            if _support_level(adjudicated) not in WEAK_SUPPORT_LEVELS:
                continue
            impact = attempt.get("domain_impact", {})
            if impact.get("test_domain_judgment") != impact.get(
                "original_domain_judgment"
            ):
                return True
    return False


def _has_adjudication_conflict(state: RoB2State) -> bool:
    for attempts in state.get("sq_support_adjudications", {}).values():
        for attempt in attempts:
            initial = attempt.get("initial_answer", {})
            adjudicated = attempt.get("adjudicated_answer", {})
            if initial.get("answer") != adjudicated.get("answer"):
                return True
            if _support_level(initial) != _support_level(adjudicated):
                return True
    return False


def _has_repeated_weak_support_pattern(state: RoB2State) -> bool:
    pivotal_weak_sqs = {
        test.get("sq_id")
        for tests in state.get("pivotality_tests", {}).values()
        for test in tests
        if test.get("pivotal") and test.get("support_level") in WEAK_SUPPORT_LEVELS
    }
    weak_sq_count = sum(
        1
        for sq_id, answer in state.get("sq_answers", {}).items()
        if answer.get("answer") != "NA"
        and _support_level(answer) in WEAK_SUPPORT_LEVELS
        and sq_id not in pivotal_weak_sqs
    )
    return weak_sq_count >= 3


def _support_level(answer: dict) -> str:
    return str(answer.get("support_level", "")).lower()
