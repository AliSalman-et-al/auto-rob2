from rob2_pipeline.judges.overall import judge_overall_artifact
from rob2_pipeline.state import RoB2State


WEAK_SUPPORT_LEVELS = {"weak", "unsupported"}
REQUIRED_DOMAINS = ("D1", "D2", "D3", "D4", "D5")
BLOCKING_PACKET_STATUSES = {"needs_retrieval_repair"}
NOT_AUTO_ACCEPTABLE_PACKET_STATUSES = {
    "needs_contradiction_resolution",
    "needs_quote_adjudication",
}
TRACEABILITY_CONSTRAINTS = {"quote_untraceable", "missing_required_evidence"}
CONTRADICTION_CONSTRAINTS = {"semantic_support_conflict"}


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
        "automation_confidence": automation_confidence_artifact(state),
        "overall_policy": overall_policy,
        "overall_judgment_artifact": {
            **judgment_artifact,
            "artifact_id": f"overall-judgment:{state.get('outcome', '')}",
            "policy": overall_policy,
            "label": judgment,
            "rationale": rationale,
        },
    }


def automation_confidence_artifact(state: RoB2State) -> dict:
    blocking_reasons = _blocking_reasons(state)
    non_acceptance_reasons = _non_acceptance_reasons(state)
    if blocking_reasons:
        status = "blocked"
    elif non_acceptance_reasons:
        status = "not_auto_acceptable"
    else:
        status = "auto_accept_candidate"
    return {
        "artifact_id": f"automation-confidence:{state.get('outcome', '')}",
        "schema_version": "automation-confidence-v1",
        "status": status,
        "blocking_reasons": blocking_reasons,
        "non_acceptance_reasons": non_acceptance_reasons,
        "completion": {
            "required_domains": list(REQUIRED_DOMAINS),
            "completed_domains": [
                domain
                for domain in REQUIRED_DOMAINS
                if state.get("domain_judgments", {}).get(domain)
            ],
        },
    }


def _blocking_reasons(state: RoB2State) -> list[dict]:
    reasons = []
    domain_judgments = state.get("domain_judgments", {})
    for domain in REQUIRED_DOMAINS:
        if not domain_judgments.get(domain):
            reasons.append(
                {
                    "kind": "incomplete_required_input",
                    "domain": domain,
                    "reason": "Required domain judgment is missing.",
                }
            )
    for sq_id, answer in state.get("sq_answers", {}).items():
        if not answer.get("classification_blocked"):
            continue
        packet_status = answer.get("packet_status", "unknown")
        if packet_status in NOT_AUTO_ACCEPTABLE_PACKET_STATUSES:
            continue
        reasons.append(
            {
                "kind": "failed_required_artifact",
                "sq_id": sq_id,
                "status": packet_status,
                "reason": answer.get("support_rationale")
                or answer.get("justification")
                or "SQ classification was blocked.",
            }
        )
    for sq_id, readiness in _packet_readiness_items(state):
        status = readiness.get("status", "")
        if status in BLOCKING_PACKET_STATUSES:
            reasons.append(
                {
                    "kind": "failed_required_artifact",
                    "sq_id": sq_id,
                    "status": status,
                    "reason": readiness.get("blocking_reason", ""),
                }
            )
    for error in state.get("errors", []) or []:
        reasons.append(
            {
                "kind": "failed_required_artifact",
                "reason": str(error),
            }
        )
    return reasons


def _non_acceptance_reasons(state: RoB2State) -> list[dict]:
    reasons = []
    reasons.extend(_pivotal_support_reasons(state))
    for sq_id, readiness in _packet_readiness_items(state):
        status = readiness.get("status", "")
        if status in NOT_AUTO_ACCEPTABLE_PACKET_STATUSES:
            reasons.append(
                {
                    "kind": "packet_not_auto_acceptable",
                    "sq_id": sq_id,
                    "status": status,
                    "reason": readiness.get("blocking_reason", ""),
                }
            )
    return reasons


def _pivotal_support_reasons(state: RoB2State) -> list[dict]:
    reasons = []
    for domain, tests in state.get("pivotality_tests", {}).items():
        for test in tests:
            if not test.get("pivotal"):
                continue
            support_level = str(test.get("support_level", "")).lower()
            constraints = test.get("constraints", [])
            constraint_types = {
                constraint.get("constraint_type") for constraint in constraints
            }
            if support_level in WEAK_SUPPORT_LEVELS:
                reasons.append(
                    {
                        "kind": "pivotal_support_below_moderate",
                        "domain": domain,
                        "sq_id": test.get("sq_id"),
                        "support_level": support_level,
                        "acceptance_status": test.get("acceptance_status"),
                    }
                )
            if constraint_types & TRACEABILITY_CONSTRAINTS:
                reasons.append(
                    {
                        "kind": "pivotal_quote_not_traceable",
                        "domain": domain,
                        "sq_id": test.get("sq_id"),
                        "constraints": sorted(
                            constraint_types & TRACEABILITY_CONSTRAINTS
                        ),
                    }
                )
            if constraint_types & CONTRADICTION_CONSTRAINTS:
                reasons.append(
                    {
                        "kind": "pivotal_contradiction_unresolved",
                        "domain": domain,
                        "sq_id": test.get("sq_id"),
                        "constraints": sorted(
                            constraint_types & CONTRADICTION_CONSTRAINTS
                        ),
                    }
                )
    return reasons


def _packet_readiness_items(state: RoB2State) -> list[tuple[str, dict]]:
    readiness_by_sq = state.get("packet_readiness") or {}
    if not readiness_by_sq:
        readiness_by_sq = {
            sq_id: packet.get("packet_readiness", {})
            for sq_id, packet in (state.get("evidence_packets") or {}).items()
        }
    return [
        (sq_id, readiness)
        for sq_id, readiness in readiness_by_sq.items()
        if isinstance(readiness, dict)
    ]


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
