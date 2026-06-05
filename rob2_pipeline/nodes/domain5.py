from rob2_pipeline.judges.domain5 import judge_domain5, judge_domain5_artifact
from rob2_pipeline.nodes.common import (
    add_domain_judgment_with_pivotality_tests,
)
from rob2_pipeline.nodes.domain_context import build_domain5_context
from rob2_pipeline.nodes.domain_classifier import has_ready_packets, run_json_sq_classifier
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
    if has_ready_packets(state, domain="d5", sq_ids=DOMAIN5_STAGE.sq_ids):
        result = run_json_sq_classifier(
            state,
            domain="d5",
            stage="sq",
            sq_ids=DOMAIN5_STAGE.sq_ids,
            node_name="domain5_sq_json",
            artifact_key="d5_sq_classifier_artifact",
            branching={
                "stage": "sq",
                "outcome_type": state.get("outcome_type", ""),
                "source_policy": "prespecification evidence before reported-result evidence",
            },
            postprocess=apply_domain5_selective_reporting_guard,
        )
    else:
        result = run_domain_sq_stage(state, DOMAIN5_STAGE)
    result["errors"] = errors
    result["human_review_priority"] = human_review_priority
    return result


def apply_domain5_selective_reporting_guard(
    state: RoB2State, sq_answers: dict[str, dict]
) -> dict[str, dict]:
    updated = dict(sq_answers)
    updated = _apply_domain5_prespecification_guard(state, updated)
    for sq_id in ("5.2", "5.3"):
        answer = dict(updated.get(sq_id, {}))
        if answer.get("answer") == "NI" and _has_prespecified_reported_outcome_evidence(
            state, answer
        ):
            answer.update(
                {
                    "answer": "PN",
                    "justification": (
                        "The available registration/protocol and report evidence "
                        "shows the assessed outcome was prespecified and reported; "
                        "no cited evidence shows result-based selection from "
                        "multiple eligible measurements or analyses."
                    ),
                    "support_level": "moderate",
                    "support_rationale": (
                        "Local D5 guard converted unsupported NI to PN where "
                        "prespecified reporting evidence is present and no "
                        "result-based selection evidence is cited."
                    ),
                    "uncertainty": True,
                    "d5_guard_applied": True,
                }
            )
            updated[sq_id] = answer
            continue
        if answer.get("answer") == "NI" and _ni_reports_no_result_selection(answer):
            answer.update(
                {
                    "answer": "PN",
                    "justification": (
                        "The cited evidence may list multiple outcomes or "
                        "measurements, but the answer identifies no evidence that "
                        "the reported result was selected on the basis of the "
                        "results."
                    ),
                    "support_level": "moderate",
                    "support_rationale": (
                        "Local D5 guard treats absence of result-based selection "
                        "evidence as probably no selective reporting when "
                        "prespecification/reporting evidence is otherwise present."
                    ),
                    "uncertainty": True,
                    "supporting_fact_artifact_ids": [],
                    "d5_guard_applied": True,
                }
            )
            updated[sq_id] = answer
            continue
        if answer.get("answer") not in {"Y", "PY"}:
            continue
        if _looks_like_nonselective_reporting_mislabeled_as_yes(answer):
            answer.update(
                {
                    "answer": "N",
                    "justification": (
                        "The cited evidence indicates the assessed outcome result "
                        "matched prespecified or reported intentions, and does not "
                        "show result-based selection from multiple eligible "
                        "measurements or analyses."
                    ),
                    "support_level": "moderate",
                    "support_rationale": (
                        "Local D5 guard corrected a Y/PY answer whose rationale "
                        "described non-selective reporting rather than selection "
                        "on the basis of the results."
                    ),
                    "uncertainty": True,
                    "supporting_fact_artifact_ids": [],
                    "d5_guard_applied": True,
                }
            )
            updated[sq_id] = answer
            continue
        if not _looks_like_other_outcome_or_generic_analysis_inference(answer):
            continue
        answer.update(
            {
                "answer": "PN",
                "justification": (
                    "The cited evidence identifies other outcomes, subgroup tables, "
                    "or generic analysis possibilities, but does not show that the "
                    f"reported {state.get('outcome', 'assessed outcome')} result was "
                    "selected on the basis of the results from multiple eligible "
                    "measurements or analyses of that same assessed outcome."
                ),
                "support_level": "moderate",
                "support_rationale": (
                    "Local D5 guard rejected an inference from other endpoint families "
                    "or generic analyses to result-based selective reporting for the "
                    "assessed outcome."
                ),
                "uncertainty": True,
                "supporting_fact_artifact_ids": [],
                "d5_guard_applied": True,
            }
        )
        updated[sq_id] = answer
    return updated


def _apply_domain5_prespecification_guard(
    state: RoB2State, sq_answers: dict[str, dict]
) -> dict[str, dict]:
    answer = dict(sq_answers.get("5.1", {}))
    if answer.get("answer") in {"Y", "PY"}:
        return sq_answers
    if not _has_prespecified_reported_outcome_evidence(state, answer):
        return sq_answers
    updated = dict(sq_answers)
    answer.update(
        {
            "answer": "Y",
            "justification": (
                "Registration/protocol evidence and the trial report identify the "
                "assessed outcome as a prespecified endpoint that was reported."
            ),
            "support_level": "moderate",
            "support_rationale": (
                "Local D5 guard used outcome-bound registry/protocol and report "
                "evidence to correct unsupported NI for SQ 5.1."
            ),
            "uncertainty": True,
            "d5_prespecification_guard_applied": True,
        }
    )
    updated["5.1"] = answer
    return updated


def _has_prespecified_reported_outcome_evidence(
    state: RoB2State, answer: dict
) -> bool:
    text = " ".join(
        str(part)
        for part in (
            answer.get("quote", ""),
            answer.get("justification", ""),
            answer.get("support_rationale", ""),
            state.get("registered_endpoint", ""),
            state.get("registered_secondary_endpoints", ""),
            state.get("registered_analysis", ""),
            state.get("ctgov_outcomes", ""),
            state.get("numerical_result", ""),
            state.get("outcome", ""),
        )
        if part
    ).casefold()
    for sq_id in ("5.1", "5.2", "5.3"):
        packet = (state.get("evidence_packets") or {}).get(sq_id, {})
        for source in packet.get("sources", []):
            text += " " + str(source.get("text", "")).casefold()
    outcome = str(state.get("outcome", "")).casefold()
    aliases = {
        outcome,
        outcome.replace("-", " "),
        "radiographic progression-free survival"
        if "progression" in outcome and "survival" in outcome
        else "",
        "rPFS".casefold() if "progression" in outcome and "survival" in outcome else "",
        "overall survival" if "overall survival" in outcome else "",
    }
    aliases = {alias for alias in aliases if alias}
    has_outcome = any(alias in text for alias in aliases)
    has_prespec = any(
        signal in text
        for signal in (
            "registered outcomes",
            "registered outcome",
            "primary:",
            "primary end point",
            "primary endpoint",
            "coprimary",
            "pre-specified",
            "prespecified",
            "protocol",
        )
    )
    has_report = any(
        signal in text
        for signal in (
            "reported",
            "hazard ratio",
            "primary end point was",
            "primary endpoint was",
            "the primary end point",
            "the primary endpoint",
        )
    )
    result_selection_signals = (
        "selected on the basis of the results",
        "based on the results",
        "post hoc",
        "post-hoc",
        "chosen after",
        "unreported time point",
        "unreported measurement",
    )
    return (
        has_outcome
        and has_prespec
        and has_report
        and not _has_positive_result_selection_signal(text, result_selection_signals)
    )


def _looks_like_other_outcome_or_generic_analysis_inference(answer: dict) -> bool:
    text = " ".join(
        str(answer.get(field, ""))
        for field in ("quote", "justification", "support_rationale")
    ).casefold()
    misleading_signals = (
        "secondary outcome",
        "secondary:",
        "other outcome",
        "other pre‑specified analyses",
        "other pre-specified analyses",
        "other prespecified analyses",
        "other endpoint",
        "endpoint families",
        "alternative endpoint",
        "psa progression",
        "skeletal event",
        "time to skeletal",
        "time to start",
        "quality of life",
        "qol",
        "subgroup",
        "protocol amendment",
        "several analyses were possible",
        "multiple eligible outcomes",
        "reports only",
        "reported only",
        "only a subset is reported",
        "only a subset was presented",
        "only a subset",
        "reported subset",
        "limited set",
        "reported results focus",
        "only the coprimary endpoints",
        "coprimary endpoints are reported",
        "multiple analyses were possible",
        "multiple eligible analyses",
    )
    result_selection_signals = (
        "selected on the basis of the results",
        "based on the results",
        "post hoc",
        "post-hoc",
        "chosen after",
        "unreported time point",
        "unreported measurement",
    )
    return any(signal in text for signal in misleading_signals) and not (
        _has_positive_result_selection_signal(text, result_selection_signals)
    )


def _ni_reports_no_result_selection(answer: dict) -> bool:
    text = " ".join(
        str(answer.get(field, ""))
        for field in ("quote", "justification", "support_rationale")
    ).casefold()
    uncertainty_signals = (
        "does not provide information",
        "lacks detail",
        "cannot be determined",
        "no evidence",
        "does not show",
        "no cited evidence",
        "whether a subset was selectively reported",
    )
    selection_context = (
        "based on the results",
        "selectively reported",
        "selective reporting",
        "selection process",
        "subset",
        "eligible outcome",
        "eligible measurement",
        "multiple outcomes",
        "multiple eligible",
    )
    return any(signal in text for signal in uncertainty_signals) and any(
        signal in text for signal in selection_context
    )


def _looks_like_nonselective_reporting_mislabeled_as_yes(answer: dict) -> bool:
    text = " ".join(
        str(answer.get(field, ""))
        for field in ("quote", "justification", "support_rationale")
    ).casefold()
    nonselective_signals = (
        "registered primary outcome matches",
        "registered outcome matches",
        "matches the outcome reported",
        "matches the outcome registered",
        "matching the outcome registered",
        "matches the outcome registered",
        "reports the primary endpoint exactly as registered",
        "same primary outcome",
        "appropriately used",
        "correctly reported",
        "pre‑specified and correctly reported",
        "pre-specified and correctly reported",
        "prespecified and correctly reported",
        "consistent description of the primary endpoint",
        "corresponds to the prespecified",
        "aligns with the prespecified",
        "fully reported",
        "without selective omission",
        "no alternative analyses",
        "no alternative measurements",
        "prespecified and reported",
        "confirming that the eligible outcome measurement was prespecified and reported",
    )
    result_selection_signals = (
        "selected on the basis of the results",
        "based on the results",
        "post hoc",
        "post-hoc",
        "chosen after",
        "only a subset",
        "omitted",
    )
    return any(signal in text for signal in nonselective_signals) and not (
        _has_positive_result_selection_signal(text, result_selection_signals)
    )


def _has_positive_result_selection_signal(
    text: str, result_selection_signals: tuple[str, ...]
) -> bool:
    negated_patterns = (
        "no result-based",
        "no result based",
        "not result-based",
        "not result based",
        "does not show result-based",
        "does not show result based",
        "no evidence of result-based",
        "no evidence of result based",
        "without result-based",
        "without result based",
    )
    if any(pattern in text for pattern in negated_patterns):
        return False
    return any(signal in text for signal in result_selection_signals)


def domain5_judge_node(state: RoB2State) -> RoB2State:
    judgment_artifact = judge_domain5_artifact(state["sq_answers"])
    judgment = judgment_artifact["label"]
    rationale = judgment_artifact["rationale"]
    update = add_domain_judgment_with_pivotality_tests(
        state, "D5", judgment, rationale, judge_domain5, DOMAIN5_STAGE.sq_ids
    )
    final_sq_answers = update.get("sq_answers", state["sq_answers"])
    guarded_sq_answers = apply_domain5_selective_reporting_guard(
        state, final_sq_answers
    )
    if guarded_sq_answers != final_sq_answers:
        update["sq_answers"] = guarded_sq_answers
        final_sq_answers = guarded_sq_answers
        judgment, rationale = judge_domain5(final_sq_answers)
        update["domain_judgments"]["D5"] = judgment
        update["domain_rationales"]["D5"] = rationale
    final_judgment = update["domain_judgments"]["D5"]
    final_rationale = update["domain_rationales"]["D5"]
    if final_judgment != judgment or final_rationale != rationale:
        judgment_artifact = judge_domain5_artifact(final_sq_answers)
    update["d5_judgment_artifact"] = {
        **judgment_artifact,
        "artifact_id": f"d5-judgment:{state.get('outcome', '')}",
        "pivotality_tests": update.get("pivotality_tests", {}).get("D5", []),
        "sq_support_adjudications": update.get("sq_support_adjudications", {}).get(
            "D5", []
        ),
    }
    return update
