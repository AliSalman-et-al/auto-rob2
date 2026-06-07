from rob2_pipeline.judges.domain1 import judge_domain1, judge_domain1_artifact
from rob2_pipeline.nodes.common import (
    add_domain_judgment_with_pivotality_tests,
    merge_sq_answers,
)
from rob2_pipeline.nodes.domain_context import build_domain1_context
from rob2_pipeline.nodes.domain_classifier import has_ready_packets, run_json_sq_classifier
from rob2_pipeline.nodes.domain_helpers import DomainSqStage, run_domain_sq_stage
from rob2_pipeline.prompts import PROMPT_DOMAIN1
from rob2_pipeline.state import RoB2State


D1_SQ_IDS = ("1.1", "1.2", "1.3")


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
    sq_ids=D1_SQ_IDS,
    source_domain="d1",
    build_prompt=build_domain1_prompt,
)


def domain1_sq_node(state: RoB2State) -> RoB2State:
    if has_ready_packets(state, domain="d1", sq_ids=DOMAIN1_STAGE.sq_ids):
        return run_json_sq_classifier(
            state,
            domain="d1",
            stage="sq",
            sq_ids=DOMAIN1_STAGE.sq_ids,
            node_name="domain1_sq_json",
            artifact_key="d1_sq_classifier_artifact",
        )
    result = run_domain_sq_stage(state, DOMAIN1_STAGE)
    if "sq_answers" in result:
        result["sq_answers"] = merge_sq_answers(
            state, _apply_domain1_controls(state, result["sq_answers"])
        )
    return result


def _apply_domain1_controls(
    state: RoB2State, sq_answers: dict[str, dict]
) -> dict[str, dict]:
    sq_answers = _apply_domain1_randomized_design_guard(state, sq_answers)
    sq_answers = _apply_domain1_concealment_guard(state, sq_answers)
    return _apply_domain1_baseline_balance_guard(sq_answers)


def _apply_domain1_randomized_design_guard(
    state: RoB2State, sq_answers: dict[str, dict]
) -> dict[str, dict]:
    answer = dict(sq_answers.get("1.1", {}))
    if answer.get("answer") in {"Y", "PY"}:
        return sq_answers
    text = _domain1_support_text(state, answer)
    if not (
        "allocation type: randomized" in text
        or "randomly assigned" in text
        or "randomised" in text
        or "randomized" in text
    ):
        return sq_answers
    updated = dict(sq_answers)
    answer.update(
        {
            "answer": "Y",
            "quote": answer.get("quote")
            if answer.get("quote") and answer.get("quote") != "No relevant text found"
            else "Allocation type: RANDOMIZED",
            "justification": (
                "Authoritative trial design evidence describes randomized "
                "allocation, satisfying SQ 1.1."
            ),
            "support_level": "moderate",
            "support_rationale": (
                "Local D1 control used explicit randomized design evidence."
            ),
            "uncertainty": True,
            "d1_randomized_design_guard_applied": True,
        }
    )
    updated["1.1"] = answer
    return updated


def _apply_domain1_concealment_guard(
    state: RoB2State, sq_answers: dict[str, dict]
) -> dict[str, dict]:
    answer = dict(sq_answers.get("1.2", {}))
    if answer.get("answer") in {"Y", "PY", "N", "PN"}:
        return sq_answers
    text = _domain1_support_text(state, answer)
    has_randomization = any(
        signal in text
        for signal in (
            "allocation type: randomized",
            "randomly assigned",
            "randomised",
            "randomized",
        )
    )
    has_masking = any(
        signal in text
        for signal in (
            "double-blind",
            "double blind",
            "quadruple",
            "masked parties",
            "placebo",
        )
    )
    has_central_assignment = any(
        signal in text
        for signal in (
            "centrally randomly assigned",
            "central random",
            "alea clinical portal",
            "interactive response",
            "interactive web",
            "central web",
            "web-based random",
            "centralized random",
            "centralised random",
        )
    )
    direct_text = " ".join(
        str(part)
        for part in (
            answer.get("quote", ""),
            answer.get("justification", ""),
            answer.get("support_rationale", ""),
            state.get("ctgov_design", ""),
        )
        if part
    ).casefold()
    no_masking = (
        "masking: none" in direct_text
        or "open-label" in direct_text
        or "open label" in direct_text
    )
    if not has_randomization or (not has_masking and not has_central_assignment):
        return sq_answers
    if no_masking and not has_central_assignment:
        return sq_answers
    updated = dict(sq_answers)
    answer.update(
        {
            "answer": "PY",
            "justification": (
                "The trial is described as randomized with masking/placebo "
                "controls or central assignment. In the absence of evidence "
                "that assignments were foreseeable, this supports probable "
                "allocation concealment."
            ),
            "support_level": "moderate",
            "support_rationale": (
                "Local D1 control accepted blinded randomized placebo design or "
                "central assignment as probable concealment evidence, not as "
                "definite procedural proof."
            ),
            "uncertainty": True,
            "d1_concealment_guard_applied": True,
        }
    )
    updated["1.2"] = answer
    return updated


def _domain1_support_text(state: RoB2State, answer: dict) -> str:
    parts = [
        answer.get("quote", ""),
        answer.get("justification", ""),
        answer.get("support_rationale", ""),
        state.get("ctgov_design", ""),
    ]
    for sq_id in D1_SQ_IDS:
        packet = (state.get("evidence_packets") or {}).get(sq_id, {})
        for source in packet.get("sources", []):
            parts.append(str(source.get("text", "")))
    text = " ".join(str(part) for part in parts if part).casefold()
    return (
        text.replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )


def _apply_domain1_baseline_balance_guard(sq_answers: dict[str, dict]) -> dict[str, dict]:
    answer = dict(sq_answers.get("1.3", {}))
    if answer.get("answer") not in {"Y", "PY"}:
        return sq_answers
    text = " ".join(
        str(answer.get(field, ""))
        for field in ("quote", "justification", "support_rationale")
    ).casefold()
    text = (
        text.replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )
    post_randomization_signals = (
        "subsequently",
        "eventually received",
        "life-prolonging therapy",
        "next-generation hormonal therapy",
        "after progression",
        "post-progression",
        "salvage treatment",
    )
    if any(signal in text for signal in post_randomization_signals):
        updated = dict(sq_answers)
        answer.update(
            {
                "answer": "NI",
                "justification": (
                    "The cited text describes post-randomization or subsequent "
                    "treatment use, not baseline prognostic imbalance at "
                    "randomization."
                ),
                "support_level": "unsupported",
                "support_rationale": (
                    "Local D1 control rejected post-randomization treatment text "
                    "as evidence for baseline imbalance."
                ),
                "uncertainty": True,
                "supporting_fact_artifact_ids": [],
                "d1_baseline_source_guard_applied": True,
            }
        )
        updated["1.3"] = answer
        return updated
    balance_signals = (
        "well balanced",
        "balanced",
        "similar between",
        "similar across",
        "no major",
        "no important imbalance",
        "no baseline imbalance",
        "no substantial imbalance",
    )
    imbalance_problem_signals = (
        "major imbalance",
        "important imbalance",
        "substantial imbalance",
        "baseline imbalance",
        "chance imbalance",
        "differences in baseline",
        "imbalanced",
    )
    if not any(signal in text for signal in balance_signals):
        return sq_answers
    if any(signal in text for signal in imbalance_problem_signals) and not any(
        signal in text
        for signal in (
            "no important imbalance",
            "no baseline imbalance",
            "no substantial imbalance",
            "no major",
        )
    ):
        return sq_answers
    updated = dict(sq_answers)
    answer.update(
        {
            "answer": "N",
            "justification": (
                "The cited rationale describes baseline balance or absence of "
                "important imbalances, which argues against a randomization "
                "problem for SQ 1.3."
            ),
            "support_level": "moderate",
            "support_rationale": (
                "Local D1 control corrected a polarity error: baseline balance "
                "does not indicate a randomization problem."
            ),
            "uncertainty": True,
            "d1_baseline_balance_guard_applied": True,
        }
    )
    updated["1.3"] = answer
    return updated


def domain1_judge_node(state: RoB2State) -> RoB2State:
    judgment_artifact = judge_domain1_artifact(state["sq_answers"])
    judgment = judgment_artifact["label"]
    rationale = judgment_artifact["rationale"]
    update = add_domain_judgment_with_pivotality_tests(
        state, "D1", judgment, rationale, judge_domain1, DOMAIN1_STAGE.sq_ids
    )
    final_sq_answers = _apply_domain1_controls(
        state, update.get("sq_answers", state["sq_answers"])
    )
    if final_sq_answers != update.get("sq_answers", state["sq_answers"]):
        update["sq_answers"] = {
            sq_id: final_sq_answers[sq_id]
            for sq_id in DOMAIN1_STAGE.sq_ids
            if sq_id in final_sq_answers
        }
        final_judgment, final_rationale = judge_domain1(final_sq_answers)
        update["domain_judgments"]["D1"] = final_judgment
        update["domain_rationales"]["D1"] = final_rationale
    final_judgment = update["domain_judgments"]["D1"]
    final_rationale = update["domain_rationales"]["D1"]
    if final_judgment != judgment or final_rationale != rationale:
        judgment_artifact = judge_domain1_artifact(final_sq_answers)
    update["d1_judgment_artifact"] = {
        **judgment_artifact,
        "artifact_id": f"d1-judgment:{state.get('outcome', '')}",
        "pivotality_tests": update.get("pivotality_tests", {}).get("D1", []),
        "sq_support_adjudications": update.get("sq_support_adjudications", {}).get(
            "D1", []
        ),
    }
    return update
