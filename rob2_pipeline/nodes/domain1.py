from rob2_pipeline.judges.domain1 import judge_domain1, judge_domain1_artifact
from rob2_pipeline.nodes.common import (
    add_domain_judgment_with_pivotality_tests,
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
    return run_domain_sq_stage(state, DOMAIN1_STAGE)


def domain1_judge_node(state: RoB2State) -> RoB2State:
    judgment_artifact = judge_domain1_artifact(state["sq_answers"])
    judgment = judgment_artifact["label"]
    rationale = judgment_artifact["rationale"]
    update = add_domain_judgment_with_pivotality_tests(
        state, "D1", judgment, rationale, judge_domain1, DOMAIN1_STAGE.sq_ids
    )
    final_sq_answers = update.get("sq_answers", state["sq_answers"])
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
