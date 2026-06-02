import json
from pathlib import Path

from rob2_pipeline.constants import DEFAULT_EFFECT_OF_INTEREST
from rob2_pipeline.graph import build_rob2_graph
from rob2_pipeline.state import RoB2State
from rob2_pipeline.state_factory import create_initial_state
from rob2_pipeline.trace import end_trace, start_trace


JSON_OUTPUT_KEYS = (
    "pdf_path",
    "is_rct",
    "rct_screen_evidence",
    "intervention",
    "comparator",
    "outcome",
    "outcome_type",
    "outcome_properties",
    "outcome_classification_support",
    "numerical_result",
    "effect_of_interest",
    "registration_number",
    "registered_endpoint",
    "registered_analysis",
    "n_randomized",
    "supplementary_paths",
    "source_documents",
    "supplement_warnings",
    "evidence",
    "rag_sources",
    "retrieval_grades",
    "evidence_packets",
    "packet_grades",
    "evidence_facts",
    "sources_consulted",
    "trial_facts",
    "sq_answers",
    "initial_domain_judgments",
    "initial_domain_rationales",
    "domain_judgments",
    "domain_rationales",
    "pivotality_tests",
    "sq_support_adjudications",
    "overall_judgment",
    "overall_rationale",
    "ni_count",
    "high_uncertainty_sqs",
    "human_review_priority",
    "evidence_validation_flags",
    "support_constraints",
    "verifier_trace",
    "verification_actions",
    "overall_policy",
    "errors",
)


def _assessment_json(state: RoB2State) -> dict:
    data = {key: state.get(key) for key in JSON_OUTPUT_KEYS}
    data["rag_sources"] = state.get("rag_chunk_metadata", {})
    return data


def run_assessment(
    pdf_path: str,
    outcome: str | None = None,
    effect_of_interest: str = DEFAULT_EFFECT_OF_INTEREST,
    output_dir: str = "outputs/",
    supplementary_paths: list[str] | None = None,
    precomputed_ingestion=None,
    trial_retrieval_indexes: dict | None = None,
) -> RoB2State:
    """
    Main entry point. Returns the completed state dict.
    Also writes: {output_dir}/{pdf_basename}_rob2_report.md
                 {output_dir}/{pdf_basename}_rob2_data.json
                 {output_dir}/{pdf_basename}_trace.json (LLM I/O for the
                     diagnostic categorizer; chunks come from rag_sources
                     in the rob2_data.json).
    """
    base = Path(pdf_path).stem
    start_trace(trial=base, outcome=outcome)

    try:
        graph = build_rob2_graph()
        state = graph.invoke(
            create_initial_state(
                pdf_path,
                outcome,
                effect_of_interest,
                supplementary_paths=supplementary_paths or [],
                precomputed_ingestion=precomputed_ingestion,
                trial_retrieval_indexes=trial_retrieval_indexes or {},
            )
        )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if state.get("markdown_report"):
            (output_path / f"{base}_rob2_report.md").write_text(
                state["markdown_report"], encoding="utf-8"
            )
        json_data = _assessment_json(state)
        (output_path / f"{base}_rob2_data.json").write_text(
            json.dumps(json_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return state
    finally:
        trace = end_trace()
        if trace is not None:
            trace.write(output_dir)
