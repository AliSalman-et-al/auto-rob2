import json
import re
from pathlib import Path

from rob2_pipeline import config
from rob2_pipeline.constants import DEFAULT_EFFECT_OF_INTEREST
from rob2_pipeline.graph import build_rob2_graph
from rob2_pipeline.state import RoB2State
from rob2_pipeline.state_factory import create_initial_state
from rob2_pipeline.trace import end_trace, start_trace
from rob2_pipeline.trial_workspace import (
    write_evidence_store_trial_workspace,
    write_parse_trial_workspace,
)


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
    "parse_artifacts",
    "supplement_warnings",
    "evidence",
    "rag_sources",
    "retrieval_grades",
    "evidence_packets",
    "packet_grades",
    "evidence_facts",
    "evidence_store",
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
        if state.get("source_documents") and state.get("parse_artifacts"):
            workspace_dir = output_path / f"{base}_trial_workspace"
            write_parse_trial_workspace(
                trial_id=base,
                workspace_dir=workspace_dir,
                source_documents=state["source_documents"],
                parse_artifacts=state["parse_artifacts"],
            )
            if state.get("evidence_store"):
                write_evidence_store_trial_workspace(
                    trial_id=base,
                    workspace_dir=workspace_dir,
                    evidence_store=state["evidence_store"],
                    upstream_artifact_paths=_evidence_store_upstream_paths(
                        workspace_dir,
                        state["parse_artifacts"],
                    ),
                    model_metadata=_model_metadata_from_state(state),
                )
        return state
    finally:
        trace = end_trace()
        if trace is not None:
            trace.write(output_dir)


def _evidence_store_upstream_paths(
    workspace_dir: Path,
    parse_artifacts: list[dict],
) -> dict[str, Path]:
    paths = {}
    for parse_artifact in parse_artifacts:
        source_id = parse_artifact.get("source_identity", {}).get("document_id")
        if not source_id:
            continue
        filename = f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', source_id)}.json"
        paths[f"{source_id}:parse-artifact"] = (
            workspace_dir / "parse_artifacts" / filename
        )
        paths[f"{source_id}:page-aware-artifacts"] = (
            workspace_dir / "page_artifacts" / filename
        )
    return paths


def _model_metadata_from_state(state: RoB2State) -> dict:
    for entry in reversed(state.get("llm_call_log", [])):
        model = entry.get("model")
        if model:
            return {"model": model}
    return {"model": config.LLM_MODEL}
