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
    write_d1_engineering_diagnostics_workspace,
    write_d1_judgment_workspace,
    write_domain_judgment_workspace,
    write_domain_sq_answer_workspace,
    write_outcome_normalization_workspace,
    write_evidence_store_trial_workspace,
    write_parse_trial_workspace,
    write_support_escalation_diagnostics_workspace,
    stable_payload_sha256,
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
    "outcome_normalization_artifact",
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
    "packet_readiness",
    "retrieval_repair_artifacts",
    "evidence_facts",
    "selected_evidence_facts",
    "evidence_store",
    "sources_consulted",
    "trial_facts",
    "sq_answers",
    "initial_domain_judgments",
    "initial_domain_rationales",
    "d1_judgment_artifact",
    "d2_judgment_artifact",
    "d3_judgment_artifact",
    "d4_judgment_artifact",
    "d5_judgment_artifact",
    "domain_judgments",
    "domain_rationales",
    "pivotality_tests",
    "micro_agent_routing_decisions",
    "sq_support_adjudications",
    "overall_judgment",
    "overall_rationale",
    "overall_judgment_artifact",
    "automation_confidence",
    "ni_count",
    "high_uncertainty_sqs",
    "human_review_priority",
    "reviewer_report",
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
        if state.get("reviewer_report"):
            (output_path / f"{base}_reviewer_report.md").write_text(
                state["reviewer_report"], encoding="utf-8"
            )
        json_data = _assessment_json(state)
        (output_path / f"{base}_rob2_data.json").write_text(
            json.dumps(json_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if state.get("source_documents") and state.get("parse_artifacts"):
            _write_workspace_artifacts(base, output_path, state)
        return state
    finally:
        trace = end_trace()
        if trace is not None:
            trace.write(output_dir)


def _write_workspace_artifacts(
    base: str,
    output_path: Path,
    state: RoB2State,
) -> None:
    trial_workspace_dir = output_path / f"{base}_trial_workspace"
    write_parse_trial_workspace(
        trial_id=base,
        workspace_dir=trial_workspace_dir,
        source_documents=state["source_documents"],
        parse_artifacts=state["parse_artifacts"],
    )
    if state.get("evidence_store"):
        write_evidence_store_trial_workspace(
            trial_id=base,
            workspace_dir=trial_workspace_dir,
            evidence_store=state["evidence_store"],
            upstream_artifact_paths=_evidence_store_upstream_paths(
                trial_workspace_dir,
                state["parse_artifacts"],
            ),
            model_metadata=_model_metadata_from_state(state),
        )

    write_outcome_normalization_workspace(
        trial_id=base,
        outcome_id=_outcome_id_from_state(state),
        workspace_root=output_path / f"{base}_outcome_workspaces",
        trial_workspace_dir=trial_workspace_dir,
        upstream_artifact_paths=_outcome_workspace_upstream_paths(
            trial_workspace_dir,
            state["parse_artifacts"],
        ),
        outcome_definition=_outcome_definition_from_state(state),
        rob2_settings=_rob2_settings_from_state(state),
        outcome_normalization_artifact=_outcome_normalization_artifact_from_state(
            state
        ),
        model_metadata=_model_metadata_from_state(state),
    )
    for domain in ("d1", "d2", "d3", "d4", "d5"):
        classifier_artifacts = _domain_classifier_artifacts_from_state(state, domain)
        if classifier_artifacts:
            write_domain_sq_answer_workspace(
                domain=domain,
                trial_id=base,
                outcome_id=_outcome_id_from_state(state),
                workspace_root=output_path / f"{base}_outcome_workspaces",
                trial_workspace_dir=trial_workspace_dir,
                upstream_artifact_paths=_outcome_workspace_upstream_paths(
                    trial_workspace_dir,
                    state["parse_artifacts"],
                ),
                outcome_definition=_outcome_definition_from_state(state),
                rob2_settings=_rob2_settings_from_state(state),
                sq_answer_artifact=_domain_sq_answer_artifact_from_state(
                    state,
                    domain,
                    classifier_artifacts,
                ),
                model_metadata=_model_metadata_from_state(state),
                contract_metadata=_domain_contract_metadata_from_state(
                    state,
                    domain,
                    classifier_artifacts,
                ),
            )
    if state.get("d1_judgment_artifact"):
        write_d1_judgment_workspace(
            trial_id=base,
            outcome_id=_outcome_id_from_state(state),
            workspace_root=output_path / f"{base}_outcome_workspaces",
            trial_workspace_dir=trial_workspace_dir,
            upstream_artifact_paths={
                **_outcome_workspace_upstream_paths(
                    trial_workspace_dir,
                    state["parse_artifacts"],
                ),
                **_domain_sq_answer_upstream_path(output_path, base, state, "d1"),
            },
            outcome_definition=_outcome_definition_from_state(state),
            rob2_settings=_rob2_settings_from_state(state),
            d1_judgment_artifact=state["d1_judgment_artifact"],
        )
    for domain in ("d2", "d3", "d4", "d5"):
        judgment_key = f"{domain}_judgment_artifact"
        if state.get(judgment_key):
            write_domain_judgment_workspace(
                domain=domain,
                trial_id=base,
                outcome_id=_outcome_id_from_state(state),
                workspace_root=output_path / f"{base}_outcome_workspaces",
                trial_workspace_dir=trial_workspace_dir,
                upstream_artifact_paths={
                    **_outcome_workspace_upstream_paths(
                        trial_workspace_dir,
                        state["parse_artifacts"],
                    ),
                    **_domain_sq_answer_upstream_path(
                        output_path,
                        base,
                        state,
                        domain,
                    ),
                },
                outcome_definition=_outcome_definition_from_state(state),
                rob2_settings=_rob2_settings_from_state(state),
                judgment_artifact=state[judgment_key],
            )
    if _d1_classifier_artifact_from_state(state) or state.get("d1_judgment_artifact"):
        write_d1_engineering_diagnostics_workspace(
            trial_id=base,
            outcome_id=_outcome_id_from_state(state),
            workspace_root=output_path / f"{base}_outcome_workspaces",
            trial_workspace_dir=trial_workspace_dir,
            upstream_artifact_paths={
                **_outcome_workspace_upstream_paths(
                    trial_workspace_dir,
                    state["parse_artifacts"],
                ),
                **_domain_sq_answer_upstream_path(output_path, base, state, "d1"),
                **_d1_judgment_upstream_path(output_path, base, state),
            },
            outcome_definition=_outcome_definition_from_state(state),
            rob2_settings=_rob2_settings_from_state(state),
            diagnostics_artifact=_d1_engineering_diagnostics_from_state(state),
        )
    if state.get("sq_support_adjudications"):
        write_support_escalation_diagnostics_workspace(
            trial_id=base,
            outcome_id=_outcome_id_from_state(state),
            workspace_root=output_path / f"{base}_outcome_workspaces",
            trial_workspace_dir=trial_workspace_dir,
            upstream_artifact_paths={
                **_outcome_workspace_upstream_paths(
                    trial_workspace_dir,
                    state["parse_artifacts"],
                ),
                **_all_sq_answer_upstream_paths(output_path, base, state),
                **_all_judgment_upstream_paths(output_path, base, state),
            },
            outcome_definition=_outcome_definition_from_state(state),
            rob2_settings=_rob2_settings_from_state(state),
            diagnostics_artifact=_support_escalation_diagnostics_from_state(state),
        )


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


def _outcome_workspace_upstream_paths(
    workspace_dir: Path,
    parse_artifacts: list[dict],
) -> dict[str, Path]:
    return {
        "trial-workspace-manifest": workspace_dir / "trial-workspace-manifest.json",
        **_evidence_store_upstream_paths(workspace_dir, parse_artifacts),
    }


def _outcome_id_from_state(state: RoB2State) -> str:
    return str(state.get("outcome") or "unspecified-outcome")


def _outcome_definition_from_state(state: RoB2State) -> dict:
    return {
        "outcome": state.get("outcome"),
        "numerical_result": state.get("numerical_result"),
        "outcome_type": state.get("outcome_type"),
        "outcome_properties": state.get("outcome_properties") or {},
        "outcome_classification_support": state.get("outcome_classification_support")
        or {},
    }


def _outcome_normalization_artifact_from_state(state: RoB2State) -> dict:
    artifact = state.get("outcome_normalization_artifact")
    if artifact:
        return artifact
    support = state.get("outcome_classification_support") or {
        "support_level": "unsupported",
        "support_rationale": "Outcome classification artifact was derived from state.",
        "quotes": [],
        "constraints": [],
    }
    return {
        "artifact_id": f"outcome-normalization:{state.get('outcome', '')}",
        "schema_version": "outcome-normalization-v1",
        "outcome": state.get("outcome", ""),
        "normalized_definition": "",
        "aliases": [],
        "outcome_type": state.get("outcome_type", "clinician-composite"),
        "outcome_properties": state.get("outcome_properties") or {},
        "binding_support": support,
        "auto_accept_blocked": support.get("support_level") in {
            "weak",
            "unsupported",
        },
        "uncertainty": support.get("support_level") in {"weak", "unsupported"},
    }


def _rob2_settings_from_state(state: RoB2State) -> dict:
    return {
        "effect_of_interest": state.get("effect_of_interest"),
        "overall_policy": state.get("overall_policy"),
    }


def _model_metadata_from_state(state: RoB2State) -> dict:
    for entry in reversed(state.get("llm_call_log", [])):
        model = entry.get("model")
        if model:
            return {"model": model}
    return {"model": config.LLM_MODEL}


def _domain_classifier_artifacts_from_state(
    state: RoB2State,
    domain: str,
) -> list[dict]:
    nested = (state.get("domain_sq_classifier_artifacts") or {}).get(domain) or {}
    if nested:
        return [
            artifact
            for _stage, artifact in sorted(nested.items())
            if isinstance(artifact, dict)
        ]
    keys_by_domain = {
        "d1": ("d1_sq_classifier_artifact",),
        "d2": (
            "d2_sq12_classifier_artifact",
            "d2_conditional_classifier_artifact",
            "d2_analysis_classifier_artifact",
        ),
        "d3": ("d3_sq_classifier_artifact",),
        "d4": ("d4_sq_classifier_artifact",),
        "d5": ("d5_sq_classifier_artifact",),
    }
    return [
        state[key]
        for key in keys_by_domain.get(domain, ())
        if isinstance(state.get(key), dict)
    ]


def _domain_sq_answer_artifact_from_state(
    state: RoB2State,
    domain: str,
    classifier_artifacts: list[dict],
) -> dict:
    answers = []
    stages = []
    branching = {}
    outcome_specific_concerns = []
    for artifact in classifier_artifacts:
        answers.extend(artifact.get("answers", []))
        stage = artifact.get("stage")
        if stage:
            stages.append(stage)
        branching[str(stage or len(stages))] = artifact.get("branching", {})
        outcome_specific_concerns.extend(artifact.get("outcome_specific_concerns", []))
    return {
        "artifact_id": f"{domain}-sq-answer-set:{_outcome_id_from_state(state)}",
        "schema_version": f"{domain}-sq-answer-set-v1",
        "classifier_schema_version": classifier_artifacts[-1].get(
            "schema_version", ""
        ),
        "classifier_prompt_version": _domain_contract_metadata_from_state(
            state,
            domain,
            classifier_artifacts,
        ).get("classifier_prompt_version", ""),
        "domain": domain,
        "stages": stages,
        "branching": branching,
        "outcome_specific_concerns": outcome_specific_concerns,
        "answers": answers,
    }


def _domain_contract_metadata_from_state(
    state: RoB2State,
    domain: str,
    classifier_artifacts: list[dict],
) -> dict:
    log_entries = [
        entry
        for entry in state.get("llm_call_log", [])
        if str(entry.get("node", "")).startswith(f"domain{domain[-1]}_")
        and str(entry.get("node", "")).endswith("_json")
    ]
    latest = log_entries[-1] if log_entries else {}
    attempts = latest.get("attempts") or []
    max_attempts = max(
        [int(attempt.get("attempt", 0) or 0) for attempt in attempts] or [1]
    )
    schema_version = (
        latest.get("schema_version")
        or classifier_artifacts[-1].get("schema_version", "")
    )
    return {
        "schema_version": f"{domain}-sq-answer-set-v1",
        "classifier_schema_version": schema_version,
        "classifier_prompt_version": latest.get("prompt_version", ""),
        "retry_policy": {"max_attempts": max_attempts},
        "model_affecting_settings": {
            "provider": latest.get("provider", config.PROVIDER_NAME),
            "model": latest.get("model", config.LLM_MODEL),
        },
    }


def _d1_classifier_artifact_from_state(state: RoB2State) -> dict | None:
    nested = (state.get("domain_sq_classifier_artifacts") or {}).get("d1") or {}
    artifact = nested.get("sq")
    if isinstance(artifact, dict):
        return artifact
    legacy = state.get("d1_sq_classifier_artifact")
    return legacy if isinstance(legacy, dict) else None


def _domain_sq_answer_upstream_path(
    output_path: Path,
    base: str,
    state: RoB2State,
    domain: str,
) -> dict[str, Path]:
    if not _domain_classifier_artifacts_from_state(state, domain):
        return {}
    outcome_dir = re.sub(r"[^A-Za-z0-9_.-]+", "_", _outcome_id_from_state(state))
    return {
        f"{domain}-sq-answer-set": output_path
        / f"{base}_outcome_workspaces"
        / outcome_dir
        / f"{domain}-sq-answers.json"
    }


def _all_sq_answer_upstream_paths(
    output_path: Path,
    base: str,
    state: RoB2State,
) -> dict[str, Path]:
    paths = {}
    for domain in ("d1", "d2", "d3", "d4", "d5"):
        paths.update(_domain_sq_answer_upstream_path(output_path, base, state, domain))
    return paths


def _d1_judgment_upstream_path(
    output_path: Path,
    base: str,
    state: RoB2State,
) -> dict[str, Path]:
    if not state.get("d1_judgment_artifact"):
        return {}
    outcome_dir = re.sub(r"[^A-Za-z0-9_.-]+", "_", _outcome_id_from_state(state))
    return {
        "d1-judgment": output_path
        / f"{base}_outcome_workspaces"
        / outcome_dir
        / "d1-judgment.json"
    }


def _domain_judgment_upstream_path(
    output_path: Path,
    base: str,
    state: RoB2State,
    domain: str,
) -> dict[str, Path]:
    if not state.get(f"{domain}_judgment_artifact"):
        return {}
    outcome_dir = re.sub(r"[^A-Za-z0-9_.-]+", "_", _outcome_id_from_state(state))
    return {
        f"{domain}-judgment": output_path
        / f"{base}_outcome_workspaces"
        / outcome_dir
        / f"{domain}-judgment.json"
    }


def _all_judgment_upstream_paths(
    output_path: Path,
    base: str,
    state: RoB2State,
) -> dict[str, Path]:
    paths = _d1_judgment_upstream_path(output_path, base, state)
    for domain in ("d2", "d3", "d4", "d5"):
        paths.update(_domain_judgment_upstream_path(output_path, base, state, domain))
    return paths


def _latest_llm_log_entry(state: RoB2State, node: str) -> dict:
    for entry in reversed(state.get("llm_call_log", [])):
        if entry.get("node") == node:
            return entry
    return {}


def _support_escalation_diagnostics_from_state(state: RoB2State) -> dict:
    outcome_id = _outcome_id_from_state(state)
    adjudications = state.get("sq_support_adjudications") or {}
    attempts = []
    for domain, domain_attempts in sorted(adjudications.items()):
        for index, attempt in enumerate(domain_attempts or [], start=1):
            llm_node = (attempt.get("provenance") or {}).get("llm_node")
            model_call = _latest_llm_log_entry(state, llm_node) if llm_node else {}
            persisted_attempt = {
                "domain": domain,
                "sq_id": attempt.get("sq_id"),
                "bounded_attempt_number": index,
                "initial_answer": attempt.get("initial_answer"),
                "adjudicated_answer": attempt.get("adjudicated_answer"),
                "changed": attempt.get("changed"),
                "changed_answer": attempt.get("changed_answer"),
                "changed_support": attempt.get("changed_support"),
                "acceptance_status": _acceptance_status_for_attempt(
                    state,
                    domain,
                    attempt.get("sq_id"),
                ),
                "provenance": attempt.get("provenance") or {},
                "constraints": attempt.get("constraints") or [],
                "model_call": _support_escalation_model_call_diagnostics(model_call),
            }
            persisted_attempt["artifact_hash"] = stable_payload_sha256(
                persisted_attempt
            )
            attempts.append(persisted_attempt)

    max_attempts = max(
        [
            int((attempt.get("model_call") or {}).get("attempt_count") or 0)
            for attempt in attempts
        ]
        or [0]
    )
    return {
        "artifact_id": f"support-escalation-diagnostics:{outcome_id}",
        "schema_version": "support-escalation-diagnostics-v1",
        "producer_version": "support-escalation-diagnostics-v1",
        "outcome_id": outcome_id,
        "reviewer_report_artifact_id": None,
        "retry_policy": {"max_attempts_per_escalation": max_attempts},
        "attempt_count": len(attempts),
        "attempts": attempts,
    }


def _support_escalation_model_call_diagnostics(log_entry: dict) -> dict:
    return {
        "node": log_entry.get("node"),
        "provider": log_entry.get("provider"),
        "model": log_entry.get("model"),
        "latency_ms": log_entry.get("latency_ms"),
        "input_tokens": log_entry.get("input_tokens"),
        "output_tokens": log_entry.get("output_tokens"),
        "cost_usd": log_entry.get("cost_usd"),
        "cache_hit": log_entry.get("cache_hit"),
        "attempt_count": len(log_entry.get("attempts") or []),
    }


def _acceptance_status_for_attempt(
    state: RoB2State,
    domain: str,
    sq_id: object,
) -> str | None:
    for test in (state.get("pivotality_tests") or {}).get(domain, []):
        if test.get("sq_id") == sq_id:
            return test.get("acceptance_status")
    return None


def _d1_engineering_diagnostics_from_state(state: RoB2State) -> dict:
    outcome_id = _outcome_id_from_state(state)
    d1_calls = [
        entry
        for entry in state.get("llm_call_log", [])
        if str(entry.get("node", "")).startswith("domain1")
    ]
    latest_call = d1_calls[-1] if d1_calls else {}
    schema_status = latest_call.get("validation_status") or "not_run"
    packet_readiness = state.get("packet_readiness") or {}
    d1_packet_status = packet_readiness.get("d1") or _packet_status_from_state(state)
    judge_artifact = state.get("d1_judgment_artifact") or {}
    parse_status = _parse_status_from_state(state)
    model_status = _model_call_status(schema_status, d1_calls)
    judge_status = "ok" if judge_artifact.get("label") else "not_run"
    statuses = {
        "parse": parse_status,
        "packet": d1_packet_status,
        "schema_validation": schema_status,
        "model_call": model_status,
        "judge": judge_status,
    }
    return {
        "artifact_id": f"d1-engineering-diagnostics:{outcome_id}",
        "schema_version": "d1-engineering-diagnostics-v1",
        "producer_version": "d1-engineering-diagnostics-v1",
        "domain": "d1",
        "outcome_id": outcome_id,
        "reviewer_report_artifact_id": None,
        "statuses": statuses,
        "parse": {
            "status": parse_status,
            "documents": [_parse_document_diagnostics(item) for item in state.get("parse_artifacts", [])],
        },
        "packets": {
            "status": d1_packet_status,
            "packet_readiness": packet_readiness,
            "packet_grades": state.get("packet_grades") or {},
            "evidence_packet_count": _evidence_packet_count(state.get("evidence_packets")),
        },
        "schema_validation": {
            "status": schema_status,
            "schema_version": latest_call.get("schema_version"),
            "prompt_version": latest_call.get("prompt_version"),
            "failure_reason": latest_call.get("failure_reason"),
            "attempts": _validation_attempts(latest_call),
        },
        "model_calls": [_model_call_diagnostics(entry) for entry in d1_calls],
        "judge": {
            "status": judge_status,
            "artifact_id": judge_artifact.get("artifact_id"),
            "schema_version": judge_artifact.get("schema_version"),
            "judge_version": judge_artifact.get("judge_version"),
            "rule_table_version": judge_artifact.get("rule_table_version"),
            "applied_rule_path": judge_artifact.get("applied_rule_path"),
            "label": judge_artifact.get("label"),
        },
        "failure_summary": _d1_failure_summary(statuses, latest_call),
    }


def _parse_status_from_state(state: RoB2State) -> str:
    parse_artifacts = state.get("parse_artifacts") or []
    if not parse_artifacts:
        return "missing"
    if any(artifact.get("diagnostics") for artifact in parse_artifacts):
        return "diagnostics_present"
    return "ok"


def _parse_document_diagnostics(parse_artifact: dict) -> dict:
    pages = parse_artifact.get("pages", [])
    provenance = parse_artifact.get("provenance") or {}
    source = parse_artifact.get("source_identity") or {}
    return {
        "document_id": source.get("document_id"),
        "document_name": source.get("document_name"),
        "document_role": source.get("document_role"),
        "parser_name": provenance.get("parser_name"),
        "parser_version": provenance.get("parser_version"),
        "adapter_name": provenance.get("adapter_name"),
        "parse_time_ms": parse_artifact.get("parse_time_ms"),
        "page_count": len(pages),
        "text_character_count": sum(len(page.get("text", "")) for page in pages),
        "diagnostic_count": len(parse_artifact.get("diagnostics") or []),
        "diagnostics": parse_artifact.get("diagnostics") or [],
    }


def _packet_status_from_state(state: RoB2State) -> str:
    if state.get("evidence_packets"):
        return "ready"
    return "missing"


def _evidence_packet_count(evidence_packets: object) -> int:
    if not isinstance(evidence_packets, dict):
        return 0
    count = 0
    for value in evidence_packets.values():
        if isinstance(value, dict):
            count += len(value)
        elif isinstance(value, list):
            count += len(value)
        elif value:
            count += 1
    return count


def _model_call_status(schema_status: object, d1_calls: list[dict]) -> str:
    if not d1_calls:
        return "not_run"
    if schema_status == "fallback":
        return "fallback"
    if schema_status in {"validated", "not_validated"}:
        return "ok"
    return str(schema_status or "unknown")


def _validation_attempts(log_entry: dict) -> list[dict]:
    return [
        {
            "attempt": attempt.get("attempt"),
            "parse_status": attempt.get("parse_status"),
            "validation_status": attempt.get("validation_status"),
            "parse_error": attempt.get("parse_error"),
            "validation_error": attempt.get("validation_error"),
            "is_repair": attempt.get("is_repair"),
        }
        for attempt in log_entry.get("attempts", [])
    ]


def _model_call_diagnostics(log_entry: dict) -> dict:
    return {
        "node": log_entry.get("node"),
        "provider": log_entry.get("provider"),
        "model": log_entry.get("model"),
        "prompt_version": log_entry.get("prompt_version"),
        "schema_version": log_entry.get("schema_version"),
        "latency_ms": log_entry.get("latency_ms"),
        "input_tokens": log_entry.get("input_tokens"),
        "output_tokens": log_entry.get("output_tokens"),
        "cached": log_entry.get("cached"),
        "cache_hit": log_entry.get("cache_hit"),
        "cost_usd": log_entry.get("cost_usd"),
        "parse_status": log_entry.get("parse_status"),
        "validation_status": log_entry.get("validation_status"),
        "attempt_count": len(log_entry.get("attempts") or []),
        "failure_reason": log_entry.get("failure_reason"),
    }


def _d1_failure_summary(statuses: dict, latest_call: dict) -> list[dict]:
    failures = []
    for stage, status in statuses.items():
        if status in {"ok", "ready", "validated"}:
            continue
        if stage == "model_call" and statuses.get("schema_validation") == "fallback":
            continue
        reason = None
        if stage in {"schema_validation", "model_call"}:
            reason = latest_call.get("failure_reason")
        failures.append({"stage": stage, "status": status, "reason": reason})
    return failures
