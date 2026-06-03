import json

from rob2_pipeline.pipeline import _assessment_json, _write_workspace_artifacts
from rob2_pipeline.trial_workspace import file_sha256


def test_assessment_json_includes_supplement_fields():
    state = {
        "pdf_path": "paper.pdf",
        "supplementary_paths": ["protocol.pdf"],
        "source_documents": [
            {
                "document_id": "supplement:001",
                "document_name": "protocol.pdf",
                "document_role": "protocol",
                "status": "parsed",
            }
        ],
        "parse_artifacts": [
            {
                "source_identity": {
                    "document_id": "primary",
                    "document_name": "paper.pdf",
                    "document_role": "primary",
                },
                "pages": [{"page_number": 1, "text": "Primary text"}],
                "diagnostics": [],
                "provenance": {
                    "parser_name": "liteparse",
                    "parser_version": "2.0.0",
                    "adapter_name": "liteparse",
                    "artifact_schema_version": "parse-artifact-v1",
                    "config": {},
                },
            }
        ],
        "supplement_warnings": [],
        "rag_chunk_metadata": {},
    }

    data = _assessment_json(state)

    assert data["supplementary_paths"] == ["protocol.pdf"]
    assert data["source_documents"][0]["document_name"] == "protocol.pdf"
    assert data["parse_artifacts"][0]["provenance"]["parser_name"] == "liteparse"
    assert data["supplement_warnings"] == []


def test_assessment_json_preserves_sq_support_metadata():
    data = _assessment_json(
        {
            "sq_answers": {
                "1.1": {
                    "answer": "Y",
                    "quote": "Randomized",
                    "justification": "Stated randomized.",
                    "uncertainty_flag": "NORMAL",
                    "support_level": "strong",
                    "support_rationale": "Direct quote supports the answer.",
                }
            },
            "rag_chunk_metadata": {},
        }
    )

    assert data["sq_answers"]["1.1"]["support_level"] == "strong"
    assert (
        data["sq_answers"]["1.1"]["support_rationale"]
        == "Direct quote supports the answer."
    )


def test_assessment_json_preserves_support_constraints():
    state = {
        "support_constraints": [
            {
                "constraint_type": "quote_untraceable",
                "sq_id": "1.1",
                "claim": {"answer": "Y", "support_level": "strong"},
                "evidence_label": "quote",
                "evidence": "Randomized centrally.",
                "reason": "quote_not_found_in_source_context",
            }
        ],
        "rag_chunk_metadata": {},
    }

    data = _assessment_json(state)

    assert data["support_constraints"] == state["support_constraints"]


def test_assessment_json_preserves_outcome_classification_support():
    state = {
        "outcome_classification_support": {
            "support_level": "strong",
            "support_rationale": "Direct outcome-bound quote.",
            "quotes": [{"quote": "Overall survival was death from any cause."}],
            "constraints": [],
        },
        "rag_chunk_metadata": {},
    }

    data = _assessment_json(state)

    assert (
        data["outcome_classification_support"]
        == state["outcome_classification_support"]
    )


def test_workspace_output_writes_outcome_manifest_with_trial_hashes(tmp_path):
    primary = tmp_path / "trial.pdf"
    primary.write_bytes(b"primary trial report")
    state = {
        "source_documents": [_source_document(primary)],
        "parse_artifacts": [_parse_artifact(primary)],
        "outcome": "Overall survival",
        "outcome_type": "vital-status",
        "outcome_properties": {"death_only_objective_event": True},
        "outcome_classification_support": {"support_level": "strong"},
        "numerical_result": "HR 0.80",
        "effect_of_interest": "ITT",
        "overall_policy": "rob2-default",
    }

    _write_workspace_artifacts("trial", tmp_path, state)

    outcome_manifest_path = (
        tmp_path
        / "trial_outcome_workspaces"
        / "Overall_survival"
        / "outcome-workspace-manifest.json"
    )
    payload = json.loads(outcome_manifest_path.read_text(encoding="utf-8"))
    trial_manifest_path = (
        tmp_path / "trial_trial_workspace" / "trial-workspace-manifest.json"
    )
    page_path = tmp_path / "trial_trial_workspace" / "page_artifacts" / "primary.json"

    assert payload["trial_id"] == "trial"
    assert payload["outcome_id"] == "Overall survival"
    assert payload["upstream_trial_workspace_hashes"][
        "trial-workspace-manifest"
    ] == file_sha256(trial_manifest_path)
    assert payload["upstream_trial_workspace_hashes"][
        "primary:page-aware-artifacts"
    ] == file_sha256(page_path)


def test_workspace_output_writes_d1_sq_answer_artifact(tmp_path):
    primary = tmp_path / "trial.pdf"
    primary.write_bytes(b"primary trial report")
    state = {
        "source_documents": [_source_document(primary)],
        "parse_artifacts": [_parse_artifact(primary)],
        "outcome": "Overall survival",
        "outcome_type": "vital-status",
        "outcome_properties": {"death_only_objective_event": True},
        "outcome_classification_support": {"support_level": "strong"},
        "effect_of_interest": "ITT",
        "overall_policy": "rob2-default",
        "d1_judgment_artifact": {
            "artifact_id": "d1-judgment:Overall survival",
            "schema_version": "d1-judgment-v1",
            "domain": "d1",
            "judge_version": "d1-judge-v1",
            "rule_table_version": "rob2-d1-rule-table-v1",
            "input_sq_answers": {
                "1.1": {"answer": "Y"},
                "1.2": {"answer": "NI"},
                "1.3": {"answer": "N"},
            },
            "applied_rule_path": "d1-row-4:any/ni/n-pn-ni",
            "label": "Some concerns",
            "rationale": "Row: Any / NI / N-PN-NI -> Some concerns (concealment unclear)",
        },
        "d1_sq_classifier_artifact": {
            "schema_version": "d1-sq-classifier-v1",
            "domain": "d1",
            "answers": [
                _d1_answer("1.1", "Y"),
                _d1_answer("1.2", "NI", support_level="unsupported"),
                _d1_answer("1.3", "N"),
            ],
        },
        "llm_call_log": [
            {
                "node": "domain1_sq_json",
                "model": "gpt-4.1",
                "prompt_version": "d1-sq-classifier-prompt-v1",
                "schema_version": "d1-sq-classifier-v1",
                "attempts": [{"attempt": 1}],
            }
        ],
    }

    _write_workspace_artifacts("trial", tmp_path, state)

    artifact_path = (
        tmp_path
        / "trial_outcome_workspaces"
        / "Overall_survival"
        / "d1-sq-answers.json"
    )
    manifest_path = artifact_path.parent / "outcome-workspace-manifest.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    judgment = json.loads(
        (artifact_path.parent / "d1-judgment.json").read_text(encoding="utf-8")
    )

    assert artifact["artifact_id"] == "d1-sq-answer-set:Overall survival"
    assert artifact["schema_version"] == "d1-sq-answer-set-v1"
    assert artifact["classifier_schema_version"] == "d1-sq-classifier-v1"
    assert artifact["classifier_prompt_version"] == "d1-sq-classifier-prompt-v1"
    assert artifact["validation"]["status"] == "validated"
    manifest_artifacts = {
        item["artifact_id"]: item for item in manifest["artifacts"]
    }
    assert manifest_artifacts[artifact["artifact_id"]]["producer"] == "d1-sq-classifier"
    assert judgment["artifact_id"] == "d1-judgment:Overall survival"
    assert judgment["judge_version"] == "d1-judge-v1"
    assert {
        "d1-sq-answer-set:Overall survival",
        "d1-judgment:Overall survival",
    }.issubset(manifest_artifacts)


def _source_document(path):
    return {
        "document_id": "primary",
        "document_name": path.name,
        "document_role": "primary",
        "source_kind": "rag_chunk",
        "path": str(path),
        "is_primary": True,
        "status": "parsed",
    }


def _parse_artifact(path):
    return {
        "source_identity": _source_document(path),
        "pages": [
            {
                "page_number": 1,
                "text": "Methods\nParticipants were randomized.\nResults\nDone.",
                "width": 612.0,
                "height": 792.0,
            }
        ],
        "diagnostics": [],
        "parse_time_ms": 17,
        "provenance": {
            "parser_name": "liteparse",
            "parser_version": "2.0.4",
            "adapter_name": "liteparse",
            "artifact_schema_version": "parse-artifact-v1",
            "config": {"ocr_enabled": False},
        },
    }


def _d1_answer(sq_id, answer, *, support_level="strong"):
    return {
        "sq_id": sq_id,
        "answer": answer,
        "quote": "computer-generated sequence",
        "justification": "The selected packet supports this answer.",
        "support_level": support_level,
        "support_rationale": "Supported by selected packet evidence.",
        "uncertainty": support_level == "unsupported",
        "packet_artifact_id": f"evidence-packet:d1:{sq_id}",
        "decision_table_artifact_id": f"decision-table:d1:{sq_id}",
        "supporting_fact_artifact_ids": [],
    }
