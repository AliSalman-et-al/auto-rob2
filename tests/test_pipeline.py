import json

from rob2_pipeline.pipeline import _assessment_json, _write_workspace_artifacts
from rob2_pipeline.ingestion.parse_artifacts import PARSE_ARTIFACT_SCHEMA_VERSION
from rob2_pipeline.state_factory import create_initial_state
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
                    "parser_name": "pymupdf+pymupdf4llm",
                    "parser_version": "pymupdf=1.26.0; pymupdf4llm=0.0.27",
                    "adapter_name": "pymupdf-sectionmap",
                    "artifact_schema_version": PARSE_ARTIFACT_SCHEMA_VERSION,
                    "config": {
                        "layout_text_engine": "pymupdf4llm",
                        "raw_character_stream_engine": "pymupdf",
                    },
                },
            }
        ],
        "supplement_warnings": [],
        "supplement_segments": [],
        "supplement_retrieval_grades": {},
        "rag_chunk_metadata": {},
    }

    data = _assessment_json(state)

    assert data["supplementary_paths"] == ["protocol.pdf"]
    assert data["source_documents"][0]["document_name"] == "protocol.pdf"
    assert (
        data["parse_artifacts"][0]["provenance"]["parser_name"]
        == "pymupdf+pymupdf4llm"
    )
    assert (
        data["parse_artifacts"][0]["provenance"]["artifact_schema_version"]
        == PARSE_ARTIFACT_SCHEMA_VERSION
    )
    assert data["supplement_warnings"] == []
    assert data["supplement_segments"] == []
    assert data["supplement_retrieval_grades"] == {}


def test_initial_state_omits_legacy_rag_fields_and_keeps_supplement_outputs():
    state = create_initial_state("paper.pdf")

    for key in (
        "rag_contexts",
        "rag_chunk_metadata",
        "trial_retrieval_indexes",
        "docling_chunks",
        "retrieval_grades",
    ):
        assert key not in state
    assert state["supplement_segments"] == []
    assert state["supplement_retrieval_grades"] == {}


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


def test_assessment_json_preserves_judgment_artifacts():
    state = {
        "d2_judgment_artifact": {
            "artifact_id": "d2-judgment:Overall survival",
            "judge_version": "d2-judge-v1",
        },
        "d3_judgment_artifact": {
            "artifact_id": "d3-judgment:Overall survival",
            "judge_version": "d3-judge-v1",
        },
        "d4_judgment_artifact": {
            "artifact_id": "d4-judgment:Overall survival",
            "judge_version": "d4-judge-v1",
        },
        "d5_judgment_artifact": {
            "artifact_id": "d5-judgment:Overall survival",
            "judge_version": "d5-judge-v1",
        },
        "overall_judgment_artifact": {
            "artifact_id": "overall-judgment:Overall survival",
            "policy": "official_rob2",
        },
        "rag_chunk_metadata": {},
    }

    data = _assessment_json(state)

    assert data["d2_judgment_artifact"] == state["d2_judgment_artifact"]
    assert data["d3_judgment_artifact"] == state["d3_judgment_artifact"]
    assert data["d4_judgment_artifact"] == state["d4_judgment_artifact"]
    assert data["d5_judgment_artifact"] == state["d5_judgment_artifact"]
    assert data["overall_judgment_artifact"] == state["overall_judgment_artifact"]


def test_assessment_json_preserves_automation_confidence():
    state = {
        "automation_confidence": {
            "artifact_id": "automation-confidence:Overall survival",
            "schema_version": "automation-confidence-v1",
            "status": "auto_accept_candidate",
            "blocking_reasons": [],
            "non_acceptance_reasons": [],
        },
        "rag_chunk_metadata": {},
    }

    data = _assessment_json(state)

    assert data["automation_confidence"] == state["automation_confidence"]


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
        "domain_sq_classifier_artifacts": {
            "d1": {
                "sq": {
                    "schema_version": "d1-sq-classifier-v1",
                    "domain": "d1",
                    "stage": "sq",
                    "branching": {},
                    "outcome_specific_concerns": [],
                    "answers": [
                        _d1_answer("1.1", "Y"),
                        _d1_answer("1.2", "NI", support_level="unsupported"),
                        _d1_answer("1.3", "N"),
                    ],
                }
            }
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


def test_workspace_output_writes_d2_sq_answer_and_judgment_artifacts(tmp_path):
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
        "d2_sq12_classifier_artifact": {
            "schema_version": "d2-sq-classifier-v1",
            "domain": "d2",
            "stage": "sq12",
            "branching": {"effect_of_interest": "ITT"},
            "answers": [_domain_answer("d2", "2.1", "Y")],
        },
        "d2_judgment_artifact": {
            "artifact_id": "d2-judgment:Overall survival",
            "schema_version": "d2-judgment-v1",
            "domain": "d2",
            "judge_version": "d2-judge-v1",
            "rule_table_version": "rob2-d2-assignment-rule-table-v1",
            "input_sq_answers": {"2.1": {"answer": "Y"}},
            "applied_rule_path": "d2-assignment:part1-low+part2-low",
            "label": "Low",
            "rationale": "Part1=Low; Part2=Low",
        },
        "llm_call_log": [
            {
                "node": "domain2_sq12_json",
                "model": "gpt-4.1",
                "prompt_version": "d2-sq12-classifier-prompt-v1",
                "schema_version": "d2-sq-classifier-v1",
                "attempts": [{"attempt": 1}],
            }
        ],
    }

    _write_workspace_artifacts("trial", tmp_path, state)

    outcome_dir = tmp_path / "trial_outcome_workspaces" / "Overall_survival"
    sq_artifact = json.loads(
        (outcome_dir / "d2-sq-answers.json").read_text(encoding="utf-8")
    )
    judgment = json.loads(
        (outcome_dir / "d2-judgment.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (outcome_dir / "outcome-workspace-manifest.json").read_text(encoding="utf-8")
    )
    manifest_artifacts = {
        item["artifact_id"]: item for item in manifest["artifacts"]
    }

    assert sq_artifact["artifact_id"] == "d2-sq-answer-set:Overall survival"
    assert sq_artifact["schema_version"] == "d2-sq-answer-set-v1"
    assert sq_artifact["classifier_schema_version"] == "d2-sq-classifier-v1"
    assert sq_artifact["classifier_prompt_version"] == "d2-sq12-classifier-prompt-v1"
    assert sq_artifact["validation"]["status"] == "validated"
    assert judgment["artifact_id"] == "d2-judgment:Overall survival"
    assert manifest_artifacts[sq_artifact["artifact_id"]]["producer"] == (
        "d2-sq-classifier"
    )
    assert manifest_artifacts[judgment["artifact_id"]]["producer"] == (
        "d2-deterministic-judge"
    )


def test_workspace_output_writes_generic_d1_artifacts_without_engineering_diagnostics(
    tmp_path,
):
    primary = tmp_path / "trial.pdf"
    primary.write_bytes(b"primary trial report")
    state = _d1_workspace_state(primary)

    _write_workspace_artifacts("trial", tmp_path, state)

    outcome_dir = (
        tmp_path
        / "trial_outcome_workspaces"
        / "Overall_survival"
    )
    manifest_path = outcome_dir / "outcome-workspace-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_artifacts = {
        item["artifact_id"]: item for item in manifest["artifacts"]
    }

    assert (outcome_dir / "d1-sq-answers.json").exists()
    assert (outcome_dir / "d1-judgment.json").exists()
    assert not (outcome_dir / "d1-engineering-diagnostics.json").exists()
    assert "d1-sq-answer-set:Overall survival" in manifest_artifacts
    assert "d1-judgment:Overall survival" in manifest_artifacts
    assert not any(
        item["producer"] == "d1-engineering-diagnostics"
        for item in manifest["artifacts"]
    )


def test_workspace_output_writes_support_escalation_diagnostics(tmp_path):
    primary = tmp_path / "trial.pdf"
    primary.write_bytes(b"primary trial report")
    state = _d1_workspace_state(primary)
    state["sq_support_adjudications"] = {
        "D1": [
            {
                "sq_id": "1.3",
                "initial_answer": {"answer": "Y", "support_level": "weak"},
                "adjudicated_answer": {"answer": "N", "support_level": "strong"},
                "changed": True,
                "changed_answer": True,
                "changed_support": True,
                "provenance": {"llm_node": "sq_support_adjudication_D1_1_3"},
            }
        ]
    }
    state["llm_call_log"].append(
        {
            "node": "sq_support_adjudication_D1_1_3",
            "provider": "openrouter",
            "model": "gpt-4.1",
            "prompt_version": "sq-support-adjudication-prompt-v1",
            "schema_version": "sq-support-adjudication-v1",
            "latency_ms": 43,
            "input_tokens": 120,
            "output_tokens": 35,
            "cost_usd": 0.0021,
            "cache_hit": False,
            "attempts": [{"attempt": 1}, {"attempt": 2, "is_repair": True}],
        }
    )

    _write_workspace_artifacts("trial", tmp_path, state)

    outcome_dir = tmp_path / "trial_outcome_workspaces" / "Overall_survival"
    diagnostics = json.loads(
        (outcome_dir / "support-escalation-diagnostics.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        (outcome_dir / "outcome-workspace-manifest.json").read_text(encoding="utf-8")
    )
    manifest_artifacts = {item["artifact_id"]: item for item in manifest["artifacts"]}

    assert diagnostics["artifact_id"] == (
        "support-escalation-diagnostics:Overall survival"
    )
    assert diagnostics["schema_version"] == "support-escalation-diagnostics-v1"
    assert diagnostics["reviewer_report_artifact_id"] is None
    assert diagnostics["retry_policy"] == {"max_attempts_per_escalation": 2}
    assert diagnostics["attempts"][0]["domain"] == "D1"
    assert diagnostics["attempts"][0]["sq_id"] == "1.3"
    assert diagnostics["attempts"][0]["bounded_attempt_number"] == 1
    assert diagnostics["attempts"][0]["artifact_hash"]
    assert diagnostics["attempts"][0]["model_call"] == {
        "node": "sq_support_adjudication_D1_1_3",
        "provider": "openrouter",
        "model": "gpt-4.1",
        "latency_ms": 43,
        "input_tokens": 120,
        "output_tokens": 35,
        "cost_usd": 0.0021,
        "cache_hit": False,
        "attempt_count": 2,
    }
    assert manifest_artifacts[diagnostics["artifact_id"]]["producer"] == (
        "support-escalation-diagnostics"
    )
    upstream_hashes = manifest_artifacts[diagnostics["artifact_id"]][
        "upstream_trial_workspace_hashes"
    ]
    assert upstream_hashes["d1-sq-answer-set"]
    assert upstream_hashes["d1-judgment"]


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
            "parser_name": "pymupdf+pymupdf4llm",
            "parser_version": "pymupdf=1.26.0; pymupdf4llm=0.0.27",
            "adapter_name": "pymupdf-sectionmap",
            "artifact_schema_version": PARSE_ARTIFACT_SCHEMA_VERSION,
            "config": {
                "layout_text_engine": "pymupdf4llm",
                "raw_character_stream_engine": "pymupdf",
            },
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


def _domain_answer(domain, sq_id, answer, *, support_level="strong"):
    return {
        "sq_id": sq_id,
        "answer": answer,
        "quote": "Participants and carers were aware.",
        "justification": "The selected packet supports this answer.",
        "support_level": support_level,
        "support_rationale": "Supported by selected packet evidence.",
        "uncertainty": support_level == "unsupported",
        "packet_artifact_id": f"evidence-packet:{domain}:{sq_id}",
        "decision_table_artifact_id": f"decision-table:{domain}:{sq_id}",
        "supporting_fact_artifact_ids": [],
    }


def _d1_workspace_state(primary):
    return {
        "source_documents": [_source_document(primary)],
        "parse_artifacts": [_parse_artifact(primary)],
        "outcome": "Overall survival",
        "outcome_type": "vital-status",
        "outcome_properties": {"death_only_objective_event": True},
        "outcome_classification_support": {"support_level": "strong"},
        "effect_of_interest": "ITT",
        "overall_policy": "rob2-default",
        "packet_readiness": {"d1": "ready"},
        "packet_grades": {"d1": {"grade": "usable", "missing_evidence": []}},
        "evidence_packets": {"d1": {"1.1": {"packet_grade": "usable"}}},
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
            "rationale": "Row: Any / NI / N-PN-NI -> Some concerns",
        },
        "domain_sq_classifier_artifacts": {
            "d1": {
                "sq": {
                    "schema_version": "d1-sq-classifier-v1",
                    "domain": "d1",
                    "stage": "sq",
                    "answers": [
                        _d1_answer("1.1", "Y"),
                        _d1_answer("1.2", "NI", support_level="unsupported"),
                        _d1_answer("1.3", "N"),
                    ],
                }
            }
        },
        "llm_call_log": [
            {
                "node": "domain1_sq_json",
                "provider": "openrouter",
                "model": "gpt-4.1",
                "prompt_version": "d1-sq-classifier-prompt-v1",
                "schema_version": "d1-sq-classifier-v1",
                "latency_ms": 12,
                "input_tokens": 100,
                "output_tokens": 50,
                "cost_usd": 0.0012,
                "parse_status": "parsed",
                "validation_status": "validated",
                "attempts": [{"attempt": 1}],
            }
        ],
    }
