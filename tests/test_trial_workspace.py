from __future__ import annotations

import json

from rob2_pipeline.ingestion.parse_artifacts import PARSE_ARTIFACT_SCHEMA_VERSION
from rob2_pipeline.trial_workspace import (
    TRIAL_WORKSPACE_MANIFEST_SCHEMA_VERSION,
    ArtifactIdentity,
    SourceIdentity,
    TrialWorkspaceManifest,
    artifact_identity,
    build_trial_workspace_manifest,
    evaluate_artifact_status,
    file_sha256,
    load_parse_trial_workspace,
    load_trial_workspace_artifacts,
    read_trial_workspace_manifest,
    write_evidence_store_trial_workspace,
    write_d1_judgment_workspace,
    write_domain_judgment_workspace,
    write_domain_sq_answer_workspace,
    write_outcome_normalization_workspace,
    write_parse_trial_workspace,
    write_trial_workspace_manifest,
)


def test_manifest_records_source_artifact_versions_and_hash_metadata(tmp_path):
    primary = tmp_path / "trial.pdf"
    primary.write_bytes(b"primary trial report")

    manifest = build_trial_workspace_manifest(
        trial_id="trial-001",
        sources=[
            SourceIdentity.from_path(
                document_id="primary",
                document_role="primary",
                path=primary,
            )
        ],
        artifacts=[
            artifact_identity(
                artifact_id="primary:parse-artifact",
                schema_version=PARSE_ARTIFACT_SCHEMA_VERSION,
                producer="pymupdf+pymupdf4llm",
                producer_version="pymupdf=1.26.0; pymupdf4llm=0.0.27",
                config_hash="config-a",
                upstream_artifact_hashes={"source:primary": file_sha256(primary)},
                content_hash="artifact-a",
            )
        ],
    )

    manifest_path = tmp_path / "trial-workspace-manifest.json"
    write_trial_workspace_manifest(manifest, manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["manifest_schema_version"] == TRIAL_WORKSPACE_MANIFEST_SCHEMA_VERSION
    assert payload["trial_id"] == "trial-001"
    assert payload["sources"][0] == {
        "document_id": "primary",
        "document_name": "trial.pdf",
        "document_role": "primary",
        "path": str(primary),
        "content_hash": file_sha256(primary),
    }
    assert payload["artifacts"][0]["artifact_id"] == "primary:parse-artifact"
    assert payload["artifacts"][0]["status"] == "fresh"
    assert payload["artifacts"][0]["schema_version"] == PARSE_ARTIFACT_SCHEMA_VERSION
    assert payload["artifacts"][0]["producer"] == "pymupdf+pymupdf4llm"
    assert (
        payload["artifacts"][0]["producer_version"]
        == "pymupdf=1.26.0; pymupdf4llm=0.0.27"
    )
    assert payload["artifacts"][0]["config_hash"] == "config-a"
    assert payload["artifacts"][0]["upstream_artifact_hashes"] == {
        "source:primary": file_sha256(primary)
    }
    assert read_trial_workspace_manifest(manifest_path) == manifest


def test_unchanged_artifact_identity_is_reusable(tmp_path):
    manifest = TrialWorkspaceManifest(
        trial_id="trial-001",
        sources=[],
        artifacts=[
            ArtifactIdentity(
                artifact_id="primary:parse-artifact",
                schema_version=PARSE_ARTIFACT_SCHEMA_VERSION,
                producer="pymupdf+pymupdf4llm",
                producer_version="pymupdf=1.26.0; pymupdf4llm=0.0.27",
                config_hash="config-a",
                upstream_artifact_hashes={"source:primary": "source-a"},
                content_hash="artifact-a",
                status="fresh",
            )
        ],
    )

    status = evaluate_artifact_status(
        manifest,
        artifact_identity(
            artifact_id="primary:parse-artifact",
            schema_version=PARSE_ARTIFACT_SCHEMA_VERSION,
            producer="pymupdf+pymupdf4llm",
            producer_version="pymupdf=1.26.0; pymupdf4llm=0.0.27",
            config_hash="config-a",
            upstream_artifact_hashes={"source:primary": "source-a"},
            content_hash="artifact-a",
        ),
    )

    assert status == "reusable"


def test_content_hash_change_marks_artifact_stale():
    manifest = TrialWorkspaceManifest(
        trial_id="trial-001",
        sources=[],
        artifacts=[
            ArtifactIdentity(
                artifact_id="primary:parse-artifact",
                schema_version=PARSE_ARTIFACT_SCHEMA_VERSION,
                producer="pymupdf+pymupdf4llm",
                producer_version="pymupdf=1.26.0; pymupdf4llm=0.0.27",
                config_hash="config-a",
                upstream_artifact_hashes={"source:primary": "source-a"},
                content_hash="artifact-a",
                status="fresh",
            )
        ],
    )

    status = evaluate_artifact_status(
        manifest,
        artifact_identity(
            artifact_id="primary:parse-artifact",
            schema_version=PARSE_ARTIFACT_SCHEMA_VERSION,
            producer="pymupdf+pymupdf4llm",
            producer_version="pymupdf=1.26.0; pymupdf4llm=0.0.27",
            config_hash="config-a",
            upstream_artifact_hashes={"source:primary": "source-b"},
            content_hash="artifact-a",
        ),
    )

    assert status == "stale"


def test_schema_version_change_marks_artifact_stale():
    manifest = TrialWorkspaceManifest(
        trial_id="trial-001",
        sources=[],
        artifacts=[
            ArtifactIdentity(
                artifact_id="primary:parse-artifact",
                schema_version=PARSE_ARTIFACT_SCHEMA_VERSION,
                producer="pymupdf+pymupdf4llm",
                producer_version="pymupdf=1.26.0; pymupdf4llm=0.0.27",
                config_hash="config-a",
                upstream_artifact_hashes={"source:primary": "source-a"},
                content_hash="artifact-a",
                status="fresh",
            )
        ],
    )

    status = evaluate_artifact_status(
        manifest,
        artifact_identity(
            artifact_id="primary:parse-artifact",
            schema_version="parse-artifact-v3",
            producer="pymupdf+pymupdf4llm",
            producer_version="pymupdf=1.26.0; pymupdf4llm=0.0.27",
            config_hash="config-a",
            upstream_artifact_hashes={"source:primary": "source-a"},
            content_hash="artifact-a",
        ),
    )

    assert status == "stale"


def test_config_hash_change_marks_artifact_stale():
    manifest = TrialWorkspaceManifest(
        trial_id="trial-001",
        sources=[],
        artifacts=[
            ArtifactIdentity(
                artifact_id="primary:parse-artifact",
                schema_version=PARSE_ARTIFACT_SCHEMA_VERSION,
                producer="pymupdf+pymupdf4llm",
                producer_version="pymupdf=1.26.0; pymupdf4llm=0.0.27",
                config_hash="config-a",
                upstream_artifact_hashes={"source:primary": "source-a"},
                content_hash="artifact-a",
                status="fresh",
            )
        ],
    )

    status = evaluate_artifact_status(
        manifest,
        artifact_identity(
            artifact_id="primary:parse-artifact",
            schema_version=PARSE_ARTIFACT_SCHEMA_VERSION,
            producer="pymupdf+pymupdf4llm",
            producer_version="pymupdf=1.26.0; pymupdf4llm=0.0.27",
            config_hash="config-b",
            upstream_artifact_hashes={"source:primary": "source-a"},
            content_hash="artifact-a",
        ),
    )

    assert status == "stale"


def test_parse_trial_workspace_persists_loadable_artifacts_and_diagnostics(tmp_path):
    primary = tmp_path / "trial.pdf"
    primary.write_bytes(b"primary trial report")
    parse_artifact = {
        "source_identity": {
            "document_id": "primary",
            "document_name": "trial.pdf",
            "document_role": "primary",
            "source_kind": "rag_chunk",
            "path": str(primary),
            "is_primary": True,
            "status": "parsed",
        },
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

    manifest = write_parse_trial_workspace(
        trial_id="trial-001",
        workspace_dir=tmp_path / "workspace",
        source_documents=[parse_artifact["source_identity"]],
        parse_artifacts=[parse_artifact],
    )

    loaded = load_trial_workspace_artifacts(tmp_path / "workspace")
    diagnostic = loaded["diagnostics"]["primary"]

    assert (tmp_path / "workspace" / "sources" / "primary.json").exists()
    assert loaded["parse_artifacts"]["primary"] == parse_artifact
    assert (
        loaded["page_artifacts"]["primary"]["sections"][0]["canonical_label"]
        == "METHODS"
    )
    assert (
        loaded["page_artifacts"]["primary"]["sections"][0]["original_heading"]
        == "Methods"
    )
    assert (
        loaded["page_artifacts"]["primary"]["chunks"][0]["original_heading"]
        == "Methods"
    )
    assert loaded["page_artifacts"]["primary"]["chunks"][0]["source_id"] == "primary"
    assert diagnostic == {
        "source_id": "primary",
        "parse_time_ms": 17,
        "page_count": 1,
        "text_character_count": 51,
        "parser": {
            "name": "pymupdf+pymupdf4llm",
            "version": "pymupdf=1.26.0; pymupdf4llm=0.0.27",
            "adapter": "pymupdf-sectionmap",
            "artifact_schema_version": PARSE_ARTIFACT_SCHEMA_VERSION,
        },
        "diagnostics": [],
    }
    assert (
        read_trial_workspace_manifest(
            tmp_path / "workspace" / "trial-workspace-manifest.json"
        )
        == manifest
    )
    assert {artifact.artifact_id for artifact in manifest.artifacts} == {
        "primary:parse-artifact",
        "primary:page-aware-artifacts",
        "primary:parser-diagnostics",
    }


def test_evidence_store_workspace_persists_jsonl_search_fields_and_hashes(tmp_path):
    parse_path = tmp_path / "workspace" / "parse_artifacts" / "primary.json"
    parse_path.parent.mkdir(parents=True)
    parse_path.write_text('{"parse": "artifact"}\n', encoding="utf-8")
    page_path = tmp_path / "workspace" / "page_artifacts" / "primary.json"
    page_path.parent.mkdir(parents=True)
    page_path.write_text('{"sections": [], "chunks": []}\n', encoding="utf-8")

    evidence_store = {
        "artifact_id": "evidence-store:TITAN:overall-survival",
        "schema_version": "1.0",
        "supported_facts": [
            {
                "artifact_id": "evidence-fact:d1:1.1:central-randomization",
                "fact_type": "randomization_sequence",
                "domain": "d1",
                "sq_ids": ["1.1"],
                "claim_type": "trial_method",
                "claim": "Participants were assigned centrally.",
                "quote": "Participants were assigned centrally.",
                "support_level": "strong",
                "support_status": "supported",
                "uncertainty": False,
                "provenance": {
                    "document_id": "primary:TITAN",
                    "document_name": "TITAN primary report",
                    "document_role": "primary",
                    "source_kind": "rag_chunk",
                    "source_path": "inputs/benchmark/TITAN.pdf",
                    "source_section": "Methods",
                    "page_numbers": [4],
                },
                "family": "randomization_allocation",
                "family_fields": {
                    "method": "central randomization",
                    "allocation_concealment": "central office",
                    "unit_of_randomization": "participant",
                },
            }
        ],
        "failed_claims": [],
        "gaps": [
            {
                "artifact_id": "evidence-gap:d3:3.1:denominator",
                "domain": "d3",
                "sq_ids": ["3.1"],
                "missing_evidence": "denominator_or_percentage",
                "reason": "No denominator was found.",
            }
        ],
    }

    manifest = write_evidence_store_trial_workspace(
        trial_id="trial-001",
        workspace_dir=tmp_path / "workspace",
        evidence_store=evidence_store,
        upstream_artifact_paths={
            "primary:parse-artifact": parse_path,
            "primary:page-aware-artifacts": page_path,
        },
        model_metadata={"provider": "openai", "model": "gpt-4.1"},
    )

    jsonl_path = tmp_path / "workspace" / "evidence_store" / "facts.jsonl"
    records = [
        json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()
    ]

    assert [record["record_kind"] for record in records] == ["fact", "gap"]
    assert records[0]["search_text"] == (
        "Participants were assigned centrally.\n"
        "Participants were assigned centrally.\n"
        "central randomization central office participant"
    )
    assert records[0]["embedding_text"] == records[0]["search_text"]
    artifact = next(
        item
        for item in manifest.artifacts
        if item.artifact_id == "evidence-store:TITAN:overall-survival"
    )
    assert artifact.schema_version == "1.0"
    assert artifact.producer == "evidence-family-mining"
    assert artifact.producer_version == "gpt-4.1"
    assert artifact.upstream_artifact_hashes == {
        "primary:page-aware-artifacts": file_sha256(page_path),
        "primary:parse-artifact": file_sha256(parse_path),
    }


def test_load_parse_trial_workspace_reuses_valid_existing_artifacts(tmp_path):
    primary = tmp_path / "trial.pdf"
    primary.write_bytes(b"primary trial report")
    source_document = _source_document(primary)
    parse_artifact = _parse_artifact(primary)
    workspace_dir = tmp_path / "workspace"
    write_parse_trial_workspace(
        trial_id="trial-001",
        workspace_dir=workspace_dir,
        source_documents=[source_document],
        parse_artifacts=[parse_artifact],
    )

    loaded = load_parse_trial_workspace(
        workspace_dir=workspace_dir,
        source_documents=[source_document],
        parser_metadata=_parser_metadata(),
    )

    assert loaded.artifact_statuses == {
        "primary:parse-artifact": "reusable",
        "primary:page-aware-artifacts": "reusable",
        "primary:parser-diagnostics": "reusable",
    }
    assert loaded.stale_artifact_ids == []
    assert loaded.reusable_artifacts["parse_artifacts"]["primary"] == parse_artifact


def test_load_parse_trial_workspace_marks_changed_source_dependents_stale(tmp_path):
    primary = tmp_path / "trial.pdf"
    primary.write_bytes(b"primary trial report")
    source_document = _source_document(primary)
    workspace_dir = tmp_path / "workspace"
    write_parse_trial_workspace(
        trial_id="trial-001",
        workspace_dir=workspace_dir,
        source_documents=[source_document],
        parse_artifacts=[_parse_artifact(primary)],
    )
    primary.write_bytes(b"updated primary trial report")

    loaded = load_parse_trial_workspace(
        workspace_dir=workspace_dir,
        source_documents=[source_document],
        parser_metadata=_parser_metadata(),
    )

    assert loaded.artifact_statuses == {
        "primary:parse-artifact": "stale",
        "primary:page-aware-artifacts": "stale",
        "primary:parser-diagnostics": "stale",
    }
    assert loaded.reusable_artifacts["parse_artifacts"] == {}
    assert loaded.stale_artifact_ids == [
        "primary:page-aware-artifacts",
        "primary:parse-artifact",
        "primary:parser-diagnostics",
    ]


def test_load_parse_trial_workspace_marks_parser_metadata_change_stale(tmp_path):
    primary = tmp_path / "trial.pdf"
    primary.write_bytes(b"primary trial report")
    source_document = _source_document(primary)
    workspace_dir = tmp_path / "workspace"
    write_parse_trial_workspace(
        trial_id="trial-001",
        workspace_dir=workspace_dir,
        source_documents=[source_document],
        parse_artifacts=[_parse_artifact(primary)],
    )

    loaded = load_parse_trial_workspace(
        workspace_dir=workspace_dir,
        source_documents=[source_document],
        parser_metadata={
            **_parser_metadata(),
            "parser_version": "pymupdf=1.26.1; pymupdf4llm=0.0.27",
        },
    )

    assert loaded.artifact_statuses["primary:parse-artifact"] == "stale"
    assert loaded.artifact_statuses["primary:page-aware-artifacts"] == "stale"
    assert loaded.artifact_statuses["primary:parser-diagnostics"] == "stale"


def test_load_parse_trial_workspace_marks_legacy_v1_schema_stale_for_v2_parser(tmp_path):
    primary = tmp_path / "trial.pdf"
    primary.write_bytes(b"primary trial report")
    source_document = _source_document(primary)
    workspace_dir = tmp_path / "workspace"
    legacy_parse_artifact = _parse_artifact(primary)
    legacy_parse_artifact["provenance"] = {
        **legacy_parse_artifact["provenance"],
        "artifact_schema_version": "parse-artifact-v1",
    }
    write_parse_trial_workspace(
        trial_id="trial-001",
        workspace_dir=workspace_dir,
        source_documents=[source_document],
        parse_artifacts=[legacy_parse_artifact],
    )

    loaded = load_parse_trial_workspace(
        workspace_dir=workspace_dir,
        source_documents=[source_document],
        parser_metadata=_parser_metadata(),
    )

    assert loaded.artifact_statuses == {
        "primary:parse-artifact": "stale",
        "primary:page-aware-artifacts": "stale",
        "primary:parser-diagnostics": "stale",
    }
    assert loaded.reusable_artifacts["parse_artifacts"] == {}


def test_load_parse_trial_workspace_reuses_only_unaffected_artifacts(tmp_path):
    primary = tmp_path / "trial.pdf"
    primary.write_bytes(b"primary trial report")
    protocol = tmp_path / "protocol.pdf"
    protocol.write_bytes(b"protocol report")
    primary_source = _source_document(primary)
    protocol_source = _source_document(
        protocol,
        document_id="protocol",
        document_role="protocol",
    )
    workspace_dir = tmp_path / "workspace"
    write_parse_trial_workspace(
        trial_id="trial-001",
        workspace_dir=workspace_dir,
        source_documents=[primary_source, protocol_source],
        parse_artifacts=[
            _parse_artifact(primary),
            _parse_artifact(
                protocol,
                document_id="protocol",
                document_role="protocol",
            ),
        ],
    )
    protocol.write_bytes(b"updated protocol report")

    loaded = load_parse_trial_workspace(
        workspace_dir=workspace_dir,
        source_documents=[primary_source, protocol_source],
        parser_metadata=_parser_metadata(),
    )

    assert loaded.artifact_statuses["primary:parse-artifact"] == "reusable"
    assert loaded.artifact_statuses["primary:page-aware-artifacts"] == "reusable"
    assert loaded.artifact_statuses["protocol:parse-artifact"] == "stale"
    assert loaded.artifact_statuses["protocol:page-aware-artifacts"] == "stale"
    assert set(loaded.reusable_artifacts["parse_artifacts"]) == {"primary"}


def test_outcome_workspace_manifest_records_trial_hashes_and_settings(tmp_path):
    trial_manifest_path = tmp_path / "trial_workspace" / "trial-workspace-manifest.json"
    trial_manifest_path.parent.mkdir(parents=True)
    trial_manifest_path.write_text('{"trial": "manifest"}\n', encoding="utf-8")
    page_path = tmp_path / "trial_workspace" / "page_artifacts" / "primary.json"
    page_path.parent.mkdir(parents=True)
    page_path.write_text('{"pages": []}\n', encoding="utf-8")

    from rob2_pipeline.trial_workspace import write_outcome_workspace_manifest

    manifest = write_outcome_workspace_manifest(
        trial_id="trial-001",
        outcome_id="overall-survival",
        workspace_root=tmp_path / "outcomes",
        trial_workspace_dir=tmp_path / "trial_workspace",
        upstream_artifact_paths={
            "trial-workspace-manifest": trial_manifest_path,
            "primary:page-aware-artifacts": page_path,
        },
        outcome_definition={"name": "Overall survival", "timepoint": "24 months"},
        rob2_settings={"effect_of_interest": "ITT", "outcome_type": "vital-status"},
    )

    payload = json.loads(
        (
            tmp_path
            / "outcomes"
            / "overall-survival"
            / "outcome-workspace-manifest.json"
        ).read_text(encoding="utf-8")
    )

    assert manifest.trial_id == "trial-001"
    assert manifest.outcome_id == "overall-survival"
    assert payload["upstream_trial_workspace_hashes"] == {
        "primary:page-aware-artifacts": file_sha256(page_path),
        "trial-workspace-manifest": file_sha256(trial_manifest_path),
    }
    assert payload["outcome_definition_hash"]
    assert payload["rob2_settings_hash"]
    assert payload["artifacts"] == []


def test_outcome_definition_or_rob2_settings_change_marks_outcome_artifact_stale(
    tmp_path,
):
    upstream = {"trial-workspace-manifest": "trial-hash-a"}
    from rob2_pipeline.trial_workspace import (
        build_outcome_artifact_identity,
        build_outcome_workspace_manifest,
        evaluate_outcome_artifact_status,
    )

    previous = build_outcome_workspace_manifest(
        trial_id="trial-001",
        outcome_id="overall-survival",
        trial_workspace_dir=tmp_path / "trial_workspace",
        upstream_trial_workspace_hashes=upstream,
        outcome_definition={"name": "Overall survival"},
        rob2_settings={"effect_of_interest": "ITT"},
        artifacts=[
            build_outcome_artifact_identity(
                artifact_id="evidence-packets",
                schema_version="evidence-packets-v1",
                producer="evidence-packet-builder",
                producer_version="1",
                content_hash="packets-a",
                upstream_trial_workspace_hashes=upstream,
                outcome_definition={"name": "Overall survival"},
                rob2_settings={"effect_of_interest": "ITT"},
            )
        ],
    )

    changed_outcome = build_outcome_artifact_identity(
        artifact_id="evidence-packets",
        schema_version="evidence-packets-v1",
        producer="evidence-packet-builder",
        producer_version="1",
        content_hash="packets-a",
        upstream_trial_workspace_hashes=upstream,
        outcome_definition={"name": "Progression-free survival"},
        rob2_settings={"effect_of_interest": "ITT"},
    )
    changed_settings = build_outcome_artifact_identity(
        artifact_id="evidence-packets",
        schema_version="evidence-packets-v1",
        producer="evidence-packet-builder",
        producer_version="1",
        content_hash="packets-a",
        upstream_trial_workspace_hashes=upstream,
        outcome_definition={"name": "Overall survival"},
        rob2_settings={"effect_of_interest": "per-protocol"},
    )

    assert evaluate_outcome_artifact_status(previous, changed_outcome) == "stale"
    assert evaluate_outcome_artifact_status(previous, changed_settings) == "stale"


def test_outcome_workspace_paths_are_separated_by_outcome_id(tmp_path):
    from rob2_pipeline.trial_workspace import outcome_workspace_dir

    root = tmp_path / "outcomes"

    assert outcome_workspace_dir(root, "Overall Survival") == root / "Overall_Survival"
    assert (
        outcome_workspace_dir(root, "Progression-Free Survival")
        == root / "Progression-Free_Survival"
    )


def test_outcome_normalization_workspace_persists_artifact_and_manifest_identity(
    tmp_path,
):
    trial_manifest_path = tmp_path / "trial_workspace" / "trial-workspace-manifest.json"
    trial_manifest_path.parent.mkdir(parents=True)
    trial_manifest_path.write_text('{"trial": "manifest"}\n', encoding="utf-8")

    artifact = {
        "artifact_id": "outcome-normalization:Overall survival",
        "schema_version": "outcome-normalization-v1",
        "outcome": "Overall survival",
        "normalized_definition": "Time from randomization to death.",
        "aliases": ["OS"],
        "outcome_type": "vital-status",
        "outcome_properties": {"objective_event": True},
        "binding_support": {
            "support_level": "weak",
            "support_rationale": "Only a partial quote supports the binding.",
            "quotes": [{"quote": "overall survival", "source": "results"}],
            "constraints": [],
        },
        "auto_accept_blocked": True,
        "uncertainty": True,
    }

    manifest = write_outcome_normalization_workspace(
        trial_id="trial-001",
        outcome_id="overall-survival",
        workspace_root=tmp_path / "outcomes",
        trial_workspace_dir=tmp_path / "trial_workspace",
        upstream_artifact_paths={"trial-workspace-manifest": trial_manifest_path},
        outcome_definition={"outcome": "Overall survival"},
        rob2_settings={"effect_of_interest": "ITT"},
        outcome_normalization_artifact=artifact,
        model_metadata={"model": "gpt-4.1"},
    )

    artifact_path = (
        tmp_path
        / "outcomes"
        / "overall-survival"
        / "outcome-normalization.json"
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert payload == artifact
    assert manifest.artifacts[0].artifact_id == "outcome-normalization:Overall survival"
    assert manifest.artifacts[0].schema_version == "outcome-normalization-v1"
    assert manifest.artifacts[0].producer == "outcome-resolver"
    assert manifest.artifacts[0].producer_version == "gpt-4.1"
    assert manifest.artifacts[0].content_hash == file_sha256(artifact_path)


def test_domain_sq_answer_workspace_persists_d1_loadable_artifact_and_contract_identity(
    tmp_path,
):
    trial_manifest_path = tmp_path / "trial_workspace" / "trial-workspace-manifest.json"
    trial_manifest_path.parent.mkdir(parents=True)
    trial_manifest_path.write_text('{"trial": "manifest"}\n', encoding="utf-8")
    packet_path = tmp_path / "outcomes" / "overall-survival" / "d1-packets.json"
    packet_path.parent.mkdir(parents=True)
    packet_path.write_text('{"packet": "v1"}\n', encoding="utf-8")

    artifact = {
        "artifact_id": "d1-sq-answer-set:overall-survival",
        "schema_version": "d1-sq-answer-set-v1",
        "classifier_schema_version": "d1-sq-classifier-v1",
        "classifier_prompt_version": "d1-sq-classifier-prompt-v1",
        "domain": "d1",
        "answers": [
            {
                "sq_id": "1.1",
                "answer": "Y",
                "quote": "computer-generated sequence",
                "justification": "The packet supports random sequence generation.",
                "support_level": "strong",
                "support_rationale": "Directly supported by the selected packet.",
                "uncertainty": False,
                "packet_artifact_id": "evidence-packet:d1:1.1",
                "decision_table_artifact_id": "decision-table:d1:1.1",
                "supporting_fact_artifact_ids": ["evidence-fact:d1:1.1:0"],
            },
            {
                "sq_id": "1.2",
                "answer": "NI",
                "quote": "No relevant text found",
                "justification": "The packet does not establish concealment.",
                "support_level": "unsupported",
                "support_rationale": "No selected fact supports allocation concealment.",
                "uncertainty": True,
                "packet_artifact_id": "evidence-packet:d1:1.2",
                "decision_table_artifact_id": "decision-table:d1:1.2",
                "supporting_fact_artifact_ids": [],
            },
            {
                "sq_id": "1.3",
                "answer": "N",
                "quote": "Baseline factors were balanced",
                "justification": "The packet supports no important baseline imbalance.",
                "support_level": "moderate",
                "support_rationale": "Supported by selected baseline evidence.",
                "uncertainty": False,
                "packet_artifact_id": "evidence-packet:d1:1.3",
                "decision_table_artifact_id": "decision-table:d1:1.3",
                "supporting_fact_artifact_ids": [],
            },
        ],
        "validation": {
            "status": "validated",
            "missing_support_metadata": [],
            "invalid_answers": [],
        },
    }

    manifest = write_domain_sq_answer_workspace(
        domain="d1",
        trial_id="trial-001",
        outcome_id="overall-survival",
        workspace_root=tmp_path / "outcomes",
        trial_workspace_dir=tmp_path / "trial_workspace",
        upstream_artifact_paths={
            "trial-workspace-manifest": trial_manifest_path,
            "d1:evidence-packets": packet_path,
        },
        outcome_definition={"outcome": "Overall survival"},
        rob2_settings={"effect_of_interest": "ITT"},
        sq_answer_artifact=artifact,
        model_metadata={"provider": "openai", "model": "gpt-4.1"},
        contract_metadata={
            "schema_version": "d1-sq-answer-set-v1",
            "classifier_schema_version": "d1-sq-classifier-v1",
            "classifier_prompt_version": "d1-sq-classifier-prompt-v1",
            "retry_policy": {"max_attempts": 2},
            "model_affecting_settings": {"temperature": 0},
        },
    )

    artifact_path = tmp_path / "outcomes" / "overall-survival" / "d1-sq-answers.json"
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    identity = manifest.artifacts[0]

    assert payload == artifact
    assert identity.artifact_id == "d1-sq-answer-set:overall-survival"
    assert identity.schema_version == "d1-sq-answer-set-v1"
    assert identity.producer == "d1-sq-classifier"
    assert identity.producer_version == "gpt-4.1"
    assert identity.upstream_trial_workspace_hashes["d1:evidence-packets"] == file_sha256(
        packet_path
    )
    assert identity.content_hash == file_sha256(artifact_path)

    changed_contract = write_domain_sq_answer_workspace(
        domain="d1",
        trial_id="trial-001",
        outcome_id="overall-survival",
        workspace_root=tmp_path / "other-outcomes",
        trial_workspace_dir=tmp_path / "trial_workspace",
        upstream_artifact_paths={
            "trial-workspace-manifest": trial_manifest_path,
            "d1:evidence-packets": packet_path,
        },
        outcome_definition={"outcome": "Overall survival"},
        rob2_settings={"effect_of_interest": "ITT"},
        sq_answer_artifact=artifact,
        model_metadata={"provider": "openai", "model": "gpt-4.1"},
        contract_metadata={
            "schema_version": "d1-sq-answer-set-v1",
            "classifier_schema_version": "d1-sq-classifier-v2",
            "classifier_prompt_version": "d1-sq-classifier-prompt-v1",
            "retry_policy": {"max_attempts": 2},
            "model_affecting_settings": {"temperature": 0},
        },
    )

    assert changed_contract.artifacts[0].config_hash != identity.config_hash


def test_domain_sq_answer_workspace_records_d1_invalid_answers_and_missing_support_metadata(
    tmp_path,
):
    trial_manifest_path = tmp_path / "trial_workspace" / "trial-workspace-manifest.json"
    trial_manifest_path.parent.mkdir(parents=True)
    trial_manifest_path.write_text('{"trial": "manifest"}\n', encoding="utf-8")

    write_domain_sq_answer_workspace(
        domain="d1",
        trial_id="trial-001",
        outcome_id="overall-survival",
        workspace_root=tmp_path / "outcomes",
        trial_workspace_dir=tmp_path / "trial_workspace",
        upstream_artifact_paths={"trial-workspace-manifest": trial_manifest_path},
        outcome_definition={"outcome": "Overall survival"},
        rob2_settings={"effect_of_interest": "ITT"},
        sq_answer_artifact={
            "artifact_id": "d1-sq-answer-set:overall-survival",
            "schema_version": "d1-sq-answer-set-v1",
            "domain": "d1",
            "answers": [
                {
                    "sq_id": "1.1",
                    "answer": "YES",
                    "quote": "computer-generated sequence",
                    "justification": "Invalid answer code should be visible.",
                    "support_level": "strong",
                    "support_rationale": "Supported.",
                    "packet_artifact_id": "evidence-packet:d1:1.1",
                    "decision_table_artifact_id": "decision-table:d1:1.1",
                },
                {
                    "sq_id": "1.2",
                    "answer": "NI",
                    "quote": "No relevant text found",
                    "justification": "Support metadata is incomplete.",
                },
            ],
        },
        model_metadata={"model": "gpt-4.1"},
        contract_metadata={
            "schema_version": "d1-sq-answer-set-v1",
            "classifier_schema_version": "d1-sq-classifier-v1",
            "classifier_prompt_version": "d1-sq-classifier-prompt-v1",
            "retry_policy": {"max_attempts": 2},
        },
    )

    payload = json.loads(
        (
            tmp_path
            / "outcomes"
            / "overall-survival"
            / "d1-sq-answers.json"
        ).read_text(encoding="utf-8")
    )

    assert payload["validation"]["status"] == "invalid"
    assert payload["validation"]["invalid_answers"] == [
        {
            "sq_id": "1.1",
            "answer": "YES",
            "reason": "Answer is not a canonical RoB 2 SQ answer.",
        }
    ]
    assert payload["validation"]["missing_support_metadata"] == [
        {
            "sq_id": "1.2",
            "missing_fields": [
                "decision_table_artifact_id",
                "packet_artifact_id",
                "support_level",
                "support_rationale",
            ],
        }
    ]


def test_d1_judgment_workspace_persists_artifact_and_invalidates_on_versions(tmp_path):
    trial_manifest_path = tmp_path / "trial_workspace" / "trial-workspace-manifest.json"
    trial_manifest_path.parent.mkdir(parents=True)
    trial_manifest_path.write_text('{"trial": "manifest"}\n', encoding="utf-8")
    sq_answer_path = tmp_path / "outcomes" / "overall-survival" / "d1-sq-answers.json"
    sq_answer_path.parent.mkdir(parents=True)
    sq_answer_path.write_text('{"answers": []}\n', encoding="utf-8")
    artifact = {
        "artifact_id": "d1-judgment:overall-survival",
        "schema_version": "d1-judgment-v1",
        "domain": "d1",
        "judge_version": "d1-judge-v1",
        "rule_table_version": "rob2-d1-rule-table-v1",
        "input_sq_answers": {
            "1.1": {"answer": "Y"},
            "1.2": {"answer": "Y"},
            "1.3": {"answer": "N"},
        },
        "applied_rule_path": "d1-row-1:y-py-ni/y-py/ni-n-pn",
        "label": "Low",
        "rationale": "Row: Y-PY-NI / Y-PY / NI-N-PN -> Low",
    }

    manifest = write_d1_judgment_workspace(
        trial_id="trial-001",
        outcome_id="overall-survival",
        workspace_root=tmp_path / "outcomes",
        trial_workspace_dir=tmp_path / "trial_workspace",
        upstream_artifact_paths={
            "trial-workspace-manifest": trial_manifest_path,
            "d1-sq-answer-set": sq_answer_path,
        },
        outcome_definition={"outcome": "Overall survival"},
        rob2_settings={"effect_of_interest": "ITT"},
        d1_judgment_artifact=artifact,
    )

    artifact_path = tmp_path / "outcomes" / "overall-survival" / "d1-judgment.json"
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    identity = manifest.artifacts[0]

    assert payload == artifact
    assert identity.artifact_id == "d1-judgment:overall-survival"
    assert identity.schema_version == "d1-judgment-v1"
    assert identity.producer == "d1-deterministic-judge"
    assert identity.producer_version == "d1-judge-v1"
    assert identity.upstream_trial_workspace_hashes["d1-sq-answer-set"] == file_sha256(
        sq_answer_path
    )
    assert identity.content_hash == file_sha256(artifact_path)

    changed_versions = write_d1_judgment_workspace(
        trial_id="trial-001",
        outcome_id="overall-survival",
        workspace_root=tmp_path / "other-outcomes",
        trial_workspace_dir=tmp_path / "trial_workspace",
        upstream_artifact_paths={
            "trial-workspace-manifest": trial_manifest_path,
            "d1-sq-answer-set": sq_answer_path,
        },
        outcome_definition={"outcome": "Overall survival"},
        rob2_settings={"effect_of_interest": "ITT"},
        d1_judgment_artifact={
            **artifact,
            "judge_version": "d1-judge-v2",
            "rule_table_version": "rob2-d1-rule-table-v2",
        },
    )

    assert changed_versions.artifacts[0].config_hash != identity.config_hash
    assert changed_versions.artifacts[0].producer_version == "d1-judge-v2"


def test_d2_sq_answer_workspace_persists_artifact_and_contract_identity(tmp_path):
    trial_manifest_path = tmp_path / "trial_workspace" / "trial-workspace-manifest.json"
    trial_manifest_path.parent.mkdir(parents=True)
    trial_manifest_path.write_text('{"trial": "manifest"}\n', encoding="utf-8")
    packet_path = tmp_path / "outcomes" / "overall-survival" / "d2-packets.json"
    packet_path.parent.mkdir(parents=True)
    packet_path.write_text('{"packet": "v1"}\n', encoding="utf-8")

    artifact = {
        "artifact_id": "d2-sq-answer-set:overall-survival",
        "schema_version": "d2-sq-answer-set-v1",
        "classifier_schema_version": "d2-sq-classifier-v1",
        "classifier_prompt_version": "d2-sq12-classifier-prompt-v1",
        "domain": "d2",
        "stage": "sq12",
        "branching": {"effect_of_interest": "ITT"},
        "answers": [
            {
                "sq_id": "2.1",
                "answer": "Y",
                "quote": "Participants and carers were aware.",
                "justification": "The selected packet supports awareness.",
                "support_level": "moderate",
                "support_rationale": "Directly supported by the selected packet.",
                "uncertainty": False,
                "packet_artifact_id": "evidence-packet:d2:2.1",
                "decision_table_artifact_id": "decision-table:d2:2.1",
                "supporting_fact_artifact_ids": ["evidence-fact:d2:2.1:0"],
            }
        ],
        "validation": {"status": "validated"},
    }

    manifest = write_domain_sq_answer_workspace(
        domain="d2",
        trial_id="trial-001",
        outcome_id="overall-survival",
        workspace_root=tmp_path / "outcomes",
        trial_workspace_dir=tmp_path / "trial_workspace",
        upstream_artifact_paths={
            "trial-workspace-manifest": trial_manifest_path,
            "d2:evidence-packets": packet_path,
        },
        outcome_definition={"outcome": "Overall survival"},
        rob2_settings={"effect_of_interest": "ITT"},
        sq_answer_artifact=artifact,
        model_metadata={"model": "gpt-4.1"},
        contract_metadata={
            "schema_version": "d2-sq-answer-set-v1",
            "classifier_schema_version": "d2-sq-classifier-v1",
            "classifier_prompt_version": "d2-sq12-classifier-prompt-v1",
            "retry_policy": {"max_attempts": 2},
        },
    )

    artifact_path = tmp_path / "outcomes" / "overall-survival" / "d2-sq-answers.json"
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    identity = manifest.artifacts[0]

    assert payload == {
        **artifact,
        "validation": {
            "status": "validated",
            "invalid_answers": [],
            "missing_support_metadata": [],
        },
    }
    assert identity.artifact_id == "d2-sq-answer-set:overall-survival"
    assert identity.schema_version == "d2-sq-answer-set-v1"
    assert identity.producer == "d2-sq-classifier"
    assert identity.producer_version == "gpt-4.1"
    assert identity.upstream_trial_workspace_hashes["d2:evidence-packets"] == file_sha256(
        packet_path
    )
    assert identity.content_hash == file_sha256(artifact_path)


def test_d2_judgment_workspace_persists_artifact_and_invalidates_on_versions(tmp_path):
    trial_manifest_path = tmp_path / "trial_workspace" / "trial-workspace-manifest.json"
    trial_manifest_path.parent.mkdir(parents=True)
    trial_manifest_path.write_text('{"trial": "manifest"}\n', encoding="utf-8")
    sq_answer_path = tmp_path / "outcomes" / "overall-survival" / "d2-sq-answers.json"
    sq_answer_path.parent.mkdir(parents=True)
    sq_answer_path.write_text('{"answers": []}\n', encoding="utf-8")
    artifact = {
        "artifact_id": "d2-judgment:overall-survival",
        "schema_version": "d2-judgment-v1",
        "domain": "d2",
        "judge_version": "d2-judge-v1",
        "rule_table_version": "rob2-d2-assignment-rule-table-v1",
        "input_sq_answers": {"2.1": {"answer": "N"}},
        "applied_rule_path": "d2-assignment:part1-low+part2-low",
        "label": "Low",
        "rationale": "Part1=Low; Part2=Low",
    }

    manifest = write_domain_judgment_workspace(
        domain="d2",
        trial_id="trial-001",
        outcome_id="overall-survival",
        workspace_root=tmp_path / "outcomes",
        trial_workspace_dir=tmp_path / "trial_workspace",
        upstream_artifact_paths={
            "trial-workspace-manifest": trial_manifest_path,
            "d2-sq-answer-set": sq_answer_path,
        },
        outcome_definition={"outcome": "Overall survival"},
        rob2_settings={"effect_of_interest": "ITT"},
        judgment_artifact=artifact,
    )

    artifact_path = tmp_path / "outcomes" / "overall-survival" / "d2-judgment.json"
    identity = manifest.artifacts[0]

    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert identity.artifact_id == "d2-judgment:overall-survival"
    assert identity.schema_version == "d2-judgment-v1"
    assert identity.producer == "d2-deterministic-judge"
    assert identity.producer_version == "d2-judge-v1"
    assert identity.upstream_trial_workspace_hashes["d2-sq-answer-set"] == file_sha256(
        sq_answer_path
    )

    changed_versions = write_domain_judgment_workspace(
        domain="d2",
        trial_id="trial-001",
        outcome_id="overall-survival",
        workspace_root=tmp_path / "other-outcomes",
        trial_workspace_dir=tmp_path / "trial_workspace",
        upstream_artifact_paths={
            "trial-workspace-manifest": trial_manifest_path,
            "d2-sq-answer-set": sq_answer_path,
        },
        outcome_definition={"outcome": "Overall survival"},
        rob2_settings={"effect_of_interest": "ITT"},
        judgment_artifact={
            **artifact,
            "judge_version": "d2-judge-v2",
            "rule_table_version": "rob2-d2-assignment-rule-table-v2",
        },
    )

    assert changed_versions.artifacts[0].config_hash != identity.config_hash
    assert changed_versions.artifacts[0].producer_version == "d2-judge-v2"


def _source_document(path, *, document_id="primary", document_role="primary"):
    return {
        "document_id": document_id,
        "document_name": path.name,
        "document_role": document_role,
        "source_kind": "rag_chunk",
        "path": str(path),
        "is_primary": document_role == "primary",
        "status": "parsed",
    }


def _parse_artifact(path, *, document_id="primary", document_role="primary"):
    return {
        "source_identity": _source_document(
            path,
            document_id=document_id,
            document_role=document_role,
        ),
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
        "provenance": {**_parser_metadata(), "adapter_name": "pymupdf-sectionmap"},
    }


def _parser_metadata():
    return {
        "parser_name": "pymupdf+pymupdf4llm",
        "parser_version": "pymupdf=1.26.0; pymupdf4llm=0.0.27",
        "artifact_schema_version": PARSE_ARTIFACT_SCHEMA_VERSION,
        "config": {
            "layout_text_engine": "pymupdf4llm",
            "raw_character_stream_engine": "pymupdf",
        },
    }
