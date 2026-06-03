from __future__ import annotations

import json

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
                artifact_id="liteparse_document",
                schema_version="liteparse-document-v1",
                producer="liteparse",
                producer_version="2.0.4",
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
    assert payload["artifacts"][0]["artifact_id"] == "liteparse_document"
    assert payload["artifacts"][0]["status"] == "fresh"
    assert payload["artifacts"][0]["schema_version"] == "liteparse-document-v1"
    assert payload["artifacts"][0]["producer"] == "liteparse"
    assert payload["artifacts"][0]["producer_version"] == "2.0.4"
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
                artifact_id="liteparse_document",
                schema_version="liteparse-document-v1",
                producer="liteparse",
                producer_version="2.0.4",
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
            artifact_id="liteparse_document",
            schema_version="liteparse-document-v1",
            producer="liteparse",
            producer_version="2.0.4",
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
                artifact_id="liteparse_document",
                schema_version="liteparse-document-v1",
                producer="liteparse",
                producer_version="2.0.4",
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
            artifact_id="liteparse_document",
            schema_version="liteparse-document-v1",
            producer="liteparse",
            producer_version="2.0.4",
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
                artifact_id="liteparse_document",
                schema_version="liteparse-document-v1",
                producer="liteparse",
                producer_version="2.0.4",
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
            artifact_id="liteparse_document",
            schema_version="liteparse-document-v2",
            producer="liteparse",
            producer_version="2.0.4",
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
                artifact_id="liteparse_document",
                schema_version="liteparse-document-v1",
                producer="liteparse",
                producer_version="2.0.4",
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
            artifact_id="liteparse_document",
            schema_version="liteparse-document-v1",
            producer="liteparse",
            producer_version="2.0.4",
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
            "parser_name": "liteparse",
            "parser_version": "2.0.4",
            "adapter_name": "liteparse",
            "artifact_schema_version": "parse-artifact-v1",
            "config": {"ocr_enabled": False},
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
    assert loaded["page_artifacts"]["primary"]["sections"][0]["heading"] == "Methods"
    assert loaded["page_artifacts"]["primary"]["chunks"][0]["source_id"] == "primary"
    assert diagnostic == {
        "source_id": "primary",
        "parse_time_ms": 17,
        "page_count": 1,
        "text_character_count": 51,
        "parser": {
            "name": "liteparse",
            "version": "2.0.4",
            "adapter": "liteparse",
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
        parser_metadata={
            "parser_name": "liteparse",
            "parser_version": "2.0.4",
            "artifact_schema_version": "parse-artifact-v1",
            "config": {"ocr_enabled": False},
        },
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
            "parser_version": "2.0.5",
        },
    )

    assert loaded.artifact_statuses["primary:parse-artifact"] == "stale"
    assert loaded.artifact_statuses["primary:page-aware-artifacts"] == "stale"
    assert loaded.artifact_statuses["primary:parser-diagnostics"] == "stale"


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
        "provenance": {**_parser_metadata(), "adapter_name": "liteparse"},
    }


def _parser_metadata():
    return {
        "parser_name": "liteparse",
        "parser_version": "2.0.4",
        "artifact_schema_version": "parse-artifact-v1",
        "config": {"ocr_enabled": False},
    }
