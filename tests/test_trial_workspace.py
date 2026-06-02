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
    load_trial_workspace_artifacts,
    read_trial_workspace_manifest,
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
    assert read_trial_workspace_manifest(
        tmp_path / "workspace" / "trial-workspace-manifest.json"
    ) == manifest
    assert {artifact.artifact_id for artifact in manifest.artifacts} == {
        "primary:parse-artifact",
        "primary:page-aware-artifacts",
        "primary:parser-diagnostics",
    }
