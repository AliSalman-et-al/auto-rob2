from __future__ import annotations

import json
from pathlib import Path

from rob2_pipeline.ingestion.parse_artifacts import (
    ParserProvenance,
    SourceParseArtifact,
)
from rob2_pipeline.trial_workspace_cli import build_workspace, inspect_workspace, main


class FakeParser:
    producer = "fake-parser"
    producer_version = "1.2.3"
    config = {"ocr_enabled": False}

    def parse(self, path):
        pdf_path = Path(path)
        return SourceParseArtifact(
            source_identity={},
            pages=[
                {
                    "page_number": 1,
                    "text": f"Methods\nParsed text from {pdf_path.name}.",
                    "width": 612.0,
                    "height": 792.0,
                }
            ],
            diagnostics=[],
            provenance=ParserProvenance(
                parser_name=self.producer,
                parser_version=self.producer_version,
                adapter_name="fake",
                artifact_schema_version="parse-artifact-v1",
                config=dict(self.config),
            ),
        )


def test_build_workspace_from_primary_and_supplement_persists_manifest(tmp_path):
    primary = tmp_path / "trial.pdf"
    primary.write_bytes(b"primary pdf")
    protocol = tmp_path / "trial_protocol.pdf"
    protocol.write_bytes(b"protocol pdf")
    workspace_dir = tmp_path / "workspace"

    manifest = build_workspace(
        primary_pdf=primary,
        workspace_dir=workspace_dir,
        supplement_pdfs=[protocol],
        trial_id="trial-001",
        parser=FakeParser(),
    )

    manifest_payload = json.loads(
        (workspace_dir / "trial-workspace-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.trial_id == "trial-001"
    assert [source.document_id for source in manifest.sources] == [
        "primary",
        "supplement:001",
    ]
    assert manifest_payload["sources"][1]["document_role"] == "protocol"
    assert json.loads(
        (workspace_dir / "sources" / "supplement_001.json").read_text(
            encoding="utf-8"
        )
    )["status"] == "parsed"
    assert (workspace_dir / "parse_artifacts" / "primary.json").exists()
    assert (workspace_dir / "parse_artifacts" / "supplement_001.json").exists()
    assert (workspace_dir / "page_artifacts" / "primary.json").exists()
    assert (workspace_dir / "diagnostics" / "supplement_001.json").exists()


def test_inspect_workspace_reports_manifest_and_artifact_status(tmp_path):
    primary = tmp_path / "trial.pdf"
    primary.write_bytes(b"primary pdf")
    protocol = tmp_path / "trial_protocol.pdf"
    protocol.write_bytes(b"protocol pdf")
    workspace_dir = tmp_path / "workspace"
    build_workspace(
        primary_pdf=primary,
        workspace_dir=workspace_dir,
        supplement_pdfs=[protocol],
        trial_id="trial-001",
        parser=FakeParser(),
    )

    status = inspect_workspace(workspace_dir)

    assert status["trial_id"] == "trial-001"
    assert status["manifest_schema_version"] == "trial-workspace-manifest-v1"
    assert status["sources"] == [
        {
            "document_id": "primary",
            "document_name": "trial.pdf",
            "document_role": "primary",
            "status": "parsed",
        },
        {
            "document_id": "supplement:001",
            "document_name": "trial_protocol.pdf",
            "document_role": "protocol",
            "status": "parsed",
        },
    ]
    assert status["artifacts"] == [
        {
            "artifact_id": "primary:page-aware-artifacts",
            "status": "fresh",
            "exists": True,
        },
        {
            "artifact_id": "primary:parse-artifact",
            "status": "fresh",
            "exists": True,
        },
        {
            "artifact_id": "primary:parser-diagnostics",
            "status": "fresh",
            "exists": True,
        },
        {
            "artifact_id": "supplement:001:page-aware-artifacts",
            "status": "fresh",
            "exists": True,
        },
        {
            "artifact_id": "supplement:001:parse-artifact",
            "status": "fresh",
            "exists": True,
        },
        {
            "artifact_id": "supplement:001:parser-diagnostics",
            "status": "fresh",
            "exists": True,
        },
    ]


def test_inspect_command_prints_workspace_status_json(tmp_path, capsys):
    primary = tmp_path / "trial.pdf"
    primary.write_bytes(b"primary pdf")
    workspace_dir = tmp_path / "workspace"
    build_workspace(
        primary_pdf=primary,
        workspace_dir=workspace_dir,
        trial_id="trial-001",
        parser=FakeParser(),
    )

    exit_code = main(["inspect", str(workspace_dir)])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["trial_id"] == "trial-001"
    assert payload["artifacts"][0]["exists"] is True
