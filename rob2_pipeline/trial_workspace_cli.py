from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rob2_pipeline.ingestion.parse_artifacts import (
    LiteParseSourceParser,
    SourceParserAdapter,
    parse_sources,
)
from rob2_pipeline.ingestion.source_catalog import (
    primary_source_document,
    supplement_source_document,
)
from rob2_pipeline.trial_workspace import (
    TrialWorkspaceManifest,
    read_trial_workspace_manifest,
    write_parse_trial_workspace,
)


MANIFEST_FILENAME = "trial-workspace-manifest.json"


def build_workspace(
    *,
    primary_pdf: str | Path,
    workspace_dir: str | Path,
    supplement_pdfs: list[str | Path] | None = None,
    trial_id: str | None = None,
    parser: SourceParserAdapter | None = None,
) -> TrialWorkspaceManifest:
    primary_path = Path(primary_pdf)
    supplements = [Path(path) for path in supplement_pdfs or []]
    sources = [primary_source_document(primary_path)]
    sources.extend(
        supplement_source_document(path, index)
        for index, path in enumerate(supplements, start=1)
    )
    parsed_sources = parse_sources(sources, parser=parser)
    parse_artifacts = [artifact.to_dict() for artifact in parsed_sources]
    return write_parse_trial_workspace(
        trial_id=trial_id or primary_path.stem,
        workspace_dir=workspace_dir,
        source_documents=[artifact.source_identity for artifact in parsed_sources],
        parse_artifacts=parse_artifacts,
    )


def inspect_workspace(workspace_dir: str | Path) -> dict[str, Any]:
    root = Path(workspace_dir)
    manifest = read_trial_workspace_manifest(root / MANIFEST_FILENAME)
    source_statuses = _read_source_statuses(root)
    return {
        "trial_id": manifest.trial_id,
        "manifest_schema_version": manifest.manifest_schema_version,
        "sources": [
            {
                "document_id": source.document_id,
                "document_name": source.document_name,
                "document_role": source.document_role,
                "status": source_statuses.get(source.document_id, "unknown"),
            }
            for source in manifest.sources
        ],
        "artifacts": [
            {
                "artifact_id": artifact.artifact_id,
                "status": artifact.status,
                "exists": _artifact_path(root, artifact.artifact_id).exists(),
            }
            for artifact in sorted(
                manifest.artifacts, key=lambda item: item.artifact_id
            )
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.command == "build":
        manifest = build_workspace(
            primary_pdf=args.primary_pdf,
            workspace_dir=args.workspace_dir,
            supplement_pdfs=args.supplement,
            trial_id=args.trial_id,
            parser=LiteParseSourceParser(),
        )
        print(
            json.dumps(
                {
                    "trial_id": manifest.trial_id,
                    "workspace_dir": str(Path(args.workspace_dir)),
                    "manifest": str(Path(args.workspace_dir) / MANIFEST_FILENAME),
                    "source_count": len(manifest.sources),
                    "artifact_count": len(manifest.artifacts),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "inspect":
        print(json.dumps(inspect_workspace(args.workspace_dir), indent=2, sort_keys=True))
        return 0
    raise ValueError(f"Unknown command: {args.command}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or inspect parser-neutral Trial Workspace artifacts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build a Trial Workspace from PDFs.")
    build.add_argument("primary_pdf", help="Primary trial report PDF.")
    build.add_argument(
        "--workspace-dir",
        required=True,
        help="Directory where workspace artifacts will be written.",
    )
    build.add_argument(
        "--supplement",
        action="append",
        default=[],
        help="Supplement PDF path. Can be passed multiple times.",
    )
    build.add_argument(
        "--trial-id",
        default=None,
        help="Trial identifier for the workspace manifest. Defaults to primary PDF stem.",
    )

    inspect = subparsers.add_parser(
        "inspect", help="Inspect a Trial Workspace manifest and artifact status."
    )
    inspect.add_argument("workspace_dir", help="Workspace directory to inspect.")
    return parser


def _read_source_statuses(root: Path) -> dict[str, str]:
    statuses: dict[str, str] = {}
    sources_dir = root / "sources"
    if not sources_dir.exists():
        return statuses
    for source_path in sorted(sources_dir.glob("*.json")):
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        source_id = payload.get("document_id")
        if source_id:
            statuses[source_id] = payload.get("status", "unknown")
    return statuses


def _artifact_path(root: Path, artifact_id: str) -> Path:
    if artifact_id.startswith("evidence-store:"):
        return root / "evidence_store" / "facts.jsonl"
    source_id, artifact_kind = artifact_id.split(":", 1)
    filename = f"{source_id.replace(':', '_')}.json"
    directory_by_kind = {
        "parse-artifact": "parse_artifacts",
        "page-aware-artifacts": "page_artifacts",
        "parser-diagnostics": "diagnostics",
    }
    return root / directory_by_kind[artifact_kind] / filename


if __name__ == "__main__":
    raise SystemExit(main())
