from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from rob2_pipeline.ingestion.parse_artifacts import (
    PAGE_AWARE_ARTIFACT_SCHEMA_VERSION,
    ParserDiagnostic,
    ParserProvenance,
    SourceParseArtifact,
    build_page_aware_artifacts,
)
from rob2_pipeline.types import SourceDocument


TRIAL_WORKSPACE_MANIFEST_SCHEMA_VERSION = "trial-workspace-manifest-v1"

ArtifactStatus = Literal["fresh", "reusable", "stale"]


@dataclass(frozen=True)
class SourceIdentity:
    document_id: str
    document_name: str
    document_role: str
    path: str
    content_hash: str

    @classmethod
    def from_path(
        cls,
        *,
        document_id: str,
        document_role: str,
        path: str | Path,
    ) -> SourceIdentity:
        source_path = Path(path)
        return cls(
            document_id=document_id,
            document_name=source_path.name,
            document_role=document_role,
            path=str(source_path),
            content_hash=file_sha256(source_path),
        )


@dataclass(frozen=True)
class ArtifactIdentity:
    artifact_id: str
    schema_version: str
    producer: str
    producer_version: str
    config_hash: str
    upstream_artifact_hashes: dict[str, str]
    content_hash: str
    status: ArtifactStatus = "fresh"


@dataclass(frozen=True)
class TrialWorkspaceManifest:
    trial_id: str
    sources: list[SourceIdentity]
    artifacts: list[ArtifactIdentity]
    manifest_schema_version: str = TRIAL_WORKSPACE_MANIFEST_SCHEMA_VERSION


@dataclass(frozen=True)
class LoadedTrialWorkspace:
    manifest: TrialWorkspaceManifest
    artifact_statuses: dict[str, ArtifactStatus]
    reusable_artifacts: dict[str, dict[str, dict]]
    stale_artifact_ids: list[str]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_sha256(content: bytes | str) -> str:
    payload = content.encode("utf-8") if isinstance(content, str) else content
    return hashlib.sha256(payload).hexdigest()


def config_sha256(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return content_sha256(payload)


def artifact_identity(
    *,
    artifact_id: str,
    schema_version: str,
    producer: str,
    producer_version: str,
    config_hash: str,
    upstream_artifact_hashes: dict[str, str],
    content_hash: str,
    status: ArtifactStatus = "fresh",
) -> ArtifactIdentity:
    return ArtifactIdentity(
        artifact_id=artifact_id,
        schema_version=schema_version,
        producer=producer,
        producer_version=producer_version,
        config_hash=config_hash,
        upstream_artifact_hashes=dict(sorted(upstream_artifact_hashes.items())),
        content_hash=content_hash,
        status=status,
    )


def build_trial_workspace_manifest(
    *,
    trial_id: str,
    sources: list[SourceIdentity],
    artifacts: list[ArtifactIdentity],
) -> TrialWorkspaceManifest:
    return TrialWorkspaceManifest(
        trial_id=trial_id,
        sources=sources,
        artifacts=artifacts,
    )


def evaluate_artifact_status(
    manifest: TrialWorkspaceManifest,
    current_identity: ArtifactIdentity,
) -> ArtifactStatus:
    previous = _find_artifact(manifest, current_identity.artifact_id)
    if previous is None:
        return "fresh"
    if _is_same_reusable_identity(previous, current_identity):
        return "reusable"
    return "stale"


def write_trial_workspace_manifest(
    manifest: TrialWorkspaceManifest,
    path: str | Path,
) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(_manifest_to_dict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_trial_workspace_manifest(path: str | Path) -> TrialWorkspaceManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return TrialWorkspaceManifest(
        manifest_schema_version=payload["manifest_schema_version"],
        trial_id=payload["trial_id"],
        sources=[SourceIdentity(**source) for source in payload.get("sources", [])],
        artifacts=[
            ArtifactIdentity(**artifact) for artifact in payload.get("artifacts", [])
        ],
    )


def write_parse_trial_workspace(
    *,
    trial_id: str,
    workspace_dir: str | Path,
    source_documents: list[SourceDocument],
    parse_artifacts: list[dict],
) -> TrialWorkspaceManifest:
    root = Path(workspace_dir)
    artifact_identities: list[ArtifactIdentity] = []
    sources = [_source_identity_from_document(source) for source in source_documents]
    source_hashes = {source.document_id: source.content_hash for source in sources}

    for source in source_documents:
        source_id = source.get("document_id", "")
        filename = _artifact_filename(source_id)
        _write_json(root / "sources" / filename, dict(source))

    for parse_artifact in parse_artifacts:
        source_id = parse_artifact["source_identity"]["document_id"]
        filename = _artifact_filename(source_id)
        parse_path = root / "parse_artifacts" / filename
        _write_json(parse_path, parse_artifact)
        artifact_identities.append(
            _file_artifact_identity(
                artifact_id=f"{source_id}:parse-artifact",
                schema_version=parse_artifact["provenance"]["artifact_schema_version"],
                producer=parse_artifact["provenance"]["parser_name"],
                producer_version=parse_artifact["provenance"]["parser_version"],
                config=parse_artifact["provenance"].get("config", {}),
                upstream_artifact_hashes={
                    f"source:{source_id}": source_hashes.get(source_id, "")
                },
                path=parse_path,
            )
        )

        page_artifacts = build_page_aware_artifacts(
            _source_parse_artifact_from_dict(parse_artifact)
        )
        page_path = root / "page_artifacts" / filename
        _write_json(page_path, page_artifacts.to_dict())
        artifact_identities.append(
            _file_artifact_identity(
                artifact_id=f"{source_id}:page-aware-artifacts",
                schema_version=page_artifacts.artifact_schema_version,
                producer=parse_artifact["provenance"]["parser_name"],
                producer_version=parse_artifact["provenance"]["parser_version"],
                config=parse_artifact["provenance"].get("config", {}),
                upstream_artifact_hashes={
                    f"{source_id}:parse-artifact": file_sha256(parse_path)
                },
                path=page_path,
            )
        )

        diagnostics = _parser_diagnostic_summary(parse_artifact)
        diagnostics_path = root / "diagnostics" / filename
        _write_json(diagnostics_path, diagnostics)
        artifact_identities.append(
            _file_artifact_identity(
                artifact_id=f"{source_id}:parser-diagnostics",
                schema_version="parser-diagnostics-v1",
                producer=parse_artifact["provenance"]["parser_name"],
                producer_version=parse_artifact["provenance"]["parser_version"],
                config=parse_artifact["provenance"].get("config", {}),
                upstream_artifact_hashes={
                    f"{source_id}:parse-artifact": file_sha256(parse_path)
                },
                path=diagnostics_path,
            )
        )

    manifest = build_trial_workspace_manifest(
        trial_id=trial_id,
        sources=sources,
        artifacts=sorted(
            artifact_identities,
            key=lambda artifact: artifact.artifact_id,
        ),
    )
    write_trial_workspace_manifest(manifest, root / "trial-workspace-manifest.json")
    return manifest


def load_trial_workspace_artifacts(workspace_dir: str | Path) -> dict[str, dict]:
    root = Path(workspace_dir)
    return {
        "sources": _load_artifact_directory(root / "sources"),
        "parse_artifacts": _load_artifact_directory(root / "parse_artifacts"),
        "page_artifacts": _load_artifact_directory(root / "page_artifacts"),
        "diagnostics": _load_artifact_directory(root / "diagnostics"),
    }


def load_parse_trial_workspace(
    *,
    workspace_dir: str | Path,
    source_documents: list[SourceDocument],
    parser_metadata: dict,
) -> LoadedTrialWorkspace:
    root = Path(workspace_dir)
    manifest = read_trial_workspace_manifest(root / "trial-workspace-manifest.json")
    current_sources = [
        _source_identity_from_document(source) for source in source_documents
    ]
    source_hashes = {source.document_id: source.content_hash for source in current_sources}
    artifact_statuses: dict[str, ArtifactStatus] = {}

    for source in current_sources:
        source_id = source.document_id
        parse_id = f"{source_id}:parse-artifact"
        parse_path = root / "parse_artifacts" / _artifact_filename(source_id)
        parse_status = _evaluate_existing_file_artifact(
            manifest=manifest,
            artifact_id=parse_id,
            schema_version=parser_metadata["artifact_schema_version"],
            producer=parser_metadata["parser_name"],
            producer_version=parser_metadata["parser_version"],
            config=parser_metadata.get("config", {}),
            upstream_artifact_hashes={f"source:{source_id}": source_hashes[source_id]},
            path=parse_path,
        )
        artifact_statuses[parse_id] = parse_status

        parse_hash = file_sha256(parse_path) if parse_path.exists() else ""
        for suffix, directory, schema_version in (
            ("page-aware-artifacts", "page_artifacts", PAGE_AWARE_ARTIFACT_SCHEMA_VERSION),
            ("parser-diagnostics", "diagnostics", "parser-diagnostics-v1"),
        ):
            artifact_id = f"{source_id}:{suffix}"
            path = root / directory / _artifact_filename(source_id)
            if parse_status != "reusable":
                artifact_statuses[artifact_id] = "stale"
                continue
            artifact_statuses[artifact_id] = _evaluate_existing_file_artifact(
                manifest=manifest,
                artifact_id=artifact_id,
                schema_version=schema_version,
                producer=parser_metadata["parser_name"],
                producer_version=parser_metadata["parser_version"],
                config=parser_metadata.get("config", {}),
                upstream_artifact_hashes={f"{source_id}:parse-artifact": parse_hash},
                path=path,
            )

    reusable_artifacts = load_trial_workspace_artifacts(root)
    _filter_reusable_artifacts(reusable_artifacts, artifact_statuses)
    stale_artifact_ids = [
        artifact_id
        for artifact_id, status in sorted(artifact_statuses.items())
        if status == "stale"
    ]
    return LoadedTrialWorkspace(
        manifest=manifest,
        artifact_statuses=artifact_statuses,
        reusable_artifacts=reusable_artifacts,
        stale_artifact_ids=stale_artifact_ids,
    )


def _find_artifact(
    manifest: TrialWorkspaceManifest,
    artifact_id: str,
) -> ArtifactIdentity | None:
    for artifact in manifest.artifacts:
        if artifact.artifact_id == artifact_id:
            return artifact
    return None


def _evaluate_existing_file_artifact(
    *,
    manifest: TrialWorkspaceManifest,
    artifact_id: str,
    schema_version: str,
    producer: str,
    producer_version: str,
    config: dict,
    upstream_artifact_hashes: dict[str, str],
    path: Path,
) -> ArtifactStatus:
    if not path.exists():
        return "stale"
    return evaluate_artifact_status(
        manifest,
        artifact_identity(
            artifact_id=artifact_id,
            schema_version=schema_version,
            producer=producer,
            producer_version=producer_version,
            config_hash=config_sha256(config),
            upstream_artifact_hashes=upstream_artifact_hashes,
            content_hash=file_sha256(path),
        ),
    )


def _filter_reusable_artifacts(
    artifacts: dict[str, dict[str, dict]],
    artifact_statuses: dict[str, ArtifactStatus],
) -> None:
    artifact_groups = {
        "parse_artifacts": "parse-artifact",
        "page_artifacts": "page-aware-artifacts",
        "diagnostics": "parser-diagnostics",
    }
    for group_name, artifact_suffix in artifact_groups.items():
        group = artifacts[group_name]
        for source_id in list(group):
            if artifact_statuses.get(f"{source_id}:{artifact_suffix}") != "reusable":
                del group[source_id]


def _is_same_reusable_identity(
    previous: ArtifactIdentity,
    current: ArtifactIdentity,
) -> bool:
    return (
        previous.schema_version == current.schema_version
        and previous.producer == current.producer
        and previous.producer_version == current.producer_version
        and previous.config_hash == current.config_hash
        and previous.upstream_artifact_hashes == current.upstream_artifact_hashes
        and previous.content_hash == current.content_hash
    )


def _manifest_to_dict(manifest: TrialWorkspaceManifest) -> dict:
    payload = asdict(manifest)
    payload["sources"] = sorted(
        payload["sources"], key=lambda item: item["document_id"]
    )
    payload["artifacts"] = sorted(
        payload["artifacts"], key=lambda item: item["artifact_id"]
    )
    return payload


def _source_identity_from_document(source: SourceDocument) -> SourceIdentity:
    path = Path(source.get("path", ""))
    if path.exists():
        content_hash = file_sha256(path)
    else:
        content_hash = ""
    return SourceIdentity(
        document_id=source.get("document_id", ""),
        document_name=source.get("document_name", path.name),
        document_role=source.get("document_role", ""),
        path=str(path),
        content_hash=content_hash,
    )


def _source_parse_artifact_from_dict(payload: dict) -> SourceParseArtifact:
    provenance = payload["provenance"]
    return SourceParseArtifact(
        source_identity=payload["source_identity"],
        pages=payload.get("pages", []),
        diagnostics=[
            ParserDiagnostic(**diagnostic)
            for diagnostic in payload.get("diagnostics", [])
        ],
        provenance=ParserProvenance(**provenance),
        parse_time_ms=payload.get("parse_time_ms", 0),
    )


def _parser_diagnostic_summary(parse_artifact: dict) -> dict:
    pages = parse_artifact.get("pages", [])
    provenance = parse_artifact["provenance"]
    return {
        "source_id": parse_artifact["source_identity"]["document_id"],
        "parse_time_ms": parse_artifact.get("parse_time_ms", 0),
        "page_count": len(pages),
        "text_character_count": sum(len(page.get("text", "")) for page in pages),
        "parser": {
            "name": provenance["parser_name"],
            "version": provenance["parser_version"],
            "adapter": provenance["adapter_name"],
        },
        "diagnostics": parse_artifact.get("diagnostics", []),
    }


def _file_artifact_identity(
    *,
    artifact_id: str,
    schema_version: str,
    producer: str,
    producer_version: str,
    config: dict,
    upstream_artifact_hashes: dict[str, str],
    path: Path,
) -> ArtifactIdentity:
    return artifact_identity(
        artifact_id=artifact_id,
        schema_version=schema_version,
        producer=producer,
        producer_version=producer_version,
        config_hash=config_sha256(config),
        upstream_artifact_hashes=upstream_artifact_hashes,
        content_hash=file_sha256(path),
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_artifact_directory(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    artifacts = {}
    for artifact_path in sorted(path.glob("*.json")):
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        source_id = (
            payload.get("source_id")
            or payload.get("source_identity", {}).get("document_id")
            or payload.get("document_id")
        )
        if not source_id:
            continue
        artifacts[source_id] = payload
    return artifacts


def _artifact_filename(artifact_id: str) -> str:
    return f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', artifact_id)}.json"


__all__ = [
    "ArtifactIdentity",
    "ArtifactStatus",
    "LoadedTrialWorkspace",
    "SourceIdentity",
    "TRIAL_WORKSPACE_MANIFEST_SCHEMA_VERSION",
    "TrialWorkspaceManifest",
    "artifact_identity",
    "build_trial_workspace_manifest",
    "config_sha256",
    "content_sha256",
    "evaluate_artifact_status",
    "file_sha256",
    "load_parse_trial_workspace",
    "load_trial_workspace_artifacts",
    "read_trial_workspace_manifest",
    "write_parse_trial_workspace",
    "write_trial_workspace_manifest",
]
