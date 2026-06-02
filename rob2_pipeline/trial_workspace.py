from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


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


def _find_artifact(
    manifest: TrialWorkspaceManifest,
    artifact_id: str,
) -> ArtifactIdentity | None:
    for artifact in manifest.artifacts:
        if artifact.artifact_id == artifact_id:
            return artifact
    return None


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


__all__ = [
    "ArtifactIdentity",
    "ArtifactStatus",
    "SourceIdentity",
    "TRIAL_WORKSPACE_MANIFEST_SCHEMA_VERSION",
    "TrialWorkspaceManifest",
    "artifact_identity",
    "build_trial_workspace_manifest",
    "config_sha256",
    "content_sha256",
    "evaluate_artifact_status",
    "file_sha256",
    "read_trial_workspace_manifest",
    "write_trial_workspace_manifest",
]
