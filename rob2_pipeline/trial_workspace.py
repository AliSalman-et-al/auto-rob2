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
from rob2_pipeline.evidence_store import EvidenceStore
from rob2_pipeline.types import SourceDocument


TRIAL_WORKSPACE_MANIFEST_SCHEMA_VERSION = "trial-workspace-manifest-v1"
OUTCOME_WORKSPACE_MANIFEST_SCHEMA_VERSION = "outcome-workspace-manifest-v1"

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


@dataclass(frozen=True)
class OutcomeArtifactIdentity:
    artifact_id: str
    schema_version: str
    producer: str
    producer_version: str
    config_hash: str
    upstream_trial_workspace_hashes: dict[str, str]
    outcome_definition_hash: str
    rob2_settings_hash: str
    content_hash: str
    status: ArtifactStatus = "fresh"


@dataclass(frozen=True)
class OutcomeWorkspaceManifest:
    trial_id: str
    outcome_id: str
    trial_workspace_dir: str
    upstream_trial_workspace_hashes: dict[str, str]
    outcome_definition_hash: str
    rob2_settings_hash: str
    artifacts: list[OutcomeArtifactIdentity]
    manifest_schema_version: str = OUTCOME_WORKSPACE_MANIFEST_SCHEMA_VERSION


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


def stable_payload_sha256(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return content_sha256(raw)


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


def build_outcome_workspace_manifest(
    *,
    trial_id: str,
    outcome_id: str,
    trial_workspace_dir: str | Path,
    upstream_trial_workspace_hashes: dict[str, str],
    outcome_definition: dict,
    rob2_settings: dict,
    artifacts: list[OutcomeArtifactIdentity],
) -> OutcomeWorkspaceManifest:
    return OutcomeWorkspaceManifest(
        trial_id=trial_id,
        outcome_id=outcome_id,
        trial_workspace_dir=str(trial_workspace_dir),
        upstream_trial_workspace_hashes=dict(
            sorted(upstream_trial_workspace_hashes.items())
        ),
        outcome_definition_hash=stable_payload_sha256(outcome_definition),
        rob2_settings_hash=stable_payload_sha256(rob2_settings),
        artifacts=artifacts,
    )


def build_outcome_artifact_identity(
    *,
    artifact_id: str,
    schema_version: str,
    producer: str,
    producer_version: str,
    content_hash: str,
    upstream_trial_workspace_hashes: dict[str, str],
    outcome_definition: dict,
    rob2_settings: dict,
    config: dict | None = None,
    status: ArtifactStatus = "fresh",
) -> OutcomeArtifactIdentity:
    return OutcomeArtifactIdentity(
        artifact_id=artifact_id,
        schema_version=schema_version,
        producer=producer,
        producer_version=producer_version,
        config_hash=config_sha256(config or {}),
        upstream_trial_workspace_hashes=dict(
            sorted(upstream_trial_workspace_hashes.items())
        ),
        outcome_definition_hash=stable_payload_sha256(outcome_definition),
        rob2_settings_hash=stable_payload_sha256(rob2_settings),
        content_hash=content_hash,
        status=status,
    )


def evaluate_outcome_artifact_status(
    manifest: OutcomeWorkspaceManifest,
    current_identity: OutcomeArtifactIdentity,
) -> ArtifactStatus:
    previous = _find_outcome_artifact(manifest, current_identity.artifact_id)
    if previous is None:
        return "fresh"
    if previous == current_identity or _is_same_outcome_reusable_identity(
        previous, current_identity
    ):
        return "reusable"
    return "stale"


def outcome_workspace_dir(workspace_root: str | Path, outcome_id: str) -> Path:
    return Path(workspace_root) / _artifact_filename(outcome_id).removesuffix(".json")


def write_outcome_workspace_manifest(
    *,
    trial_id: str,
    outcome_id: str,
    workspace_root: str | Path,
    trial_workspace_dir: str | Path,
    upstream_artifact_paths: dict[str, str | Path],
    outcome_definition: dict,
    rob2_settings: dict,
    artifacts: list[OutcomeArtifactIdentity] | None = None,
) -> OutcomeWorkspaceManifest:
    upstream_hashes = {
        artifact_id: file_sha256(path)
        for artifact_id, path in sorted(upstream_artifact_paths.items())
    }
    manifest = build_outcome_workspace_manifest(
        trial_id=trial_id,
        outcome_id=outcome_id,
        trial_workspace_dir=trial_workspace_dir,
        upstream_trial_workspace_hashes=upstream_hashes,
        outcome_definition=outcome_definition,
        rob2_settings=rob2_settings,
        artifacts=artifacts or [],
    )
    manifest_path = (
        outcome_workspace_dir(workspace_root, outcome_id)
        / "outcome-workspace-manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(_manifest_to_dict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def write_outcome_normalization_workspace(
    *,
    trial_id: str,
    outcome_id: str,
    workspace_root: str | Path,
    trial_workspace_dir: str | Path,
    upstream_artifact_paths: dict[str, str | Path],
    outcome_definition: dict,
    rob2_settings: dict,
    outcome_normalization_artifact: dict,
    model_metadata: dict,
) -> OutcomeWorkspaceManifest:
    root = outcome_workspace_dir(workspace_root, outcome_id)
    artifact_path = root / "outcome-normalization.json"
    _write_json(artifact_path, outcome_normalization_artifact)

    upstream_hashes = {
        artifact_id: file_sha256(path)
        for artifact_id, path in sorted(upstream_artifact_paths.items())
    }
    model_name = str(model_metadata.get("model") or "unknown-model")
    identity = build_outcome_artifact_identity(
        artifact_id=outcome_normalization_artifact["artifact_id"],
        schema_version=outcome_normalization_artifact["schema_version"],
        producer="outcome-resolver",
        producer_version=model_name,
        content_hash=file_sha256(artifact_path),
        upstream_trial_workspace_hashes=upstream_hashes,
        outcome_definition=outcome_definition,
        rob2_settings=rob2_settings,
        config={
            "schema_version": outcome_normalization_artifact["schema_version"],
            "model_metadata": model_metadata,
        },
    )
    return write_outcome_workspace_manifest(
        trial_id=trial_id,
        outcome_id=outcome_id,
        workspace_root=workspace_root,
        trial_workspace_dir=trial_workspace_dir,
        upstream_artifact_paths=upstream_artifact_paths,
        outcome_definition=outcome_definition,
        rob2_settings=rob2_settings,
        artifacts=[identity],
    )


def write_d1_sq_answer_workspace(
    *,
    trial_id: str,
    outcome_id: str,
    workspace_root: str | Path,
    trial_workspace_dir: str | Path,
    upstream_artifact_paths: dict[str, str | Path],
    outcome_definition: dict,
    rob2_settings: dict,
    d1_sq_answer_artifact: dict,
    model_metadata: dict,
    contract_metadata: dict,
) -> OutcomeWorkspaceManifest:
    root = outcome_workspace_dir(workspace_root, outcome_id)
    artifact_path = root / "d1-sq-answers.json"
    artifact = _d1_sq_answer_artifact_with_validation(d1_sq_answer_artifact)
    _write_json(artifact_path, artifact)

    upstream_hashes = {
        artifact_id: file_sha256(path)
        for artifact_id, path in sorted(upstream_artifact_paths.items())
    }
    model_name = str(model_metadata.get("model") or "unknown-model")
    schema_version = str(
        contract_metadata.get("schema_version")
        or artifact.get("schema_version")
        or "d1-sq-answer-set-v1"
    )
    identity = build_outcome_artifact_identity(
        artifact_id=artifact.get("artifact_id", f"d1-sq-answer-set:{outcome_id}"),
        schema_version=schema_version,
        producer="d1-sq-classifier",
        producer_version=model_name,
        content_hash=file_sha256(artifact_path),
        upstream_trial_workspace_hashes=upstream_hashes,
        outcome_definition=outcome_definition,
        rob2_settings=rob2_settings,
        config={
            "contract_metadata": contract_metadata,
            "model_metadata": model_metadata,
        },
    )
    artifacts = _existing_outcome_artifacts(workspace_root, outcome_id, identity)
    return write_outcome_workspace_manifest(
        trial_id=trial_id,
        outcome_id=outcome_id,
        workspace_root=workspace_root,
        trial_workspace_dir=trial_workspace_dir,
        upstream_artifact_paths=upstream_artifact_paths,
        outcome_definition=outcome_definition,
        rob2_settings=rob2_settings,
        artifacts=[*artifacts, identity],
    )


def write_d1_judgment_workspace(
    *,
    trial_id: str,
    outcome_id: str,
    workspace_root: str | Path,
    trial_workspace_dir: str | Path,
    upstream_artifact_paths: dict[str, str | Path],
    outcome_definition: dict,
    rob2_settings: dict,
    d1_judgment_artifact: dict,
) -> OutcomeWorkspaceManifest:
    root = outcome_workspace_dir(workspace_root, outcome_id)
    artifact_path = root / "d1-judgment.json"
    _write_json(artifact_path, d1_judgment_artifact)

    upstream_hashes = {
        artifact_id: file_sha256(path)
        for artifact_id, path in sorted(upstream_artifact_paths.items())
        if Path(path).exists()
    }
    identity = build_outcome_artifact_identity(
        artifact_id=d1_judgment_artifact.get("artifact_id", f"d1-judgment:{outcome_id}"),
        schema_version=d1_judgment_artifact.get("schema_version", "d1-judgment-v1"),
        producer="d1-deterministic-judge",
        producer_version=d1_judgment_artifact.get("judge_version", "unknown-judge"),
        content_hash=file_sha256(artifact_path),
        upstream_trial_workspace_hashes=upstream_hashes,
        outcome_definition=outcome_definition,
        rob2_settings=rob2_settings,
        config={
            "judge_version": d1_judgment_artifact.get("judge_version", ""),
            "rule_table_version": d1_judgment_artifact.get("rule_table_version", ""),
            "schema_version": d1_judgment_artifact.get("schema_version", ""),
        },
    )
    artifacts = _existing_outcome_artifacts(workspace_root, outcome_id, identity)
    return write_outcome_workspace_manifest(
        trial_id=trial_id,
        outcome_id=outcome_id,
        workspace_root=workspace_root,
        trial_workspace_dir=trial_workspace_dir,
        upstream_artifact_paths=upstream_artifact_paths,
        outcome_definition=outcome_definition,
        rob2_settings=rob2_settings,
        artifacts=[*artifacts, identity],
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


def read_outcome_workspace_manifest(path: str | Path) -> OutcomeWorkspaceManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return OutcomeWorkspaceManifest(
        manifest_schema_version=payload["manifest_schema_version"],
        trial_id=payload["trial_id"],
        outcome_id=payload["outcome_id"],
        trial_workspace_dir=payload["trial_workspace_dir"],
        upstream_trial_workspace_hashes=payload.get(
            "upstream_trial_workspace_hashes", {}
        ),
        outcome_definition_hash=payload["outcome_definition_hash"],
        rob2_settings_hash=payload["rob2_settings_hash"],
        artifacts=[
            OutcomeArtifactIdentity(**artifact)
            for artifact in payload.get("artifacts", [])
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


def write_evidence_store_trial_workspace(
    *,
    trial_id: str,
    workspace_dir: str | Path,
    evidence_store: dict,
    upstream_artifact_paths: dict[str, str | Path],
    model_metadata: dict,
) -> TrialWorkspaceManifest:
    root = Path(workspace_dir)
    store = EvidenceStore.model_validate(evidence_store)
    jsonl_path = root / "evidence_store" / "facts.jsonl"
    _write_jsonl(
        jsonl_path,
        _evidence_store_jsonl_records(store),
    )

    manifest_path = root / "trial-workspace-manifest.json"
    if manifest_path.exists():
        previous_manifest = read_trial_workspace_manifest(manifest_path)
        sources = previous_manifest.sources
        artifacts = [
            artifact
            for artifact in previous_manifest.artifacts
            if artifact.artifact_id != store.artifact_id
        ]
    else:
        sources = []
        artifacts = []

    upstream_hashes = {
        artifact_id: file_sha256(path)
        for artifact_id, path in sorted(upstream_artifact_paths.items())
    }
    model_name = str(model_metadata.get("model") or "unknown-model")
    artifacts.append(
        _file_artifact_identity(
            artifact_id=store.artifact_id,
            schema_version=store.schema_version,
            producer="evidence-family-mining",
            producer_version=model_name,
            config={
                "schema_version": store.schema_version,
                "model_metadata": model_metadata,
            },
            upstream_artifact_hashes=upstream_hashes,
            path=jsonl_path,
        )
    )

    manifest = build_trial_workspace_manifest(
        trial_id=trial_id,
        sources=sources,
        artifacts=artifacts,
    )
    write_trial_workspace_manifest(manifest, manifest_path)
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
    source_hashes = {
        source.document_id: source.content_hash for source in current_sources
    }
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
            (
                "page-aware-artifacts",
                "page_artifacts",
                PAGE_AWARE_ARTIFACT_SCHEMA_VERSION,
            ),
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


def _find_outcome_artifact(
    manifest: OutcomeWorkspaceManifest,
    artifact_id: str,
) -> OutcomeArtifactIdentity | None:
    for artifact in manifest.artifacts:
        if artifact.artifact_id == artifact_id:
            return artifact
    return None


def _existing_outcome_artifacts(
    workspace_root: str | Path,
    outcome_id: str,
    current_identity: OutcomeArtifactIdentity,
) -> list[OutcomeArtifactIdentity]:
    manifest_path = (
        outcome_workspace_dir(workspace_root, outcome_id)
        / "outcome-workspace-manifest.json"
    )
    if not manifest_path.exists():
        return []
    manifest = read_outcome_workspace_manifest(manifest_path)
    return [
        artifact
        for artifact in manifest.artifacts
        if artifact.artifact_id != current_identity.artifact_id
    ]


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


def _is_same_outcome_reusable_identity(
    previous: OutcomeArtifactIdentity,
    current: OutcomeArtifactIdentity,
) -> bool:
    return (
        previous.schema_version == current.schema_version
        and previous.producer == current.producer
        and previous.producer_version == current.producer_version
        and previous.config_hash == current.config_hash
        and previous.upstream_trial_workspace_hashes
        == current.upstream_trial_workspace_hashes
        and previous.outcome_definition_hash == current.outcome_definition_hash
        and previous.rob2_settings_hash == current.rob2_settings_hash
        and previous.content_hash == current.content_hash
    )


def _manifest_to_dict(
    manifest: TrialWorkspaceManifest | OutcomeWorkspaceManifest,
) -> dict:
    payload = asdict(manifest)
    if "sources" in payload:
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


def _d1_sq_answer_artifact_with_validation(artifact: dict) -> dict:
    invalid_answers = []
    missing_support_metadata = []
    valid_answers = {"Y", "PY", "PN", "N", "NI"}
    required_support_fields = {
        "support_level",
        "support_rationale",
        "packet_artifact_id",
        "decision_table_artifact_id",
    }
    for answer in artifact.get("answers", []):
        sq_id = str(answer.get("sq_id", ""))
        if answer.get("answer") not in valid_answers:
            invalid_answers.append(
                {
                    "sq_id": sq_id,
                    "answer": answer.get("answer"),
                    "reason": "Answer is not a canonical RoB 2 SQ answer.",
                }
            )
        missing = [
            field
            for field in sorted(required_support_fields)
            if not str(answer.get(field, "")).strip()
        ]
        if missing:
            missing_support_metadata.append({"sq_id": sq_id, "missing_fields": missing})

    validation = dict(artifact.get("validation") or {})
    validation["invalid_answers"] = invalid_answers
    validation["missing_support_metadata"] = missing_support_metadata
    if invalid_answers or missing_support_metadata:
        validation["status"] = "invalid"
    else:
        validation["status"] = validation.get("status") or "validated"
    return {**artifact, "validation": validation}


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


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(record, sort_keys=True, separators=(",", ":")) for record in records
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _evidence_store_jsonl_records(store: EvidenceStore) -> list[dict]:
    records = []
    for fact in store.supported_facts:
        payload = fact.model_dump(mode="json")
        payload["record_kind"] = "fact"
        payload["search_text"] = _fact_search_text(payload)
        payload["embedding_text"] = payload["search_text"]
        records.append(payload)
    for fact in store.failed_claims:
        payload = fact.model_dump(mode="json")
        payload["record_kind"] = "failed_claim"
        payload["search_text"] = _fact_search_text(payload)
        payload["embedding_text"] = payload["search_text"]
        records.append(payload)
    for gap in store.gaps:
        payload = gap.model_dump(mode="json")
        payload["record_kind"] = "gap"
        payload["search_text"] = _gap_search_text(payload)
        payload["embedding_text"] = payload["search_text"]
        records.append(payload)
    return records


def _fact_search_text(fact: dict) -> str:
    fields = [
        str(fact.get("claim", "")).strip(),
        str(fact.get("quote", "")).strip(),
        _family_fields_search_text(fact.get("family_fields")),
    ]
    return "\n".join(field for field in fields if field)


def _family_fields_search_text(family_fields: dict | None) -> str:
    if not family_fields:
        return ""
    return " ".join(str(value).strip() for value in family_fields.values() if value)


def _gap_search_text(gap: dict) -> str:
    fields = [
        str(gap.get("missing_evidence", "")).strip(),
        str(gap.get("reason", "")).strip(),
    ]
    return "\n".join(field for field in fields if field)


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
    "OUTCOME_WORKSPACE_MANIFEST_SCHEMA_VERSION",
    "OutcomeArtifactIdentity",
    "OutcomeWorkspaceManifest",
    "SourceIdentity",
    "TRIAL_WORKSPACE_MANIFEST_SCHEMA_VERSION",
    "TrialWorkspaceManifest",
    "artifact_identity",
    "build_outcome_artifact_identity",
    "build_outcome_workspace_manifest",
    "build_trial_workspace_manifest",
    "config_sha256",
    "content_sha256",
    "evaluate_outcome_artifact_status",
    "evaluate_artifact_status",
    "file_sha256",
    "load_parse_trial_workspace",
    "load_trial_workspace_artifacts",
    "outcome_workspace_dir",
    "read_outcome_workspace_manifest",
    "read_trial_workspace_manifest",
    "stable_payload_sha256",
    "write_outcome_workspace_manifest",
    "write_d1_judgment_workspace",
    "write_d1_sq_answer_workspace",
    "write_outcome_normalization_workspace",
    "write_evidence_store_trial_workspace",
    "write_parse_trial_workspace",
    "write_trial_workspace_manifest",
]
