"""Bounded retrieval repair artifacts for incomplete SQ evidence packets."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rob2_pipeline.nodes.evidence_contracts import CONTRACTS, EvidenceContract
from rob2_pipeline.nodes.evidence_packets import build_packet_for_contract
from rob2_pipeline.state import RoB2State
from rob2_pipeline.types import EvidencePacket, RetrievalRepairArtifact


class RetrievalRepairQueryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    purpose: Literal["required_evidence", "outcome_binding", "source_hierarchy"]
    query: str = Field(min_length=1)
    required_evidence: list[str] = Field(default_factory=list)
    preferred_source_roles: list[str] = Field(default_factory=list)


class RetrievalRepairQueryPayloadRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queries: list[RetrievalRepairQueryRecord] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def query_purposes_are_exact(self) -> RetrievalRepairQueryPayloadRecord:
        purposes = [query.purpose for query in self.queries]
        if purposes != ["required_evidence", "outcome_binding", "source_hierarchy"]:
            raise ValueError("repair payload must contain the three allowed queries")
        return self


class RetrievalRepairArtifactRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    sq_id: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    outcome: str = Field(min_length=1)
    packet_artifact_id: str = Field(min_length=1)
    trigger_conditions: list[
        Literal["low_packet_confidence", "missing_required_evidence"]
    ] = Field(min_length=1)
    query_payload: RetrievalRepairQueryPayloadRecord
    before_packet_status: dict
    after_packet_status: dict
    source_changes: dict

    @model_validator(mode="after")
    def artifact_identity_matches_packet(self) -> RetrievalRepairArtifactRecord:
        expected = f"retrieval-repair:{self.domain}:{self.sq_id}"
        if self.artifact_id != expected:
            raise ValueError("retrieval repair artifact_id must match domain and sq_id")
        expected_packet = f"evidence-packet:{self.domain}:{self.sq_id}"
        if self.packet_artifact_id != expected_packet:
            raise ValueError("packet_artifact_id must match domain and sq_id")
        return self


def retrieval_repair_node(state: RoB2State) -> dict:
    artifacts: dict[str, RetrievalRepairArtifact] = {}
    updated_packets: dict[str, EvidencePacket] = {}
    updated_grades: dict[str, dict] = {}
    updated_readiness: dict[str, dict] = {}
    for sq_id, packet in sorted((state.get("evidence_packets") or {}).items()):
        contract = CONTRACTS.get(sq_id)
        if contract is None:
            continue
        trigger_conditions = _trigger_conditions(packet)
        if not trigger_conditions:
            continue
        repaired_packet = build_packet_for_contract(state, contract)
        artifact = _build_repair_artifact(
            state=state,
            contract=contract,
            packet=packet,
            repaired_packet=repaired_packet,
            trigger_conditions=trigger_conditions,
        )
        validated = RetrievalRepairArtifactRecord.model_validate(artifact)
        artifacts[sq_id] = validated.model_dump()
        updated_packets[sq_id] = repaired_packet
        updated_grades[sq_id] = repaired_packet.get("packet_grade", {})
        updated_readiness[sq_id] = repaired_packet.get("packet_readiness", {})
    return {
        "retrieval_repair_artifacts": artifacts,
        "evidence_packets": updated_packets,
        "packet_grades": updated_grades,
        "packet_readiness": updated_readiness,
    }


def _trigger_conditions(packet: EvidencePacket) -> list[str]:
    readiness = packet.get("packet_readiness") or {}
    if readiness.get("status") != "needs_retrieval_repair":
        return []
    conditions = []
    confidence = float(packet.get("retrieval_confidence", 0.0))
    if confidence < 0.35:
        conditions.append("low_packet_confidence")
    missing = packet.get("missing_evidence") or (
        (packet.get("packet_grade") or {}).get("missing_evidence") or []
    )
    if missing:
        conditions.append("missing_required_evidence")
    return conditions


def _build_repair_artifact(
    *,
    state: RoB2State,
    contract: EvidenceContract,
    packet: EvidencePacket,
    repaired_packet: EvidencePacket,
    trigger_conditions: list[str],
) -> RetrievalRepairArtifact:
    before = _packet_status(packet)
    after = _packet_status(repaired_packet)
    before_source_ids = [_source_id(source) for source in packet.get("sources", [])]
    after_source_ids = [
        _source_id(source) for source in repaired_packet.get("sources", [])
    ]
    artifact = RetrievalRepairArtifact(
        artifact_id=f"retrieval-repair:{contract.domain}:{contract.sq_id}",
        schema_version="1.0",
        sq_id=contract.sq_id,
        domain=contract.domain,
        outcome=str(state.get("outcome", packet.get("outcome", ""))),
        packet_artifact_id=f"evidence-packet:{contract.domain}:{contract.sq_id}",
        trigger_conditions=trigger_conditions,
        query_payload={"queries": _repair_queries(state, contract, packet)},
        before_packet_status=before,
        after_packet_status=after,
        source_changes={
            "before_source_ids": before_source_ids,
            "after_source_ids": after_source_ids,
            "added_source_ids": [
                source_id
                for source_id in after_source_ids
                if source_id not in before_source_ids
            ],
            "removed_source_ids": [
                source_id
                for source_id in before_source_ids
                if source_id not in after_source_ids
            ],
        },
    )
    return artifact


def _repair_queries(
    state: RoB2State, contract: EvidenceContract, packet: EvidencePacket
) -> list[dict]:
    outcome = str(state.get("outcome", packet.get("outcome", ""))).strip()
    missing = packet.get("missing_evidence") or list(contract.required_evidence)
    role_terms = " ".join(_preferred_roles(contract.domain))
    evidence_terms = " ".join(missing)
    base = f"SQ {contract.sq_id} {outcome}".strip()
    return [
        {
            "purpose": "required_evidence",
            "query": f"{base} {evidence_terms} {' '.join(contract.terms[:5])}".strip(),
            "required_evidence": list(missing),
            "preferred_source_roles": _preferred_roles(contract.domain),
        },
        {
            "purpose": "outcome_binding",
            "query": f"{base} outcome-specific evidence {evidence_terms}".strip(),
            "required_evidence": list(missing),
            "preferred_source_roles": _preferred_roles(contract.domain),
        },
        {
            "purpose": "source_hierarchy",
            "query": f"{base} {role_terms} {evidence_terms}".strip(),
            "required_evidence": list(missing),
            "preferred_source_roles": _preferred_roles(contract.domain),
        },
    ]


def _packet_status(packet: EvidencePacket) -> dict:
    readiness = packet.get("packet_readiness") or {}
    grade = packet.get("packet_grade") or {}
    return {
        "status": readiness.get("status", "unknown"),
        "retrieval_confidence": packet.get("retrieval_confidence", 0.0),
        "missing_evidence": packet.get("missing_evidence")
        or grade.get("missing_evidence")
        or [],
        "negative_flags": packet.get("negative_flags", []),
        "source_count": len(packet.get("sources", [])),
    }


def _preferred_roles(domain: str) -> list[str]:
    by_domain = {
        "d1": ["primary", "protocol", "appendix"],
        "d2": ["primary", "protocol", "sap", "appendix"],
        "d3": ["primary", "appendix", "sap"],
        "d4": ["primary", "protocol", "sap", "appendix"],
        "d5": ["protocol", "sap", "registry", "primary", "appendix"],
    }
    return by_domain.get(domain, ["primary"])


def _source_id(source: dict) -> str:
    document_id = source.get("document_id") or "unknown"
    section = source.get("section") or "unknown"
    pages = ",".join(str(page) for page in source.get("page_numbers", []))
    return f"{document_id}:{section}:{pages}"
