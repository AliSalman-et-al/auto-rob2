"""Runtime schemas for quote-grounded evidence artifacts."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rob2_pipeline.types import LLMCallLogEntry


SupportLevel = Literal["strong", "moderate", "weak", "unsupported"]
SupportStatus = Literal["supported", "failed"]
ClaimType = Literal[
    "trial_method",
    "outcome_measurement",
    "analysis",
    "result_reporting",
    "registry",
    "other",
]
EvidenceFamily = Literal["randomization_allocation", "prespecification"]


class RandomizationAllocationFields(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: str = Field(min_length=1)
    allocation_concealment: str = Field(min_length=1)
    unit_of_randomization: str = Field(min_length=1)


class PrespecificationFields(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_type: Literal["registry", "protocol", "sap"]
    identifier: str = Field(min_length=1)
    prespecified_outcome: str = Field(min_length=1)
    prespecified_analysis: str = Field(min_length=1)


FamilyFields = RandomizationAllocationFields | PrespecificationFields


class EvidenceProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    document_name: str = Field(min_length=1)
    document_role: str = Field(min_length=1)
    source_kind: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    source_section: str = Field(min_length=1)
    page_numbers: list[int] = Field(default_factory=list)
    retrieval_date: str = ""
    api_response_hash: str = ""


class EvidenceFactRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    artifact_id: str = Field(min_length=1)
    fact_type: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    sq_ids: list[str] = Field(min_length=1)
    claim_type: ClaimType
    claim: str = Field(min_length=1)
    quote: str = ""
    support_level: SupportLevel
    support_status: SupportStatus
    uncertainty: bool
    provenance: EvidenceProvenance
    family: EvidenceFamily | None = None
    family_fields: FamilyFields | None = None
    failure_reason: str = ""

    @model_validator(mode="after")
    def supported_facts_require_quote(self) -> EvidenceFactRecord:
        if self.support_level == "unsupported" and self.support_status == "supported":
            raise ValueError(
                "unsupported claims cannot be selected as supporting facts"
            )
        if self.support_status == "supported" and not self.quote.strip():
            raise ValueError("supported evidence facts require a quote")
        if self.support_status == "failed" and not self.failure_reason.strip():
            raise ValueError("failed evidence claims require a failure_reason")
        if self.family == "randomization_allocation" and not isinstance(
            self.family_fields, RandomizationAllocationFields
        ):
            raise ValueError(
                "randomization_allocation facts require method, "
                "allocation_concealment, and unit_of_randomization"
            )
        if self.family == "prespecification" and not isinstance(
            self.family_fields, PrespecificationFields
        ):
            raise ValueError(
                "prespecification facts require artifact_type, identifier, "
                "prespecified_outcome, and prespecified_analysis"
            )
        if self.family is None and self.family_fields is not None:
            raise ValueError("family_fields require a family")
        return self


class EvidenceGap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    sq_ids: list[str] = Field(default_factory=list)
    missing_evidence: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class EvidencePacketRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    artifact_id: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    sq_id: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    outcome: str = Field(min_length=1)
    required_evidence: list[str] = Field(min_length=1)
    sources: list[dict] = Field(default_factory=list)
    candidate_facts: list[dict] = Field(default_factory=list)
    gaps: list[EvidenceGap] = Field(default_factory=list)
    failed_claims: list[EvidenceFactRecord] = Field(default_factory=list)
    contradictions: list[dict] = Field(default_factory=list)
    text: str = ""
    retrieval_confidence: float = Field(ge=0.0, le=1.0)
    missing_evidence: list[str] = Field(default_factory=list)
    negative_flags: list[str] = Field(default_factory=list)
    packet_grade: dict

    @model_validator(mode="after")
    def packet_identity_matches_sq(self) -> EvidencePacketRecord:
        expected_prefix = f"evidence-packet:{self.domain}:{self.sq_id}"
        if self.artifact_id != expected_prefix:
            raise ValueError("packet artifact_id must match domain and sq_id")
        return self


class EvidenceStore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    supported_facts: list[EvidenceFactRecord] = Field(default_factory=list)
    failed_claims: list[EvidenceFactRecord] = Field(default_factory=list)
    gaps: list[EvidenceGap] = Field(default_factory=list)

    @model_validator(mode="after")
    def facts_match_their_store_bucket(self) -> EvidenceStore:
        if any(fact.support_status != "supported" for fact in self.supported_facts):
            raise ValueError("supported_facts may only contain supported facts")
        if any(fact.support_status != "failed" for fact in self.failed_claims):
            raise ValueError("failed_claims may only contain failed claims")
        return self


FAMILY_BY_SQ: dict[str, EvidenceFamily] = {
    "1.1": "randomization_allocation",
    "1.2": "randomization_allocation",
    "5.1": "prespecification",
    "5.2": "prespecification",
}

FAMILY_SCHEMA_TEXT: dict[EvidenceFamily, str] = {
    "randomization_allocation": (
        "family_fields must contain method, allocation_concealment, "
        "and unit_of_randomization."
    ),
    "prespecification": (
        "family_fields must contain artifact_type (registry, protocol, or sap), "
        "identifier, prespecified_outcome, and prespecified_analysis."
    ),
}


EvidenceFamilyCall = Callable[
    [dict, str, str],
    tuple[str, list[LLMCallLogEntry], object],
]


def select_evidence_facts(
    *,
    evidence_store: EvidenceStore | dict,
    outcome: str,
    sq_family: EvidenceFamily,
    sq_ids: list[str],
    raw_packets: dict,
) -> dict:
    """Select typed facts for one outcome and SQ family.

    Raw packet sources are returned only as a marked repair substrate when no
    typed supported facts survive family, SQ, and outcome filtering.
    """

    store = (
        evidence_store
        if isinstance(evidence_store, EvidenceStore)
        else EvidenceStore.model_validate(evidence_store)
    )
    requested_sq_ids = set(sq_ids)
    selected: list[dict] = []
    excluded: list[dict] = []

    for fact in store.supported_facts:
        if fact.family != sq_family:
            continue
        if requested_sq_ids.isdisjoint(fact.sq_ids):
            continue
        if _is_wrong_outcome_fact(fact, outcome):
            excluded.append(
                {
                    "artifact_id": fact.artifact_id,
                    "exclusion_reason": "wrong_outcome_context",
                }
            )
            continue
        selected.append(fact.model_dump())

    if selected:
        return {
            "outcome": outcome,
            "sq_family": sq_family,
            "sq_ids": sq_ids,
            "retrieval_substrate": "typed_facts",
            "fallback_used": False,
            "selected_facts": selected,
            "excluded_facts": excluded,
            "missing_family_facts": [],
            "raw_fallback_sources": [],
        }

    return {
        "outcome": outcome,
        "sq_family": sq_family,
        "sq_ids": sq_ids,
        "retrieval_substrate": "raw_chunk_fallback",
        "fallback_used": True,
        "selected_facts": [],
        "excluded_facts": excluded,
        "missing_family_facts": [sq_family],
        "raw_fallback_sources": _raw_fallback_sources(raw_packets, sq_ids),
    }


def mine_evidence_families(
    state: dict,
    call_fn: EvidenceFamilyCall,
    *,
    max_sources_per_sq: int = 3,
) -> dict:
    """Mine family-typed evidence facts from selected packet source zones.

    The LLM receives only the bounded source zones already selected for an SQ
    packet. Deterministic code validates schema/provenance and controls retry
    and fallback; it does not infer semantic fact meaning from source text.
    """

    supported: list[EvidenceFactRecord] = []
    failed: list[EvidenceFactRecord] = []
    llm_log: list[LLMCallLogEntry] = []

    for sq_id, packet in sorted(state.get("evidence_packets", {}).items()):
        family = FAMILY_BY_SQ.get(sq_id)
        if family is None:
            continue
        zones = _selected_zones(packet, max_sources_per_sq)
        if not zones:
            continue

        prompt = _build_family_prompt(state, sq_id, family, zones)
        response, log, _parsed = call_fn(
            state, prompt, f"evidence_family_mining_{sq_id.replace('.', '_')}"
        )
        llm_log.extend(log)
        try:
            next_supported, next_failed = _validate_family_response(
                response, family, sq_id
            )
            supported.extend(next_supported)
            failed.extend(next_failed)
            continue
        except Exception as exc:  # noqa: BLE001
            repair_prompt = _build_repair_prompt(prompt, exc)

        repair_response, repair_log, _repair_parsed = call_fn(
            state,
            repair_prompt,
            f"evidence_family_mining_{sq_id.replace('.', '_')}_repair",
        )
        llm_log.extend(repair_log)
        try:
            next_supported, next_failed = _validate_family_response(
                repair_response, family, sq_id
            )
            supported.extend(next_supported)
            failed.extend(next_failed)
        except Exception as repair_exc:  # noqa: BLE001
            failed.append(_failed_family_claim(state, sq_id, family, zones, repair_exc))

    store = EvidenceStore(
        artifact_id=_store_artifact_id(state),
        schema_version="1.0",
        supported_facts=supported,
        failed_claims=failed,
        gaps=[],
    )
    selected_facts = _select_family_packets(
        store,
        outcome=str(state.get("outcome", "")),
        raw_packets=state.get("evidence_packets", {}),
    )
    return {
        "evidence_store": store.model_dump(),
        "selected_evidence_facts": selected_facts,
    }


def _select_family_packets(
    store: EvidenceStore,
    *,
    outcome: str,
    raw_packets: dict,
) -> dict:
    packets = {}
    for sq_id, family in sorted(FAMILY_BY_SQ.items()):
        if sq_id not in raw_packets and not any(
            fact.family == family and sq_id in fact.sq_ids
            for fact in store.supported_facts
        ):
            continue
        packets[sq_id] = select_evidence_facts(
            evidence_store=store,
            outcome=outcome,
            sq_family=family,
            sq_ids=[sq_id],
            raw_packets=raw_packets,
        )
    return packets


def _is_wrong_outcome_fact(fact: EvidenceFactRecord, outcome: str) -> bool:
    fields = fact.family_fields
    if not isinstance(fields, PrespecificationFields):
        return False
    return _normalize_outcome(fields.prespecified_outcome) != _normalize_outcome(
        outcome
    )


def _normalize_outcome(value: str) -> str:
    return " ".join(value.casefold().replace("-", " ").split())


def _raw_fallback_sources(raw_packets: dict, sq_ids: list[str]) -> list[dict]:
    sources: list[dict] = []
    for sq_id in sq_ids:
        packet = raw_packets.get(sq_id, {})
        for source in packet.get("sources", []):
            fallback_source = dict(source)
            fallback_source["fallback_reason"] = "missing_typed_family_facts"
            sources.append(fallback_source)
    return sources


def _selected_zones(packet: dict, max_sources: int) -> list[dict]:
    zones = []
    for index, source in enumerate(packet.get("sources", [])[:max_sources], start=1):
        text = str(source.get("text", "")).strip()
        if not text:
            continue
        zones.append(
            {
                "zone_id": f"source-{index}",
                "text": text,
                "provenance": {
                    "document_id": source.get("document_id", "primary"),
                    "document_name": source.get("document_name", "Primary paper"),
                    "document_role": source.get("document_role", "primary"),
                    "source_kind": source.get("source_kind", "rag_chunk"),
                    "source_path": source.get("source_path") or "unknown",
                    "source_section": source.get("section") or "Unknown",
                    "page_numbers": source.get("page_numbers", []),
                    "retrieval_date": source.get("retrieval_date", ""),
                    "api_response_hash": source.get("api_response_hash", ""),
                },
            }
        )
    return zones


def _build_family_prompt(
    state: dict, sq_id: str, family: EvidenceFamily, zones: list[dict]
) -> str:
    return "\n".join(
        [
            "Extract typed evidence facts from the bounded source zones only.",
            "Do not use outside knowledge, full text, regex keyword matches, or "
            "unstated context to decide semantic fact meaning.",
            f"Outcome: {state.get('outcome', 'Not reported')}",
            f"SQ ID: {sq_id}",
            f"Evidence family: {family}",
            f"Schema: {FAMILY_SCHEMA_TEXT[family]}",
            "Return JSON only in this shape: "
            '{"facts": [{EvidenceFactRecord fields including provenance, family, '
            "and family_fields}]}",
            "Selected source zones:",
            json.dumps(zones, indent=2),
        ]
    )


def _build_repair_prompt(original_prompt: str, exc: Exception) -> str:
    return "\n\n".join(
        [
            f"Your previous evidence-family extraction was invalid: {exc}",
            "Return JSON only. Preserve semantic claims only when they are "
            "supported by the selected source zones.",
            original_prompt,
        ]
    )


def _validate_family_response(
    response: str, family: EvidenceFamily, sq_id: str
) -> tuple[list[EvidenceFactRecord], list[EvidenceFactRecord]]:
    payload = json.loads(response)
    raw_facts = payload["facts"]
    facts = [EvidenceFactRecord.model_validate(raw) for raw in raw_facts]
    supported: list[EvidenceFactRecord] = []
    failed: list[EvidenceFactRecord] = []
    for fact in facts:
        if sq_id not in fact.sq_ids:
            raise ValueError(f"fact {fact.artifact_id!r} does not include SQ {sq_id}")
        if fact.support_status == "failed":
            failed.append(fact)
            continue
        if fact.family != family:
            raise ValueError(f"expected family {family!r}, got {fact.family!r}")
        supported.append(fact)
    return supported, failed


def _failed_family_claim(
    state: dict, sq_id: str, family: EvidenceFamily, zones: list[dict], exc: Exception
) -> EvidenceFactRecord:
    provenance = dict(zones[0]["provenance"])
    provenance["source_path"] = provenance.get("source_path") or "unknown"
    provenance["source_section"] = provenance.get("source_section") or "Unknown"
    return EvidenceFactRecord(
        artifact_id=f"evidence-fact:{sq_id}:family-mining-failed",
        fact_type=f"{family}_extraction",
        domain=f"d{sq_id.split('.')[0]}",
        sq_ids=[sq_id],
        claim_type="other",
        claim=f"Evidence-family mining failed for {family}.",
        quote="",
        support_level="unsupported",
        support_status="failed",
        uncertainty=True,
        provenance=EvidenceProvenance.model_validate(provenance),
        family=None,
        family_fields=None,
        failure_reason=f"Evidence-family validation failed after retry: {exc}",
    )


def _store_artifact_id(state: dict) -> str:
    trial = Path(str(state.get("pdf_path", "assessment"))).stem or "assessment"
    outcome = str(state.get("outcome", "unknown-outcome")).casefold()
    outcome_slug = "-".join(outcome.split()) or "unknown-outcome"
    return f"evidence-store:{trial}:{outcome_slug}"
