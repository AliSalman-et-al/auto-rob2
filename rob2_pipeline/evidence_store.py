"""Runtime schemas for quote-grounded evidence artifacts."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
EvidenceFamily = Literal[
    "randomization_allocation",
    "masking_awareness",
    "deviations_adherence",
    "analysis_population",
    "missing_outcome_data",
    "outcome_measurement",
    "prespecification",
    "result_reporting",
]


class _FamilyStringFields(BaseModel):
    @field_validator("*", mode="before")
    @classmethod
    def coerce_descriptive_field(cls, value):
        if value is None:
            return "Not reported"
        if isinstance(value, bool):
            return "Yes" if value else "No"
        return str(value)


class RandomizationAllocationFields(_FamilyStringFields):
    model_config = ConfigDict(extra="forbid")

    method: str = Field(min_length=1)
    allocation_concealment: str = Field(min_length=1)
    unit_of_randomization: str = Field(min_length=1)


class PrespecificationFields(_FamilyStringFields):
    model_config = ConfigDict(extra="forbid")

    artifact_type: Literal["registry", "protocol", "sap"]
    identifier: str = Field(min_length=1)
    prespecified_outcome: str = Field(min_length=1)
    prespecified_analysis: str = Field(min_length=1)


class MaskingAwarenessFields(_FamilyStringFields):
    model_config = ConfigDict(extra="forbid")

    participant_awareness: str = Field(min_length=1)
    personnel_awareness: str = Field(min_length=1)
    masking_method: str = Field(min_length=1)
    awareness_context: str = Field(min_length=1)


class DeviationsAdherenceFields(_FamilyStringFields):
    model_config = ConfigDict(extra="forbid")

    awareness_status: str = Field(min_length=1)
    deviation_description: str = Field(min_length=1)
    adherence_population: str = Field(min_length=1)
    analysis_population: str = Field(min_length=1)
    outcome_impact: str = Field(min_length=1)


class AnalysisPopulationFields(_FamilyStringFields):
    model_config = ConfigDict(extra="forbid")

    population_label: str = Field(min_length=1)
    included_participants: str = Field(min_length=1)
    excluded_participants: str = Field(min_length=1)
    analysis_principle: str = Field(min_length=1)
    exclusion_impact: str = Field(min_length=1)


class MissingOutcomeDataFields(_FamilyStringFields):
    model_config = ConfigDict(extra="forbid")

    randomized_count: str = Field(min_length=1)
    outcome_data_count: str = Field(min_length=1)
    missing_count: str = Field(min_length=1)
    missing_reason: str = Field(min_length=1)
    analysis_handling: str = Field(min_length=1)


class OutcomeMeasurementFields(_FamilyStringFields):
    model_config = ConfigDict(extra="forbid")

    assessed_outcome: str = Field(min_length=1)
    measurement_method: str = Field(min_length=1)
    measurement_timing: str = Field(min_length=1)
    assessor_awareness: str = Field(min_length=1)
    influence_risk: str = Field(min_length=1)


class ResultReportingFields(_FamilyStringFields):
    model_config = ConfigDict(extra="forbid")

    reported_outcome: str = Field(min_length=1)
    reported_measurement: str = Field(min_length=1)
    reported_analysis: str = Field(min_length=1)
    result_metric: str = Field(min_length=1)
    matches_prespecification: str = Field(min_length=1)


FamilyFields = (
    RandomizationAllocationFields
    | MaskingAwarenessFields
    | DeviationsAdherenceFields
    | AnalysisPopulationFields
    | MissingOutcomeDataFields
    | OutcomeMeasurementFields
    | PrespecificationFields
    | ResultReportingFields
)


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
    family_fields: FamilyFields | dict[str, Any] | None = None
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
        if self.support_status == "failed":
            return self
        if self.family == "randomization_allocation" and not isinstance(
            self.family_fields, RandomizationAllocationFields
        ):
            raise ValueError(
                "randomization_allocation facts require method, "
                "allocation_concealment, and unit_of_randomization"
            )
        if self.family == "masking_awareness" and not isinstance(
            self.family_fields, MaskingAwarenessFields
        ):
            raise ValueError(
                "masking_awareness facts require participant_awareness, "
                "personnel_awareness, masking_method, and awareness_context"
            )
        if self.family == "deviations_adherence" and not isinstance(
            self.family_fields, DeviationsAdherenceFields
        ):
            raise ValueError(
                "deviations_adherence facts require awareness_status, "
                "deviation_description, adherence_population, analysis_population, "
                "and outcome_impact"
            )
        if self.family == "analysis_population" and not isinstance(
            self.family_fields, AnalysisPopulationFields
        ):
            raise ValueError(
                "analysis_population facts require population_label, "
                "included_participants, excluded_participants, analysis_principle, "
                "and exclusion_impact"
            )
        if self.family == "missing_outcome_data" and not isinstance(
            self.family_fields, MissingOutcomeDataFields
        ):
            raise ValueError(
                "missing_outcome_data facts require randomized_count, "
                "outcome_data_count, missing_count, missing_reason, "
                "and analysis_handling"
            )
        if self.family == "outcome_measurement" and not isinstance(
            self.family_fields, OutcomeMeasurementFields
        ):
            raise ValueError(
                "outcome_measurement facts require assessed_outcome, "
                "measurement_method, measurement_timing, assessor_awareness, "
                "and influence_risk"
            )
        if self.family == "prespecification" and not isinstance(
            self.family_fields, PrespecificationFields
        ):
            raise ValueError(
                "prespecification facts require artifact_type, identifier, "
                "prespecified_outcome, and prespecified_analysis"
            )
        if self.family == "result_reporting" and not isinstance(
            self.family_fields, ResultReportingFields
        ):
            raise ValueError(
                "result_reporting facts require reported_outcome, "
                "reported_measurement, reported_analysis, result_metric, "
                "and matches_prespecification"
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


class DecisionTableRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    artifact_id: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    sq_id: str = Field(min_length=1)
    allowed_answers: list[str] = Field(min_length=1)
    rows: list[dict] = Field(default_factory=list)
    default_insufficient_evidence_answer: str = Field(min_length=1)
    classifier_instruction: str = Field(min_length=1)


class PacketContractRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    sq_id: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    required_evidence: list[str] = Field(min_length=1)
    allowed_answers: list[str] = Field(min_length=1)
    outcome_binding_status: Literal["outcome_bound", "trial_level"]
    source_hierarchy: list[str] = Field(min_length=1)
    needs_denominator: bool = False
    needs_prespecification: bool = False


class EvidencePacketRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    artifact_id: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    sq_id: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    outcome: str = Field(min_length=1)
    required_evidence: list[str] = Field(min_length=1)
    contract: PacketContractRecord
    sources: list[dict] = Field(default_factory=list)
    candidate_facts: list[dict] = Field(default_factory=list)
    gaps: list[EvidenceGap] = Field(default_factory=list)
    failed_claims: list[EvidenceFactRecord] = Field(default_factory=list)
    contradictions: list[dict] = Field(default_factory=list)
    decision_table: DecisionTableRecord
    text: str = ""
    retrieval_confidence: float = Field(ge=0.0, le=1.0)
    missing_evidence: list[str] = Field(default_factory=list)
    negative_flags: list[str] = Field(default_factory=list)
    provenance_warnings: list[str] = Field(default_factory=list)
    packet_grade: dict

    @model_validator(mode="after")
    def packet_identity_matches_sq(self) -> EvidencePacketRecord:
        expected_prefix = f"evidence-packet:{self.domain}:{self.sq_id}"
        if self.artifact_id != expected_prefix:
            raise ValueError("packet artifact_id must match domain and sq_id")
        expected_contract = f"packet-contract:{self.domain}:{self.sq_id}"
        if self.contract.artifact_id != expected_contract:
            raise ValueError("contract artifact_id must match domain and sq_id")
        if self.contract.allowed_answers != self.decision_table.allowed_answers:
            raise ValueError("contract allowed_answers must match decision table")
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
    "2.1": "masking_awareness",
    "2.2": "masking_awareness",
    "2.3": "deviations_adherence",
    "2.4": "deviations_adherence",
    "2.5": "deviations_adherence",
    "2.6": "analysis_population",
    "2.7": "analysis_population",
    "3.1": "missing_outcome_data",
    "3.2": "missing_outcome_data",
    "3.3": "missing_outcome_data",
    "3.4": "missing_outcome_data",
    "4.1": "outcome_measurement",
    "4.2": "outcome_measurement",
    "4.3": "outcome_measurement",
    "4.4": "outcome_measurement",
    "4.5": "outcome_measurement",
    "5.1": "prespecification",
    "5.2": "result_reporting",
    "5.3": "result_reporting",
}

FAMILY_SCHEMA_TEXT: dict[EvidenceFamily, str] = {
    "randomization_allocation": (
        "family_fields must contain method, allocation_concealment, "
        "and unit_of_randomization."
    ),
    "masking_awareness": (
        "family_fields must contain participant_awareness, personnel_awareness, "
        "masking_method, and awareness_context."
    ),
    "deviations_adherence": (
        "family_fields must contain awareness_status, deviation_description, "
        "adherence_population, analysis_population, and outcome_impact."
    ),
    "analysis_population": (
        "family_fields must contain population_label, included_participants, "
        "excluded_participants, analysis_principle, and exclusion_impact."
    ),
    "missing_outcome_data": (
        "family_fields must contain randomized_count, outcome_data_count, "
        "missing_count, missing_reason, and analysis_handling."
    ),
    "outcome_measurement": (
        "family_fields must contain assessed_outcome, measurement_method, "
        "measurement_timing, assessor_awareness, and influence_risk."
    ),
    "prespecification": (
        "family_fields must contain artifact_type (registry, protocol, or sap), "
        "identifier, prespecified_outcome, and prespecified_analysis."
    ),
    "result_reporting": (
        "family_fields must contain reported_outcome, reported_measurement, "
        "reported_analysis, result_metric, and matches_prespecification."
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

    for family, sq_ids, zones in _family_mining_jobs(
        state.get("evidence_packets", {}),
        max_sources_per_sq,
    ):
        if not zones:
            continue
        prompt = _build_family_prompt(state, sq_ids, family, zones)
        node_suffix = str(family).replace("_", "-")
        try:
            response, log, _parsed = call_fn(
                state, prompt, f"evidence_family_mining_{node_suffix}"
            )
        except Exception as exc:  # noqa: BLE001
            failed.append(_failed_family_claim(state, sq_ids, family, zones, exc))
            continue
        llm_log.extend(log)
        try:
            next_supported, next_failed = _validate_family_response(
                response, family, sq_ids
            )
            supported.extend(next_supported)
            failed.extend(next_failed)
            continue
        except Exception as exc:  # noqa: BLE001
            repair_prompt = _build_repair_prompt(prompt, exc)

        repair_response, repair_log, _repair_parsed = call_fn(
            state,
            repair_prompt,
            f"evidence_family_mining_{node_suffix}_repair",
        )
        llm_log.extend(repair_log)
        try:
            next_supported, next_failed = _validate_family_response(
                repair_response, family, sq_ids
            )
            supported.extend(next_supported)
            failed.extend(next_failed)
        except Exception as repair_exc:  # noqa: BLE001
            failed.append(_failed_family_claim(state, sq_ids, family, zones, repair_exc))

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


def _family_mining_jobs(
    evidence_packets: dict,
    max_sources_per_sq: int,
) -> list[tuple[EvidenceFamily, list[str], list[dict]]]:
    grouped: dict[EvidenceFamily, dict[str, Any]] = {}
    seen_zone_keys: dict[EvidenceFamily, set[tuple]] = {}
    for sq_id, packet in sorted(evidence_packets.items()):
        family = FAMILY_BY_SQ.get(sq_id)
        if family is None:
            continue
        group = grouped.setdefault(family, {"sq_ids": [], "zones": []})
        seen = seen_zone_keys.setdefault(family, set())
        group["sq_ids"].append(sq_id)
        for zone in _selected_zones(packet, max_sources_per_sq):
            zone_key = _zone_identity(zone)
            if zone_key in seen:
                continue
            seen.add(zone_key)
            zone["sq_ids"] = [sq_id]
            zone["family"] = family
            zone["zone_id"] = f"{family}:{len(group['zones']) + 1}"
            group["zones"].append(zone)
    return [
        (family, group["sq_ids"], group["zones"])
        for family, group in grouped.items()
    ]


def _zone_identity(zone: dict) -> tuple:
    provenance = zone.get("provenance") or {}
    return (
        provenance.get("document_id", ""),
        provenance.get("source_kind", ""),
        provenance.get("source_section", ""),
        tuple(provenance.get("page_numbers") or []),
        zone.get("text", ""),
    )


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
    fact_outcome = ""
    if isinstance(fields, PrespecificationFields):
        fact_outcome = fields.prespecified_outcome
    elif isinstance(fields, OutcomeMeasurementFields):
        fact_outcome = fields.assessed_outcome
    elif isinstance(fields, ResultReportingFields):
        fact_outcome = fields.reported_outcome
    if not fact_outcome:
        return False
    return _normalize_outcome(fact_outcome) != _normalize_outcome(outcome)


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
                    "source_kind": source.get("source_kind") or "unknown",
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
    state: dict, sq_ids: list[str], family: EvidenceFamily, zones: list[dict]
) -> str:
    domain = f"d{sq_ids[0].split('.')[0]}"
    return "\n".join(
        [
            "Extract typed evidence facts from the bounded source zones only.",
            "Do not use outside knowledge, full text, regex keyword matches, or "
            "unstated context to decide semantic fact meaning.",
            f"Outcome: {state.get('outcome', 'Not reported')}",
            f"SQ IDs: {', '.join(sq_ids)}",
            f"Evidence family: {family}",
            f"Schema: {FAMILY_SCHEMA_TEXT[family]}",
            "Extract facts for this evidence family once, then assign each fact "
            "to every SQ ID it directly supports. A fact may support multiple "
            "listed SQ IDs only when the same quote supports all of them.",
            "Return JSON only. Do not omit required keys. Every facts[] item "
            "must be a complete EvidenceFactRecord object with exactly these "
            "top-level keys:",
            (
                "artifact_id, fact_type, domain, sq_ids, claim_type, claim, "
                "quote, support_level, support_status, uncertainty, provenance, "
                "family, family_fields, failure_reason"
            ),
            "Use support_status='supported' only when quote is a direct source "
            "quote supporting claim. Family fields are required schema fields, "
            "not a requirement that every methodological detail is reported; "
            "when a supported quote establishes the claim but a field is not "
            "reported in the selected zones, set that field to 'Not reported' "
            "instead of failing the whole fact. When the selected zones do not "
            "support any claim for this family, return one failed fact with support_status='failed', "
            "support_level='unsupported', quote='', family set to the evidence "
            "family, family_fields=null, provenance copied from the most relevant "
            "selected source zone, and a specific failure_reason.",
            "Use claim_type as one of: trial_method, outcome_measurement, "
            "analysis, result_reporting, registry, other.",
            "Use support_level as one of: strong, moderate, weak, unsupported.",
            "Use support_status as one of: supported, failed.",
            "Use uncertainty as a JSON boolean.",
            "Template for one supported fact:",
            json.dumps(
                {
                    "facts": [
                        {
                            "artifact_id": f"evidence-fact:{domain}:{family}:1",
                            "fact_type": f"{family}_fact",
                            "domain": domain,
                            "sq_ids": [sq_ids[0]],
                            "claim_type": "trial_method",
                            "claim": "One sentence claim supported by the quote.",
                            "quote": "Exact quote copied from one selected source zone.",
                            "support_level": "moderate",
                            "support_status": "supported",
                            "uncertainty": False,
                            "provenance": {
                                "document_id": "copy from selected zone provenance",
                                "document_name": "copy from selected zone provenance",
                                "document_role": "copy from selected zone provenance",
                                "source_kind": "copy from selected zone provenance",
                                "source_path": "copy from selected zone provenance",
                                "source_section": "copy from selected zone provenance",
                                "page_numbers": [],
                                "retrieval_date": "",
                                "api_response_hash": "",
                            },
                            "family": family,
                            "family_fields": {},
                            "failure_reason": "",
                        }
                    ]
                },
                indent=2,
            ),
            "Template for one failed fact:",
            json.dumps(
                {
                    "facts": [
                        {
                            "artifact_id": f"evidence-fact:{domain}:{family}:failed",
                            "fact_type": f"{family}_fact",
                            "domain": domain,
                            "sq_ids": [sq_ids[0]],
                            "claim_type": "other",
                            "claim": "Selected source zones do not support a claim for this SQ.",
                            "quote": "",
                            "support_level": "unsupported",
                            "support_status": "failed",
                            "uncertainty": True,
                            "provenance": {
                                "document_id": "copy from selected zone provenance",
                                "document_name": "copy from selected zone provenance",
                                "document_role": "copy from selected zone provenance",
                                "source_kind": "copy from selected zone provenance",
                                "source_path": "copy from selected zone provenance",
                                "source_section": "copy from selected zone provenance",
                                "page_numbers": [],
                                "retrieval_date": "",
                                "api_response_hash": "",
                            },
                            "family": family,
                            "family_fields": None,
                            "failure_reason": "Specific missing evidence reason.",
                        }
                    ]
                },
                indent=2,
            ),
            "Selected source zones:",
            json.dumps(zones, indent=2),
        ]
    )


def _build_repair_prompt(original_prompt: str, exc: Exception) -> str:
    failure_summary = _summarize_validation_error(str(exc))
    return "\n\n".join(
        [
            f"Your previous evidence-family extraction was invalid: {failure_summary}",
            "Return JSON only. Preserve semantic claims only when they are "
            "supported by the selected source zones. Every facts[] item must "
            "include artifact_id, fact_type, domain, sq_ids, claim_type, claim, "
            "quote, support_level, support_status, uncertainty, provenance, "
            "family, family_fields, and failure_reason.",
            "For failed facts, do not use empty provenance fields. Copy provenance "
            "from the most relevant selected source zone and set family_fields to null.",
            original_prompt,
        ]
    )


def _summarize_validation_error(message: str, *, max_lines: int = 8) -> str:
    lines = [line for line in message.splitlines() if line.strip()]
    if len(lines) <= max_lines:
        return message
    return "\n".join([*lines[:max_lines], f"... ({len(lines) - max_lines} more lines)"])


def _validate_family_response(
    response: str, family: EvidenceFamily, sq_ids: list[str]
) -> tuple[list[EvidenceFactRecord], list[EvidenceFactRecord]]:
    payload = json.loads(response)
    raw_facts = payload["facts"]
    facts = [EvidenceFactRecord.model_validate(raw) for raw in raw_facts]
    supported: list[EvidenceFactRecord] = []
    failed: list[EvidenceFactRecord] = []
    allowed_sq_ids = set(sq_ids)
    for fact in facts:
        if set(fact.sq_ids).isdisjoint(allowed_sq_ids):
            raise ValueError(
                f"fact {fact.artifact_id!r} does not include any requested SQ ID"
            )
        if fact.support_status == "failed":
            failed.append(fact)
            continue
        if fact.family != family:
            raise ValueError(f"expected family {family!r}, got {fact.family!r}")
        supported.append(fact)
    return supported, failed


def _failed_family_claim(
    state: dict, sq_ids: list[str], family: EvidenceFamily, zones: list[dict], exc: Exception
) -> EvidenceFactRecord:
    provenance = dict(zones[0]["provenance"])
    provenance["source_path"] = provenance.get("source_path") or "unknown"
    provenance["source_section"] = provenance.get("source_section") or "Unknown"
    domain = f"d{sq_ids[0].split('.')[0]}"
    return EvidenceFactRecord(
        artifact_id=f"evidence-fact:{domain}:{family}:family-mining-failed",
        fact_type=f"{family}_extraction",
        domain=domain,
        sq_ids=sq_ids,
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
