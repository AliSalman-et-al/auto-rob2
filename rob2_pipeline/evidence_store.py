"""Runtime schemas for quote-grounded evidence artifacts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class EvidenceProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    document_name: str = Field(min_length=1)
    document_role: str = Field(min_length=1)
    source_kind: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    source_section: str = Field(min_length=1)
    page_numbers: list[int] = Field(default_factory=list)


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
    failure_reason: str = ""

    @model_validator(mode="after")
    def supported_facts_require_quote(self) -> EvidenceFactRecord:
        if self.support_status == "supported" and not self.quote.strip():
            raise ValueError("supported evidence facts require a quote")
        if self.support_status == "failed" and not self.failure_reason.strip():
            raise ValueError("failed evidence claims require a failure_reason")
        return self


class EvidenceGap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    sq_ids: list[str] = Field(default_factory=list)
    missing_evidence: str = Field(min_length=1)
    reason: str = Field(min_length=1)


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
