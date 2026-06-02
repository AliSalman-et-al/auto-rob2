from typing import NotRequired, TypedDict


class LLMCallLogEntry(TypedDict):
    node: str
    prompt_length_chars: int
    response_length_chars: int
    latency_ms: int
    cache_hit: bool
    model: NotRequired[str]
    input_tokens: NotRequired[int]
    output_tokens: NotRequired[int]
    cached: NotRequired[bool]
    suspected_parse_failures: NotRequired[list[str]]
    chunk_sources: NotRequired[list[str]]


class ChunkMeta(TypedDict, total=False):
    text: str
    section: str
    page_numbers: list[int]
    score: float
    document_id: str
    document_name: str
    document_role: str
    source_kind: str
    source_path: str


class SourceDocument(TypedDict, total=False):
    document_id: str
    document_name: str
    document_role: str
    source_kind: str
    path: str
    is_primary: bool
    status: str
    error: str


class ParseArtifact(TypedDict, total=False):
    source_identity: SourceDocument
    pages: list[dict]
    diagnostics: list[dict]
    provenance: dict


class OutcomeProperties(TypedDict):
    objective_event: bool
    clinician_judged: bool
    patient_reported: bool
    composite: bool
    time_to_event: bool
    safety_harm: bool
    lab_or_imaging_threshold: bool
    blinded_adjudication: bool


class TrialFacts(TypedDict, total=False):
    randomization: str
    allocation_concealment: str
    masking: str
    protocol_deviations: str
    protocol_amendments: str
    analysis_populations: str
    source: str


class RetrievalGrade(TypedDict):
    relevance: float
    coverage: float
    missing_evidence: list[str]
    retry_recommended: bool


class PacketSource(TypedDict, total=False):
    text: str
    section: str
    page_numbers: list[int]
    score: float
    matched_terms: list[str]
    source_kind: str
    document_id: str
    document_name: str
    document_role: str
    source_path: str


class EvidenceFact(TypedDict, total=False):
    fact_type: str
    domain: str
    sq_ids: list[str]
    claim: str
    quote: str
    source_section: str
    page_numbers: list[int]
    confidence: float
    support_status: str
    missing_reason: str
    document_id: str
    document_name: str
    document_role: str
    source_kind: str
    source_path: str


class EvidencePacket(TypedDict, total=False):
    sq_id: str
    domain: str
    required_evidence: list[str]
    sources: list[PacketSource]
    candidate_facts: list[EvidenceFact]
    text: str
    retrieval_confidence: float
    missing_evidence: list[str]
    negative_flags: list[str]
    packet_grade: RetrievalGrade


class EvidenceValidationFlag(TypedDict):
    sq_id: str
    issue: str
    quote: str


class SupportConstraint(TypedDict, total=False):
    constraint_type: str
    sq_id: str
    claim: dict
    evidence_label: str
    evidence: str
    reason: str
    provenance: dict


class VerifierTraceEntry(TypedDict, total=False):
    node: str
    sq_id: str
    action: str
    reason: str
    before: dict
    after: dict


class PivotalityTest(TypedDict):
    sq_id: str
    original_answer: str
    support_level: str
    conservative_test_answer: str
    original_domain_judgment: str
    test_domain_judgment: str
    pivotal: bool
    acceptance_status: str
    constraints: NotRequired[list[SupportConstraint]]


class SqSupportAdjudication(TypedDict, total=False):
    sq_id: str
    initial_answer: dict
    adjudicated_answer: dict
    domain_impact: dict
    changed: bool
    changed_answer: bool
    changed_support: bool
    rationale: str
    constraints: list[SupportConstraint]
    provenance: dict
    llm_node: str
