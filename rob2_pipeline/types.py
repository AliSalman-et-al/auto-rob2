from typing import Any, NotRequired, TypedDict


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
    provider: NotRequired[str]
    prompt_version: NotRequired[str]
    schema_version: NotRequired[str]
    parse_status: NotRequired[str]
    validation_status: NotRequired[str]
    attempts: NotRequired[list[dict[str, Any]]]
    fallback_artifact: NotRequired[dict[str, Any]]
    failure_reason: NotRequired[str]
    suspected_parse_failures: NotRequired[list[str]]
    chunk_sources: NotRequired[list[str]]


class ChunkMeta(TypedDict, total=False):
    text: str
    section: str
    original_heading: str
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
    retrieval_date: str
    api_response_hash: str


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


class OutcomeNormalizationArtifact(TypedDict, total=False):
    artifact_id: str
    schema_version: str
    outcome: str
    normalized_definition: str
    aliases: list[str]
    outcome_type: str
    outcome_properties: OutcomeProperties
    binding_support: dict
    auto_accept_blocked: bool
    uncertainty: bool


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


class PacketReadiness(TypedDict, total=False):
    artifact_id: str
    schema_version: str
    sq_id: str
    status: str
    mechanical_completeness: dict
    semantic_adequacy: dict
    blocking_reason: str


class PacketSource(TypedDict, total=False):
    text: str
    section: str
    original_heading: str
    page_numbers: list[int]
    score: float
    matched_terms: list[str]
    source_kind: str
    document_id: str
    document_name: str
    document_role: str
    source_path: str
    retrieval_date: str
    api_response_hash: str


class EvidenceFact(TypedDict, total=False):
    artifact_id: str
    fact_type: str
    domain: str
    sq_ids: list[str]
    claim_type: str
    claim: str
    quote: str
    source_section: str
    page_numbers: list[int]
    confidence: float
    support_level: str
    support_status: str
    uncertainty: bool
    missing_reason: str
    failure_reason: str
    document_id: str
    document_name: str
    document_role: str
    source_kind: str
    source_path: str
    retrieval_date: str
    api_response_hash: str
    provenance: dict
    family: str
    family_fields: dict


class EvidenceGap(TypedDict, total=False):
    artifact_id: str
    domain: str
    sq_ids: list[str]
    missing_evidence: str
    reason: str


class DecisionTableRow(TypedDict, total=False):
    answer: str
    rule: str
    allowed_by_packet: bool
    supporting_facts: list[dict]
    evidence_gaps: list[dict]
    insufficient_evidence_default: bool


class DecisionTable(TypedDict, total=False):
    artifact_id: str
    schema_version: str
    sq_id: str
    allowed_answers: list[str]
    rows: list[DecisionTableRow]
    default_insufficient_evidence_answer: str
    classifier_instruction: str


class PacketContract(TypedDict, total=False):
    artifact_id: str
    schema_version: str
    sq_id: str
    domain: str
    required_evidence: list[str]
    allowed_answers: list[str]
    outcome_binding_status: str
    source_hierarchy: list[str]
    needs_denominator: bool
    needs_prespecification: bool


class EvidenceStoreArtifact(TypedDict, total=False):
    artifact_id: str
    schema_version: str
    supported_facts: list[EvidenceFact]
    failed_claims: list[EvidenceFact]
    gaps: list[EvidenceGap]


class EvidencePacket(TypedDict, total=False):
    artifact_id: str
    schema_version: str
    sq_id: str
    domain: str
    outcome: str
    required_evidence: list[str]
    contract: PacketContract
    sources: list[PacketSource]
    candidate_facts: list[EvidenceFact]
    gaps: list[EvidenceGap]
    failed_claims: list[EvidenceFact]
    contradictions: list[dict]
    decision_table: DecisionTable
    text: str
    retrieval_confidence: float
    missing_evidence: list[str]
    negative_flags: list[str]
    packet_grade: RetrievalGrade
    packet_readiness: PacketReadiness


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


class MicroAgentRoutingDecision(TypedDict):
    sq_id: str
    status: str
    route: str
    trigger_conditions: list[str]
    reason: str


class RetrievalRepairArtifact(TypedDict, total=False):
    artifact_id: str
    schema_version: str
    sq_id: str
    domain: str
    outcome: str
    packet_artifact_id: str
    trigger_conditions: list[str]
    query_payload: dict
    before_packet_status: dict
    after_packet_status: dict
    source_changes: dict


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
