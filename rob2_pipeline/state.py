import operator
from typing import Annotated, Any, TypedDict

from rob2_pipeline.models import PaperEvidence
from rob2_pipeline.types import (
    EvidenceFact,
    EvidencePacket,
    EvidenceValidationFlag,
    LLMCallLogEntry,
    OutcomeProperties,
    PivotalityTest,
    RetrievalGrade,
    SourceDocument,
    SqSupportAdjudication,
    SupportConstraint,
    TrialFacts,
    VerifierTraceEntry,
)


def merge_dicts(left: dict, right: dict) -> dict:
    merged = dict(left or {})
    merged.update(right or {})
    return merged


def take_latest(_left, right):
    return right


class RoB2State(TypedDict, total=False):
    # === INPUT ===
    pdf_path: Annotated[str, take_latest]
    full_text: Annotated[str, take_latest]
    evidence: Annotated[PaperEvidence, take_latest]
    docling_doc: Annotated[Any, take_latest]
    docling_chunks: Annotated[list, take_latest]
    precomputed_ingestion: Annotated[Any, take_latest]
    trial_retrieval_indexes: Annotated[dict, take_latest]
    supplementary_paths: Annotated[list[str], take_latest]
    source_documents: Annotated[list[SourceDocument], take_latest]
    supplement_warnings: Annotated[list[str], take_latest]
    rag_contexts: Annotated[dict[str, str], take_latest]
    rag_chunk_metadata: Annotated[dict, merge_dicts]
    retrieval_grades: Annotated[dict[str, RetrievalGrade], merge_dicts]
    evidence_packets: Annotated[dict[str, EvidencePacket], merge_dicts]
    evidence_facts: Annotated[dict[str, list[EvidenceFact]], merge_dicts]
    packet_grades: Annotated[dict[str, RetrievalGrade], merge_dicts]
    verification_actions: Annotated[list[dict], take_latest]

    # === PRELIMINARY INFO ===
    is_rct: Annotated[bool, take_latest]
    rct_screen_evidence: Annotated[str, take_latest]
    intervention: Annotated[str, take_latest]
    comparator: Annotated[str, take_latest]
    outcome: Annotated[str, take_latest]
    outcome_type: Annotated[str, take_latest]
    outcome_properties: Annotated[OutcomeProperties, take_latest]
    numerical_result: Annotated[str, take_latest]
    effect_of_interest: Annotated[str, take_latest]
    registration_number: Annotated[str, take_latest]
    registered_endpoint: Annotated[str, take_latest]
    registered_secondary_endpoints: Annotated[str, take_latest]
    registered_analysis: Annotated[str, take_latest]
    ctgov_outcomes: Annotated[str, take_latest]
    ctgov_design: Annotated[str, take_latest]
    ctgov_description: Annotated[str, take_latest]
    ctgov_flow: Annotated[str, take_latest]
    n_randomized: Annotated[str, take_latest]
    sources_consulted: Annotated[list[str], take_latest]
    trial_facts: Annotated[TrialFacts, take_latest]

    # === SIGNALING QUESTION ANSWERS ===
    sq_answers: Annotated[dict[str, dict], merge_dicts]

    # === DOMAIN JUDGMENTS (set by deterministic nodes) ===
    initial_domain_judgments: Annotated[dict[str, str], merge_dicts]
    initial_domain_rationales: Annotated[dict[str, str], merge_dicts]
    domain_judgments: Annotated[dict[str, str], merge_dicts]
    domain_rationales: Annotated[dict[str, str], merge_dicts]
    pivotality_tests: Annotated[dict[str, list[PivotalityTest]], merge_dicts]
    sq_support_adjudications: Annotated[
        dict[str, list[SqSupportAdjudication]], merge_dicts
    ]

    # === OVERALL ===
    overall_judgment: Annotated[str, take_latest]
    overall_rationale: Annotated[str, take_latest]

    # === QUALITY FLAGS ===
    ni_count: Annotated[int, take_latest]
    high_uncertainty_sqs: Annotated[list[str], take_latest]
    human_review_priority: Annotated[str, take_latest]
    evidence_validation_flags: Annotated[list[EvidenceValidationFlag], take_latest]
    support_constraints: Annotated[list[SupportConstraint], take_latest]
    verifier_trace: Annotated[list[VerifierTraceEntry], take_latest]
    overall_policy: Annotated[str, take_latest]

    # === OUTPUT ===
    markdown_report: Annotated[str, take_latest]

    # === METADATA ===
    errors: Annotated[list[str], take_latest]
    llm_call_log: Annotated[list[LLMCallLogEntry], operator.add]
