from rob2_pipeline.constants import DEFAULT_EFFECT_OF_INTEREST, NOT_REPORTED
from rob2_pipeline.models import empty_paper_evidence
from rob2_pipeline.state import RoB2State


DEFAULT_OUTCOME_PROPERTIES = {
    "objective_event": False,
    "clinician_judged": True,
    "patient_reported": False,
    "composite": False,
    "time_to_event": False,
    "safety_harm": False,
    "lab_or_imaging_threshold": False,
    "blinded_adjudication": False,
}


def create_initial_state(
    pdf_path: str,
    outcome: str | None = None,
    effect_of_interest: str = DEFAULT_EFFECT_OF_INTEREST,
    **kwargs,
) -> RoB2State:
    return {
        "pdf_path": pdf_path,
        "full_text": "",
        "evidence": empty_paper_evidence(),
        "docling_doc": None,
        "docling_chunks": [],
        "precomputed_ingestion": kwargs.get("precomputed_ingestion"),
        "trial_retrieval_indexes": kwargs.get("trial_retrieval_indexes") or {},
        "supplementary_paths": list(kwargs.get("supplementary_paths") or []),
        "source_documents": [],
        "supplement_warnings": [],
        "rag_contexts": {},
        "rag_chunk_metadata": {},
        "retrieval_grades": {},
        "evidence_packets": {},
        "evidence_facts": {},
        "packet_grades": {},
        "verification_actions": [],
        "is_rct": False,
        "rct_screen_evidence": "",
        "intervention": NOT_REPORTED,
        "comparator": NOT_REPORTED,
        "outcome": outcome or "",
        "outcome_type": "clinician-composite",
        "outcome_properties": dict(DEFAULT_OUTCOME_PROPERTIES),
        "outcome_classification_support": {
            "support_level": "unsupported",
            "support_rationale": "Outcome classification has not been resolved.",
            "quotes": [],
            "constraints": [],
        },
        "outcome_normalization_artifact": {
            "artifact_id": f"outcome-normalization:{outcome or ''}",
            "schema_version": "outcome-normalization-v1",
            "outcome": outcome or "",
            "normalized_definition": "",
            "aliases": [],
            "outcome_type": "clinician-composite",
            "outcome_properties": dict(DEFAULT_OUTCOME_PROPERTIES),
            "binding_support": {
                "support_level": "unsupported",
                "support_rationale": "Outcome classification has not been resolved.",
                "quotes": [],
                "constraints": [],
            },
            "auto_accept_blocked": True,
            "uncertainty": True,
        },
        "numerical_result": NOT_REPORTED,
        "effect_of_interest": effect_of_interest,
        "registration_number": NOT_REPORTED,
        "registered_endpoint": NOT_REPORTED,
        "registered_secondary_endpoints": kwargs.get(
            "registered_secondary_endpoints", "Not reported"
        ),
        "registered_analysis": NOT_REPORTED,
        "ctgov_outcomes": kwargs.get(
            "ctgov_outcomes", "(ClinicalTrials.gov data not yet retrieved)"
        ),
        "n_randomized": NOT_REPORTED,
        "sources_consulted": [],
        "trial_facts": {},
        "sq_answers": {},
        "domain_judgments": {},
        "domain_rationales": {},
        "overall_judgment": "",
        "overall_rationale": "",
        "ni_count": 0,
        "high_uncertainty_sqs": [],
        "human_review_priority": "HIGH",
        "evidence_validation_flags": [],
        "support_constraints": [],
        "verifier_trace": [],
        "overall_policy": kwargs.get("overall_policy", "official_rob2"),
        "markdown_report": "",
        "errors": [],
        "llm_call_log": [],
    }
