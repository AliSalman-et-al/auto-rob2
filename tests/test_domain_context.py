from rob2_pipeline.models import empty_paper_evidence
from rob2_pipeline.nodes.domain_context import (
    build_domain1_context,
    build_domain2_analysis_context,
    build_domain2_conditional_context,
    build_domain2_sq12_context,
    build_domain4_context,
    build_domain5_context,
)


def test_domain1_context_preserves_structured_trial_and_packet_text():
    evidence = empty_paper_evidence()
    evidence["d1_randomization"]["text"] = "Randomization used a central system."
    evidence["baseline_table"]["text"] = "Baseline factors were balanced."
    evidence["consort_flow"]["text"] = "All randomized patients were included."
    state = {
        "evidence": evidence,
        "rag_contexts": {"d1": "legacy generic retrieval context."},
        "trial_facts": {
            "randomization": "Trial fact randomization.",
            "allocation_concealment": "Trial fact concealment.",
        },
        "evidence_packets": {
            "1.1": {
                "domain": "d1",
                "required_evidence": ["sequence_generation"],
                "missing_evidence": [],
                "negative_flags": [],
                "sources": [
                    {
                        "document_role": "primary",
                        "document_name": "paper.pdf",
                        "page_numbers": [2],
                        "section": "Methods",
                        "text": "Computer-generated sequence.",
                    }
                ],
            }
        },
        "ctgov_design": "Registry design text.",
    }

    context = build_domain1_context(state)

    assert "Randomization used a central system" in context.randomization_text
    assert "Trial fact concealment" in context.randomization_text
    assert context.baseline_text == "Baseline factors were balanced."
    assert context.consort_text == "All randomized patients were included."
    assert "SQ 1.1 verified evidence packet" in context.rag_text
    assert "legacy generic retrieval context" not in context.rag_text
    assert context.ctgov_design == "Registry design text."


def test_domain2_stage_contexts_preserve_stage_specific_inputs():
    evidence = empty_paper_evidence()
    evidence["d2_blinding"]["text"] = "Open-label trial."
    evidence["methods"]["text"] = "Protocol methods."
    evidence["results"]["text"] = "Protocol deviations were balanced."
    evidence["d4_outcome_meas"]["text"] = "ITT analysis population."
    state = {
        "evidence": evidence,
        "rag_contexts": {
            "d2_blinding": "RAG blinding.",
            "d2_deviations": "RAG deviations.",
            "d2_analysis": "RAG analysis.",
        },
        "trial_facts": {
            "masking": "Trial fact masking.",
            "protocol_deviations": "Trial fact deviations.",
            "protocol_amendments": "Trial fact amendments.",
            "analysis_populations": "Trial fact analysis.",
        },
        "evidence_packets": {},
        "sq_answers": {"2.1": {"answer": "Y"}, "2.2": {"answer": "PY"}},
        "effect_of_interest": "per-protocol",
        "ctgov_design": "Registry masking data.",
    }

    sq12 = build_domain2_sq12_context(state)
    conditional = build_domain2_conditional_context(state)
    analysis = build_domain2_analysis_context(state)

    assert "Open-label trial" in sq12.blinding_text
    assert "Trial fact masking" in sq12.blinding_text
    assert sq12.methods_text == "Protocol methods."
    assert sq12.ctgov_design == "Registry masking data."
    assert conditional.sq_2_1 == "Y"
    assert conditional.sq_2_2 == "PY"
    assert "Trial fact amendments" in conditional.deviations_text
    assert conditional.concomitant_text == "Protocol methods."
    assert analysis.effect_of_interest == "per-protocol"
    assert analysis.analysis_text == "ITT analysis population."
    assert "Trial fact analysis" in analysis.results_text


def test_domain4_context_includes_all_prompt_fields_without_legacy_rag_keys():
    evidence = empty_paper_evidence()
    evidence["d4_outcome_meas"]["text"] = "Radiographic progression was assessed."
    evidence["d2_blinding"]["text"] = "Outcome assessors were blinded."
    state = {
        "evidence": evidence,
        "rag_contexts": {
            "d4_measurement": "RAG measurement context.",
            "d4_assessor": "RAG assessor context.",
        },
        "evidence_packets": {},
        "sq_answers": {"2.1": {"answer": "N"}},
        "outcome_type": "clinician-composite",
    }

    context = build_domain4_context(state)

    assert context.sq_2_1 == "N"
    assert context.outcome_type == "clinician-composite"
    assert context.outcome_measurement_text == "Radiographic progression was assessed."
    assert context.blinding_text == "Outcome assessors were blinded."
    assert "RAG measurement context" not in context.rag_text
    assert "RAG assessor context" not in context.rag_text


def test_domain5_context_preserves_registry_and_reported_outcome_binding():
    evidence = empty_paper_evidence()
    evidence["d5_registration"]["text"] = "NCT registration reported OS."
    evidence["d4_outcome_meas"]["text"] = "Overall survival was measured."
    evidence["results"]["text"] = "Hazard ratio was reported."
    state = {
        "evidence": evidence,
        "rag_contexts": {"d5": "RAG registration context."},
        "evidence_packets": {},
        "outcome": "Overall Survival",
        "outcome_type": "vital-status",
        "numerical_result": "HR 0.80",
        "registration_number": "NCT12345678",
        "registered_endpoint": "Overall Survival",
        "registered_secondary_endpoints": "Progression-Free Survival",
        "ctgov_outcomes": "PRIMARY: Overall Survival",
        "ctgov_description": "Registry objective text.",
    }

    context = build_domain5_context(state)

    assert context.reported_endpoint == "Overall Survival"
    assert context.registered_endpoint == "Overall Survival"
    assert context.registered_secondary_endpoints == "Progression-Free Survival"
    assert context.ctgov_outcomes == "PRIMARY: Overall Survival"
    assert context.ctgov_description == "Registry objective text."
    assert context.registration_text == "NCT registration reported OS."
    assert context.sap_text == "Overall survival was measured."
    assert context.results_text == "Hazard ratio was reported."
    assert context.rag_text == ""
