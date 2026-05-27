"""Tests for prompt CT.gov metadata insertion points."""


def test_prompt_domain1_accepts_ctgov_design():
    from rob2_pipeline.prompts import PROMPT_DOMAIN1

    result = PROMPT_DOMAIN1.format(
        intervention="Drug A plus usual care",
        comparator="usual care alone",
        outcome="Functional Recovery",
        randomization_text="Patients were randomized.",
        baseline_text="",
        consort_text="",
        rag_text="",
        ctgov_design="Authoritative ClinicalTrials.gov registry design metadata:\n  Allocation type: RANDOMIZED",
    )

    assert "RANDOMIZED" in result
    assert "registry" in result.lower()


def test_prompt_domain2_sq12_accepts_ctgov_design():
    from rob2_pipeline.prompts import PROMPT_DOMAIN2_SQ12

    result = PROMPT_DOMAIN2_SQ12.format(
        intervention="Drug A plus usual care",
        comparator="usual care alone",
        outcome="Functional Recovery",
        blinding_text="Open-label trial.",
        methods_text="",
        rag_text="",
        ctgov_design="  Masking: NONE (masked parties: not specified)",
    )

    assert "NONE" in result


def test_registry_prompt_guidance_does_not_overclaim_indirect_metadata():
    from rob2_pipeline.prompts import (
        PROMPT_DOMAIN1,
        PROMPT_DOMAIN2_SQ12,
        PROMPT_DOMAIN5,
    )

    assert "central randomization infrastructure" not in PROMPT_DOMAIN1
    assert "direct evidence that a random method was used" not in PROMPT_DOMAIN1
    assert "Masking = DOUBLE or QUADRUPLE supports N or PN" not in PROMPT_DOMAIN2_SQ12
    assert "analysis plan was pre-specified" not in PROMPT_DOMAIN5
    assert "endpoints were specified in the registry" not in PROMPT_DOMAIN5


def test_prompt_domain3_accepts_ctgov_flow():
    from rob2_pipeline.prompts import PROMPT_DOMAIN3

    result = PROMPT_DOMAIN3.format(
        intervention="Drug A plus usual care",
        comparator="usual care alone",
        outcome="Functional Recovery",
        n_randomized="790",
        consort_text="",
        missing_data_text="",
        sensitivity_text="",
        rag_text="",
        ctgov_flow="Participant flow:\n  STARTED: Drug A plus usual care: 397, Usual care alone: 393",
    )

    assert "397" in result


def test_prompt_domain5_accepts_ctgov_description():
    from rob2_pipeline.prompts import PROMPT_DOMAIN5

    result = PROMPT_DOMAIN5.format(
        intervention="Drug A plus usual care",
        comparator="usual care alone",
        outcome="Functional Recovery",
        outcome_type="vital-status",
        numerical_result="HR 0.61",
        registration_number="NCT00309985",
        registered_endpoint="Functional Recovery",
        registered_secondary_endpoints="Symptom Score",
        reported_endpoint="Functional Recovery",
        ctgov_outcomes="PRIMARY: Functional Recovery",
        ctgov_description="Authoritative ClinicalTrials.gov registry description:\nPRIMARY OBJECTIVE: Functional recovery.",
        registration_text="",
        sap_text="",
        results_text="",
        rag_text="",
    )

    assert "PRIMARY OBJECTIVE" in result


def test_domain_prompts_label_primary_evidence_and_retrieved_context():
    from rob2_pipeline.prompts import PROMPT_DOMAIN1

    result = PROMPT_DOMAIN1.format(
        intervention="Drug A plus usual care",
        comparator="usual care alone",
        outcome="Functional Recovery",
        randomization_text="Central allocation by statistical center.",
        baseline_text="Groups were balanced.",
        consort_text="All participants accounted for.",
        rag_text="Generic randomized trial sentence.",
        ctgov_design="",
    )

    assert "PRIMARY EVIDENCE" in result
    assert "ADDITIONAL RETRIEVED CONTEXT" in result
    assert "Central allocation by statistical center." in result
    assert "Generic randomized trial sentence." in result


def test_domain1_prompt_guides_stratified_cooperative_group_concealment_inference():
    from rob2_pipeline.prompts import PROMPT_DOMAIN1

    assert "stratified randomization" in PROMPT_DOMAIN1
    assert "PY rather than NI for Q1.2" in PROMPT_DOMAIN1


def test_domain4_prompt_guides_outcome_specific_q44_reasoning():
    from rob2_pipeline.prompts import PROMPT_DOMAIN4

    result = PROMPT_DOMAIN4.format(
        intervention="Drug A plus usual care",
        comparator="usual care alone",
        outcome="Composite Clinical Response",
        outcome_type="clinician-composite",
        sq_2_1="Y",
        outcome_measurement_text=(
            "Functional recovery was defined as return to usual activities. "
            "Composite clinical response was based on clinician-rated symptoms and repeat testing."
        ),
        blinding_text="Open-label trial.",
        rag_text="",
    )

    assert "definitions for multiple outcomes" in result
    assert "based only on the definition for Composite Clinical Response" in result
    assert "outcomes involving judgment" in result
    assert "all-cause mortality" in result
    assert "mechanical" in result


def test_domain_prompts_include_canonical_methodology_blocks():
    from rob2_pipeline.prompts import (
        PROMPT_DOMAIN1,
        PROMPT_DOMAIN2_CONDITIONAL,
        PROMPT_DOMAIN3,
        PROMPT_DOMAIN4,
    )

    assert "CANONICAL RoB 2 METHODOLOGY" in PROMPT_DOMAIN1
    assert "SQ 1.2" in PROMPT_DOMAIN1
    assert "CANONICAL RoB 2 METHODOLOGY" in PROMPT_DOMAIN2_CONDITIONAL
    assert "SQ 2.3" in PROMPT_DOMAIN2_CONDITIONAL
    assert "N/PN requires affirmative evidence" in PROMPT_DOMAIN2_CONDITIONAL
    assert "CANONICAL RoB 2 METHODOLOGY" in PROMPT_DOMAIN3
    assert "SQ 3.4" in PROMPT_DOMAIN3
    assert "CANONICAL RoB 2 METHODOLOGY" in PROMPT_DOMAIN4
    assert "SQ 4.5" in PROMPT_DOMAIN4


def test_domain_prompts_do_not_duplicate_rendered_sq_rule_cards():
    from rob2_pipeline.prompts import (
        PROMPT_DOMAIN1,
        PROMPT_DOMAIN2_ADHERING_ANALYSIS,
        PROMPT_DOMAIN2_ANALYSIS,
        PROMPT_DOMAIN2_CONDITIONAL,
        PROMPT_DOMAIN3,
        PROMPT_DOMAIN4,
        PROMPT_DOMAIN5,
    )

    duplicated_local_headings = [
        "1.1 Was the allocation sequence random?",
        "2.3 Were there deviations from the intended intervention",
        "3.1 Were data for this outcome available",
        "4.1 Was the method of measuring the outcome inappropriate?",
        "5.1 Were the data that produced this result analysed",
    ]
    combined = "\n".join(
        [
            PROMPT_DOMAIN1,
            PROMPT_DOMAIN2_CONDITIONAL,
            PROMPT_DOMAIN2_ANALYSIS,
            PROMPT_DOMAIN2_ADHERING_ANALYSIS,
            PROMPT_DOMAIN3,
            PROMPT_DOMAIN4,
            PROMPT_DOMAIN5,
        ]
    )

    for heading in duplicated_local_headings:
        assert heading not in combined


def test_domain5_prompt_includes_methodology_and_preserves_outcome_scope():
    from rob2_pipeline.prompts import PROMPT_DOMAIN5

    result = PROMPT_DOMAIN5.format(
        intervention="Drug A plus usual care",
        comparator="usual care alone",
        outcome="Composite Clinical Response",
        outcome_type="clinician-composite",
        numerical_result="HR 0.61",
        registration_number="NCT00309985",
        registered_endpoint="Composite Clinical Response",
        registered_secondary_endpoints="Composite Clinical Response",
        reported_endpoint="Composite Clinical Response",
        ctgov_outcomes="SECONDARY: Composite Clinical Response",
        ctgov_description="",
        registration_text="",
        sap_text="",
        results_text="",
        rag_text="",
    )

    assert "CANONICAL RoB 2 METHODOLOGY" in result
    assert "SQ 5.2" in result
    assert (
        "You are assessing Domain 5 for the specific outcome: Composite Clinical Response"
        in result
    )
