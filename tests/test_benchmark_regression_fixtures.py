import pytest

from rob2_pipeline.models import empty_paper_evidence
from rob2_pipeline.nodes.outcome_resolver import outcome_resolver_node
from rob2_pipeline.nodes.sq_control import (
    apply_domain2_conditional_control,
    apply_domain3_control,
    apply_domain4_control,
)


def _open_label_pfs_state(trial: str) -> dict:
    return {
        "trial_name": trial,
        "outcome": "Progression-free survival",
        "outcome_code": "PFS",
        "outcome_type": "clinician-composite",
        "outcome_properties": {"blinded_adjudication": False},
        "masking_facts": {
            "outcome_assessor_awareness": {
                "status": "aware",
                "quotes": [{"quote": f"{trial} was open label."}],
            },
            "blinded_adjudication": {"status": "absent"},
        },
        "rag_contexts": {
            "d4_measurement": "Progression was assessed by investigators using RECIST.",
        },
        "sq_answers": {},
    }


def _d4_sq_answers() -> dict:
    return {
        "4.1": {"answer": "N", "quote": "RECIST criteria"},
        "4.2": {"answer": "N", "quote": "same imaging schedule"},
        "4.3": {"answer": "NI", "quote": "Auto-set: no quote"},
        "4.4": {"answer": "NI", "quote": "No relevant text found"},
        "4.5": {"answer": "NI", "quote": "No relevant text found"},
    }


@pytest.mark.parametrize("trial", ["STAMPEDE", "PEACE-1", "CHAARTED"])
def test_benchmark_pfs_d4_does_not_depend_on_d2_sq_execution_order(trial):
    state_without_d2 = _open_label_pfs_state(trial)
    state_with_d2 = {
        **state_without_d2,
        "sq_answers": {
            "2.1": {"answer": "Y", "quote": f"{trial} was open label."},
            "2.2": {"answer": "Y", "quote": "Investigators were aware."},
        },
    }

    without_d2 = apply_domain4_control(state_without_d2, _d4_sq_answers())
    with_d2 = apply_domain4_control(state_with_d2, _d4_sq_answers())

    assert without_d2 == with_d2
    assert without_d2["4.3"]["answer"] == "PY"
    assert without_d2["4.4"]["answer"] == "PY"
    assert without_d2["4.5"]["answer"] == "PN"


@pytest.mark.parametrize(
    ("trial", "outcome_code", "outcome"),
    [
        ("STAMPEDE", "OS", "Overall survival"),
        ("STAMPEDE", "PFS", "Progression-free survival"),
        ("PEACE-1", "OS", "Overall survival"),
        ("PEACE-1", "PFS", "Progression-free survival"),
        ("CHAARTED", "PFS", "Progression-free survival"),
    ],
)
def test_benchmark_time_to_event_d3_blocks_unsupported_completeness_answers(
    trial, outcome_code, outcome
):
    state = {
        "trial_name": trial,
        "outcome": outcome,
        "outcome_code": outcome_code,
        "outcome_properties": {"time_to_event": True},
        "evidence": {
            "d3_missing_data": {
                "text": (
                    "The primary analysis used the intention-to-treat population. "
                    "Survival was estimated with Kaplan-Meier methods and patients "
                    "without events were censored at the last assessment."
                ),
                "tables": "",
            },
            "consort_flow": {"text": "", "tables": ""},
            "results": {"text": "", "tables": ""},
        },
        "rag_contexts": {"d3": ""},
        "ctgov_flow": "",
    }
    sq_answers = {
        "3.1": {
            "answer": "Y",
            "quote": "patients without events were censored at the last assessment",
            "completeness_calculation": "Not calculable from available text",
            "justification": "The analysis described censoring rules.",
        }
    }

    result = apply_domain3_control(state, sq_answers)

    assert result["3.1"]["answer"] == "NI"
    assert result["3.2"]["answer"] != "NA"


@pytest.mark.parametrize(
    ("trial", "outcome_code", "outcome"),
    [
        ("STAMPEDE", "AE", "Adverse events"),
        ("PEACE-1", "AE", "Adverse events"),
    ],
)
def test_benchmark_aware_ae_d2_blocks_unsupported_no_deviation_answers(
    trial, outcome_code, outcome
):
    state = {
        "trial_name": trial,
        "outcome": outcome,
        "outcome_code": outcome_code,
        "effect_of_interest": "ITT",
        "masking_facts": {
            "participant_awareness": {"status": "aware"},
            "personnel_awareness": {"status": "aware"},
        },
        "evidence": empty_paper_evidence(),
        "rag_contexts": {
            "d2_deviations": "The trial was open label and analysed by intention to treat.",
        },
        "trial_facts": {"protocol_deviations": ""},
        "sq_answers": {
            "2.1": {"answer": "Y", "quote": "open label"},
            "2.2": {"answer": "Y", "quote": "open label"},
        },
    }
    sq_answers = {
        "2.3": {
            "answer": "N",
            "quote": "No relevant text found",
            "justification": "The report did not mention deviations.",
        }
    }

    result = apply_domain2_conditional_control(state, sq_answers)

    assert result["2.3"]["answer"] == "NI"
    assert result["2.4"]["answer"] == "NA"
    assert result["2.5"]["answer"] == "NA"


@pytest.mark.parametrize(
    ("trial", "outcome_code", "outcome", "primary_text", "ctgov_outcomes", "expected_type"),
    [
        (
            "STAMPEDE",
            "OS",
            "Overall survival",
            "Overall survival was defined as death from any cause.",
            "PRIMARY: Radiographic progression-free survival; SECONDARY: Overall Survival",
            "vital-status",
        ),
        (
            "STAMPEDE",
            "PFS",
            "Progression-free survival",
            "Progression-free survival was time to radiographic progression or death.",
            "PRIMARY: Patient-reported quality of life; SECONDARY: Progression-free survival",
            "clinician-composite",
        ),
        (
            "STAMPEDE",
            "AE",
            "Adverse events",
            "Adverse events were graded according to CTCAE criteria.",
            "PRIMARY: Overall Survival; SECONDARY: Serious adverse events",
            "clinician-graded",
        ),
        (
            "PEACE-1",
            "OS",
            "Overall survival",
            "Overall survival was defined as death from any cause.",
            "PRIMARY: Radiographic progression-free survival; SECONDARY: Overall Survival",
            "vital-status",
        ),
        (
            "PEACE-1",
            "PFS",
            "Progression-free survival",
            "Radiographic progression-free survival was time to progression or death.",
            "PRIMARY: Quality of life; SECONDARY: Progression-free survival",
            "clinician-composite",
        ),
        (
            "PEACE-1",
            "AE",
            "Adverse events",
            "Adverse events were graded according to CTCAE criteria.",
            "PRIMARY: Overall Survival; SECONDARY: Serious adverse events",
            "clinician-graded",
        ),
        (
            "CHAARTED",
            "PFS",
            "Progression-free survival",
            "Progression-free survival was defined as progression or death.",
            "PRIMARY: Overall survival; SECONDARY: Progression-free survival",
            "clinician-composite",
        ),
    ],
)
def test_benchmark_registry_enriches_without_overriding_clearer_endpoint_sources(
    trial,
    outcome_code,
    outcome,
    primary_text,
    ctgov_outcomes,
    expected_type,
):
    result = outcome_resolver_node(
        {
            "trial_name": trial,
            "outcome": outcome,
            "outcome_code": outcome_code,
            "evidence": {
                "abstract": {"text": primary_text, "tables": [], "source": "primary"},
            },
            "ctgov_outcomes": ctgov_outcomes,
            "errors": [],
        }
    )

    support_priorities = [
        item["source_priority"] for item in result["outcome_resolution"]["support"]
    ]
    assert result["outcome_type"] == expected_type
    assert 2 in support_priorities
    assert 4 in support_priorities
