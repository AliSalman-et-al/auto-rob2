from rob2_pipeline.nodes.sq_control import (
    apply_domain2_analysis_control,
    apply_domain2_conditional_control,
    apply_domain2_sq12_control,
    classify_d2_deviation_evidence,
    apply_domain3_control,
    apply_domain4_control,
    classify_d3_completeness_support,
    next_domain2_stage,
)


def test_domain2_sq12_sets_later_questions_na_when_itt_trial_has_no_awareness():
    sq_answers = {"2.1": {"answer": "N"}, "2.2": {"answer": "PN"}}

    result = apply_domain2_sq12_control({"effect_of_interest": "ITT"}, sq_answers)

    assert result["2.3"]["answer"] == "NA"
    assert result["2.4"]["answer"] == "NA"
    assert result["2.5"]["answer"] == "NA"


def test_next_domain2_stage_routes_per_protocol_to_conditional():
    result = next_domain2_stage(
        {
            "effect_of_interest": "per-protocol",
            "sq_answers": {"2.1": {"answer": "N"}, "2.2": {"answer": "N"}},
        }
    )

    assert result == "conditional"


def test_domain2_conditional_sets_later_questions_na_after_no_deviations():
    sq_answers = {"2.3": {"answer": "N"}, "2.4": {"answer": "Y"}}

    result = apply_domain2_conditional_control(
        {"effect_of_interest": "ITT"}, sq_answers
    )

    assert result["2.4"]["answer"] == "NA"
    assert result["2.5"]["answer"] == "NA"


def test_domain2_deviation_gate_accepts_affirmative_no_deviation_evidence():
    sq_answers = {
        "2.3": {
            "answer": "N",
            "quote": "No protocol deviations occurred during the trial.",
            "justification": "The report directly states no deviations occurred.",
        },
        "2.4": {"answer": "Y"},
    }

    result = apply_domain2_conditional_control(_aware_itt_state(), sq_answers)

    assert result["2.3"]["answer"] == "N"
    assert result["2.4"]["answer"] == "NA"
    assert result["2.5"]["answer"] == "NA"


def test_domain2_deviation_gate_blocks_generic_silence_as_no_deviations():
    sq_answers = {
        "2.3": {
            "answer": "N",
            "quote": "No relevant text found",
            "justification": "The report did not mention protocol deviations.",
        }
    }

    result = apply_domain2_conditional_control(_aware_itt_state(), sq_answers)

    assert result["2.3"]["answer"] == "NI"


def test_domain2_deviation_gate_detects_non_adherence_when_sq23_denies_deviations():
    sq_answers = {
        "2.3": {
            "answer": "PN",
            "quote": "Treatment discontinuation occurred in 17% of the intervention arm.",
            "justification": "No protocol deviations were reported.",
        }
    }

    result = apply_domain2_conditional_control(_aware_itt_state(), sq_answers)

    assert result["2.3"]["answer"] == "Y"
    assert "deviations present" in result["2.3"]["justification"]


def test_domain2_deviation_classifier_covers_deviation_signal_types():
    examples = [
        ("Patients crossed over from control to active treatment.", "deviations_present"),
        ("Non-protocol co-interventions were permitted after progression.", "deviations_present"),
        ("Rescue therapy was used more often in the placebo arm.", "deviations_present"),
        ("Dose interruptions were imbalanced between groups.", "deviations_present"),
        ("No major protocol deviations were reported.", "affirmative_no_deviations"),
        ("Methods were described in the protocol and results were summarized.", "insufficient"),
        (
            "No protocol deviations occurred, but 14 patients crossed over to active treatment.",
            "contradictory",
        ),
    ]

    for text, expected in examples:
        assert classify_d2_deviation_evidence({}, text)["classification"] == expected


def test_domain2_analysis_sets_sq27_na_when_sq26_is_probably_yes():
    sq_answers = {"2.6": {"answer": "PY"}, "2.7": {"answer": "Y"}}

    result = apply_domain2_analysis_control({"effect_of_interest": "ITT"}, sq_answers)

    assert result["2.7"]["answer"] == "NA"


def _aware_itt_state():
    return {
        "effect_of_interest": "ITT",
        "sq_answers": {"2.1": {"answer": "Y"}, "2.2": {"answer": "Y"}},
    }


def test_domain3_sets_remaining_questions_na_when_missing_data_not_problematic():
    sq_answers = {
        "3.1": {
            "answer": "Y",
            "quote": "Outcome data were available for 100 of 100 randomized participants.",
        }
    }

    result = apply_domain3_control({}, sq_answers)

    assert result["3.2"]["answer"] == "NA"
    assert result["3.3"]["answer"] == "NA"
    assert result["3.4"]["answer"] == "NA"


def test_domain3_time_to_event_requires_direct_missingness_evidence():
    state = {
        "outcome_properties": {"time_to_event": True},
        "evidence": {
            "d3_missing_data": {
                "text": "The primary analysis used the intention-to-treat population. Overall survival was estimated with Kaplan-Meier methods; patients without events were censored at last follow-up.",
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
            "quote": "patients without events were censored at last follow-up",
            "completeness_calculation": "100/100 = 100%",
            "justification": "The survival analysis censored participants without events.",
        }
    }

    result = apply_domain3_control(state, sq_answers)

    assert result["3.1"]["answer"] == "NI"
    assert result["3.2"]["answer"] != "NA"


def test_domain3_time_to_event_accepts_direct_completeness_evidence():
    state = {
        "outcome_properties": {"time_to_event": True},
        "evidence": {
            "d3_missing_data": {
                "text": "Vital status was ascertained for 198 of 200 randomized participants; two participants were lost to follow-up before the survival analysis cutoff.",
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
            "answer": "PY",
            "quote": "Vital status was ascertained for 198 of 200 randomized participants",
            "completeness_calculation": "198/200 = 99.0%",
            "justification": "Nearly all participants had survival status ascertained.",
        }
    }

    result = apply_domain3_control(state, sq_answers)

    assert result["3.1"]["answer"] == "PY"
    assert result["3.2"]["answer"] == "NA"
    assert result["3.3"]["answer"] == "NA"
    assert result["3.4"]["answer"] == "NA"


def test_domain3_completeness_gate_accepts_direct_denominator_support():
    text = "Outcome data were available for 198 of 200 randomized participants."

    result = classify_d3_completeness_support({}, text)

    assert result["classification"] == "sufficient"


def test_domain3_completeness_gate_blocks_itt_only_support():
    sq_answers = {
        "3.1": {
            "answer": "Y",
            "quote": "The primary analysis used the intention-to-treat population.",
            "justification": "ITT analysis included all randomized participants.",
        }
    }

    result = apply_domain3_control({}, sq_answers)

    assert result["3.1"]["answer"] == "NI"
    assert result["3.2"]["answer"] != "NA"


def test_domain3_completeness_gate_blocks_censoring_only_support():
    sq_answers = {
        "3.1": {
            "answer": "PY",
            "quote": "Patients without events were censored at the date of last assessment.",
            "justification": "Censoring rules were described.",
        }
    }

    result = apply_domain3_control(
        {"outcome_properties": {"time_to_event": True}}, sq_answers
    )

    assert result["3.1"]["answer"] == "NI"


def test_domain3_completeness_gate_blocks_missing_denominator_support():
    sq_answers = {
        "3.1": {
            "answer": "Y",
            "quote": "Missing outcome data were uncommon.",
            "justification": "The report says missingness was uncommon.",
        }
    }

    result = apply_domain3_control({}, sq_answers)

    assert result["3.1"]["answer"] == "NI"


def test_domain3_completeness_gate_flags_contradictory_denominators():
    text = (
        "A total of 200 participants were randomized. "
        "The primary outcome analysis included 160 participants with outcome data."
    )

    result = classify_d3_completeness_support({}, text)

    assert result["classification"] == "contradictory"


def test_domain3_completeness_gate_blocks_safety_population_exclusions():
    sq_answers = {
        "3.1": {
            "answer": "Y",
            "quote": "The safety population included 180 patients who received at least one dose.",
            "justification": "Safety analyses excluded untreated randomized participants.",
        }
    }

    result = apply_domain3_control({}, sq_answers)

    assert result["3.1"]["answer"] == "NI"


def test_domain4_open_label_patient_reported_outcome_sets_assessor_awareness():
    sq_answers = {
        "4.1": {"answer": "N", "quote": "No measurement issue"},
        "4.2": {"answer": "N", "quote": "No definition issue"},
        "4.3": {"answer": "NI", "quote": "Auto-set: no quote"},
        "4.4": {"answer": "NI"},
    }
    state = {
        "outcome_type": "patient-reported",
        "masking_facts": {
            "participant_awareness": {
                "status": "aware",
                "quotes": [{"quote": "Open-label study"}],
            }
        },
        "sq_answers": {
            "2.1": {"answer": "Y", "quote": "Open-label study"},
            "2.2": {"answer": "N", "quote": "Outcome assessors masked"},
        },
    }

    result = apply_domain4_control(state, sq_answers)

    assert result["4.3"]["answer"] == "Y"
    assert result["4.3"]["quote"] == "Open-label study"


def test_domain4_objective_os_stays_on_objective_path_before_correction():
    sq_answers = {
        "4.1": {"answer": "N", "quote": "Vital status"},
        "4.2": {"answer": "N", "quote": "Same follow-up"},
        "4.3": {"answer": "NI", "quote": "No relevant text found"},
        "4.4": {"answer": "NI", "quote": "No relevant text found"},
        "4.5": {"answer": "NI", "quote": "No relevant text found"},
    }
    state = {
        "outcome": "Overall Survival",
        "outcome_code": "OS",
        "outcome_type": "clinician-composite",
        "outcome_properties": {
            "objective_event": True,
            "patient_reported": False,
            "safety_harm": False,
            "blinded_adjudication": False,
        },
        "sq_answers": {
            "2.1": {"answer": "Y", "quote": "Open-label trial"},
            "2.2": {"answer": "Y", "quote": "Investigators aware"},
        },
    }

    result = apply_domain4_control(state, sq_answers)

    assert result["4.3"]["answer"] == "NI"
    assert result["4.4"]["answer"] == "N"
    assert result["4.5"]["answer"] == "NA"


def test_domain4_open_label_pfs_without_blinded_adjudication_defaults_to_some_concerns_path():
    sq_answers = {
        "4.1": {"answer": "N", "quote": "RECIST criteria"},
        "4.2": {"answer": "N", "quote": "same imaging schedule"},
        "4.3": {"answer": "NI", "quote": "No relevant text found"},
        "4.4": {"answer": "NI", "quote": "No relevant text found"},
        "4.5": {"answer": "NI", "quote": "No relevant text found"},
    }
    state = {
        "outcome": "Progression-free survival",
        "outcome_code": "PFS",
        "outcome_type": "clinician-composite",
        "outcome_properties": {"blinded_adjudication": False},
        "masking_facts": {
            "outcome_assessor_awareness": {
                "status": "aware",
                "quotes": [{"quote": "Investigators aware"}],
            },
            "blinded_adjudication": {"status": "absent"},
        },
        "rag_contexts": {
            "d4_measurement": "Progression was assessed by investigators using RECIST.",
        },
        "sq_answers": {
            "2.1": {"answer": "Y", "quote": "Open-label trial"},
            "2.2": {"answer": "Y", "quote": "Investigators aware"},
        },
    }

    result = apply_domain4_control(state, sq_answers)

    assert result["4.3"]["answer"] == "PY"
    assert result["4.4"]["answer"] == "PY"
    assert result["4.5"]["answer"] == "PN"


def test_domain4_open_label_pfs_control_is_identical_without_d2_sq_answers():
    sq_answers = {
        "4.1": {"answer": "N", "quote": "RECIST criteria"},
        "4.2": {"answer": "N", "quote": "same imaging schedule"},
        "4.3": {"answer": "NI", "quote": "Auto-set: no quote"},
        "4.4": {"answer": "NI", "quote": "No relevant text found"},
        "4.5": {"answer": "NI", "quote": "No relevant text found"},
    }
    state = {
        "outcome": "Progression-free survival",
        "outcome_code": "PFS",
        "outcome_type": "clinician-composite",
        "outcome_properties": {"blinded_adjudication": False},
        "masking_facts": {
            "outcome_assessor_awareness": {
                "status": "aware",
                "quotes": [{"quote": "Outcome assessors were aware of assignment."}],
            },
            "blinded_adjudication": {
                "status": "absent",
                "quotes": [{"quote": "No blinded independent adjudication was used."}],
            },
        },
        "rag_contexts": {
            "d4_measurement": "Progression was assessed by investigators using RECIST.",
        },
        "sq_answers": {},
    }
    state_with_d2 = {
        **state,
        "sq_answers": {
            "2.1": {"answer": "Y", "quote": "Open-label trial"},
            "2.2": {"answer": "Y", "quote": "Investigators aware"},
        },
    }

    without_d2 = apply_domain4_control(state, sq_answers)
    with_d2 = apply_domain4_control(state_with_d2, sq_answers)

    assert without_d2 == with_d2
    assert without_d2["4.3"]["quote"] == "Outcome assessors were aware of assignment."


def test_domain4_open_label_pfs_with_blinded_adjudication_does_not_override_answers():
    sq_answers = {
        "4.1": {"answer": "N", "quote": "RECIST criteria"},
        "4.2": {"answer": "N", "quote": "same imaging schedule"},
        "4.3": {"answer": "N", "quote": "Blinded independent central review"},
        "4.4": {"answer": "N", "quote": "Blinded independent central review"},
        "4.5": {"answer": "NA", "quote": "Not applicable"},
    }
    state = {
        "outcome": "Progression-free survival",
        "outcome_code": "PFS",
        "outcome_type": "clinician-composite",
        "outcome_properties": {"blinded_adjudication": True},
        "rag_contexts": {
            "d4_measurement": "Progression was assessed by blinded independent central review.",
        },
        "sq_answers": {
            "2.1": {"answer": "Y", "quote": "Open-label trial"},
            "2.2": {"answer": "Y", "quote": "Investigators aware"},
        },
    }

    result = apply_domain4_control(state, sq_answers)

    assert result["4.3"]["answer"] == "N"
    assert result["4.4"]["answer"] == "NA"
    assert result["4.5"]["answer"] == "NA"
