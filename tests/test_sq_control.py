from rob2_pipeline.nodes.sq_control import (
    apply_domain2_analysis_control,
    apply_domain2_conditional_control,
    apply_domain2_sq12_control,
    apply_domain3_control,
    apply_domain4_control,
    next_domain2_stage,
)


def test_domain2_sq12_sets_later_questions_na_when_itt_trial_has_no_awareness():
    sq_answers = {"2.1": {"answer": "N"}, "2.2": {"answer": "PN"}}

    result = apply_domain2_sq12_control({"effect_of_interest": "ITT"}, sq_answers)

    assert result["2.3"]["answer"] == "NA"
    assert result["2.4"]["answer"] == "NA"
    assert result["2.5"]["answer"] == "NA"
    assert result["2.5"]["support_level"] == "unsupported"
    assert result["2.5"]["support_rationale"] == "Not applicable"


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


def test_domain2_conditional_rejects_design_text_as_actual_deviation():
    sq_answers = {
        "2.3": {
            "answer": "Y",
            "quote": "Masking: NONE (masked parties: not specified)",
            "justification": (
                "The trial was open-label and the protocol described eligibility criteria."
            ),
            "support_rationale": "Design and eligibility text only.",
        },
        "2.4": {"answer": "Y"},
        "2.5": {"answer": "Y"},
    }

    result = apply_domain2_conditional_control(
        {"effect_of_interest": "ITT"}, sq_answers
    )

    assert result["2.3"]["answer"] == "N"
    assert result["2.3"]["d2_actual_deviation_guard_applied"] is True
    assert result["2.4"]["answer"] == "NA"
    assert result["2.5"]["answer"] == "NA"


def test_domain2_conditional_rejects_standard_of_care_plus_as_deviation():
    sq_answers = {
        "2.3": {
            "answer": "Y",
            "quote": "standard of care plus radiotherapy plus abiraterone",
            "justification": (
                "The intervention was added to standard of care in an open-label design."
            ),
            "support_rationale": "Trial design text only.",
        },
        "2.4": {"answer": "Y"},
        "2.5": {"answer": "Y"},
    }

    result = apply_domain2_conditional_control(
        {"effect_of_interest": "ITT"}, sq_answers
    )

    assert result["2.3"]["answer"] == "N"
    assert result["2.4"]["answer"] == "NA"
    assert result["2.5"]["answer"] == "NA"


def test_domain2_conditional_rejects_nonrandom_docetaxel_stratum_as_deviation():
    sq_answers = {
        "2.1": {"answer": "Y"},
        "2.2": {"answer": "Y"},
        "2.3": {
            "answer": "Y",
            "quote": (
                "As the patients were not randomly assigned according to "
                "docetaxel prescription, toxicities recorded in the ADT without "
                "docetaxel and ADT with docetaxel populations are not directly "
                "comparable."
            ),
            "justification": (
                "Non-random docetaxel prescription indicates deviations caused "
                "by trial context."
            ),
            "support_level": "strong",
        },
        "2.4": {"answer": "Y"},
        "2.5": {"answer": "Y"},
    }

    result = apply_domain2_conditional_control({}, sq_answers)

    assert result["2.3"]["answer"] == "N"
    assert result["2.3"]["d2_actual_deviation_guard_applied"] is True
    assert result["2.4"]["answer"] == "NA"
    assert result["2.5"]["answer"] == "NA"


def test_domain2_analysis_sets_sq27_na_when_sq26_is_probably_yes():
    sq_answers = {"2.6": {"answer": "PY"}, "2.7": {"answer": "Y"}}

    result = apply_domain2_analysis_control({"effect_of_interest": "ITT"}, sq_answers)

    assert result["2.7"]["answer"] == "NA"


def test_domain2_analysis_safety_guard_sets_no_analysis_failure_impact():
    sq_answers = {
        "2.6": {"answer": "NI"},
        "2.7": {
            "answer": "NI",
            "quote": "Allocation type: RANDOMIZED",
            "justification": "No exclusion impact evidence.",
        },
    }
    state = {
        "outcome": "Adverse Events",
        "outcome_properties": {"safety_harm": True},
        "evidence": {
            "results": {
                "text": (
                    "The safety population includes patients who actually received "
                    "the assigned treatment. Table 3: adverse events in the safety population."
                )
            }
        },
    }

    result = apply_domain2_analysis_control(state, sq_answers)

    assert result["2.7"]["answer"] == "PN"
    assert result["2.7"]["d2_safety_analysis_guard_applied"] is True


def test_domain3_sets_remaining_questions_na_when_missing_data_not_problematic():
    sq_answers = {"3.1": {"answer": "Y"}}

    result = apply_domain3_control({}, sq_answers)

    assert result["3.2"]["answer"] == "NA"
    assert result["3.3"]["answer"] == "NA"
    assert result["3.4"]["answer"] == "NA"


def test_domain3_time_to_event_guard_rejects_endpoint_events_as_likely_missingness():
    sq_answers = {
        "3.1": {"answer": "PN"},
        "3.2": {"answer": "N"},
        "3.3": {"answer": "Y"},
        "3.4": {
            "answer": "Y",
            "quote": "The primary end point was radiographic progression-free survival.",
            "justification": (
                "Differential missingness due to death or disease progression "
                "makes missingness likely informative."
            ),
            "support_level": "strong",
        },
    }

    result = apply_domain3_control(
        {"outcome": "Progression-Free Survival"}, sq_answers
    )

    assert result["3.4"]["answer"] == "PN"
    assert result["3.4"]["d3_time_to_event_guard_applied"] is True
    assert "observed endpoint events" in result["3.4"]["justification"]


def test_domain3_time_to_event_guard_rejects_endpoint_events_as_incomplete_data():
    sq_answers = {
        "3.1": {
            "answer": "PN",
            "quote": "Withdrawal - Death: Treatment: 117, Control: 197",
            "justification": "Many participants withdrew due to death.",
            "support_level": "strong",
        },
        "3.2": {"answer": "N"},
        "3.3": {"answer": "Y"},
        "3.4": {"answer": "Y"},
    }

    result = apply_domain3_control(
        {"outcome": "Progression-Free Survival"}, sq_answers
    )

    assert result["3.1"]["answer"] == "PY"
    assert result["3.1"]["d3_time_to_event_completeness_guard_applied"] is True
    assert result["3.2"]["answer"] == "NA"
    assert result["3.3"]["answer"] == "NA"
    assert result["3.4"]["answer"] == "NA"


def test_domain3_time_to_event_guard_preserves_true_censoring_evidence():
    sq_answers = {
        "3.1": {"answer": "PN"},
        "3.2": {"answer": "N"},
        "3.3": {"answer": "Y"},
        "3.4": {
            "answer": "Y",
            "quote": "Participants were censored when they switched treatment.",
            "justification": (
                "Treatment switching caused censoring before outcome ascertainment."
            ),
            "support_level": "strong",
        },
    }

    result = apply_domain3_control(
        {"outcome": "Progression-Free Survival"}, sq_answers
    )

    assert result["3.4"]["answer"] == "Y"
    assert "d3_time_to_event_guard_applied" not in result["3.4"]


def test_domain3_safety_guard_rejects_efficacy_flow_as_harms_missingness():
    sq_answers = {
        "3.1": {
            "answer": "N",
            "quote": "(lost to follow-up or consent withdrawn) 228 ... overall population for efficacy analysis",
            "justification": (
                "Lost to follow-up in the efficacy flow could affect adverse events."
            ),
            "support_level": "moderate",
        },
        "3.2": {"answer": "N"},
        "3.3": {"answer": "NI"},
        "3.4": {"answer": "NI"},
    }
    state = {
        "outcome": "Adverse Events",
        "outcome_properties": {"safety_harm": True},
        "evidence": {
            "results": {
                "text": (
                    "Table 3: Adverse events in the safety population. "
                    "The safety population includes patients who actually received "
                    "the assigned treatment. Any adverse events were reported."
                )
            }
        },
    }

    result = apply_domain3_control(state, sq_answers)

    assert result["3.1"]["answer"] == "PY"
    assert result["3.1"]["d3_safety_outcome_binding_guard_applied"] is True


def test_domain3_safety_guard_replaces_wrong_support_even_when_answer_low():
    sq_answers = {
        "3.1": {
            "answer": "PY",
            "quote": (
                "589 overallpopulation for efficacy analysis 583 overallpopulation "
                "for efficacy analysis; lost to follow-up or consentwithdrawn"
            ),
            "justification": (
                "The efficacy population counts show most participants have "
                "outcome data."
            ),
            "support_level": "strong",
        }
    }
    state = {
        "outcome": "Adverse Events",
        "outcome_properties": {"safety_harm": True},
        "evidence": {
            "results": {
                "text": (
                    "Table 3: Adverse events in the safety population. "
                    "The safety population includes patients who actually "
                    "received the assigned treatment."
                )
            }
        },
    }

    result = apply_domain3_control(state, sq_answers)

    assert result["3.1"]["answer"] == "PY"
    assert result["3.1"]["support_level"] == "moderate"
    assert "safety population" in result["3.1"]["quote"].casefold()
    assert result["3.1"]["d3_safety_outcome_binding_guard_applied"] is True


def test_domain3_safety_guard_uses_safety_analysis_counts():
    sq_answers = {
        "3.1": {
            "answer": "NI",
            "quote": "No relevant text found",
            "justification": "Classifier fallback.",
            "support_level": "unsupported",
        }
    }
    count_text = (
        "355 included inthe ADT with docetaxel population for efficacy analysis "
        "350included inthe ADT with docetaxel population for safety analysis "
        "347included inthe ADT with docetaxel population for safety analysis "
        "Table 3: Adverse events in the safety population."
    )
    state = {
        "outcome": "Adverse Events",
        "outcome_properties": {"safety_harm": True},
        "evidence": {"results": {"text": count_text}},
    }

    result = apply_domain3_control(state, sq_answers)

    assert result["3.1"]["answer"] == "PY"
    assert result["3.1"]["support_level"] == "moderate"
    assert "350/355" in result["3.1"]["justification"]
    assert "347/355" in result["3.1"]["justification"]
    assert result["3.1"]["d3_safety_outcome_binding_guard_applied"] is True
    assert result["3.2"]["answer"] == "NA"
    assert result["3.3"]["answer"] == "NA"
    assert result["3.4"]["answer"] == "NA"


def test_domain4_open_label_patient_reported_outcome_sets_assessor_awareness():
    sq_answers = {
        "4.1": {"answer": "N", "quote": "No measurement issue"},
        "4.2": {"answer": "N", "quote": "No definition issue"},
        "4.3": {"answer": "NI", "quote": "Auto-set: no quote"},
        "4.4": {"answer": "NI"},
    }
    state = {
        "outcome_type": "patient-reported",
        "sq_answers": {
            "2.1": {"answer": "Y", "quote": "Open-label study"},
            "2.2": {"answer": "N", "quote": "Outcome assessors masked"},
        },
    }

    result = apply_domain4_control(state, sq_answers)

    assert result["4.3"]["answer"] == "Y"
    assert result["4.3"]["quote"] == "Open-label study"
    assert result["4.3"]["support_level"] == "moderate"
    assert result["4.3"]["support_rationale"]


def test_domain4_objective_outcome_control_inherits_classification_support():
    sq_answers = {
        "4.1": {"answer": "N", "quote": "No measurement issue"},
        "4.2": {"answer": "N", "quote": "No definition issue"},
        "4.3": {"answer": "NI"},
        "4.4": {"answer": "NI"},
    }
    state = {
        "outcome_type": "vital-status",
        "outcome_classification_support": {
            "support_level": "weak",
            "support_rationale": "Only indirect evidence supports vital-status classification.",
        },
        "sq_answers": {},
    }

    result = apply_domain4_control(state, sq_answers)

    assert result["4.4"]["answer"] == "N"
    assert result["4.4"]["support_level"] == "weak"
    assert (
        result["4.4"]["support_rationale"]
        == "Only indirect evidence supports vital-status classification."
    )


def test_domain4_objective_outcome_corrects_unclear_differential_measurement():
    sq_answers = {
        "4.1": {"answer": "N", "quote": "All time-to-event endpoints were determined."},
        "4.2": {"answer": "NI", "quote": "No group-specific differences reported."},
        "4.3": {"answer": "NI"},
        "4.4": {"answer": "NI"},
    }
    state = {
        "outcome_type": "vital-status",
        "outcome_classification_support": {
            "support_level": "moderate",
            "support_rationale": "Overall survival is a vital-status endpoint.",
        },
        "sq_answers": {},
    }

    result = apply_domain4_control(state, sq_answers)

    assert result["4.2"]["answer"] == "PN"
    assert result["4.2"]["support_level"] == "moderate"
    assert "differential ascertainment" in result["4.2"]["justification"]
    assert result["4.4"]["answer"] == "N"


def test_domain4_vital_status_control_sets_suitable_measurement_when_unspecified():
    sq_answers = {
        "4.1": {"answer": "NI", "quote": "All time-to-event endpoints."},
        "4.2": {"answer": "NI"},
        "4.3": {"answer": "PY"},
        "4.4": {"answer": "NI"},
        "4.5": {"answer": "NI"},
    }
    state = {
        "outcome": "Overall Survival",
        "outcome_type": "vital-status",
        "outcome_classification_support": {
            "support_level": "moderate",
            "support_rationale": "Overall survival is vital status.",
        },
        "sq_answers": {
            "2.1": {"answer": "Y", "quote": "Open-label trial."},
            "2.2": {"answer": "Y", "quote": "Open-label trial."},
        },
    }

    result = apply_domain4_control(state, sq_answers)

    assert result["4.1"]["answer"] == "N"
    assert result["4.1"]["d4_objective_control_applied"] is True
    assert result["4.2"]["answer"] == "PN"
    assert result["4.4"]["answer"] == "N"
    assert result["4.5"]["answer"] == "NA"


def test_domain4_safety_guard_separates_awareness_from_likely_influence():
    sq_answers = {
        "4.1": {"answer": "NI"},
        "4.2": {"answer": "NI"},
        "4.3": {
            "answer": "Y",
            "quote": "Neither the investigators nor the patients were masked.",
        },
        "4.4": {"answer": "NI"},
        "4.5": {
            "answer": "Y",
            "quote": "Neither the investigators nor the patients were masked.",
            "justification": (
                "Given the open-label nature, assessors are likely influenced."
            ),
            "support_level": "strong",
        },
    }
    state = {
        "outcome": "Adverse Events",
        "outcome_type": "clinician-graded",
        "outcome_properties": {"safety_harm": True},
        "sq_answers": {
            "2.1": {"answer": "Y", "quote": "Open-label trial."},
            "2.2": {"answer": "Y", "quote": "Open-label trial."},
        },
        "evidence": {
            "results": {
                "text": (
                    "Adverse events were graded on the basis of the National "
                    "Cancer Institute Common Terminology Criteria."
                )
            }
        },
    }

    result = apply_domain4_control(state, sq_answers)

    assert result["4.1"]["answer"] == "N"
    assert result["4.2"]["answer"] == "PN"
    assert result["4.5"]["answer"] == "PN"
    assert result["4.5"]["d4_safety_influence_guard_applied"] is True


def test_domain4_safety_guard_rejects_efficacy_text_for_influence():
    sq_answers = {
        "4.1": {"answer": "N"},
        "4.2": {"answer": "PN"},
        "4.3": {
            "answer": "PY",
            "quote": "Neither the investigators nor the patients were masked.",
        },
        "4.4": {
            "answer": "PN",
            "quote": (
                "CRPC was defined as either radiographic progression or a "
                "confirmed PSA rise."
            ),
            "justification": (
                "The definition of CRPC involves objective radiographic and PSA "
                "criteria, so the outcome is mostly objective."
            ),
            "support_level": "strong",
            "supporting_fact_artifact_ids": ["fact"],
        },
        "4.5": {"answer": "NA"},
    }
    state = {
        "outcome": "Adverse Events",
        "outcome_type": "clinician-graded",
        "outcome_properties": {"safety_harm": True},
        "sq_answers": {
            "2.1": {"answer": "Y", "quote": "Open-label trial."},
            "2.2": {"answer": "Y", "quote": "Open-label trial."},
        },
    }

    result = apply_domain4_control(state, sq_answers)

    assert result["4.4"]["answer"] == "NI"
    assert result["4.4"]["d4_safety_influence_source_guard_applied"] is True
    assert result["4.5"]["answer"] == "PN"


def test_domain4_safety_guard_separates_ctcae_possible_from_likely_influence():
    sq_answers = {
        "4.1": {"answer": "N"},
        "4.2": {"answer": "PN"},
        "4.3": {
            "answer": "PY",
            "quote": "Neither the investigators nor the patients were masked.",
        },
        "4.4": {
            "answer": "PN",
            "quote": "Neither the investigators nor the patients were masked.",
            "justification": (
                "Adverse events were graded using standardized CTCAE criteria, "
                "which are largely objective; although assessors were unblinded, "
                "the grading system limits the potential for bias."
            ),
            "support_level": "moderate",
        },
        "4.5": {"answer": "NA"},
    }
    state = {
        "outcome": "Adverse Events",
        "outcome_type": "clinician-graded",
        "outcome_properties": {"safety_harm": True},
        "sq_answers": {
            "2.1": {"answer": "Y", "quote": "Open-label trial."},
            "2.2": {"answer": "Y", "quote": "Open-label trial."},
        },
    }

    result = apply_domain4_control(state, sq_answers)

    assert result["4.4"]["answer"] == "PY"
    assert result["4.4"]["d4_safety_possible_influence_guard_applied"] is True
    assert result["4.5"]["answer"] == "PN"
    assert result["4.5"]["d4_safety_influence_guard_applied"] is True
