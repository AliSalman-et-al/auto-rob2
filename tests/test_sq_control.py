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


def test_domain2_analysis_sets_sq27_na_when_sq26_is_probably_yes():
    sq_answers = {"2.6": {"answer": "PY"}, "2.7": {"answer": "Y"}}

    result = apply_domain2_analysis_control({"effect_of_interest": "ITT"}, sq_answers)

    assert result["2.7"]["answer"] == "NA"


def test_domain3_sets_remaining_questions_na_when_missing_data_not_problematic():
    sq_answers = {"3.1": {"answer": "Y"}}

    result = apply_domain3_control({}, sq_answers)

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
