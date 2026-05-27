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

    result = apply_domain2_conditional_control({"effect_of_interest": "ITT"}, sq_answers)

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
