from rob2_pipeline.nodes.domain5 import apply_domain5_selective_reporting_guard


def test_d5_guard_downgrades_other_outcome_inference():
    answers = {
        "5.2": {
            "answer": "Y",
            "quote": "SECONDARY: Time to Clinical Progression; QOL Change",
            "justification": (
                "The trial registered several eligible outcome measurements, but "
                "the published report presents only overall survival."
            ),
            "support_level": "strong",
            "support_rationale": "Multiple eligible outcomes were pre-specified.",
            "uncertainty": False,
            "supporting_fact_artifact_ids": ["fact"],
        }
    }

    guarded = apply_domain5_selective_reporting_guard(
        {"outcome": "Overall Survival"}, answers
    )

    assert guarded["5.2"]["answer"] == "PN"
    assert guarded["5.2"]["d5_guard_applied"] is True
    assert guarded["5.2"]["supporting_fact_artifact_ids"] == []
    assert "same assessed outcome" in guarded["5.2"]["justification"]


def test_d5_guard_downgrades_subset_reporting_inference_without_result_selection():
    answers = {
        "5.2": {
            "answer": "Y",
            "quote": (
                "The coprimary endpoints were radiographic progression-free "
                "survival and overall survival."
            ),
            "justification": (
                "The registry lists many eligible outcomes, but the published "
                "paper reports only a subset (the coprimary endpoints) without "
                "justification, indicating selective reporting of a subset of "
                "pre-specified measurements."
            ),
            "support_level": "strong",
            "support_rationale": (
                "The reported results focus on a limited set of registered "
                "outcomes."
            ),
            "uncertainty": False,
        }
    }

    guarded = apply_domain5_selective_reporting_guard(
        {"outcome": "Adverse Events"}, answers
    )

    assert guarded["5.2"]["answer"] == "PN"
    assert guarded["5.2"]["d5_guard_applied"] is True


def test_d5_guard_converts_ni_when_answer_finds_no_result_selection():
    answers = {
        "5.2": {
            "answer": "NI",
            "quote": (
                "Registered outcomes from ClinicalTrials.gov: PRIMARY: Survival; "
                "SECONDARY: adverse events."
            ),
            "justification": (
                "While the registry lists multiple eligible outcome measurements, "
                "the packet does not provide information on whether a subset was "
                "selectively reported based on the results, so the level of "
                "reporting bias cannot be determined."
            ),
            "support_level": "moderate",
            "support_rationale": (
                "The evidence confirms eligible outcomes but lacks detail on the "
                "selection process for the reported outcomes."
            ),
            "supporting_fact_artifact_ids": ["fact"],
        }
    }

    guarded = apply_domain5_selective_reporting_guard(
        {"outcome": "Adverse Events"}, answers
    )

    assert guarded["5.2"]["answer"] == "PN"
    assert guarded["5.2"]["d5_guard_applied"] is True
    assert guarded["5.2"]["supporting_fact_artifact_ids"] == []


def test_d5_guard_downgrades_other_endpoint_family_analysis_inference():
    answers = {
        "5.3": {
            "answer": "Y",
            "quote": "The risk of radiographic progression or death was reduced.",
            "justification": (
                "Multiple eligible analyses such as radiographic progression-free "
                "survival, PSA progression, and time to skeletal event were possible, "
                "yet only the primary rPFS analysis is presented."
            ),
            "support_level": "strong",
            "support_rationale": "Other endpoint families are listed in the registry.",
            "uncertainty": False,
        }
    }

    guarded = apply_domain5_selective_reporting_guard(
        {"outcome": "Progression-Free Survival"}, answers
    )

    assert guarded["5.3"]["answer"] == "PN"
    assert guarded["5.3"]["d5_guard_applied"] is True


def test_d5_guard_preserves_result_based_selection_evidence():
    answers = {
        "5.3": {
            "answer": "Y",
            "quote": "The reported analysis was selected on the basis of the results.",
            "justification": (
                "The analysis was selected on the basis of the results from "
                "multiple eligible analyses of overall survival."
            ),
            "support_level": "strong",
            "support_rationale": "Direct result-based selection evidence.",
            "uncertainty": False,
        }
    }

    guarded = apply_domain5_selective_reporting_guard(
        {"outcome": "Overall Survival"}, answers
    )

    assert guarded["5.3"]["answer"] == "Y"
    assert "d5_guard_applied" not in guarded["5.3"]


def test_d5_guard_corrects_nonselective_reporting_mislabeled_as_yes():
    answers = {
        "5.3": {
            "answer": "Y",
            "quote": "The primary end point was radiographic progression-free survival.",
            "justification": (
                "The reported analysis corresponds to the prespecified primary "
                "endpoint and no alternative analyses are presented."
            ),
            "support_level": "strong",
            "support_rationale": (
                "The result aligns with the prespecified analysis plan and was "
                "fully reported without selective omission."
            ),
            "uncertainty": False,
            "supporting_fact_artifact_ids": ["fact"],
        }
    }

    guarded = apply_domain5_selective_reporting_guard(
        {"outcome": "Progression-Free Survival"}, answers
    )

    assert guarded["5.3"]["answer"] == "N"
    assert guarded["5.3"]["d5_guard_applied"] is True
    assert "does not show result-based selection" in guarded["5.3"]["justification"]


def test_d5_guard_corrects_registered_match_mislabeled_as_yes():
    answers = {
        "5.2": {
            "answer": "Y",
            "quote": "The primary end point was radiographic progression-free survival.",
            "justification": (
                "Both the ClinicalTrials.gov registration and the trial manuscript "
                "explicitly state the primary outcome, confirming that the eligible "
                "outcome measurement was pre-specified and correctly reported."
            ),
            "support_level": "strong",
            "support_rationale": (
                "Consistent description of the primary endpoint across registry "
                "and primary paper provides strong evidence of eligible outcome "
                "measurement."
            ),
            "uncertainty": False,
        }
    }

    guarded = apply_domain5_selective_reporting_guard(
        {"outcome": "Progression-Free Survival"}, answers
    )

    assert guarded["5.2"]["answer"] == "N"
    assert guarded["5.2"]["d5_guard_applied"] is True


def test_d5_guard_corrects_unsupported_prespecification_ni_when_registry_matches():
    answers = {
        "5.1": {
            "answer": "NI",
            "quote": "No relevant text found",
            "justification": "The classifier cited evidence outside the packet.",
            "support_level": "unsupported",
        },
        "5.2": {"answer": "N", "support_level": "strong"},
        "5.3": {
            "answer": "NI",
            "quote": "No relevant text found",
            "justification": "No alternative analysis evidence was cited.",
            "support_level": "unsupported",
        },
    }
    state = {
        "outcome": "Progression-Free Survival",
        "registered_endpoint": "PRIMARY: Radiographic Progression-Free Survival (rPFS)",
        "numerical_result": "The primary end point was radiographic progression-free survival; hazard ratio 0.39.",
    }

    guarded = apply_domain5_selective_reporting_guard(state, answers)

    assert guarded["5.1"]["answer"] == "Y"
    assert guarded["5.1"]["d5_prespecification_guard_applied"] is True
    assert guarded["5.3"]["answer"] == "PN"
    assert guarded["5.3"]["d5_guard_applied"] is True
