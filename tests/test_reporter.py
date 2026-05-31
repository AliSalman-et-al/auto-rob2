import pytest

from rob2_pipeline.nodes.reporter import report_formatter_node


@pytest.mark.parametrize(
    ("sq_id", "support_level", "support_rationale"),
    [
        ("1.1", "strong", "Direct randomization text supports the answer."),
        ("1.2", "moderate", "Allocation text is relevant but indirect."),
        ("1.3", "weak", "Baseline text only weakly supports the answer."),
        ("2.1", "unsupported", "No cited evidence supports awareness."),
        ("2.5", "unsupported", "Not applicable"),
    ],
)
def test_markdown_report_shows_sq_support_metadata(
    sq_id, support_level, support_rationale
):
    state = {
        "sq_answers": {
            sq_id: {
                "answer": "NA" if sq_id == "2.5" else "Y",
                "quote": "Relevant quoted text.",
                "justification": "Reviewer-facing justification.",
                "support_level": support_level,
                "support_rationale": support_rationale,
            }
        },
        "domain_judgments": {},
        "domain_rationales": {},
    }

    report = report_formatter_node(state)["markdown_report"]

    assert f"**{support_level.title()}**: {support_rationale}" in report


def test_markdown_report_keeps_sq_content_readable_with_support_metadata():
    state = {
        "sq_answers": {
            "1.1": {
                "answer": "PY",
                "quote": "Participants were randomly assigned.",
                "justification": "The quote describes random assignment.",
                "support_level": "moderate",
                "support_rationale": "Randomization is stated without method details.",
            }
        },
        "domain_judgments": {"D1": "Some concerns"},
        "domain_rationales": {"D1": "Sequence details were limited."},
    }

    report = report_formatter_node(state)["markdown_report"]

    assert "Participants were randomly assigned." in report
    assert "The quote describes random assignment." in report
    assert "**Domain 1 judgment: Some concerns**" in report
    assert '{"support_level"' not in report
    assert "'support_level'" not in report


def test_markdown_report_summarizes_changed_sq_support_adjudication():
    state = {
        "sq_answers": {
            "1.3": {
                "answer": "N",
                "quote": "Baseline characteristics were well balanced.",
                "justification": "Adjudication found no baseline concern.",
                "support_level": "strong",
                "support_rationale": "Direct baseline table supports the final answer.",
            }
        },
        "domain_judgments": {"D1": "Low"},
        "domain_rationales": {"D1": "No randomization concerns remained."},
        "pivotality_tests": {
            "D1": [
                {
                    "sq_id": "1.3",
                    "original_answer": "Y",
                    "support_level": "weak",
                    "conservative_test_answer": "N",
                    "original_domain_judgment": "Some concerns",
                    "test_domain_judgment": "Low",
                    "pivotal": True,
                }
            ]
        },
        "sq_support_adjudications": {
            "D1": [
                {
                    "sq_id": "1.3",
                    "initial_answer": {
                        "answer": "Y",
                        "support_level": "weak",
                        "support_rationale": "Baseline imbalance claim was indirect.",
                    },
                    "adjudicated_answer": {
                        "answer": "N",
                        "support_level": "strong",
                        "support_rationale": "Direct baseline table supports no concern.",
                    },
                    "changed": True,
                }
            ]
        },
    }

    report = report_formatter_node(state)["markdown_report"]

    assert "## SQ support adjudication" in report
    assert "D1 SQ 1.3" in report
    assert "triggered adjudication" in report
    assert "Answer Y -> N" in report
    assert "support Weak -> Strong" in report
    assert "Direct baseline table supports no concern." in report
    assert "conservative_test_answer" not in report


def test_markdown_report_summarizes_unchanged_sq_support_adjudication():
    state = {
        "sq_answers": {
            "5.1": {
                "answer": "NI",
                "quote": "No analysis plan was available.",
                "justification": "Prespecification could not be verified.",
                "support_level": "moderate",
                "support_rationale": "The report lacks enough methods detail.",
            }
        },
        "domain_judgments": {"D5": "Some concerns"},
        "domain_rationales": {"D5": "Prespecification was unclear."},
        "sq_support_adjudications": {
            "D5": [
                {
                    "sq_id": "5.1",
                    "initial_answer": {
                        "answer": "NI",
                        "support_level": "weak",
                        "support_rationale": "No cited prespecification evidence.",
                    },
                    "adjudicated_answer": {
                        "answer": "NI",
                        "support_level": "moderate",
                        "support_rationale": "The absence of a plan supports NI, but not a stronger answer.",
                    },
                    "changed": False,
                }
            ]
        },
    }

    report = report_formatter_node(state)["markdown_report"]

    assert "D5 SQ 5.1 triggered adjudication (unchanged)" in report
    assert "Answer NI -> NI" in report
    assert "support Weak -> Moderate" in report
    assert "The absence of a plan supports NI" in report


def test_markdown_report_marks_unresolved_sq_support_adjudication():
    state = {
        "sq_answers": {
            "3.3": {
                "answer": "PY",
                "quote": "Reasons for missingness were incompletely reported.",
                "justification": "Missingness could depend on the true value.",
                "support_level": "weak",
                "support_rationale": "Only indirect missingness evidence was available.",
            }
        },
        "domain_judgments": {"D3": "Some concerns"},
        "domain_rationales": {"D3": "Missing outcome data remain uncertain."},
        "sq_support_adjudications": {
            "D3": [
                {
                    "sq_id": "3.3",
                    "initial_answer": {
                        "answer": "PY",
                        "support_level": "weak",
                        "support_rationale": "Reasons were not fully reported.",
                    },
                    "adjudicated_answer": {
                        "answer": "PY",
                        "support_level": "weak",
                        "support_rationale": "Adjudication still found only indirect support.",
                        "residual_uncertainty": "Missingness reasons remain incompletely reported.",
                    },
                    "changed": False,
                }
            ]
        },
    }

    report = report_formatter_node(state)["markdown_report"]

    assert "D3 SQ 3.3 triggered adjudication (uncertainty remains)" in report
    assert "support Weak -> Weak" in report
    assert "Missingness reasons remain incompletely reported." in report
