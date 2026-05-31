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
