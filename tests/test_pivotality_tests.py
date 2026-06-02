import copy

import pytest

from rob2_pipeline.pipeline import _assessment_json
from rob2_pipeline.nodes.domain1 import domain1_judge_node
from rob2_pipeline.nodes.domain2 import domain2_judge_node
from rob2_pipeline.nodes.domain3 import domain3_judge_node
from rob2_pipeline.nodes.domain4 import domain4_judge_node
from rob2_pipeline.nodes.domain5 import domain5_judge_node


def _answer(answer: str, support_level: str) -> dict:
    return {
        "answer": answer,
        "quote": "trial report",
        "justification": "supported by cited trial text",
        "uncertainty_flag": "NORMAL",
        "support_level": support_level,
    }


def test_domain_judge_records_pivotality_test_without_mutating_sq_answer():
    sq_answers = {
        "1.1": _answer("Y", "strong"),
        "1.2": _answer("Y", "strong"),
        "1.3": _answer("Y", "weak"),
    }
    original = copy.deepcopy(sq_answers)

    result = domain1_judge_node(
        {
            "sq_answers": sq_answers,
            "domain_judgments": {},
            "domain_rationales": {},
        }
    )

    assert sq_answers == original
    assert result["initial_domain_judgments"]["D1"] == "Some concerns"
    assert result["domain_judgments"]["D1"] == "Some concerns"
    assert result["pivotality_tests"]["D1"] == [
        {
            "sq_id": "1.3",
            "original_answer": "Y",
            "support_level": "weak",
            "conservative_test_answer": "NI",
            "original_domain_judgment": "Some concerns",
            "test_domain_judgment": "Low",
            "pivotal": True,
            "acceptance_status": "needs_adjudication",
        }
    ]


@pytest.mark.parametrize(
    ("domain", "judge_node", "sq_answers", "weak_sq_id", "original", "tested"),
    [
        (
            "D1",
            domain1_judge_node,
            {"1.1": ("Y", "strong"), "1.2": ("Y", "strong"), "1.3": ("Y", "weak")},
            "1.3",
            "Some concerns",
            "Low",
        ),
        (
            "D2",
            domain2_judge_node,
            {
                "2.1": ("N", "strong"),
                "2.2": ("N", "strong"),
                "2.6": ("Y", "weak"),
                "2.7": ("Y", "strong"),
            },
            "2.6",
            "Low",
            "High",
        ),
        (
            "D3",
            domain3_judge_node,
            {
                "3.1": ("N", "strong"),
                "3.2": ("N", "strong"),
                "3.3": ("Y", "strong"),
                "3.4": ("N", "weak"),
            },
            "3.4",
            "Some concerns",
            "High",
        ),
        (
            "D4",
            domain4_judge_node,
            {
                "4.1": ("N", "strong"),
                "4.2": ("N", "strong"),
                "4.3": ("Y", "strong"),
                "4.4": ("Y", "strong"),
                "4.5": ("N", "weak"),
            },
            "4.5",
            "Some concerns",
            "High",
        ),
        (
            "D5",
            domain5_judge_node,
            {"5.1": ("Y", "strong"), "5.2": ("N", "strong"), "5.3": ("N", "weak")},
            "5.3",
            "Low",
            "Some concerns",
        ),
    ],
)
def test_pivotal_weak_answers_are_audited_for_each_domain(
    domain, judge_node, sq_answers, weak_sq_id, original, tested
):
    result = judge_node(_state(sq_answers))

    assert result["domain_judgments"][domain] == original
    assert result["pivotality_tests"][domain] == [
        {
            "sq_id": weak_sq_id,
            "original_answer": sq_answers[weak_sq_id][0],
            "support_level": "weak",
            "conservative_test_answer": "NI",
            "original_domain_judgment": original,
            "test_domain_judgment": tested,
            "pivotal": True,
            "acceptance_status": "needs_adjudication",
        }
    ]


@pytest.mark.parametrize(
    ("domain", "judge_node", "sq_answers", "weak_sq_id", "judgment"),
    [
        (
            "D1",
            domain1_judge_node,
            {
                "1.1": ("Y", "weak"),
                "1.2": ("Y", "strong"),
                "1.3": ("N", "strong"),
            },
            "1.1",
            "Low",
        ),
        (
            "D2",
            domain2_judge_node,
            {
                "2.1": ("N", "strong"),
                "2.2": ("N", "strong"),
                "2.6": ("Y", "strong"),
                "2.7": ("N", "weak"),
            },
            "2.7",
            "Low",
        ),
        (
            "D3",
            domain3_judge_node,
            {"3.1": ("N", "weak"), "3.2": ("Y", "strong")},
            "3.1",
            "Low",
        ),
        (
            "D4",
            domain4_judge_node,
            {
                "4.1": ("N", "weak"),
                "4.2": ("N", "strong"),
                "4.3": ("N", "strong"),
            },
            "4.1",
            "Low",
        ),
        (
            "D5",
            domain5_judge_node,
            {"5.1": ("Y", "weak"), "5.2": ("Y", "strong"), "5.3": ("N", "strong")},
            "5.1",
            "High",
        ),
    ],
)
def test_non_pivotal_weak_answers_are_audited_for_each_domain(
    domain, judge_node, sq_answers, weak_sq_id, judgment
):
    result = judge_node(_state(sq_answers))

    assert result["pivotality_tests"][domain][0] == {
        "sq_id": weak_sq_id,
        "original_answer": sq_answers[weak_sq_id][0],
        "support_level": "weak",
        "conservative_test_answer": "NI",
        "original_domain_judgment": judgment,
        "test_domain_judgment": judgment,
        "pivotal": False,
        "acceptance_status": "accepted",
    }


def test_strong_and_moderate_answers_do_not_receive_pivotality_tests():
    result = domain1_judge_node(
        _state(
            {
                "1.1": ("Y", "strong"),
                "1.2": ("Y", "moderate"),
                "1.3": ("Y", "strong"),
            }
        )
    )

    assert "pivotality_tests" not in result


def test_unsupported_answers_receive_pivotality_tests():
    result = domain5_judge_node(
        _state(
            {
                "5.1": ("Y", "strong"),
                "5.2": ("N", "strong"),
                "5.3": ("N", "unsupported"),
            }
        )
    )

    assert result["pivotality_tests"]["D5"][0]["support_level"] == "unsupported"
    assert (
        result["pivotality_tests"]["D5"][0]["acceptance_status"]
        == "needs_adjudication"
    )


def test_constrained_answers_receive_visible_non_blocking_audit_records():
    result = domain1_judge_node(
        {
            "sq_answers": {
                "1.1": _answer("Y", "strong"),
                "1.2": _answer("Y", "strong"),
                "1.3": _answer("N", "strong"),
            },
            "support_constraints": [
                {
                    "constraint_type": "quote_untraceable",
                    "sq_id": "1.1",
                    "reason": "The cited quote was not found in source text.",
                }
            ],
            "domain_judgments": {},
            "domain_rationales": {},
        }
    )

    assert result["pivotality_tests"]["D1"][0] == {
        "sq_id": "1.1",
        "original_answer": "Y",
        "support_level": "strong",
        "conservative_test_answer": "NI",
        "original_domain_judgment": "Low",
        "test_domain_judgment": "Low",
        "pivotal": False,
        "acceptance_status": "accepted",
        "constraints": [
            {
                "constraint_type": "quote_untraceable",
                "sq_id": "1.1",
                "reason": "The cited quote was not found in source text.",
            }
        ],
    }


def test_pivotality_tests_are_in_json_output():
    state = {
        "pivotality_tests": {
            "D1": [
                {
                    "sq_id": "1.3",
                    "original_answer": "Y",
                    "support_level": "unsupported",
                    "conservative_test_answer": "NI",
                    "original_domain_judgment": "Some concerns",
                    "test_domain_judgment": "Low",
                    "pivotal": True,
                    "acceptance_status": "needs_adjudication",
                }
            ]
        }
    }

    assert _assessment_json(state)["pivotality_tests"] == state["pivotality_tests"]


def _state(sq_answers: dict[str, tuple[str, str]]) -> dict:
    return {
        "sq_answers": {
            sq_id: _answer(answer, support_level)
            for sq_id, (answer, support_level) in sq_answers.items()
        },
        "domain_judgments": {},
        "domain_rationales": {},
    }
