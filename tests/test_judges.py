from types import SimpleNamespace

import pytest

from rob2_pipeline.llm_contracts import JsonContractResult
from rob2_pipeline.judges import (
    judge_domain1,
    judge_domain1_artifact,
    judge_domain2,
    judge_domain2_artifact,
    judge_domain3,
    judge_domain3_artifact,
    judge_domain4,
    judge_domain4_artifact,
    judge_domain5,
    judge_domain5_artifact,
    judge_overall,
    judge_overall_artifact,
)
from rob2_pipeline.nodes.domain1 import domain1_judge_node
from rob2_pipeline.nodes.domain2 import domain2_judge_node
from rob2_pipeline.nodes.domain3 import domain3_judge_node
from rob2_pipeline.models import empty_paper_evidence
from rob2_pipeline.nodes.domain4 import domain4_judge_node, domain4_sq_node
from rob2_pipeline.nodes.domain5 import domain5_judge_node
from rob2_pipeline.nodes.overall import overall_judge_node
from rob2_pipeline.prompts import (
    PROMPT_DOMAIN2_ADHERING_ANALYSIS,
    PROMPT_DOMAIN2_ADHERING_CONDITIONAL,
    PROMPT_DOMAIN5,
)


def sq(**answers):
    return {key: {"answer": value} for key, value in answers.items()}


@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        (sq(**{"1.1": "Y", "1.2": "Y", "1.3": "N"}), "Low"),
        (sq(**{"1.1": "Y", "1.2": "Y", "1.3": "Y"}), "Some concerns"),
        (sq(**{"1.1": "N", "1.2": "Y", "1.3": "PY"}), "Some concerns"),
        (sq(**{"1.1": "Y", "1.2": "NI", "1.3": "PN"}), "Some concerns"),
        (sq(**{"1.1": "Y", "1.2": "NI", "1.3": "Y"}), "High"),
        (sq(**{"1.1": "Y", "1.2": "PN", "1.3": "N"}), "High"),
        ({}, "Some concerns"),
    ],
)
def test_judge_domain1(answers, expected):
    assert judge_domain1(answers)[0] == expected


def test_judge_domain1_artifact_records_rule_path_and_inputs():
    answers = sq(**{"1.1": "Y", "1.2": "Y", "1.3": "N"})

    artifact = judge_domain1_artifact(answers)

    assert artifact["judge_version"] == "d1-judge-v1"
    assert artifact["rule_table_version"] == "rob2-d1-rule-table-v1"
    assert artifact["input_sq_answers"] == answers
    assert artifact["applied_rule_path"] == "d1-row-1:y-py-ni/y-py/ni-n-pn"
    assert artifact["label"] == "Low"
    assert artifact["rationale"] == "Row: Y-PY-NI / Y-PY / NI-N-PN -> Low"


def test_domain1_judge_node_emits_versioned_judgment_artifact():
    result = domain1_judge_node(
        {
            "outcome": "Overall survival",
            "sq_answers": sq(**{"1.1": "Y", "1.2": "Y", "1.3": "N"}),
            "domain_judgments": {},
            "domain_rationales": {},
        }
    )

    artifact = result["d1_judgment_artifact"]
    assert artifact["artifact_id"] == "d1-judgment:Overall survival"
    assert artifact["schema_version"] == "d1-judgment-v1"
    assert artifact["label"] == result["domain_judgments"]["D1"]
    assert artifact["rationale"] == result["domain_rationales"]["D1"]


@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        (sq(**{"2.1": "N", "2.2": "PN", "2.6": "Y"}), "Low"),
        (sq(**{"2.1": "Y", "2.2": "N", "2.6": "Y"}), "Some concerns"),
        (sq(**{"2.1": "Y", "2.2": "NI", "2.6": "Y"}), "Some concerns"),
        (sq(**{"2.1": "Y", "2.2": "Y", "2.3": "N", "2.6": "Y"}), "Low"),
        (
            sq(
                **{
                    "2.1": "Y",
                    "2.2": "Y",
                    "2.3": "Y",
                    "2.4": "Y",
                    "2.5": "PY",
                    "2.6": "Y",
                }
            ),
            "Some concerns",
        ),
        (sq(**{"2.1": "Y", "2.2": "Y", "2.3": "NI", "2.6": "Y"}), "Some concerns"),
        (
            sq(
                **{
                    "2.1": "Y",
                    "2.2": "Y",
                    "2.3": "Y",
                    "2.4": "Y",
                    "2.5": "Y",
                    "2.6": "Y",
                }
            ),
            "Some concerns",
        ),
        (
            sq(
                **{
                    "2.1": "Y",
                    "2.2": "Y",
                    "2.3": "Y",
                    "2.4": "Y",
                    "2.5": "N",
                    "2.6": "N",
                    "2.7": "Y",
                }
            ),
            "High",
        ),
        (
            sq(
                **{
                    "2.1": "Y",
                    "2.2": "Y",
                    "2.3": "Y",
                    "2.4": "N",
                    "2.5": "N",
                    "2.6": "Y",
                }
            ),
            "Some concerns",
        ),
        (sq(**{"2.1": "N", "2.2": "N", "2.6": "N", "2.7": "N"}), "Some concerns"),
        (sq(**{"2.1": "N", "2.2": "N", "2.6": "N", "2.7": "Y"}), "High"),
        (
            sq(
                **{
                    "2.1": "NI",
                    "2.2": "NI",
                    "2.3": "NI",
                    "2.4": "NI",
                    "2.5": "NI",
                    "2.6": "NI",
                    "2.7": "NI",
                }
            ),
            "High",
        ),
        ({}, "Some concerns"),
        (
            sq(
                **{
                    "2.1": "NA",
                    "2.2": "NA",
                    "2.3": "NA",
                    "2.4": "NA",
                    "2.5": "NA",
                    "2.6": "NA",
                    "2.7": "NA",
                }
            ),
            "Some concerns",
        ),
    ],
)
def test_judge_domain2(answers, expected):
    assert judge_domain2(answers)[0] == expected


def test_judge_domain2_artifact_records_rule_path_and_inputs():
    answers = sq(**{"2.1": "N", "2.2": "PN", "2.6": "Y"})

    artifact = judge_domain2_artifact(answers, effect_of_interest="ITT")

    assert artifact["judge_version"] == "d2-judge-v1"
    assert artifact["rule_table_version"] == "rob2-d2-assignment-rule-table-v1"
    assert artifact["input_sq_answers"] == {
        "2.1": answers["2.1"],
        "2.2": answers["2.2"],
        "2.3": {},
        "2.4": {},
        "2.5": {},
        "2.6": answers["2.6"],
        "2.7": {},
    }
    assert artifact["applied_rule_path"] == "d2-assignment:part1-low+part2-low"
    assert artifact["label"] == "Low"
    assert "Part1=Low" in artifact["rationale"]


def test_domain2_judge_node_emits_versioned_judgment_artifact():
    result = domain2_judge_node(
        {
            "outcome": "Overall survival",
            "effect_of_interest": "ITT",
            "sq_answers": sq(**{"2.1": "N", "2.2": "PN", "2.6": "Y"}),
            "domain_judgments": {},
            "domain_rationales": {},
        }
    )

    artifact = result["d2_judgment_artifact"]
    assert artifact["artifact_id"] == "d2-judgment:Overall survival"
    assert artifact["schema_version"] == "d2-judgment-v1"
    assert artifact["label"] == result["domain_judgments"]["D2"]
    assert artifact["rationale"] == result["domain_rationales"]["D2"]


@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        (
            sq(
                **{
                    "2.1": "N",
                    "2.2": "N",
                    "2.3": "NA",
                    "2.4": "N",
                    "2.5": "N",
                    "2.6": "NA",
                }
            ),
            "Low",
        ),
        (
            sq(
                **{
                    "2.1": "Y",
                    "2.2": "Y",
                    "2.3": "Y",
                    "2.4": "N",
                    "2.5": "N",
                    "2.6": "NA",
                }
            ),
            "Low",
        ),
        (
            sq(
                **{
                    "2.1": "Y",
                    "2.2": "Y",
                    "2.3": "N",
                    "2.4": "N",
                    "2.5": "N",
                    "2.6": "Y",
                }
            ),
            "Some concerns",
        ),
        (
            sq(
                **{
                    "2.1": "N",
                    "2.2": "N",
                    "2.3": "NA",
                    "2.4": "Y",
                    "2.5": "N",
                    "2.6": "Y",
                }
            ),
            "Some concerns",
        ),
        (
            sq(
                **{
                    "2.1": "N",
                    "2.2": "N",
                    "2.3": "NA",
                    "2.4": "N",
                    "2.5": "Y",
                    "2.6": "N",
                }
            ),
            "High",
        ),
        (
            sq(
                **{
                    "2.1": "Y",
                    "2.2": "Y",
                    "2.3": "NI",
                    "2.4": "N",
                    "2.5": "N",
                    "2.6": "NI",
                }
            ),
            "High",
        ),
    ],
)
def test_judge_domain2_per_protocol(answers, expected):
    assert judge_domain2(answers, "per-protocol")[0] == expected


@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        (sq(**{"3.1": "Y"}), "Low"),
        (sq(**{"3.1": "N", "3.2": "Y"}), "Low"),
        (sq(**{"3.1": "N", "3.2": "N", "3.3": "N"}), "Low"),
        (sq(**{"3.1": "N", "3.2": "N", "3.3": "Y", "3.4": "N"}), "Some concerns"),
        (sq(**{"3.1": "N", "3.2": "N", "3.3": "Y", "3.4": "Y"}), "High"),
        (sq(**{"3.1": "NI", "3.2": "N", "3.3": "NI", "3.4": "NI"}), "High"),
        ({}, "Some concerns"),
        (sq(**{"3.1": "NA", "3.2": "NA", "3.3": "NA", "3.4": "NA"}), "Some concerns"),
    ],
)
def test_judge_domain3(answers, expected):
    assert judge_domain3(answers)[0] == expected


def test_d3_d5_domain_artifacts_record_versions_rule_paths_and_inputs():
    d3 = judge_domain3_artifact(sq(**{"3.1": "Y"}))
    d4 = judge_domain4_artifact(sq(**{"4.1": "N", "4.2": "N", "4.3": "N"}))
    d5 = judge_domain5_artifact(sq(**{"5.1": "Y", "5.2": "N", "5.3": "N"}))

    assert d3["judge_version"] == "d3-judge-v1"
    assert d3["rule_table_version"] == "rob2-d3-rule-table-v1"
    assert d3["input_sq_answers"]["3.1"] == {"answer": "Y"}
    assert d3["applied_rule_path"] == "d3:nearly-complete-data"
    assert d4["judge_version"] == "d4-judge-v1"
    assert d4["input_sq_answers"]["4.3"] == {"answer": "N"}
    assert d4["applied_rule_path"] == "d4:no-assessor-awareness"
    assert d5["judge_version"] == "d5-judge-v1"
    assert d5["input_sq_answers"]["5.1"] == {"answer": "Y"}
    assert d5["applied_rule_path"] == "d5:prespecified-and-not-selective"


@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        (sq(**{"4.1": "N", "4.2": "N", "4.3": "N"}), "Low"),
        (sq(**{"4.1": "N", "4.2": "N", "4.3": "Y", "4.4": "N"}), "Low"),
        (sq(**{"4.1": "N", "4.2": "N", "4.3": "Y", "4.4": "N", "4.5": "NA"}), "Low"),
        (
            sq(**{"4.1": "N", "4.2": "N", "4.3": "Y", "4.4": "Y", "4.5": "N"}),
            "Some concerns",
        ),
        (
            sq(**{"4.1": "N", "4.2": "N", "4.3": "Y", "4.4": "PY", "4.5": "N"}),
            "Some concerns",
        ),
        (
            sq(**{"4.1": "N", "4.2": "N", "4.3": "Y", "4.4": "PY", "4.5": "PN"}),
            "Some concerns",
        ),
        (sq(**{"4.1": "N", "4.2": "N", "4.3": "Y", "4.4": "PY", "4.5": "NI"}), "High"),
        (sq(**{"4.1": "N", "4.2": "N", "4.3": "Y", "4.4": "Y", "4.5": "Y"}), "High"),
        (sq(**{"4.1": "N", "4.2": "NI", "4.3": "N"}), "Some concerns"),
        (sq(**{"4.1": "N", "4.2": "NI", "4.3": "Y", "4.4": "N"}), "Some concerns"),
        (
            sq(**{"4.1": "N", "4.2": "NI", "4.3": "Y", "4.4": "Y", "4.5": "N"}),
            "Some concerns",
        ),
        (sq(**{"4.1": "N", "4.2": "NI", "4.3": "Y", "4.4": "Y", "4.5": "Y"}), "High"),
        (sq(**{"4.1": "Y", "4.2": "N"}), "High"),
        (sq(**{"4.1": "N", "4.2": "Y"}), "High"),
        (
            sq(**{"4.1": "NI", "4.2": "NI", "4.3": "NI", "4.4": "NI", "4.5": "NI"}),
            "High",
        ),
        ({}, "Some concerns"),
        (
            sq(**{"4.1": "NA", "4.2": "NA", "4.3": "NA", "4.4": "NA", "4.5": "NA"}),
            "Some concerns",
        ),
    ],
)
def test_judge_domain4(answers, expected):
    assert judge_domain4(answers)[0] == expected


@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        (sq(**{"5.1": "Y", "5.2": "N", "5.3": "N"}), "Low"),
        (sq(**{"5.1": "N", "5.2": "N", "5.3": "N"}), "Some concerns"),
        (sq(**{"5.1": "Y", "5.2": "N", "5.3": "NI"}), "Some concerns"),
        (sq(**{"5.1": "Y", "5.2": "NI", "5.3": "N"}), "Some concerns"),
        (sq(**{"5.1": "Y", "5.2": "NI", "5.3": "NI"}), "Some concerns"),
        (sq(**{"5.1": "Y", "5.2": "Y", "5.3": "N"}), "High"),
        (sq(**{"5.1": "Y", "5.2": "N", "5.3": "PY"}), "High"),
        ({}, "Some concerns"),
        (sq(**{"5.1": "NA", "5.2": "NA", "5.3": "NA"}), "Some concerns"),
    ],
)
def test_judge_domain5(answers, expected):
    assert judge_domain5(answers)[0] == expected


@pytest.mark.parametrize(
    ("domains", "expected", "rationale_part"),
    [
        (
            {"D1": "Low", "D2": "Low", "D3": "Low", "D4": "Low", "D5": "Low"},
            "Low",
            "Low in all",
        ),
        (
            {"D1": "Low", "D2": "Some concerns", "D3": "Low", "D4": "Low", "D5": "Low"},
            "Some concerns",
            "1 domain",
        ),
        (
            {
                "D1": "Some concerns",
                "D2": "Some concerns",
                "D3": "Low",
                "D4": "Low",
                "D5": "Low",
            },
            "Some concerns",
            "2 domains with Some concerns",
        ),
        (
            {
                "D1": "Some concerns",
                "D2": "Some concerns",
                "D3": "Some concerns",
                "D4": "Low",
                "D5": "Low",
            },
            "Some concerns",
            "substantially lower confidence",
        ),
        (
            {"D1": "Low", "D2": "High", "D3": "Low", "D4": "Low", "D5": "Low"},
            "High",
            "D2",
        ),
    ],
)
def test_judge_overall(domains, expected, rationale_part):
    judgment, rationale = judge_overall(domains)
    assert judgment == expected
    assert rationale_part in rationale


def test_judge_overall_artifact_records_policy_and_domain_inputs():
    domains = {"D1": "Low", "D2": "Low", "D3": "Low", "D4": "Low", "D5": "Low"}

    artifact = judge_overall_artifact(domains, policy="official_rob2")

    assert artifact["judge_version"] == "overall-judge-v1"
    assert artifact["policy"] == "official_rob2"
    assert artifact["input_domain_judgments"] == domains
    assert artifact["applied_rule_path"] == "overall:all-low"
    assert artifact["label"] == "Low"


def test_overall_node_exposes_benchmark_reference_policy():
    result = overall_judge_node(
        {
            "domain_judgments": {
                "D1": "Low",
                "D2": "Low",
                "D3": "Low",
                "D4": "Some concerns",
                "D5": "Low",
            },
            "sq_answers": {},
            "overall_policy": "benchmark_reference",
        }
    )

    assert result["overall_judgment"] == "Low"
    assert result["overall_policy"] == "benchmark_reference"
    assert result["overall_judgment_artifact"]["policy"] == "benchmark_reference"
    assert (
        result["overall_judgment_artifact"]["applied_rule_path"]
        == "overall:benchmark-reference-at-most-one-some-concern"
    )


def test_overall_priority_ignores_moderate_and_isolated_non_pivotal_weak_support():
    result = overall_judge_node(
        {
            "domain_judgments": {
                "D1": "Low",
                "D2": "Low",
                "D3": "Low",
                "D4": "Low",
                "D5": "Low",
            },
            "sq_answers": {
                "1.1": {
                    "answer": "PY",
                    "support_level": "moderate",
                    "uncertainty_flag": "NORMAL",
                },
                "1.2": {
                    "answer": "PY",
                    "support_level": "weak",
                    "uncertainty_flag": "NORMAL",
                },
            },
            "pivotality_tests": {
                "D1": [
                    {
                        "sq_id": "1.2",
                        "original_answer": "PY",
                        "support_level": "weak",
                        "conservative_test_answer": "NI",
                        "original_domain_judgment": "Low",
                        "test_domain_judgment": "Low",
                        "pivotal": False,
                    }
                ]
            },
        }
    )

    assert result["human_review_priority"] == "LOW"


def test_automation_confidence_auto_accepts_moderate_pivotal_support():
    result = overall_judge_node(
        {
            "outcome": "Overall survival",
            "domain_judgments": _complete_low_domain_judgments(),
            "sq_answers": {
                "1.3": {
                    "answer": "N",
                    "support_level": "moderate",
                    "uncertainty_flag": "NORMAL",
                }
            },
            "pivotality_tests": {
                "D1": [
                    {
                        "sq_id": "1.3",
                        "original_answer": "N",
                        "support_level": "moderate",
                        "conservative_test_answer": "NI",
                        "original_domain_judgment": "Low",
                        "test_domain_judgment": "Some concerns",
                        "pivotal": True,
                        "acceptance_status": "accepted",
                    }
                ]
            },
        }
    )

    confidence = result["automation_confidence"]
    assert confidence["status"] == "auto_accept_candidate"
    assert confidence["non_acceptance_reasons"] == []
    assert confidence["completion"]["completed_domains"] == ["D1", "D2", "D3", "D4", "D5"]


def test_automation_confidence_records_non_pivotal_weak_without_blocking():
    result = overall_judge_node(
        {
            "domain_judgments": _complete_low_domain_judgments(),
            "sq_answers": {
                "1.1": {
                    "answer": "Y",
                    "support_level": "weak",
                    "uncertainty_flag": "NORMAL",
                }
            },
            "pivotality_tests": {
                "D1": [
                    {
                        "sq_id": "1.1",
                        "original_answer": "Y",
                        "support_level": "weak",
                        "conservative_test_answer": "NI",
                        "original_domain_judgment": "Low",
                        "test_domain_judgment": "Low",
                        "pivotal": False,
                        "acceptance_status": "accepted",
                    }
                ]
            },
        }
    )

    assert result["automation_confidence"]["status"] == "auto_accept_candidate"


def test_automation_confidence_rejects_non_low_overall_judgment():
    result = overall_judge_node(
        {
            "domain_judgments": {
                "D1": "Low",
                "D2": "Low",
                "D3": "Low",
                "D4": "Low",
                "D5": "Some concerns",
            },
            "sq_answers": {
                "5.1": {"answer": "NI", "support_level": "unsupported"},
                "5.2": {"answer": "NI", "support_level": "weak"},
                "5.3": {"answer": "N", "support_level": "moderate"},
            },
        }
    )

    confidence = result["automation_confidence"]
    assert confidence["status"] == "not_auto_acceptable"
    assert any(
        reason["kind"] == "overall_judgment_not_low"
        for reason in confidence["non_acceptance_reasons"]
    )


def test_automation_confidence_rejects_unresolved_support_constraints():
    result = overall_judge_node(
        {
            "domain_judgments": _complete_low_domain_judgments(),
            "sq_answers": {
                "3.1": {
                    "answer": "Y",
                    "support_level": "strong",
                    "uncertainty_flag": "NORMAL",
                }
            },
            "support_constraints": [
                {
                    "constraint_type": "missing_required_evidence",
                    "sq_id": "3.1",
                    "reason": "D3 completeness answer lacks a denominator.",
                }
            ],
        }
    )

    confidence = result["automation_confidence"]
    assert confidence["status"] == "not_auto_acceptable"
    assert any(
        reason["kind"] == "unresolved_support_constraint"
        for reason in confidence["non_acceptance_reasons"]
    )


def test_automation_confidence_rejects_local_guardrail_corrections():
    result = overall_judge_node(
        {
            "domain_judgments": _complete_low_domain_judgments(),
            "sq_answers": {
                "5.2": {
                    "answer": "PN",
                    "support_level": "moderate",
                    "d5_guard_applied": True,
                    "support_rationale": "Guard corrected endpoint-family overreach.",
                }
            },
        }
    )

    confidence = result["automation_confidence"]
    assert confidence["status"] == "not_auto_acceptable"
    assert any(
        reason["kind"] == "local_guardrail_applied"
        for reason in confidence["non_acceptance_reasons"]
    )


def test_automation_confidence_rejects_degraded_retrieval_fallback():
    result = overall_judge_node(
        {
            "domain_judgments": _complete_low_domain_judgments(),
            "sq_answers": {},
            "evidence": {
                "warnings": [
                    "RAG vector retrieval failed; used lexical chunk retrieval fallback: missing dependency."
                ]
            },
        }
    )

    confidence = result["automation_confidence"]
    assert confidence["status"] == "not_auto_acceptable"
    assert any(
        reason["kind"] == "degraded_retrieval"
        for reason in confidence["non_acceptance_reasons"]
    )


def test_automation_confidence_rejects_pivotal_weak_or_untraceable_support():
    result = overall_judge_node(
        {
            "domain_judgments": _complete_low_domain_judgments(),
            "sq_answers": {
                "5.3": {
                    "answer": "N",
                    "support_level": "weak",
                    "uncertainty_flag": "NORMAL",
                }
            },
            "pivotality_tests": {
                "D5": [
                    {
                        "sq_id": "5.3",
                        "original_answer": "N",
                        "support_level": "weak",
                        "conservative_test_answer": "NI",
                        "original_domain_judgment": "Low",
                        "test_domain_judgment": "Some concerns",
                        "pivotal": True,
                        "acceptance_status": "needs_adjudication",
                        "constraints": [
                            {
                                "constraint_type": "quote_untraceable",
                                "sq_id": "5.3",
                                "reason": "The cited quote was not found.",
                            }
                        ],
                    }
                ]
            },
        }
    )

    confidence = result["automation_confidence"]
    assert confidence["status"] == "not_auto_acceptable"
    assert {
        reason["kind"] for reason in confidence["non_acceptance_reasons"]
    } == {"pivotal_support_below_moderate", "pivotal_quote_not_traceable"}


def test_automation_confidence_blocks_only_incomplete_or_failed_required_artifacts():
    result = overall_judge_node(
        {
            "domain_judgments": {
                "D1": "Low",
                "D2": "Low",
                "D3": "Low",
                "D4": "Low",
            },
            "sq_answers": {
                "5.1": {
                    "answer": "NI",
                    "support_level": "unsupported",
                    "uncertainty_flag": "HIGH",
                    "classification_blocked": True,
                    "packet_status": "needs_retrieval_repair",
                    "support_rationale": "Required prespecification evidence is missing.",
                }
            },
            "packet_readiness": {
                "5.1": {
                    "status": "needs_retrieval_repair",
                    "blocking_reason": "Selected packet sources do not cover required evidence.",
                }
            },
        }
    )

    confidence = result["automation_confidence"]
    assert confidence["status"] == "blocked"
    assert {reason["kind"] for reason in confidence["blocking_reasons"]} == {
        "incomplete_required_input",
        "failed_required_artifact",
    }


def test_automation_confidence_does_not_block_on_info_errors():
    result = overall_judge_node(
        {
            "domain_judgments": _complete_low_domain_judgments(),
            "sq_answers": {},
            "errors": [
                "INFO: outcome_type normalized from 'clinician-composite' to 'clinician-graded' using outcome-bound LLM resolution."
            ],
        }
    )

    confidence = result["automation_confidence"]
    assert confidence["blocking_reasons"] == []
    assert confidence["status"] == "auto_accept_candidate"


def test_automation_confidence_ignores_not_applicable_packet_failures():
    result = overall_judge_node(
        {
            "domain_judgments": _complete_low_domain_judgments(),
            "sq_answers": {
                "2.3": {
                    "answer": "N",
                    "support_level": "moderate",
                    "uncertainty_flag": "NORMAL",
                },
                "2.4": {
                    "answer": "NA",
                    "support_level": "unsupported",
                    "uncertainty_flag": "NORMAL",
                },
            },
            "packet_readiness": {
                "2.4": {
                    "status": "needs_retrieval_repair",
                    "blocking_reason": "Selected packet sources do not cover impact evidence.",
                }
            },
        }
    )

    confidence = result["automation_confidence"]
    assert confidence["status"] == "auto_accept_candidate"
    assert confidence["blocking_reasons"] == []


@pytest.mark.parametrize(
    "adjudicated_answer",
    [
        {
            "answer": "Y",
            "support_level": "weak",
            "support_rationale": "Evidence remains indirect.",
        },
        {
            "answer": "Y",
            "support_level": "unsupported",
            "support_rationale": "No selected evidence supports this answer.",
        },
    ],
)
def test_unresolved_pivotal_weak_support_raises_review_priority(adjudicated_answer):
    result = overall_judge_node(
        {
            "domain_judgments": {
                "D1": "Low",
                "D2": "Low",
                "D3": "Low",
                "D4": "Low",
                "D5": "Low",
            },
            "sq_answers": {
                "1.3": {
                    "answer": "Y",
                    "support_level": adjudicated_answer["support_level"],
                    "uncertainty_flag": "NORMAL",
                }
            },
            "sq_support_adjudications": {
                "D1": [
                    {
                        "sq_id": "1.3",
                        "initial_answer": {
                            "answer": "Y",
                            "support_level": "weak",
                        },
                        "adjudicated_answer": adjudicated_answer,
                        "domain_impact": {
                            "original_domain_judgment": "Low",
                            "test_answer": "NI",
                            "test_domain_judgment": "Some concerns",
                        },
                        "changed": False,
                    }
                ]
            },
        }
    )

    assert result["human_review_priority"] == "HIGH"


def test_adjudication_conflict_raises_review_priority():
    result = overall_judge_node(
        {
            "domain_judgments": {
                "D1": "Low",
                "D2": "Low",
                "D3": "Low",
                "D4": "Low",
                "D5": "Low",
            },
            "sq_answers": {
                "1.3": {
                    "answer": "N",
                    "support_level": "strong",
                    "uncertainty_flag": "NORMAL",
                }
            },
            "sq_support_adjudications": {
                "D1": [
                    {
                        "sq_id": "1.3",
                        "initial_answer": {
                            "answer": "Y",
                            "support_level": "weak",
                        },
                        "adjudicated_answer": {
                            "answer": "N",
                            "support_level": "strong",
                        },
                        "changed": True,
                    }
                ]
            },
        }
    )

    assert result["human_review_priority"] == "HIGH"


def test_repeated_weak_support_patterns_raise_review_priority():
    result = overall_judge_node(
        {
            "domain_judgments": {
                "D1": "Low",
                "D2": "Low",
                "D3": "Low",
                "D4": "Low",
                "D5": "Low",
            },
            "sq_answers": {
                "1.1": {
                    "answer": "PY",
                    "support_level": "weak",
                    "uncertainty_flag": "NORMAL",
                },
                "2.1": {
                    "answer": "PY",
                    "support_level": "weak",
                    "uncertainty_flag": "NORMAL",
                },
                "3.1": {
                    "answer": "PY",
                    "support_level": "weak",
                    "uncertainty_flag": "NORMAL",
                },
            },
        }
    )

    assert result["human_review_priority"] == "HIGH"


def test_domain_nodes_do_not_override_algorithm_by_outcome_label():
    d3_state = {
        "outcome": "Progression-Free Survival",
        "sq_answers": sq(**{"3.1": "Y"}),
        "domain_judgments": {},
        "domain_rationales": {},
    }
    assert domain3_judge_node(d3_state)["domain_judgments"]["D3"] == "Low"

    d4_state = {
        "outcome": "Progression-Free Survival",
        "sq_answers": sq(**{"2.1": "Y", "4.1": "N", "4.2": "N", "4.3": "N"}),
        "domain_judgments": {},
        "domain_rationales": {},
    }
    assert domain4_judge_node(d4_state)["domain_judgments"]["D4"] == "Low"

    d5_state = {
        "outcome": "Progression-Free Survival",
        "registration_number": "NCT00000000",
        "sq_answers": sq(**{"5.1": "NI", "5.2": "N", "5.3": "N"}),
        "domain_judgments": {},
        "domain_rationales": {},
    }
    assert domain5_judge_node(d5_state)["domain_judgments"]["D5"] == "Some concerns"


def test_d3_judge_node_reapplies_branch_control_after_adjudication(monkeypatch):
    def fake_add_domain_judgment_with_pivotality_tests(*args, **kwargs):
        return {
            "sq_answers": {
                "3.1": {"answer": "PY", "support_level": "moderate"},
                "3.2": {"answer": "NI", "support_level": "weak"},
                "3.3": {"answer": "NI", "support_level": "weak"},
                "3.4": {"answer": "Y", "support_level": "strong"},
            },
            "domain_judgments": {"D3": "Low"},
            "domain_rationales": {"D3": "3.1=Y/PY (nearly complete data) -> Low"},
            "pivotality_tests": {"D3": []},
            "sq_support_adjudications": {"D3": []},
        }

    monkeypatch.setattr(
        "rob2_pipeline.nodes.domain3.add_domain_judgment_with_pivotality_tests",
        fake_add_domain_judgment_with_pivotality_tests,
    )

    result = domain3_judge_node(
        {
            "outcome": "Progression-Free Survival",
            "sq_answers": {
                "3.1": {"answer": "PY", "support_level": "moderate"},
                "3.4": {"answer": "Y", "support_level": "strong"},
            },
            "domain_judgments": {},
            "domain_rationales": {},
        }
    )

    assert result["sq_answers"]["3.2"]["answer"] == "NA"
    assert result["sq_answers"]["3.3"]["answer"] == "NA"
    assert result["sq_answers"]["3.4"]["answer"] == "NA"


def test_d1_judge_corrects_baseline_balance_polarity_after_adjudication(monkeypatch):
    def fake_add_domain_judgment_with_pivotality_tests(*args, **kwargs):
        return {
            "sq_answers": {
                "1.1": {"answer": "Y", "support_level": "strong"},
                "1.2": {"answer": "Y", "support_level": "strong"},
                "1.3": {
                    "answer": "Y",
                    "quote": "Baseline characteristics were well balanced.",
                    "justification": (
                        "Baseline characteristics show no major discrepancies."
                    ),
                    "support_level": "strong",
                },
            },
            "domain_judgments": {"D1": "Some concerns"},
            "domain_rationales": {"D1": "Baseline imbalance concern."},
            "pivotality_tests": {"D1": []},
            "sq_support_adjudications": {"D1": []},
        }

    monkeypatch.setattr(
        "rob2_pipeline.nodes.domain1.add_domain_judgment_with_pivotality_tests",
        fake_add_domain_judgment_with_pivotality_tests,
    )

    result = domain1_judge_node(
        {
            "outcome": "Overall Survival",
            "sq_answers": {
                "1.1": {"answer": "Y", "support_level": "strong"},
                "1.2": {"answer": "Y", "support_level": "strong"},
                "1.3": {"answer": "Y", "support_level": "strong"},
            },
            "domain_judgments": {},
            "domain_rationales": {},
        }
    )

    assert result["sq_answers"]["1.3"]["answer"] == "N"
    assert result["sq_answers"]["1.3"]["d1_baseline_balance_guard_applied"] is True
    assert result["domain_judgments"]["D1"] == "Low"


def test_d1_judge_uses_authoritative_randomized_design_evidence(monkeypatch):
    def fake_add_domain_judgment_with_pivotality_tests(*args, **kwargs):
        return {
            "sq_answers": {
                "1.1": {
                    "answer": "NI",
                    "quote": "No relevant text found",
                    "justification": "Packet did not include sequence evidence.",
                    "support_level": "unsupported",
                },
                "1.2": {"answer": "NI", "support_level": "unsupported"},
                "1.3": {"answer": "NI", "support_level": "unsupported"},
            },
            "domain_judgments": {"D1": "Some concerns"},
            "domain_rationales": {"D1": "Sequence unclear."},
            "pivotality_tests": {"D1": []},
            "sq_support_adjudications": {"D1": []},
        }

    monkeypatch.setattr(
        "rob2_pipeline.nodes.domain1.add_domain_judgment_with_pivotality_tests",
        fake_add_domain_judgment_with_pivotality_tests,
    )

    result = domain1_judge_node(
        {
            "outcome": "Adverse Events",
            "ctgov_design": "Allocation type: RANDOMIZED",
            "sq_answers": {
                "1.1": {"answer": "NI", "support_level": "unsupported"},
                "1.2": {"answer": "NI", "support_level": "unsupported"},
                "1.3": {"answer": "NI", "support_level": "unsupported"},
            },
            "domain_judgments": {},
            "domain_rationales": {},
        }
    )

    assert result["sq_answers"]["1.1"]["answer"] == "Y"
    assert result["sq_answers"]["1.1"]["d1_randomized_design_guard_applied"] is True


def test_d1_judge_uses_blinded_randomized_placebo_as_probable_concealment(
    monkeypatch,
):
    def fake_add_domain_judgment_with_pivotality_tests(*args, **kwargs):
        return {
            "sq_answers": {
                "1.1": {"answer": "Y", "support_level": "strong"},
                "1.2": {
                    "answer": "NI",
                    "quote": "double‑blind placebo-controlled randomized trial",
                    "justification": "No central randomization details.",
                    "support_level": "unsupported",
                },
                "1.3": {"answer": "NI", "support_level": "unsupported"},
            },
            "domain_judgments": {"D1": "Some concerns"},
            "domain_rationales": {"D1": "Concealment unclear."},
            "pivotality_tests": {"D1": []},
            "sq_support_adjudications": {"D1": []},
        }

    monkeypatch.setattr(
        "rob2_pipeline.nodes.domain1.add_domain_judgment_with_pivotality_tests",
        fake_add_domain_judgment_with_pivotality_tests,
    )

    result = domain1_judge_node(
        {
            "outcome": "Progression-Free Survival",
            "ctgov_design": "Allocation type: RANDOMIZED; Masking: QUADRUPLE",
            "sq_answers": {
                "1.1": {"answer": "Y", "support_level": "strong"},
                "1.2": {"answer": "NI", "support_level": "unsupported"},
                "1.3": {"answer": "NI", "support_level": "unsupported"},
            },
            "domain_judgments": {},
            "domain_rationales": {},
        }
    )

    assert result["sq_answers"]["1.2"]["answer"] == "PY"
    assert result["sq_answers"]["1.2"]["d1_concealment_guard_applied"] is True
    assert result["domain_judgments"]["D1"] == "Low"


def test_d1_judge_uses_central_randomization_despite_open_label(monkeypatch):
    def fake_add_domain_judgment_with_pivotality_tests(*args, **kwargs):
        return {
            "sq_answers": {
                "1.1": {"answer": "Y", "support_level": "strong"},
                "1.2": {
                    "answer": "NI",
                    "quote": (
                        "Eligible patients were centrally randomly assigned "
                        "in the Alea Clinical Portal."
                    ),
                    "justification": "Open-label trial; concealment not stated.",
                    "support_level": "unsupported",
                },
                "1.3": {"answer": "NI", "support_level": "unsupported"},
            },
            "domain_judgments": {"D1": "Some concerns"},
            "domain_rationales": {"D1": "Concealment unclear."},
            "pivotality_tests": {"D1": []},
            "sq_support_adjudications": {"D1": []},
        }

    monkeypatch.setattr(
        "rob2_pipeline.nodes.domain1.add_domain_judgment_with_pivotality_tests",
        fake_add_domain_judgment_with_pivotality_tests,
    )

    result = domain1_judge_node(
        {
            "outcome": "Adverse Events",
            "ctgov_design": "Allocation type: RANDOMIZED; Masking: NONE",
            "sq_answers": {
                "1.1": {"answer": "Y", "support_level": "strong"},
                "1.2": {"answer": "NI", "support_level": "unsupported"},
                "1.3": {"answer": "NI", "support_level": "unsupported"},
            },
            "domain_judgments": {},
            "domain_rationales": {},
        }
    )

    assert result["sq_answers"]["1.2"]["answer"] == "PY"
    assert result["sq_answers"]["1.2"]["d1_concealment_guard_applied"] is True
    assert result["domain_judgments"]["D1"] == "Low"


def test_d1_judge_rejects_post_randomization_treatment_as_baseline_imbalance(
    monkeypatch,
):
    def fake_add_domain_judgment_with_pivotality_tests(*args, **kwargs):
        return {
            "sq_answers": {
                "1.1": {"answer": "Y", "support_level": "strong"},
                "1.2": {"answer": "Y", "support_level": "strong"},
                "1.3": {
                    "answer": "Y",
                    "quote": (
                        "treated by at least one life-prolonging therapy ... "
                        "subsequently ... next-generation hormonal therapy"
                    ),
                    "justification": (
                        "Post-progression treatment differed between groups."
                    ),
                    "support_level": "strong",
                    "supporting_fact_artifact_ids": ["fact"],
                },
            },
            "domain_judgments": {"D1": "Some concerns"},
            "domain_rationales": {"D1": "Baseline imbalance concern."},
            "pivotality_tests": {"D1": []},
            "sq_support_adjudications": {"D1": []},
        }

    monkeypatch.setattr(
        "rob2_pipeline.nodes.domain1.add_domain_judgment_with_pivotality_tests",
        fake_add_domain_judgment_with_pivotality_tests,
    )

    result = domain1_judge_node(
        {
            "outcome": "Adverse Events",
            "sq_answers": {
                "1.1": {"answer": "Y", "support_level": "strong"},
                "1.2": {"answer": "Y", "support_level": "strong"},
                "1.3": {"answer": "Y", "support_level": "strong"},
            },
            "domain_judgments": {},
            "domain_rationales": {},
        }
    )

    assert result["sq_answers"]["1.3"]["answer"] == "NI"
    assert result["sq_answers"]["1.3"]["d1_baseline_source_guard_applied"] is True
    assert result["domain_judgments"]["D1"] == "Low"


def test_d2_judge_reapplies_actual_deviation_guard_after_adjudication(monkeypatch):
    def fake_add_domain_judgment_with_pivotality_tests(*args, **kwargs):
        return {
            "sq_answers": {
                "2.1": {"answer": "Y", "support_level": "strong"},
                "2.2": {"answer": "Y", "support_level": "strong"},
                "2.3": {
                    "answer": "Y",
                    "quote": "Masking: NONE (masked parties: not specified)",
                    "justification": "Open-label design and eligibility criteria.",
                    "support_level": "strong",
                },
                "2.4": {"answer": "Y", "support_level": "strong"},
                "2.5": {"answer": "Y", "support_level": "strong"},
                "2.6": {"answer": "Y", "support_level": "strong"},
                "2.7": {"answer": "NA", "support_level": "unsupported"},
            },
            "domain_judgments": {"D2": "Some concerns"},
            "domain_rationales": {"D2": "Deviation concern."},
            "pivotality_tests": {"D2": []},
            "sq_support_adjudications": {"D2": []},
        }

    monkeypatch.setattr(
        "rob2_pipeline.nodes.domain2.add_domain_judgment_with_pivotality_tests",
        fake_add_domain_judgment_with_pivotality_tests,
    )

    result = domain2_judge_node(
        {
            "outcome": "Overall Survival",
            "effect_of_interest": "ITT",
            "sq_answers": {
                "2.1": {"answer": "Y", "support_level": "strong"},
                "2.2": {"answer": "Y", "support_level": "strong"},
                "2.3": {"answer": "Y", "support_level": "strong"},
                "2.6": {"answer": "Y", "support_level": "strong"},
            },
            "domain_judgments": {},
            "domain_rationales": {},
        }
    )

    assert result["sq_answers"]["2.3"]["answer"] == "N"
    assert result["sq_answers"]["2.4"]["answer"] == "NA"
    assert result["sq_answers"]["2.5"]["answer"] == "NA"
    assert result["domain_judgments"]["D2"] == "Low"


def test_d5_judge_node_reapplies_selective_reporting_guard_after_adjudication(
    monkeypatch,
):
    def fake_add_domain_judgment_with_pivotality_tests(*args, **kwargs):
        return {
            "sq_answers": {
                "5.1": {"answer": "Y", "support_level": "strong"},
                "5.2": {
                    "answer": "PY",
                    "quote": "The primary end point was radiographic progression-free survival.",
                    "justification": (
                        "The registration lists alternative endpoint families, "
                        "but no result-based selection of this same outcome is shown."
                    ),
                    "support_level": "weak",
                },
                "5.3": {"answer": "N", "support_level": "moderate"},
            },
            "domain_judgments": {"D5": "High"},
            "domain_rationales": {
                "D5": "5.2 or 5.3 = Y/PY (selective result reporting) -> High"
            },
            "pivotality_tests": {"D5": []},
            "sq_support_adjudications": {"D5": []},
        }

    monkeypatch.setattr(
        "rob2_pipeline.nodes.domain5.add_domain_judgment_with_pivotality_tests",
        fake_add_domain_judgment_with_pivotality_tests,
    )

    result = domain5_judge_node(
        {
            "outcome": "Progression-Free Survival",
            "sq_answers": {
                "5.1": {"answer": "Y", "support_level": "strong"},
                "5.2": {"answer": "N", "support_level": "moderate"},
                "5.3": {"answer": "N", "support_level": "moderate"},
            },
            "domain_judgments": {},
            "domain_rationales": {},
        }
    )

    assert result["sq_answers"]["5.2"]["answer"] == "PN"
    assert result["sq_answers"]["5.2"]["d5_guard_applied"] is True
    assert result["domain_judgments"]["D5"] == "Low"


def _complete_low_domain_judgments():
    return {"D1": "Low", "D2": "Low", "D3": "Low", "D4": "Low", "D5": "Low"}


def _domain_stage_result(parsed: dict[str, dict]) -> JsonContractResult:
    return JsonContractResult(
        artifact={
            "answers": [
                {
                    "sq_id": sq_id,
                    "support_level": "moderate",
                    "support_rationale": "Test fixture support.",
                    **answer,
                }
                for sq_id, answer in parsed.items()
            ]
        },
        log=[],
        status="validated",
        failure_reason=None,
    )


def test_domain4_autosets_clinician_assessor_awareness_in_open_label_trial(monkeypatch):
    def fake_call_json_contract_llm(state, prompt, node_name, **kwargs):
        parsed = {
            "4.1": {
                "answer": "N",
                "quote": "measurement",
                "justification": "standard",
                "uncertainty_flag": "NORMAL",
            },
            "4.2": {
                "answer": "N",
                "quote": "method",
                "justification": "same",
                "uncertainty_flag": "NORMAL",
            },
            "4.3": {
                "answer": "NI",
                "quote": "No relevant text found",
                "justification": "unclear",
                "uncertainty_flag": "HIGH",
            },
            "4.4": {
                "answer": "NI",
                "quote": "No relevant text found",
                "justification": "unclear",
                "uncertainty_flag": "HIGH",
            },
            "4.5": {
                "answer": "NI",
                "quote": "No relevant text found",
                "justification": "unclear",
                "uncertainty_flag": "HIGH",
            },
        }
        return _domain_stage_result(parsed)

    monkeypatch.setattr(
        "rob2_pipeline.nodes.domain_helpers.call_json_contract_llm",
        fake_call_json_contract_llm,
    )
    state = {
        "intervention": "Drug A",
        "comparator": "Placebo",
        "outcome": "Progression-free survival",
        "outcome_type": "clinician-graded",
        "evidence": empty_paper_evidence(),
        "rag_contexts": {
            "d4_measurement": "RECIST assessment",
            "d4_assessor": "open-label",
        },
        "sq_answers": {
            "2.1": {"answer": "N"},
            "2.2": {"answer": "Y", "quote": "open-label"},
        },
    }

    result = domain4_sq_node(state)

    assert result["sq_answers"]["4.3"]["answer"] == "PY"
    assert "clinician grading" in result["sq_answers"]["4.3"]["justification"]


def test_domain4_autosets_objective_outcome_uninfluenced_when_awareness_unknown(
    monkeypatch,
):
    def fake_call_json_contract_llm(state, prompt, node_name, **kwargs):
        parsed = {
            "4.1": {
                "answer": "N",
                "quote": "vital status",
                "justification": "appropriate",
                "uncertainty_flag": "NORMAL",
            },
            "4.2": {
                "answer": "N",
                "quote": "same method",
                "justification": "same",
                "uncertainty_flag": "NORMAL",
            },
            "4.3": {
                "answer": "NI",
                "quote": "Not reported",
                "justification": "unknown",
                "uncertainty_flag": "NORMAL",
            },
            "4.4": {
                "answer": "NI",
                "quote": "Not reported",
                "justification": "unknown",
                "uncertainty_flag": "NORMAL",
            },
            "4.5": {
                "answer": "NI",
                "quote": "Not reported",
                "justification": "unknown",
                "uncertainty_flag": "NORMAL",
            },
        }
        return _domain_stage_result(parsed)

    monkeypatch.setattr(
        "rob2_pipeline.nodes.domain_helpers.call_json_contract_llm",
        fake_call_json_contract_llm,
    )
    state = {
        "intervention": "Drug A",
        "comparator": "Placebo",
        "outcome": "Overall Survival",
        "outcome_type": "vital-status",
        "evidence": empty_paper_evidence(),
        "rag_contexts": {
            "d4_measurement": "overall survival",
            "d4_assessor": "vital status",
        },
        "sq_answers": {"2.1": {"answer": "Y"}, "2.2": {"answer": "Y"}},
    }

    result = domain4_sq_node(state)

    assert result["sq_answers"]["4.3"]["answer"] == "NI"
    assert result["sq_answers"]["4.4"]["answer"] == "N"
    assert result["sq_answers"]["4.5"]["answer"] == "NA"


def test_domain4_normalizes_invalid_assessor_na_for_objective_outcome(monkeypatch):
    def fake_call_json_contract_llm(state, prompt, node_name, **kwargs):
        parsed = {
            "4.1": {
                "answer": "N",
                "quote": "vital status",
                "justification": "appropriate",
                "uncertainty_flag": "NORMAL",
            },
            "4.2": {
                "answer": "N",
                "quote": "same method",
                "justification": "same",
                "uncertainty_flag": "NORMAL",
            },
            "4.3": {
                "answer": "NA",
                "quote": "Not applicable",
                "justification": "invalid skip",
                "uncertainty_flag": "NORMAL",
            },
            "4.4": {
                "answer": "NA",
                "quote": "Not applicable",
                "justification": "invalid skip",
                "uncertainty_flag": "NORMAL",
            },
            "4.5": {
                "answer": "NA",
                "quote": "Not applicable",
                "justification": "invalid skip",
                "uncertainty_flag": "NORMAL",
            },
        }
        return _domain_stage_result(parsed)

    monkeypatch.setattr(
        "rob2_pipeline.nodes.domain_helpers.call_json_contract_llm",
        fake_call_json_contract_llm,
    )
    state = {
        "intervention": "Drug A",
        "comparator": "Placebo",
        "outcome": "Overall Survival",
        "outcome_type": "vital-status",
        "evidence": empty_paper_evidence(),
        "rag_contexts": {
            "d4_measurement": "overall survival",
            "d4_assessor": "vital status",
        },
        "sq_answers": {},
    }

    result = domain4_sq_node(state)

    assert result["sq_answers"]["4.3"]["answer"] == "NI"
    assert result["sq_answers"]["4.4"]["answer"] == "N"
    assert result["sq_answers"]["4.5"]["answer"] == "NA"


def test_domain4_judge_reapplies_objective_control_after_adjudication(monkeypatch):
    def fake_adjudication(state, prompt, node_name, **kwargs):
        del state, prompt, kwargs
        sq_id = node_name.rsplit("_", 2)[-2] + "." + node_name.rsplit("_", 1)[-1]
        answers = {
            "4.1": {
                "sq_id": "4.1",
                "answer": "N",
                "quote": "Kaplan-Meier estimates were used.",
                "justification": "Standard survival method.",
                "uncertainty_flag": "NORMAL",
                "support_level": "strong",
                "support_rationale": "Suitable time-to-event method.",
                "residual_uncertainty": "None.",
                "quote_traceability_status": "traceable",
            },
            "4.2": {
                "sq_id": "4.2",
                "answer": "NI",
                "quote": "No group-specific differences reported.",
                "justification": "No differential method details.",
                "uncertainty_flag": "NORMAL",
                "support_level": "unsupported",
                "support_rationale": "Unclear before objective-outcome control.",
                "residual_uncertainty": "Unclear.",
                "quote_traceability_status": "traceable",
            },
        }
        artifact = answers.get(
            sq_id,
            {
                "sq_id": sq_id,
                "answer": "NI",
                "quote": "No relevant text found",
                "justification": "No change.",
                "uncertainty_flag": "HIGH",
                "support_level": "unsupported",
                "support_rationale": "No change.",
                "residual_uncertainty": "No change.",
                "quote_traceability_status": "traceability_not_assessed",
            },
        )
        return SimpleNamespace(
            artifact=artifact,
            log=[],
            status="validated",
            failure_reason=None,
        )

    monkeypatch.setattr(
        "rob2_pipeline.nodes.common.call_json_contract_llm",
        fake_adjudication,
    )
    state = {
        "outcome": "Overall Survival",
        "outcome_type": "vital-status",
        "outcome_classification_support": {
            "support_level": "moderate",
            "support_rationale": "Overall survival is objective vital status.",
        },
        "evidence_packets": {"4.1": {}, "4.2": {}, "4.3": {}, "4.4": {}, "4.5": {}},
        "support_constraints": [],
        "sq_answers": {
            "4.1": {
                "answer": "NI",
                "quote": "No relevant text found",
                "justification": "Insufficient.",
                "support_level": "unsupported",
                "support_rationale": "Insufficient.",
            },
            "4.2": {
                "answer": "NI",
                "quote": "No relevant text found",
                "justification": "Insufficient.",
                "support_level": "unsupported",
                "support_rationale": "Insufficient.",
            },
            "4.3": {
                "answer": "PY",
                "quote": "Open label.",
                "justification": "Assessors probably aware.",
                "support_level": "moderate",
                "support_rationale": "Open-label design.",
            },
            "4.4": {
                "answer": "N",
                "quote": "Death endpoint.",
                "justification": "Objective.",
                "support_level": "moderate",
                "support_rationale": "Objective endpoint.",
            },
            "4.5": {
                "answer": "NI",
                "quote": "No relevant text found",
                "justification": "Insufficient.",
                "support_level": "unsupported",
                "support_rationale": "Insufficient.",
            },
        },
    }

    result = domain4_judge_node(state)

    assert result["sq_answers"]["4.1"]["answer"] == "N"
    assert result["sq_answers"]["4.2"]["answer"] == "PN"
    assert result["domain_judgments"]["D4"] == "Low"
    assert result["d4_judgment_artifact"]["input_sq_answers"]["4.2"]["answer"] == "PN"


def test_domain4_judge_reapplies_safety_controls_without_adjudication(monkeypatch):
    def fake_add_domain_judgment_with_pivotality_tests(*args, **kwargs):
        return {
            "sq_answers": {
                "4.1": {"answer": "NI", "support_level": "unsupported"},
                "4.2": {"answer": "NI", "support_level": "unsupported"},
                "4.3": {"answer": "Y", "support_level": "strong"},
                "4.4": {"answer": "NI", "support_level": "unsupported"},
                "4.5": {
                    "answer": "Y",
                    "quote": "Neither the investigators nor the patients were masked.",
                    "justification": "Open-label, so likely influenced.",
                    "support_level": "strong",
                },
            },
            "domain_judgments": {"D4": "High"},
            "domain_rationales": {"D4": "4.5=Y -> High"},
            "pivotality_tests": {"D4": []},
            "sq_support_adjudications": {"D4": []},
        }

    monkeypatch.setattr(
        "rob2_pipeline.nodes.domain4.add_domain_judgment_with_pivotality_tests",
        fake_add_domain_judgment_with_pivotality_tests,
    )

    result = domain4_judge_node(
        {
            "outcome": "Adverse Events",
            "outcome_type": "clinician-graded",
            "outcome_properties": {"safety_harm": True},
            "evidence": {
                "results": {
                    "text": (
                        "Adverse events were graded on the basis of the National "
                        "Cancer Institute Common Terminology Criteria."
                    )
                }
            },
            "sq_answers": {
                "2.1": {"answer": "Y", "quote": "Open-label trial."},
                "2.2": {"answer": "Y", "quote": "Open-label trial."},
                "4.1": {"answer": "NI"},
                "4.2": {"answer": "NI"},
                "4.3": {"answer": "Y"},
                "4.4": {"answer": "NI"},
                "4.5": {"answer": "Y"},
            },
            "domain_judgments": {},
            "domain_rationales": {},
        }
    )

    assert result["sq_answers"]["4.1"]["answer"] == "N"
    assert result["sq_answers"]["4.2"]["answer"] == "PN"
    assert result["sq_answers"]["4.5"]["answer"] == "PN"
    assert result["domain_judgments"]["D4"] == "Some concerns"


def test_prompts_include_skill_domain2_and_domain5_guidance():
    assert "effect of adhering to intervention" in PROMPT_DOMAIN2_ADHERING_CONDITIONAL
    assert "instrumental variable" in PROMPT_DOMAIN2_ADHERING_ANALYSIS
    assert "selected, on the basis of the results" in PROMPT_DOMAIN5
