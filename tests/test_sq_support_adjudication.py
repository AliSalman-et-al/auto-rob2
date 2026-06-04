import pytest

from rob2_pipeline.nodes.domain1 import domain1_judge_node
from rob2_pipeline.nodes.domain3 import domain3_judge_node
from rob2_pipeline.nodes.domain4 import domain4_judge_node
from rob2_pipeline.llm_contracts import JsonContractResult
from rob2_pipeline.pipeline import _assessment_json


def _answer(answer: str, support_level: str = "weak") -> dict:
    return {
        "answer": answer,
        "quote": "trial report",
        "justification": "original justification",
        "uncertainty_flag": "NORMAL",
        "support_level": support_level,
        "support_rationale": "original support",
    }


def _json_contract_result(
    answer: dict, sq_id: str = "1.3", node_name: str = "sq_support_adjudication_D1_1_3"
) -> JsonContractResult:
    return JsonContractResult(
        artifact={
            "sq_id": sq_id,
            "answer": answer["answer"],
            "quote": answer["quote"],
            "justification": answer["justification"],
            "uncertainty_flag": answer["uncertainty_flag"],
            "support_level": answer["support_level"],
            "support_rationale": answer["support_rationale"],
            "residual_uncertainty": answer.get(
                "residual_uncertainty", "No residual uncertainty reported."
            ),
            "quote_traceability_status": answer.get(
                "quote_traceability_status", "traceability_not_assessed"
            ),
        },
        log=[{"node": node_name, "cache_hit": False}],
        status="validated",
    )


def _stub_adjudication_contract(monkeypatch, calls: list, answers: dict[str, dict]):
    def fake_call_json_contract_llm(
        state,
        prompt,
        node_name,
        *,
        schema_model,
        schema_version,
        prompt_version,
        fallback_factory,
        max_attempts=2,
    ):
        sq_id = node_name.rsplit("_", 2)[-2] + "." + node_name.rsplit("_", 1)[-1]
        chunk_sources = []
        for meta in (state.get("rag_chunk_metadata", {}).get("d1", []) or [])[:5]:
            pages = meta.get("page_numbers") or []
            chunk_sources.append(
                f"[page {pages[0] if pages else '?'}, {meta.get('section') or 'Unknown'}]"
            )
        calls.append(
            {
                "node_name": node_name,
                "parse_sq_ids": [sq_id],
                "prompt": prompt,
                "chunk_sources": chunk_sources,
            }
        )
        return _json_contract_result(answers[sq_id], sq_id=sq_id, node_name=node_name)

    monkeypatch.setattr(
        "rob2_pipeline.nodes.common.call_json_contract_llm",
        fake_call_json_contract_llm,
    )


def test_adjudication_prompt_requests_json_contract(monkeypatch):
    calls = []

    def fake_call_json_contract_llm(
        state,
        prompt,
        node_name,
        *,
        schema_model,
        schema_version,
        prompt_version,
        fallback_factory,
        max_attempts=2,
    ):
        calls.append(
            {
                "prompt": prompt,
                "node_name": node_name,
                "schema_model": schema_model,
                "schema_version": schema_version,
                "prompt_version": prompt_version,
                "max_attempts": max_attempts,
            }
        )
        return _json_contract_result(
            {
                "answer": "N",
                "quote": "Baseline characteristics were well balanced.",
                "justification": "Baseline data do not suggest a randomization problem.",
                "uncertainty_flag": "NORMAL",
                "support_level": "strong",
                "support_rationale": "Direct baseline evidence supports no concern.",
                "residual_uncertainty": "No important residual uncertainty.",
            }
        )

    monkeypatch.setattr(
        "rob2_pipeline.nodes.common.call_json_contract_llm",
        fake_call_json_contract_llm,
    )

    domain1_judge_node(
        {
            "outcome": "overall survival",
            "sq_answers": {
                "1.1": _answer("Y", "strong"),
                "1.2": _answer("Y", "strong"),
                "1.3": _answer("Y", "weak"),
            },
            "domain_judgments": {},
            "domain_rationales": {},
            "evidence_packets": {"1.3": {"sources": []}},
        }
    )

    assert calls[0]["node_name"] == "sq_support_adjudication_D1_1_3"
    assert calls[0]["schema_version"]
    assert calls[0]["prompt_version"]
    assert "Return only JSON" in calls[0]["prompt"]
    assert "XML" not in calls[0]["prompt"]
    assert "<sq_1_3>" not in calls[0]["prompt"]


def test_pivotal_weak_answer_triggers_targeted_adjudication(monkeypatch):
    calls = []
    _stub_adjudication_contract(
        monkeypatch,
        calls,
        {
            "1.3": {
                "answer": "N",
                "quote": "Baseline characteristics were well balanced.",
                "justification": "Baseline data do not suggest a randomization problem.",
                "uncertainty_flag": "NORMAL",
                "support_level": "strong",
                "support_rationale": "Direct baseline evidence supports no concern.",
                "residual_uncertainty": "No important residual uncertainty.",
            }
        },
    )

    result = domain1_judge_node(
        {
            "intervention": "ADT plus abiraterone",
            "comparator": "ADT",
            "outcome": "overall survival",
            "sq_answers": {
                "1.1": _answer("Y", "strong"),
                "1.2": _answer("Y", "strong"),
                "1.3": _answer("Y", "weak"),
            },
            "domain_judgments": {},
            "domain_rationales": {},
            "evidence_packets": {
                "1.3": {
                    "sources": [
                        {
                            "text": "Baseline characteristics were well balanced.",
                            "section": "Results",
                            "page_numbers": [5],
                        }
                    ],
                    "missing_evidence": [],
                }
            },
            "rag_chunk_metadata": {
                "d1": [{"section": "Results", "page_numbers": [5]}],
            },
        }
    )

    assert calls == [
        {
            "node_name": "sq_support_adjudication_D1_1_3",
            "parse_sq_ids": ["1.3"],
            "prompt": calls[0]["prompt"],
            "chunk_sources": ["[page 5, Results]"],
        }
    ]
    assert "1.1" not in calls[0]["prompt"]
    assert "SQ 1.3" in calls[0]["prompt"]
    assert result["sq_answers"]["1.3"]["answer"] == "N"
    assert result["domain_judgments"]["D1"] == "Low"
    assert (
        result["sq_support_adjudications"]["D1"][0]["initial_answer"]["answer"] == "Y"
    )
    assert (
        result["sq_support_adjudications"]["D1"][0]["adjudicated_answer"]["answer"]
        == "N"
    )


def test_non_pivotal_weak_answer_does_not_trigger_adjudication(monkeypatch):
    calls = []

    def fake_call_fn(*args, **kwargs):
        calls.append(args)
        return "", [], {}

    monkeypatch.setattr("rob2_pipeline.nodes.common.call_node_llm", fake_call_fn)

    result = domain1_judge_node(
        {
            "sq_answers": {
                "1.1": _answer("Y", "weak"),
                "1.2": _answer("Y", "strong"),
                "1.3": _answer("N", "strong"),
            },
            "domain_judgments": {},
            "domain_rationales": {},
        }
    )

    assert calls == []
    assert "sq_support_adjudications" not in result


def test_unresolved_pivotal_weak_answer_is_audit_limited(monkeypatch):
    _stub_adjudication_contract(
        monkeypatch,
        [],
        {
            "1.3": {
                "answer": "Y",
                "quote": "No direct baseline table was available.",
                "justification": "The concern remains weakly supported.",
                "uncertainty_flag": "HIGH",
                "support_level": "weak",
                "support_rationale": "Evidence remains indirect.",
                "residual_uncertainty": "The pivotal concern is unresolved.",
            }
        },
    )

    result = domain1_judge_node(
        {
            "outcome": "overall survival",
            "sq_answers": {
                "1.1": _answer("Y", "strong"),
                "1.2": _answer("Y", "strong"),
                "1.3": _answer("Y", "weak"),
            },
            "domain_judgments": {},
            "domain_rationales": {},
            "evidence_packets": {
                "1.3": {
                    "sources": [
                        {
                            "text": "No direct baseline table was available.",
                            "section": "Results",
                            "page_numbers": [5],
                        }
                    ]
                }
            },
        }
    )

    assert result["initial_domain_judgments"]["D1"] == "Some concerns"
    assert result["domain_judgments"]["D1"] == "Some concerns"
    assert result["pivotality_tests"]["D1"][0]["acceptance_status"] == "audit_limited"


def test_malformed_adjudication_output_does_not_update_assessment_artifacts(
    monkeypatch,
):
    def fake_call_json_contract_llm(*args, **kwargs):
        return JsonContractResult(
            artifact={
                "sq_id": "1.3",
                "answer": "LOW",
                "quote": "Baseline characteristics were well balanced.",
                "justification": "Attempts to write an invalid answer.",
                "uncertainty_flag": "NORMAL",
                "support_level": "strong",
                "support_rationale": "Invalid schema should be rejected.",
                "residual_uncertainty": "Invalid output was rejected.",
                "domain_judgments": {"D1": "Low"},
            },
            log=[{"node": "sq_support_adjudication_D1_1_3", "cache_hit": False}],
            status="fallback",
            failure_reason="Schema validation failed: answer; domain_judgments",
        )

    monkeypatch.setattr(
        "rob2_pipeline.nodes.common.call_json_contract_llm",
        fake_call_json_contract_llm,
    )

    result = domain1_judge_node(
        {
            "outcome": "overall survival",
            "sq_answers": {
                "1.1": _answer("Y", "strong"),
                "1.2": _answer("Y", "strong"),
                "1.3": _answer("Y", "weak"),
            },
            "domain_judgments": {},
            "domain_rationales": {},
            "evidence_packets": {
                "1.3": {
                    "sources": [
                        {
                            "text": "Baseline characteristics were well balanced.",
                            "section": "Results",
                            "page_numbers": [5],
                        }
                    ],
                }
            },
        }
    )

    assert result["sq_answers"]["1.3"]["answer"] == "Y"
    assert result["domain_judgments"]["D1"] == "Some concerns"
    attempt = result["sq_support_adjudications"]["D1"][0]
    assert attempt["changed"] is False
    assert attempt["validation_status"] == "rejected"
    assert any("answer" in error for error in attempt["validation_errors"])
    assert any("domain_judgments" in error for error in attempt["validation_errors"])
    assert result["pivotality_tests"]["D1"][0]["acceptance_status"] == "audit_limited"


@pytest.mark.parametrize(
    (
        "adjudicated_answer",
        "expected_changed_answer",
        "expected_changed_support",
        "expected_acceptance_status",
    ),
    [
        (_answer("N", "weak"), True, False, "audit_limited"),
        (_answer("Y", "strong"), False, True, "accepted"),
        (_answer("N", "strong"), True, True, "accepted"),
        (_answer("Y", "weak"), False, False, "audit_limited"),
    ],
)
def test_adjudication_records_answer_and_support_change_flags(
    monkeypatch,
    adjudicated_answer,
    expected_changed_answer,
    expected_changed_support,
    expected_acceptance_status,
):
    adjudicated_answer = dict(adjudicated_answer)
    adjudicated_answer["support_rationale"] = "adjudicated support"

    _stub_adjudication_contract(monkeypatch, [], {"1.3": adjudicated_answer})

    result = domain1_judge_node(
        {
            "outcome": "overall survival",
            "sq_answers": {
                "1.1": _answer("Y", "strong"),
                "1.2": _answer("Y", "strong"),
                "1.3": _answer("Y", "weak"),
            },
            "domain_judgments": {},
            "domain_rationales": {},
            "evidence_packets": {"1.3": {"sources": []}},
        }
    )

    attempt = result["sq_support_adjudications"]["D1"][0]
    assert attempt["changed_answer"] is expected_changed_answer
    assert attempt["changed_support"] is expected_changed_support
    assert attempt["changed"] is (expected_changed_answer or expected_changed_support)
    assert attempt["rationale"] == "adjudicated support"
    assert attempt["provenance"]["llm_node"] == "sq_support_adjudication_D1_1_3"
    assert (
        result["pivotality_tests"]["D1"][0]["acceptance_status"]
        == expected_acceptance_status
    )


def test_adjudication_records_semantic_support_without_overwriting_traceability(
    monkeypatch,
):
    _stub_adjudication_contract(
        monkeypatch,
        [],
        {
            "1.3": {
                "answer": "N",
                "quote": "Baseline characteristics were well balanced.",
                "justification": "The selected quote does not support a baseline imbalance concern.",
                "uncertainty_flag": "NORMAL",
                "support_level": "strong",
                "support_rationale": "The quote semantically supports no concern.",
                "residual_uncertainty": "No important residual uncertainty.",
            }
        },
    )

    result = domain1_judge_node(
        {
            "outcome": "overall survival",
            "sq_answers": {
                "1.1": _answer("Y", "strong"),
                "1.2": _answer("Y", "strong"),
                "1.3": {
                    **_answer("Y", "weak"),
                    "quote_traceability_status": "traceable",
                },
            },
            "domain_judgments": {},
            "domain_rationales": {},
            "evidence_packets": {
                "1.3": {
                    "packet_readiness": {"status": "ready"},
                    "sources": [
                        {
                            "text": "Baseline characteristics were well balanced.",
                            "section": "Results",
                            "page_numbers": [5],
                        }
                    ],
                }
            },
        }
    )

    attempt = result["sq_support_adjudications"]["D1"][0]
    assert attempt["semantic_support_decision"] == {
        "support_level": "strong",
        "rationale": "The quote semantically supports no concern.",
        "residual_uncertainty": "No important residual uncertainty.",
    }
    assert attempt["effect_on_sq_status"] == {
        "initial_answer": "Y",
        "adjudicated_answer": "N",
        "changed_answer": True,
        "changed_support": True,
    }
    assert attempt["effect_on_packet_status"] == {
        "initial_status": "ready",
        "adjudicated_status": "accepted",
    }
    assert attempt["traceability_status"] == {
        "initial": "traceable",
        "adjudicated": "traceable",
    }


def test_pivotal_support_constraint_is_included_in_adjudication_prompt(monkeypatch):
    calls = []
    _stub_adjudication_contract(monkeypatch, calls, {"1.3": _answer("N", "strong")})

    result = domain1_judge_node(
        {
            "outcome": "overall survival",
            "sq_answers": {
                "1.1": _answer("Y", "strong"),
                "1.2": _answer("Y", "strong"),
                "1.3": _answer("Y", "strong"),
            },
            "support_constraints": [
                {
                    "constraint_type": "quote_untraceable",
                    "sq_id": "1.3",
                    "evidence_label": "baseline table",
                    "reason": "The cited quote was not found in source text.",
                    "provenance": {"node": "quote_verifier"},
                }
            ],
            "domain_judgments": {},
            "domain_rationales": {},
            "evidence_packets": {"1.3": {"sources": []}},
        }
    )

    assert result["sq_support_adjudications"]["D1"][0]["constraints"] == [
        {
            "constraint_type": "quote_untraceable",
            "sq_id": "1.3",
            "evidence_label": "baseline table",
            "reason": "The cited quote was not found in source text.",
            "provenance": {"node": "quote_verifier"},
        }
    ]
    assert "Support constraints:" in calls[0]["prompt"]
    assert "quote_untraceable" in calls[0]["prompt"]
    assert "The cited quote was not found in source text." in calls[0]["prompt"]
    assert "Original domain judgment" not in calls[0]["prompt"]
    assert "Alternative-answer domain judgment" not in calls[0]["prompt"]


def test_assessment_json_includes_adjudication_audit_and_final_sq_state():
    state = {
        "sq_answers": {"1.3": _answer("N", "strong")},
        "sq_support_adjudications": {
            "D1": [
                {
                    "sq_id": "1.3",
                    "initial_answer": _answer("Y", "weak"),
                    "adjudicated_answer": _answer("N", "strong"),
                    "domain_impact": {
                        "original_domain_judgment": "Some concerns",
                        "test_domain_judgment": "Low",
                    },
                    "changed": True,
                }
            ]
        },
        "rag_chunk_metadata": {},
    }

    data = _assessment_json(state)

    assert data["sq_answers"]["1.3"]["answer"] == "N"
    assert data["sq_support_adjudications"] == state["sq_support_adjudications"]


@pytest.mark.parametrize(
    (
        "case_name",
        "judge_node",
        "domain",
        "weak_sq_id",
        "sq_answers",
        "adjudicated_answer",
        "expected_judgment",
    ),
    [
        (
            "PEACE-1 D1 allocation concealment",
            domain1_judge_node,
            "D1",
            "1.2",
            {
                "1.1": _answer("Y", "strong"),
                "1.2": _answer("NI", "weak"),
                "1.3": _answer("N", "strong"),
            },
            {
                "answer": "PY",
                "quote": "Allocation was performed centrally before assignment.",
                "justification": "Open-label wording does not undermine concealment before enrolment.",
                "uncertainty_flag": "NORMAL",
                "support_level": "moderate",
                "support_rationale": "Central allocation supports probable concealment.",
            },
            "Low",
        ),
        (
            "STAMPEDE D3 completeness support",
            domain3_judge_node,
            "D3",
            "3.1",
            {
                "3.1": _answer("N", "weak"),
                "3.2": _answer("N", "strong"),
                "3.3": _answer("Y", "strong"),
                "3.4": _answer("N", "strong"),
            },
            {
                "answer": "Y",
                "quote": "Vital status was available for nearly all randomized participants.",
                "justification": "Completeness evidence supports low missing outcome data.",
                "uncertainty_flag": "NORMAL",
                "support_level": "strong",
                "support_rationale": "Direct completeness evidence supports the answer.",
            },
            "Low",
        ),
        (
            "D4 open-label influence",
            domain4_judge_node,
            "D4",
            "4.5",
            {
                "4.1": _answer("N", "strong"),
                "4.2": _answer("N", "strong"),
                "4.3": _answer("Y", "strong"),
                "4.4": _answer("Y", "strong"),
                "4.5": _answer("Y", "weak"),
            },
            {
                "answer": "N",
                "quote": "Outcomes were assessed using standardized radiographic criteria.",
                "justification": "Assessor awareness alone does not show likely influence.",
                "uncertainty_flag": "NORMAL",
                "support_level": "moderate",
                "support_rationale": "Standardized criteria reduce likely influence despite open label design.",
            },
            "Some concerns",
        ),
    ],
)
def test_benchmark_failure_modes_use_adjudicated_pivotal_answer(
    monkeypatch,
    case_name,
    judge_node,
    domain,
    weak_sq_id,
    sq_answers,
    adjudicated_answer,
    expected_judgment,
):
    calls = []
    _stub_adjudication_contract(
        monkeypatch, calls, {weak_sq_id: adjudicated_answer}
    )

    result = judge_node(
        {
            "outcome": case_name,
            "sq_answers": sq_answers,
            "domain_judgments": {},
            "domain_rationales": {},
            "evidence_packets": {
                weak_sq_id: {
                    "sources": [
                        {
                            "text": adjudicated_answer["quote"],
                            "section": "Methods",
                            "page_numbers": [3],
                        }
                    ]
                }
            },
        }
    )

    assert (
        calls[0]["node_name"]
        == f"sq_support_adjudication_{domain}_{weak_sq_id.replace('.', '_')}"
    )
    assert calls[0]["parse_sq_ids"] == [weak_sq_id]
    assert result["sq_answers"][weak_sq_id]["answer"] == adjudicated_answer["answer"]
    assert result["domain_judgments"][domain] == expected_judgment
