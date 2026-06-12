import json
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from rob2_pipeline.benchmark import (
    _required_supplement_failures,
    classify_mismatches,
    compare_judgments,
    find_supplements_for_trial,
    load_reference,
    run_benchmark,
    summarize_benchmark,
    write_benchmark_report,
)


def _benchmark_fixture_output(
    *,
    final_domains: dict[str, str],
    overall: str,
    human_review_priority: str,
    adjudication_domain: str | None = None,
    sq_id: str | None = None,
    initial_answer: str = "",
    final_answer: str = "",
    initial_support: str = "",
    final_support: str = "",
    original_domain_judgment: str = "",
    test_domain_judgment: str = "",
    acceptance_status: str = "",
    support_constraints: list[dict] | None = None,
) -> dict:
    output = {
        "domain_judgments": final_domains,
        "overall_judgment": overall,
        "human_review_priority": human_review_priority,
        "source_documents": [],
        "support_constraints": support_constraints or [],
    }
    if adjudication_domain is None:
        return output

    assert sq_id is not None
    output["pivotality_tests"] = {
        adjudication_domain: [
            {
                "sq_id": sq_id,
                "original_answer": initial_answer,
                "support_level": initial_support,
                "conservative_test_answer": "NI",
                "original_domain_judgment": original_domain_judgment,
                "test_domain_judgment": test_domain_judgment,
                "pivotal": True,
                "acceptance_status": acceptance_status,
            }
        ]
    }
    output["sq_support_adjudications"] = {
        adjudication_domain: [
            {
                "sq_id": sq_id,
                "initial_answer": {
                    "answer": initial_answer,
                    "support_level": initial_support,
                },
                "adjudicated_answer": {
                    "answer": final_answer,
                    "support_level": final_support,
                },
                "domain_impact": {
                    "original_domain_judgment": original_domain_judgment,
                    "test_domain_judgment": test_domain_judgment,
                },
                "changed": initial_answer != final_answer
                or initial_support != final_support,
            }
        ]
    }
    return output


def test_load_reference_strips_whitespace():
    csv_text = (
        "Trial,D1,D2,D3,D4,D5,Overall Risk\n"
        " CHAARTED , L , S , L , L , H , Some Concerns \n"
    )

    with patch("pathlib.Path.open", return_value=StringIO(csv_text)):
        data = load_reference(Path("dummy.csv"))

    assert data == {
        "CHAARTED": {
            "D1": "L",
            "D2": "S",
            "D3": "L",
            "D4": "L",
            "D5": "H",
            "Overall Risk": "Some Concerns",
        }
    }


def test_compare_judgments_case_and_compact_normalization():
    pipeline = {
        "domain_judgments": {
            "D1": " low ",
            "D2": "Some concerns",
            "D3": "HIGH",
            "D4": "Low",
            "D5": "Some Concerns",
        },
        "overall_judgment": " high ",
    }
    reference = {
        "D1": "L",
        "D2": "s",
        "D3": "h",
        "D4": " L ",
        "D5": "S",
        "Overall Risk": "H",
    }

    assert compare_judgments(pipeline, reference) == {
        "D1": True,
        "D2": True,
        "D3": True,
        "D4": True,
        "D5": True,
        "Overall": True,
    }


def test_load_reference_preserves_optional_gold_sq_labels():
    csv_text = (
        "Trial,D1,D2,D3,D4,D5,Overall Risk,SQ 1.1,sq_1_2,2.1\n"
        " CHAARTED , L , S , L , L , H , Some Concerns , Y , PY , N \n"
    )

    with patch("pathlib.Path.open", return_value=StringIO(csv_text)):
        data = load_reference(Path("dummy.csv"))

    assert data["CHAARTED"]["sq_answers"] == {
        "1.1": "Y",
        "1.2": "PY",
        "2.1": "N",
    }


def test_compare_judgments_includes_sq_agreement_only_when_gold_labels_exist():
    pipeline = {
        "domain_judgments": {
            "D1": "Low",
            "D2": "Low",
            "D3": "Low",
            "D4": "Low",
            "D5": "Low",
        },
        "overall_judgment": "Low",
        "sq_answers": {
            "1.1": {"answer": "Y"},
            "1.2": {"answer": "N"},
            "2.1": {"answer": "PY"},
        },
    }
    reference = {
        "D1": "Low",
        "D2": "Low",
        "D3": "Low",
        "D4": "Low",
        "D5": "Low",
        "Overall Risk": "Low",
        "sq_answers": {"1.1": "Y", "1.2": "PY"},
    }

    comparison = compare_judgments(pipeline, reference)

    assert comparison["SQ"] == {"1.1": True, "1.2": False}
    assert "2.1" not in comparison["SQ"]


def test_compare_judgments_omits_sq_agreement_without_gold_labels():
    pipeline = {
        "domain_judgments": {
            "D1": "Low",
            "D2": "Low",
            "D3": "Low",
            "D4": "Low",
            "D5": "Low",
        },
        "overall_judgment": "Low",
        "sq_answers": {"1.1": {"answer": "Y"}},
    }
    reference = {
        "D1": "Low",
        "D2": "Low",
        "D3": "Low",
        "D4": "Low",
        "D5": "Low",
        "Overall Risk": "Low",
    }

    assert "SQ" not in compare_judgments(pipeline, reference)


def test_run_benchmark_scores_final_judgments_and_records_adjudication_metrics(
    tmp_path, monkeypatch
):
    pdf_dir = tmp_path / "benchmark"
    pdf_dir.mkdir()
    (pdf_dir / "TITAN.pdf").write_bytes(b"pdf")

    reference_csv = tmp_path / "ref.csv"
    reference_csv.write_text(
        "Trial,D1,D2,D3,D4,D5,Overall Risk\nTITAN,Low,Low,Low,Low,Low,Low\n",
        encoding="utf-8",
    )

    def fake_run_assessment(**kwargs):
        assessment_dir = Path(kwargs["output_dir"])
        assessment_dir.mkdir(parents=True)
        assessment_dir.joinpath("TITAN_rob2_data.json").write_text(
            json.dumps(
                {
                    "domain_judgments": {
                        "D1": "Low",
                        "D2": "Low",
                        "D3": "Low",
                        "D4": "Low",
                        "D5": "Low",
                    },
                    "overall_judgment": "Low",
                    "pivotality_tests": {
                        "D1": [
                            {
                                "sq_id": "1.3",
                                "support_level": "weak",
                                "pivotal": True,
                                "original_domain_judgment": "Some concerns",
                                "test_domain_judgment": "Low",
                            }
                        ],
                        "D5": [
                            {
                                "sq_id": "5.1",
                                "support_level": "unsupported",
                                "pivotal": False,
                                "original_domain_judgment": "Low",
                                "test_domain_judgment": "Low",
                            }
                        ],
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
                                "domain_impact": {
                                    "original_domain_judgment": "Some concerns",
                                    "test_domain_judgment": "Low",
                                },
                                "llm_node": "sq_support_adjudication_D1_1_3",
                            }
                        ]
                    },
                    "source_documents": [],
                }
            ),
            encoding="utf-8",
        )
        assessment_dir.joinpath("TITAN_trace.json").write_text(
            json.dumps(
                {
                    "llm_calls": [
                        {"node": "domain1_sq", "latency_ms": 100},
                        {
                            "node": "sq_support_adjudication_D1_1_3",
                            "latency_ms": 40,
                            "input_tokens": 7,
                            "output_tokens": 2,
                        },
                    ],
                    "node_spans": [],
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr("rob2_pipeline.benchmark.run_assessment", fake_run_assessment)

    results = run_benchmark(
        pdf_dir=pdf_dir,
        reference_csvs={"OS": reference_csv},
        outcome_map=[{"trial": "TITAN", "outcome_code": "OS"}],
        output_dir=tmp_path / "out",
    )
    summary = summarize_benchmark(results)

    assert results[0]["comparison"]["D1"] is True
    assert results[0]["pipeline"]["initial_domain_judgments"]["D1"] == "Some concerns"
    assert results[0]["pipeline"]["domain_judgments"]["D1"] == "Low"
    assert results[0]["adjudication_metrics"]["initial_final_deltas"] == {
        "domain_judgments": {"D1": {"initial": "Some concerns", "final": "Low"}},
        "overall_judgment": {"initial": "Some concerns", "final": "Low"},
    }
    assert results[0]["adjudication_metrics"]["weak_sq_answers"] == 1
    assert results[0]["adjudication_metrics"]["unsupported_sq_answers"] == 1
    assert results[0]["adjudication_metrics"]["pivotality_tests"] == {
        "total": 2,
        "pivotal": 1,
        "non_pivotal": 1,
    }
    assert results[0]["adjudication_metrics"]["sq_support_adjudications"] == {
        "total": 1,
        "changed_answer": 1,
        "changed_support": 1,
        "changed_answer_or_support": 1,
    }
    assert results[0]["timing"]["adjudication_llm_calls"] == 1
    assert results[0]["timing"]["adjudication_llm_total_ms"] == 40
    assert summary["adjudication_metrics"]["weak_sq_answers"] == 1
    assert summary["adjudication_metrics"]["unsupported_sq_answers"] == 1
    assert summary["timing"]["total_adjudication_llm_calls"] == 1


def test_run_benchmark_regression_fixtures_cover_known_undercalling_patterns(
    tmp_path, monkeypatch
):
    pdf_dir = tmp_path / "benchmark"
    pdf_dir.mkdir()
    for trial in ("PEACE-1", "STAMPEDE", "ARASENS", "TITAN"):
        (pdf_dir / f"{trial}.pdf").write_bytes(b"pdf")

    reference_csv = tmp_path / "ref.csv"
    reference_csv.write_text(
        "\n".join(
            [
                "Trial,D1,D2,D3,D4,D5,Overall Risk",
                "PEACE-1,Low,Some concerns,Low,Low,Low,Some concerns",
                "STAMPEDE,Low,Low,Some concerns,Low,Low,Some concerns",
                "ARASENS,Low,Low,Low,Some concerns,Low,Some concerns",
                "TITAN,Low,Low,Low,Low,Low,Low",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    fixtures = {
        ("PEACE-1", "OS"): _benchmark_fixture_output(
            final_domains={
                "D1": "Low",
                "D2": "Some concerns",
                "D3": "Low",
                "D4": "Low",
                "D5": "Low",
            },
            overall="Some concerns",
            human_review_priority="HIGH",
            adjudication_domain="D2",
            sq_id="2.6",
            initial_answer="Y",
            final_answer="NI",
            initial_support="weak",
            final_support="strong",
            original_domain_judgment="Low",
            test_domain_judgment="High",
            acceptance_status="accepted",
        ),
        ("STAMPEDE", "OS"): _benchmark_fixture_output(
            final_domains={
                "D1": "Low",
                "D2": "Low",
                "D3": "Some concerns",
                "D4": "Low",
                "D5": "Low",
            },
            overall="Some concerns",
            human_review_priority="HIGH",
            adjudication_domain="D3",
            sq_id="3.1",
            initial_answer="Y",
            final_answer="PY",
            initial_support="unsupported",
            final_support="unsupported",
            original_domain_judgment="Low",
            test_domain_judgment="Some concerns",
            acceptance_status="audit_limited",
        ),
        ("ARASENS", "OS"): _benchmark_fixture_output(
            final_domains={
                "D1": "Low",
                "D2": "Low",
                "D3": "Low",
                "D4": "Low",
                "D5": "Low",
            },
            overall="Low",
            human_review_priority="LOW",
        ),
        ("TITAN", "OS"): _benchmark_fixture_output(
            final_domains={
                "D1": "Low",
                "D2": "Low",
                "D3": "Low",
                "D4": "Some concerns",
                "D5": "Low",
            },
            overall="Some concerns",
            human_review_priority="HIGH",
            adjudication_domain="D4",
            sq_id="4.4",
            initial_answer="N",
            final_answer="Y",
            initial_support="weak",
            final_support="strong",
            original_domain_judgment="Low",
            test_domain_judgment="Some concerns",
            acceptance_status="accepted",
            support_constraints=[
                {
                    "constraint_type": "wrong_outcome_context",
                    "sq_id": "outcome_classification",
                    "reason": "Trial-wide PFS language contaminated the assessed adverse-event outcome classification.",
                }
            ],
        ),
    }

    def fake_run_assessment(**kwargs):
        pdf_path = Path(kwargs["pdf_path"])
        outcome_code = "OS"
        assessment_dir = Path(kwargs["output_dir"])
        assessment_dir.mkdir(parents=True)
        assessment_dir.joinpath(f"{pdf_path.stem}_rob2_data.json").write_text(
            json.dumps(fixtures[(pdf_path.stem, outcome_code)]),
            encoding="utf-8",
        )
        assessment_dir.joinpath(f"{pdf_path.stem}_trace.json").write_text(
            json.dumps({"llm_calls": [], "node_spans": []}),
            encoding="utf-8",
        )

    monkeypatch.setattr("rob2_pipeline.benchmark.run_assessment", fake_run_assessment)

    results = run_benchmark(
        pdf_dir=pdf_dir,
        reference_csvs={"OS": reference_csv},
        outcome_map=[
            {"trial": "PEACE-1", "outcome_code": "OS"},
            {"trial": "STAMPEDE", "outcome_code": "OS"},
            {"trial": "ARASENS", "outcome_code": "OS"},
            {"trial": "TITAN", "outcome_code": "OS"},
        ],
        output_dir=tmp_path / "out",
    )
    by_trial = {result["trial"]: result for result in results}
    summary = summarize_benchmark(results)

    assert by_trial["PEACE-1"]["pipeline"]["domain_judgments"]["D2"] == "Some concerns"
    assert by_trial["PEACE-1"]["pipeline"]["initial_domain_judgments"]["D2"] == "Low"
    assert (
        by_trial["PEACE-1"]["adjudication_metrics"]["sq_support_adjudications"][
            "changed_answer"
        ]
        == 1
    )
    assert by_trial["PEACE-1"]["pipeline"]["human_review_priority"] == "HIGH"

    assert by_trial["STAMPEDE"]["pipeline"]["domain_judgments"]["D3"] == "Some concerns"
    assert by_trial["STAMPEDE"]["adjudication_metrics"]["unsupported_sq_answers"] == 1
    assert by_trial["STAMPEDE"]["audit_caught_mismatches"] == {}

    assert by_trial["ARASENS"]["comparison"]["D4"] is False
    assert by_trial["ARASENS"]["audit_caught_mismatches"]["D4"] is False

    assert by_trial["TITAN"]["comparison"]["D4"] is False
    assert by_trial["TITAN"]["audit_caught_mismatches"]["D4"] is True
    assert by_trial["TITAN"]["pipeline"]["human_review_priority"] == "HIGH"

    assert summary["audit_caught_mismatches"]["D4"] == {"caught": 1, "total": 2}
    assert summary["adjudication_metrics"]["weak_sq_answers"] == 2
    assert summary["adjudication_metrics"]["unsupported_sq_answers"] == 1


def test_run_benchmark_reuses_trial_artifacts_across_outcomes(tmp_path, monkeypatch):
    pdf_dir = tmp_path / "benchmark"
    pdf_dir.mkdir()
    (pdf_dir / "TITAN.pdf").write_bytes(b"pdf")
    reference_csv = tmp_path / "ref.csv"
    reference_csv.write_text(
        "Trial,D1,D2,D3,D4,D5,Overall Risk\nTITAN,Low,Low,Low,Low,Low,Low\n",
        encoding="utf-8",
    )
    calls = []

    def fake_run_assessment(**kwargs):
        calls.append(kwargs)
        assessment_dir = Path(kwargs["output_dir"])
        assessment_dir.mkdir(parents=True)
        (assessment_dir / "TITAN_rob2_data.json").write_text(
            json.dumps(
                {
                    "domain_judgments": {
                        "D1": "Low",
                        "D2": "Low",
                        "D3": "Low",
                        "D4": "Low",
                        "D5": "Low",
                    },
                    "overall_judgment": "Low",
                    "source_documents": [],
                }
            ),
            encoding="utf-8",
        )
        return {
            "full_text": "Trial text",
            "evidence": {"warnings": []},
            "source_documents": [],
            "supplement_warnings": [],
            "supplement_segments": [],
            "supplement_retrieval_grades": {},
            "parse_artifacts": [],
        }

    monkeypatch.setattr("rob2_pipeline.benchmark.run_assessment", fake_run_assessment)

    run_benchmark(
        pdf_dir=pdf_dir,
        reference_csvs={"OS": reference_csv, "PFS": reference_csv},
        outcome_map=[
            {"trial": "TITAN", "outcome_code": "OS"},
            {"trial": "TITAN", "outcome_code": "PFS"},
        ],
        output_dir=tmp_path / "out",
    )

    assert calls[0]["precomputed_ingestion"] is None
    assert calls[1]["precomputed_ingestion"] is not None
    assert "trial_retrieval_indexes" not in calls[0]
    assert "trial_retrieval_indexes" not in calls[1]


def test_run_benchmark_scores_gold_evidence_fixtures_when_present(
    tmp_path, monkeypatch
):
    pdf_dir = tmp_path / "benchmark"
    pdf_dir.mkdir()
    (pdf_dir / "TITAN.pdf").write_bytes(b"pdf")
    reference_csv = tmp_path / "ref.csv"
    reference_csv.write_text(
        "Trial,D1,D2,D3,D4,D5,Overall Risk\nTITAN,Low,Low,Low,Low,Low,Low\n",
        encoding="utf-8",
    )
    gold_evidence = tmp_path / "gold_evidence.json"
    gold_evidence.write_text(
        json.dumps(
            {
                "TITAN": {
                    "OS": {
                        "1.1": [
                            {"page": 3, "snippet": "randomized centrally"},
                            {"page": 4, "snippet": "permuted blocks"},
                        ],
                        "3.1": [{"page": 8, "snippet": "data were available"}],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    def fake_run_assessment(**kwargs):
        assessment_dir = Path(kwargs["output_dir"])
        assessment_dir.mkdir(parents=True)
        (assessment_dir / "TITAN_rob2_data.json").write_text(
            json.dumps(
                {
                    "domain_judgments": {
                        "D1": "Low",
                        "D2": "Low",
                        "D3": "Low",
                        "D4": "Low",
                        "D5": "Low",
                    },
                    "overall_judgment": "Low",
                    "evidence_packets": {
                        "1.1": {
                            "sources": [
                                {
                                    "page_numbers": [3],
                                    "text": "Trial participants were randomized centrally.",
                                }
                            ]
                        },
                        "3.1": {"sources": []},
                    },
                    "source_documents": [],
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr("rob2_pipeline.benchmark.run_assessment", fake_run_assessment)

    results = run_benchmark(
        pdf_dir=pdf_dir,
        reference_csvs={"OS": reference_csv},
        outcome_map=[
            {"trial": "TITAN", "outcome_code": "OS"},
            {"trial": "MISSING", "outcome_code": "OS"},
        ],
        output_dir=tmp_path / "out",
        gold_evidence_path=gold_evidence,
    )
    summary = summarize_benchmark(results)
    write_benchmark_report(results, summary, tmp_path / "out" / "benchmark_report.md")
    benchmark_json = json.loads(
        (tmp_path / "out" / "benchmark_results.json").read_text(encoding="utf-8")
    )

    assert results[0]["gold_evidence"]["fixture_found"] is True
    assert results[0]["gold_evidence"]["retrieval_recall"] == {
        "matched": 0,
        "total": 0,
        "rate": None,
    }
    assert results[0]["gold_evidence"]["packet_evidence_recall"] == {
        "matched": 1,
        "total": 3,
        "rate": 1 / 3,
    }
    assert results[1]["skipped"] is True
    assert summary["gold_evidence"]["retrieval_recall"] == {
        "matched": 0,
        "total": 0,
        "rate": None,
    }
    assert benchmark_json["assessments"][0]["gold_evidence"][
        "packet_evidence_recall"
    ] == {"matched": 1, "total": 3, "rate": 1 / 3}
    assert benchmark_json["aggregate"]["gold_evidence"]["fixtures_evaluated"] == 1
    report = (tmp_path / "out" / "benchmark_report.md").read_text(encoding="utf-8")
    assert "## Gold Evidence Recall" in report
    assert "| Retrieval | 0.0% (0/0) |" in report
    assert "| Packet evidence | 33.3% (1/3) |" in report


def test_summarize_benchmark_agreement_and_confusion_dicts():
    results = [
        {
            "trial": "A",
            "skipped": False,
            "error": None,
            "comparison": {
                "D1": True,
                "D2": False,
                "D3": True,
                "D4": True,
                "D5": True,
                "Overall": True,
            },
            "reference": {
                "D1": "Low",
                "D2": "Some concerns",
                "D3": "High",
                "D4": "Low",
                "D5": "Low",
                "Overall Risk": "Some concerns",
            },
            "pipeline": {
                "domain_judgments": {
                    "D1": "Low",
                    "D2": "Low",
                    "D3": "High",
                    "D4": "Low",
                    "D5": "Low",
                },
                "overall_judgment": "Some concerns",
            },
        },
        {
            "trial": "B",
            "skipped": False,
            "error": None,
            "comparison": {
                "D1": False,
                "D2": True,
                "D3": True,
                "D4": True,
                "D5": True,
                "Overall": False,
            },
            "reference": {
                "D1": "Low",
                "D2": "Low",
                "D3": "Low",
                "D4": "Low",
                "D5": "Low",
                "Overall Risk": "Low",
            },
            "pipeline": {
                "domain_judgments": {
                    "D1": "High",
                    "D2": "Low",
                    "D3": "Low",
                    "D4": "Low",
                    "D5": "Low",
                },
                "overall_judgment": "High",
            },
        },
        {"trial": "C", "skipped": True, "error": None, "comparison": {}},
    ]

    summary = summarize_benchmark(results)

    assert summary["evaluated_trials"] == 2
    assert summary["agreement_counts"]["D1"] == {"matches": 1, "total": 2}
    assert summary["agreement_rates"]["Overall"] == 0.5
    assert summary["confusion_matrices"]["D1"]["Low"]["Low"] == 1
    assert summary["confusion_matrices"]["D1"]["Low"]["High"] == 1
    assert summary["confusion_matrices"]["Overall"]["Low"]["High"] == 1


def test_summarize_benchmark_counts_optional_sq_agreement():
    results = [
        {
            "trial": "A",
            "skipped": False,
            "error": None,
            "comparison": {
                "D1": True,
                "D2": True,
                "D3": True,
                "D4": True,
                "D5": True,
                "Overall": True,
                "SQ": {"1.1": True, "1.2": False},
            },
            "reference": {
                "D1": "Low",
                "D2": "Low",
                "D3": "Low",
                "D4": "Low",
                "D5": "Low",
                "Overall Risk": "Low",
                "sq_answers": {"1.1": "Y", "1.2": "PY"},
            },
            "pipeline": {
                "domain_judgments": {
                    "D1": "Low",
                    "D2": "Low",
                    "D3": "Low",
                    "D4": "Low",
                    "D5": "Low",
                },
                "overall_judgment": "Low",
                "sq_answers": {"1.1": {"answer": "Y"}, "1.2": {"answer": "N"}},
            },
        },
        {
            "trial": "B",
            "skipped": False,
            "error": None,
            "comparison": {
                "D1": True,
                "D2": True,
                "D3": True,
                "D4": True,
                "D5": True,
                "Overall": True,
            },
            "reference": {
                "D1": "Low",
                "D2": "Low",
                "D3": "Low",
                "D4": "Low",
                "D5": "Low",
                "Overall Risk": "Low",
            },
            "pipeline": {
                "domain_judgments": {
                    "D1": "Low",
                    "D2": "Low",
                    "D3": "Low",
                    "D4": "Low",
                    "D5": "Low",
                },
                "overall_judgment": "Low",
            },
        },
    ]

    summary = summarize_benchmark(results)

    assert summary["sq_agreement_counts"] == {
        "1.1": {"matches": 1, "total": 1},
        "1.2": {"matches": 0, "total": 1},
    }
    assert summary["sq_agreement_rates"] == {"1.1": 1.0, "1.2": 0.0}


def test_write_benchmark_report_renders_optional_sq_agreement(tmp_path):
    results = [
        {
            "id": "A:OS",
            "trial": "A",
            "outcome": "Outcome A",
            "cohort": "unspecified",
            "skipped": False,
            "error": None,
            "notes": "",
            "comparison": {
                "D1": True,
                "D2": True,
                "D3": True,
                "D4": True,
                "D5": True,
                "Overall": True,
                "SQ": {"1.1": True, "1.2": False},
            },
            "reference": {
                "D1": "Low",
                "D2": "Low",
                "D3": "Low",
                "D4": "Low",
                "D5": "Low",
                "Overall Risk": "Low",
                "sq_answers": {"1.1": "Y", "1.2": "PY"},
            },
            "pipeline": {
                "domain_judgments": {
                    "D1": "Low",
                    "D2": "Low",
                    "D3": "Low",
                    "D4": "Low",
                    "D5": "Low",
                },
                "overall_judgment": "Low",
                "sq_answers": {"1.1": {"answer": "Y"}, "1.2": {"answer": "N"}},
            },
        }
    ]
    summary = summarize_benchmark(results)

    write_benchmark_report(results, summary, tmp_path / "benchmark_report.md")

    report = (tmp_path / "benchmark_report.md").read_text(encoding="utf-8")
    assert "## SQ Agreement" in report
    assert "| 1.1 | 100.0% (1/1) |" in report
    assert "| 1.2 | 0.0% (0/1) |" in report


def test_summarize_benchmark_counts_audit_caught_label_mismatches():
    results = [
        {
            "trial": "AuditLimited",
            "skipped": False,
            "error": None,
            "comparison": {
                "D1": False,
                "D2": True,
                "D3": True,
                "D4": True,
                "D5": True,
                "Overall": False,
            },
            "reference": {
                "D1": "Low",
                "D2": "Low",
                "D3": "Low",
                "D4": "Low",
                "D5": "Low",
                "Overall Risk": "Low",
            },
            "pipeline": {
                "domain_judgments": {
                    "D1": "Some concerns",
                    "D2": "Low",
                    "D3": "Low",
                    "D4": "Low",
                    "D5": "Low",
                },
                "overall_judgment": "Some concerns",
            },
            "audit_caught_mismatches": {"D1": True, "Overall": True},
        },
        {
            "trial": "Uncaught",
            "skipped": False,
            "error": None,
            "comparison": {
                "D1": False,
                "D2": True,
                "D3": True,
                "D4": True,
                "D5": True,
                "Overall": False,
            },
            "reference": {
                "D1": "Low",
                "D2": "Low",
                "D3": "Low",
                "D4": "Low",
                "D5": "Low",
                "Overall Risk": "Low",
            },
            "pipeline": {
                "domain_judgments": {
                    "D1": "High",
                    "D2": "Low",
                    "D3": "Low",
                    "D4": "Low",
                    "D5": "Low",
                },
                "overall_judgment": "High",
            },
            "audit_caught_mismatches": {"D1": False, "Overall": False},
        },
    ]

    summary = summarize_benchmark(results)

    assert summary["agreement_counts"]["D1"] == {"matches": 0, "total": 2}
    assert summary["audit_caught_mismatches"]["D1"] == {"caught": 1, "total": 2}
    assert summary["audit_caught_mismatches"]["Overall"] == {
        "caught": 1,
        "total": 2,
    }


def test_summarize_benchmark_groups_metrics_by_cohort():
    results = [
        {
            "trial": "A",
            "cohort": "calibration",
            "skipped": False,
            "error": None,
            "comparison": {
                "D1": True,
                "D2": True,
                "D3": True,
                "D4": True,
                "D5": True,
                "Overall": True,
            },
            "reference": {
                "D1": "Low",
                "D2": "Low",
                "D3": "Low",
                "D4": "Low",
                "D5": "Low",
                "Overall Risk": "Low",
            },
            "pipeline": {
                "domain_judgments": {
                    "D1": "Low",
                    "D2": "Low",
                    "D3": "Low",
                    "D4": "Low",
                    "D5": "Low",
                },
                "overall_judgment": "Low",
            },
        },
        {
            "trial": "B",
            "cohort": "validation",
            "skipped": False,
            "error": None,
            "comparison": {
                "D1": False,
                "D2": True,
                "D3": True,
                "D4": True,
                "D5": True,
                "Overall": False,
            },
            "reference": {
                "D1": "Low",
                "D2": "Low",
                "D3": "Low",
                "D4": "Low",
                "D5": "Low",
                "Overall Risk": "Low",
            },
            "pipeline": {
                "domain_judgments": {
                    "D1": "High",
                    "D2": "Low",
                    "D3": "Low",
                    "D4": "Low",
                    "D5": "Low",
                },
                "overall_judgment": "High",
            },
        },
    ]

    summary = summarize_benchmark(results)

    assert summary["cohorts"]["calibration"]["evaluated_trials"] == 1
    assert summary["cohorts"]["calibration"]["agreement_counts"]["Overall"] == {
        "matches": 1,
        "total": 1,
    }
    assert summary["cohorts"]["validation"]["evaluated_trials"] == 1
    assert summary["cohorts"]["validation"]["agreement_counts"]["Overall"] == {
        "matches": 0,
        "total": 1,
    }


def test_write_benchmark_report_hides_unspecified_cohort_when_no_labels(tmp_path):
    results = [
        {
            "id": "TRIAL1:OS",
            "trial": "TRIAL1",
            "outcome": "Outcome A",
            "cohort": "unspecified",
            "skipped": False,
            "error": None,
            "notes": "",
            "comparison": {
                "D1": True,
                "D2": True,
                "D3": True,
                "D4": True,
                "D5": True,
                "Overall": True,
            },
            "reference": {
                "D1": "Low",
                "D2": "Low",
                "D3": "Low",
                "D4": "Low",
                "D5": "Low",
                "Overall Risk": "Low",
            },
            "pipeline": {
                "domain_judgments": {
                    "D1": "Low",
                    "D2": "Low",
                    "D3": "Low",
                    "D4": "Low",
                    "D5": "Low",
                },
                "overall_judgment": "Low",
            },
        }
    ]
    summary = summarize_benchmark(results)

    write_benchmark_report(results, summary, tmp_path / "benchmark_report.md")

    report = (tmp_path / "benchmark_report.md").read_text(encoding="utf-8")
    assert "## Cohort Agreement" not in report
    assert "| Trial | Outcome | D1 | D2 | D3 | D4 | D5 | Overall | Notes |" in report
    assert "unspecified" not in report


def test_run_benchmark_attaches_timing_from_trace_file(tmp_path, monkeypatch):
    pdf_dir = tmp_path / "benchmark"
    pdf_dir.mkdir()
    (pdf_dir / "TITAN.pdf").write_bytes(b"pdf")

    reference_csv = tmp_path / "ref.csv"
    reference_csv.write_text(
        "Trial,D1,D2,D3,D4,D5,Overall Risk\nTITAN,Low,Low,Low,Low,Low,Low\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    def fake_run_assessment(**kwargs):
        assessment_dir = Path(kwargs["output_dir"])
        assessment_dir.mkdir(parents=True)
        assessment_dir.joinpath("TITAN_rob2_data.json").write_text(
            '{"domain_judgments": {}, "overall_judgment": ""}',
            encoding="utf-8",
        )
        assessment_dir.joinpath("TITAN_trace.json").write_text(
            json.dumps(
                {
                    "llm_calls": [
                        "malformed call entry",
                        {
                            "node": "domain3_sq",
                            "latency_ms": 120,
                            "cache_hit": True,
                            "input_tokens": 4,
                            "output_tokens": 2,
                            "is_repair": False,
                            "parse_error": None,
                        },
                        {
                            "node": "domain3_sq",
                            "latency_ms": 80,
                            "cache_hit": False,
                            "input_tokens": 3,
                            "output_tokens": 1,
                            "is_repair": True,
                            "parse_error": "bad parse",
                        },
                        {
                            "node": "pdf_ingest",
                            "latency_ms": 50,
                            "cache_hit": False,
                            "input_tokens": 1,
                            "output_tokens": 1,
                            "is_repair": False,
                            "parse_error": None,
                        },
                    ],
                    "node_spans": [
                        {
                            "node": "domain3_sq",
                            "status": "ok",
                            "timestamp_start": "2026-05-22T00:00:00Z",
                            "timestamp_end": "2026-05-22T00:00:00Z",
                            "duration_ms": 200,
                            "error": None,
                        },
                        {
                            "node": "pdf_ingest",
                            "status": "ok",
                            "timestamp_start": "2026-05-22T00:00:00Z",
                            "timestamp_end": "2026-05-22T00:00:00Z",
                            "duration_ms": 300,
                            "error": None,
                        },
                        {
                            "node": "pdf_ingest",
                            "status": "error",
                            "timestamp_start": "2026-05-22T00:00:00Z",
                            "timestamp_end": "2026-05-22T00:00:00Z",
                            "duration_ms": 25,
                            "error": "boom",
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )

    perf_counter_values = iter([100.0, 100.5])
    monkeypatch.setattr(
        "rob2_pipeline.benchmark.time.perf_counter",
        lambda: next(perf_counter_values),
    )
    monkeypatch.setattr("rob2_pipeline.benchmark.run_assessment", fake_run_assessment)

    results = run_benchmark(
        pdf_dir=pdf_dir,
        reference_csvs={"OS": reference_csv},
        outcome_map=[{"trial": "TITAN", "outcome_code": "OS"}],
        output_dir=output_dir,
    )

    timing = results[0]["timing"]
    assert timing["trace_available"] is True
    assert timing["total_wall_ms"] == 500
    assert timing["node_total_ms"] == 525
    assert timing["llm_total_ms"] == 250
    assert timing["non_llm_estimated_ms"] == 250
    assert timing["llm_calls"] == 3
    assert timing["llm_cache_hits"] == 1
    assert timing["llm_repairs"] == 1
    assert timing["llm_parse_errors"] == 1
    assert timing["slowest_nodes"][0] == {
        "node": "pdf_ingest",
        "duration_ms": 300,
        "status": "ok",
    }
    assert timing["llm_by_node"]["domain3_sq"] == {
        "calls": 2,
        "latency_ms": 200,
        "input_tokens": 7,
        "output_tokens": 3,
        "cache_hits": 1,
        "repairs": 1,
        "parse_errors": 1,
    }


def test_run_benchmark_uses_wall_time_when_trace_is_missing(tmp_path, monkeypatch):
    pdf_dir = tmp_path / "benchmark"
    pdf_dir.mkdir()
    (pdf_dir / "TITAN.pdf").write_bytes(b"pdf")

    reference_csv = tmp_path / "ref.csv"
    reference_csv.write_text(
        "Trial,D1,D2,D3,D4,D5,Overall Risk\nTITAN,Low,Low,Low,Low,Low,Low\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    def fake_run_assessment(**kwargs):
        assessment_dir = Path(kwargs["output_dir"])
        assessment_dir.mkdir(parents=True)
        assessment_dir.joinpath("TITAN_rob2_data.json").write_text(
            '{"domain_judgments": {}, "overall_judgment": ""}',
            encoding="utf-8",
        )

    perf_counter_values = iter([200.0, 200.125])
    monkeypatch.setattr(
        "rob2_pipeline.benchmark.time.perf_counter",
        lambda: next(perf_counter_values),
    )
    monkeypatch.setattr("rob2_pipeline.benchmark.run_assessment", fake_run_assessment)

    results = run_benchmark(
        pdf_dir=pdf_dir,
        reference_csvs={"OS": reference_csv},
        outcome_map=[{"trial": "TITAN", "outcome_code": "OS"}],
        output_dir=output_dir,
    )

    timing = results[0]["timing"]
    assert timing["trace_available"] is False
    assert timing["trace_error"] == "trace file not found"
    assert timing["total_wall_ms"] == 125
    assert timing["node_total_ms"] == 0
    assert timing["llm_total_ms"] == 0
    assert timing["non_llm_estimated_ms"] == 125
    assert timing["llm_calls"] == 0
    assert timing["llm_cache_hits"] == 0
    assert timing["llm_repairs"] == 0
    assert timing["llm_parse_errors"] == 0
    assert timing["slowest_nodes"] == []
    assert timing["llm_by_node"] == {}


def test_summarize_benchmark_includes_timing_aggregates():
    results = [
        {
            "trial": "A",
            "outcome": "Outcome A",
            "comparison": {},
            "reference": {},
            "pipeline": {},
            "timing": {
                "total_wall_ms": 1000,
                "trace_available": True,
                "node_total_ms": 900,
                "llm_total_ms": 400,
                "non_llm_estimated_ms": 600,
                "llm_calls": 2,
                "llm_cache_hits": 1,
                "llm_repairs": 0,
                "llm_parse_errors": 0,
                "slowest_nodes": [
                    {"node": "pdf_ingest", "duration_ms": 600, "status": "ok"}
                ],
                "node_spans": [
                    {"node": "pdf_ingest", "status": "ok", "duration_ms": 600},
                    {"node": "domain3_sq", "status": "ok", "duration_ms": 300},
                ],
                "llm_by_node": {
                    "pdf_ingest": {
                        "calls": 1,
                        "latency_ms": 400,
                        "input_tokens": 10,
                        "output_tokens": 2,
                        "cache_hits": 1,
                        "repairs": 0,
                        "parse_errors": 0,
                    }
                },
            },
        },
        {
            "trial": "B",
            "outcome": "Outcome B",
            "comparison": {},
            "reference": {},
            "pipeline": {},
            "timing": {
                "total_wall_ms": 2000,
                "trace_available": True,
                "node_total_ms": 1500,
                "llm_total_ms": 1200,
                "non_llm_estimated_ms": 800,
                "llm_calls": 4,
                "llm_cache_hits": 0,
                "llm_repairs": 1,
                "llm_parse_errors": 1,
                "slowest_nodes": [
                    {"node": "domain3_sq", "duration_ms": 900, "status": "error"}
                ],
                "node_spans": [
                    {"node": "pdf_ingest", "status": "ok", "duration_ms": 300},
                    {"node": "domain3_sq", "status": "error", "duration_ms": 900},
                    {"node": "quote_verify", "status": "ok", "duration_ms": 100},
                ],
                "llm_by_node": {
                    "domain3_sq": {
                        "calls": 2,
                        "latency_ms": 1000,
                        "input_tokens": 5,
                        "output_tokens": 1,
                        "cache_hits": 0,
                        "repairs": 1,
                        "parse_errors": 1,
                    },
                    "quote_verify": {
                        "calls": 1,
                        "latency_ms": 200,
                        "input_tokens": 2,
                        "output_tokens": 1,
                        "cache_hits": 0,
                        "repairs": 0,
                        "parse_errors": 0,
                    },
                },
            },
        },
        {"trial": "C", "skipped": True, "error": None, "comparison": {}},
    ]

    summary = summarize_benchmark(results)

    assert summary["timing"]["evaluated_runs"] == 2
    assert summary["timing"]["total_wall_ms"] == 3000
    assert summary["timing"]["mean_wall_ms"] == 1500
    assert summary["timing"]["median_wall_ms"] == 1500
    assert summary["timing"]["total_llm_latency_ms"] == 1600
    assert summary["timing"]["total_llm_calls"] == 6
    assert summary["timing"]["total_llm_cache_hits"] == 1
    assert summary["timing"]["node_aggregates"]["domain3_sq"] == {
        "calls": 2,
        "total_duration_ms": 1200,
        "mean_duration_ms": 600,
        "max_duration_ms": 900,
        "error_count": 1,
    }
    assert summary["timing"]["slowest_runs"][0]["trial"] == "B"


def test_write_benchmark_report_renders_timing_summary(tmp_path):
    results = [
        {
            "id": "A:OS",
            "trial": "A",
            "outcome": "Outcome A",
            "cohort": "unspecified",
            "skipped": False,
            "error": None,
            "notes": "",
            "comparison": {
                "D1": True,
                "D2": True,
                "D3": True,
                "D4": True,
                "D5": True,
                "Overall": True,
            },
            "reference": {
                "D1": "Low",
                "D2": "Low",
                "D3": "Low",
                "D4": "Low",
                "D5": "Low",
                "Overall Risk": "Low",
            },
            "pipeline": {
                "domain_judgments": {
                    "D1": "Low",
                    "D2": "Low",
                    "D3": "Low",
                    "D4": "Low",
                    "D5": "Low",
                },
                "overall_judgment": "Low",
            },
            "timing": {
                "total_wall_ms": 1000,
                "trace_available": True,
                "node_total_ms": 900,
                "llm_total_ms": 400,
                "non_llm_estimated_ms": 600,
                "llm_calls": 2,
                "llm_cache_hits": 1,
                "llm_repairs": 0,
                "llm_parse_errors": 0,
                "slowest_nodes": [
                    {"node": "pdf_ingest", "duration_ms": 600, "status": "ok"}
                ],
                "node_spans": [
                    {
                        "node": "pdf_ingest",
                        "status": "ok",
                        "timestamp_start": "2026-05-22T00:00:00Z",
                        "timestamp_end": "2026-05-22T00:00:00Z",
                        "duration_ms": 600,
                        "error": None,
                    },
                    {
                        "node": "domain3_sq",
                        "status": "ok",
                        "timestamp_start": "2026-05-22T00:00:00Z",
                        "timestamp_end": "2026-05-22T00:00:00Z",
                        "duration_ms": 300,
                        "error": None,
                    },
                ],
                "llm_by_node": {
                    "pdf_ingest": {
                        "calls": 1,
                        "latency_ms": 400,
                        "input_tokens": 10,
                        "output_tokens": 2,
                        "cache_hits": 1,
                        "repairs": 0,
                        "parse_errors": 0,
                    }
                },
            },
        }
    ]
    summary = summarize_benchmark(results)

    write_benchmark_report(results, summary, tmp_path / "benchmark_report.md")

    report = (tmp_path / "benchmark_report.md").read_text(encoding="utf-8")
    assert "## Timing Summary" in report
    assert "### Slowest Runs" in report
    assert "### Node Timing" in report
    assert (
        "| Trial | Outcome | Wall Time | LLM Time | Estimated Non-LLM | LLM Calls | Cache Hits | Slowest Node |"
        in report
    )
    assert "| Node | Calls | Total Time | Mean Time | Max Time | Errors |" in report
    assert "1.0s" in report
    assert "0.4s" in report
    benchmark_json = json.loads(
        (tmp_path / "benchmark_results.json").read_text(encoding="utf-8")
    )
    public_timing = benchmark_json["results"][0]["timing"]
    assert "_node_spans" not in public_timing
    assert "node_spans" not in public_timing
    assert "_node_spans" not in json.dumps(benchmark_json["summary"])
    assert "node_spans" not in json.dumps(benchmark_json["summary"])


def test_write_benchmark_report_emits_machine_readable_schema(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    primary_pdf = workspace / "TITAN.pdf"
    primary_pdf.write_bytes(b"primary pdf")
    assessment_dir = tmp_path / "out" / "TITAN_os"
    assessment_dir.mkdir(parents=True)
    assessment_json = assessment_dir / "TITAN_rob2_data.json"
    assessment_json.write_text('{"overall_judgment": "Low"}', encoding="utf-8")

    results = [
        {
            "id": "TITAN:OS",
            "trial": "TITAN",
            "outcome_code": "OS",
            "outcome": "Overall Survival",
            "cohort": "calibration",
            "pdf_path": str(primary_pdf),
            "assessment_output_dir": str(assessment_dir),
            "assessment_artifacts": {
                "rob2_data_json": str(assessment_json),
            },
            "skipped": False,
            "error": None,
            "notes": "",
            "comparison": {
                "D1": True,
                "D2": True,
                "D3": True,
                "D4": True,
                "D5": True,
                "Overall": True,
            },
            "reference": {
                "D1": "Low",
                "D2": "Low",
                "D3": "Low",
                "D4": "Low",
                "D5": "Low",
                "Overall Risk": "Low",
            },
            "pipeline": {
                "domain_judgments": {
                    "D1": "Low",
                    "D2": "Low",
                    "D3": "Low",
                    "D4": "Low",
                    "D5": "Low",
                },
                "overall_judgment": "Low",
            },
            "packet_quality": {
                "D1": {
                    "grade": "strong",
                    "packet_readiness": {"status": "ready"},
                },
            },
            "schema_failures": [
                {"domain": "D4", "error": "missing required answer"}
            ],
            "support_constraints": [
                {
                    "constraint_type": "quote_untraceable",
                    "sq_id": "4.1",
                    "domain": "D4",
                    "reason": "quoted text was not found",
                },
                {
                    "constraint_type": "semantic_support_conflict",
                    "sq_id": "5.1",
                    "domain": "D5",
                    "reason": "quote does not support claim",
                },
            ],
            "timing": {
                "total_wall_ms": 100,
                "llm_total_ms": 40,
                "llm_calls": 2,
                "llm_cache_hits": 1,
                "llm_repairs": 1,
                "llm_parse_errors": 0,
                "llm_input_tokens": 25,
                "llm_output_tokens": 9,
                "llm_cost_usd": 0.0017,
                "slowest_nodes": [],
                "_node_spans": [{"node": "private", "duration_ms": 1}],
            },
        }
    ]
    summary = summarize_benchmark(results)

    write_benchmark_report(results, summary, tmp_path / "out" / "benchmark_report.md")

    benchmark_json = json.loads(
        (tmp_path / "out" / "benchmark_results.json").read_text(encoding="utf-8")
    )
    schema = benchmark_json["schema"]
    assert schema["schema_name"] == "auto_rob2_benchmark_result"
    assert schema["schema_version"] == 1
    assert "aggregate" in schema["sections"]
    assert "assessments" in schema["sections"]
    assert schema["diagnostics"]["classification"] == "engineering_only"
    assert set(schema["diagnostics"]["fields"]) >= {
        "timing",
        "parser_metrics",
        "cache_reuse",
        "packet_statuses",
        "quote_traceability",
        "schema_validation_failures",
        "llm_latency",
        "llm_usage",
        "cost_metadata",
    }

    manifest = benchmark_json["artifact_manifest"]
    assert manifest["workspace"]["path"] == str((tmp_path / "out").resolve())
    assert len(manifest["workspace"]["sha256"]) == 64
    assert manifest["assessments"][0]["id"] == "TITAN:OS"
    assert (
        len(manifest["assessments"][0]["artifacts"]["rob2_data_json"]["sha256"]) == 64
    )

    assessment = benchmark_json["assessments"][0]
    assert assessment["agreement"]["comparison"]["Overall"] is True
    assert assessment["packet_quality"] == {
        "D1": {"grade": "strong", "packet_readiness": {"status": "ready"}}
    }
    assert assessment["schema_failures"] == [
        {"domain": "D4", "error": "missing required answer"}
    ]
    assert assessment["diagnostics"]["timing"]["llm_calls"] == 2
    assert assessment["diagnostics"]["parser_metrics"] == {
        "llm_repairs": 1,
        "llm_parse_errors": 0,
        "schema_validation_failures": 1,
    }
    assert assessment["diagnostics"]["packet_statuses"] == {
        "D1": {"status": "ready", "grade": "strong"}
    }
    assert assessment["diagnostics"]["quote_traceability"] == {
        "quote_raw_pdf_only": 0,
        "quote_untraceable": 1,
        "semantic_support_conflict": 1,
        "failures": [
            {
                "constraint_type": "quote_untraceable",
                "sq_id": "4.1",
                "domain": "D4",
                "reason": "quoted text was not found",
            },
            {
                "constraint_type": "semantic_support_conflict",
                "sq_id": "5.1",
                "domain": "D5",
                "reason": "quote does not support claim",
            },
        ],
    }
    assert assessment["diagnostics"]["schema_validation_failures"] == [
        {"domain": "D4", "error": "missing required answer"}
    ]
    assert assessment["diagnostics"]["llm_usage"] == {
        "input_tokens": 25,
        "output_tokens": 9,
    }
    assert assessment["diagnostics"]["cost_metadata"] == {
        "input_tokens": 25,
        "output_tokens": 9,
        "estimated_cost_usd": 0.0017,
    }
    aggregate_diagnostics = benchmark_json["aggregate"]["diagnostics"]
    assert aggregate_diagnostics["parser_metrics"] == {
        "llm_repairs": 1,
        "llm_parse_errors": 0,
        "schema_validation_failures": 1,
    }
    assert aggregate_diagnostics["cache_reuse"] == {"llm_cache_hits": 1}
    assert aggregate_diagnostics["packet_statuses"] == {
        "by_status": {"ready": 1},
        "by_grade": {"strong": 1},
    }
    assert aggregate_diagnostics["quote_traceability"] == {
        "quote_raw_pdf_only": 0,
        "quote_untraceable": 1,
        "semantic_support_conflict": 1,
        "failure_count": 2,
    }
    assert aggregate_diagnostics["llm_usage"] == {
        "input_tokens": 25,
        "output_tokens": 9,
    }
    assert aggregate_diagnostics["cost_metadata"] == {"estimated_cost_usd": 0.0017}
    assert "_node_spans" not in json.dumps(assessment["diagnostics"])
    assert "timing" not in assessment["agreement"]
    assert "Performance Warnings" not in (
        tmp_path / "out" / "benchmark_report.md"
    ).read_text(encoding="utf-8")


def test_write_benchmark_report_renders_adjudication_summary(tmp_path):
    results = [
        {
            "id": "A:OS",
            "trial": "A",
            "outcome": "Outcome A",
            "cohort": "unspecified",
            "skipped": False,
            "error": None,
            "notes": "",
            "comparison": {
                "D1": True,
                "D2": True,
                "D3": True,
                "D4": True,
                "D5": True,
                "Overall": True,
            },
            "reference": {
                "D1": "Low",
                "D2": "Low",
                "D3": "Low",
                "D4": "Low",
                "D5": "Low",
                "Overall Risk": "Low",
            },
            "pipeline": {
                "domain_judgments": {
                    "D1": "Low",
                    "D2": "Low",
                    "D3": "Low",
                    "D4": "Low",
                    "D5": "Low",
                },
                "overall_judgment": "Low",
                "initial_domain_judgments": {
                    "D1": "Some concerns",
                    "D2": "Low",
                    "D3": "Low",
                    "D4": "Low",
                    "D5": "Low",
                },
                "initial_overall_judgment": "Some concerns",
            },
            "adjudication_metrics": {
                "weak_sq_answers": 1,
                "unsupported_sq_answers": 1,
                "pivotality_tests": {"total": 2, "pivotal": 1, "non_pivotal": 1},
                "sq_support_adjudications": {
                    "total": 1,
                    "changed_answer": 1,
                    "changed_support": 1,
                    "changed_answer_or_support": 1,
                },
                "initial_final_deltas": {
                    "domain_judgments": {
                        "D1": {"initial": "Some concerns", "final": "Low"}
                    },
                    "overall_judgment": {
                        "initial": "Some concerns",
                        "final": "Low",
                    },
                },
            },
            "timing": {
                "total_wall_ms": 100,
                "llm_total_ms": 40,
                "adjudication_llm_calls": 1,
                "adjudication_llm_total_ms": 40,
                "adjudication_llm_input_tokens": 7,
                "adjudication_llm_output_tokens": 2,
                "slowest_nodes": [],
                "node_spans": [],
            },
        }
    ]
    summary = summarize_benchmark(results)

    write_benchmark_report(results, summary, tmp_path / "benchmark_report.md")

    report = (tmp_path / "benchmark_report.md").read_text(encoding="utf-8")
    assert "## Adjudication Summary" in report
    assert "- Weak SQ answers: 1" in report
    assert "- Unsupported SQ answers: 1" in report
    assert "- Pivotality tests: 2 total; 1 pivotal; 1 non-pivotal" in report
    assert (
        "- SQ support adjudications: 1 total; 1 changed answer; 1 changed support"
        in report
    )
    assert (
        "- Adjudication LLM calls: 1 (0.0s latency; 7 input tokens; 2 output tokens)"
        in report
    )
    assert "| Field | Initial | Final | Count |" in report
    assert "| D1 | Some concerns | Low | 1 |" in report
    assert "| Overall | Some concerns | Low | 1 |" in report


def test_write_benchmark_report_renders_separate_engineering_report(tmp_path):
    results = [
        {
            "id": "TITAN:OS",
            "trial": "TITAN",
            "outcome": "Overall Survival",
            "outcome_code": "OS",
            "cohort": "calibration",
            "skipped": False,
            "error": None,
            "notes": "",
            "assessment_artifacts": {
                "rob2_data_json": str(tmp_path / "TITAN_rob2_data.json"),
            },
            "comparison": {
                "D1": True,
                "D2": False,
                "D3": True,
                "D4": True,
                "D5": True,
                "Overall": False,
            },
            "reference": {
                "D1": "Low",
                "D2": "Low",
                "D3": "Low",
                "D4": "Low",
                "D5": "Low",
                "Overall Risk": "Low",
            },
            "pipeline": {
                "domain_judgments": {
                    "D1": "Low",
                    "D2": "High",
                    "D3": "Low",
                    "D4": "Low",
                    "D5": "Low",
                },
                "overall_judgment": "High",
            },
            "packet_quality": {
                "D2": {
                    "packet_grade": "insufficient",
                    "packet_readiness": {"status": "needs_retrieval_repair"},
                }
            },
            "schema_failures": [{"domain": "D2", "error": "missing sq answer"}],
            "support_constraints": [
                {
                    "constraint_type": "quote_untraceable",
                    "sq_id": "2.6",
                    "domain": "D2",
                    "reason": "quote was not found",
                }
            ],
            "mismatch_classification": {
                "D2": {"category": "packet", "signals": ["packet_grade:insufficient"]},
                "Overall": {
                    "category": "reference_ambiguity",
                    "signals": ["audit_caught_mismatch"],
                },
            },
            "timing": {
                "total_wall_ms": 1200,
                "llm_total_ms": 400,
                "llm_calls": 3,
                "llm_cache_hits": 1,
                "llm_repairs": 1,
                "llm_parse_errors": 0,
                "llm_input_tokens": 100,
                "llm_output_tokens": 25,
                "llm_cost_usd": 0.02,
                "slowest_nodes": [
                    {"node": "domain2_analysis", "duration_ms": 300, "status": "ok"}
                ],
                "_node_spans": [
                    {"node": "domain2_analysis", "duration_ms": 300, "status": "ok"}
                ],
            },
        }
    ]
    summary = summarize_benchmark(results)

    write_benchmark_report(results, summary, tmp_path / "benchmark_report.md")

    engineering_report = (tmp_path / "engineering_report.md").read_text(
        encoding="utf-8"
    )
    assert engineering_report.startswith("# Engineering Benchmark Report")
    assert "## Agreement" in engineering_report
    assert "| D2 | 0.0% (0/1) |" in engineering_report
    assert "## Mismatch Diagnostics" in engineering_report
    assert "| D2 | packet | packet_grade:insufficient |" in engineering_report
    assert "## Artifact Status" in engineering_report
    assert "| TITAN:OS | rob2_data_json |" in engineering_report
    assert "## Packet Quality" in engineering_report
    assert "| TITAN:OS | D2 | needs_retrieval_repair | insufficient |" in engineering_report
    assert "## Timing, Cache, Model, And Cost Diagnostics" in engineering_report
    assert "- Total LLM calls: 3" in engineering_report
    assert "- Total cache hits: 1" in engineering_report
    assert "- Estimated LLM cost: $0.0200" in engineering_report
    assert "quote_untraceable: 1" in engineering_report
    assert "_node_spans" not in engineering_report


def test_engineering_report_renders_numeric_packet_quality(tmp_path):
    results = [
        {
            "id": "ARCHES:PFS",
            "trial": "ARCHES",
            "outcome": "Progression-Free Survival",
            "skipped": False,
            "error": None,
            "comparison": {
                "D1": True,
                "D2": True,
                "D3": True,
                "D4": True,
                "D5": True,
                "Overall": True,
            },
            "reference": {},
            "pipeline": {},
            "packet_quality": {
                "2.3": {
                    "relevance": 0.65,
                    "coverage": 1.0,
                    "missing_evidence": [],
                    "retry_recommended": False,
                },
                "5.1": {
                    "relevance": 1.0,
                    "coverage": 1.0,
                    "missing_evidence": ["protocol_or_registration"],
                    "retry_recommended": True,
                },
            },
            "timing": {"slowest_nodes": []},
        }
    ]
    summary = summarize_benchmark(results)

    write_benchmark_report(results, summary, tmp_path / "benchmark_report.md")

    benchmark_json = json.loads(
        (tmp_path / "benchmark_results.json").read_text(encoding="utf-8")
    )
    assert benchmark_json["assessments"][0]["diagnostics"]["packet_statuses"] == {
        "2.3": {"status": "ready", "grade": "moderate"},
        "5.1": {"status": "needs_retrieval_repair", "grade": "insufficient"},
    }
    engineering_report = (tmp_path / "engineering_report.md").read_text(
        encoding="utf-8"
    )
    assert "| ARCHES:PFS | 2.3 | ready | moderate |" in engineering_report
    assert (
        "| ARCHES:PFS | 5.1 | needs_retrieval_repair | insufficient |"
        in engineering_report
    )


def test_write_benchmark_report_renders_audit_caught_mismatch_summary(tmp_path):
    results = [
        {
            "id": "A:OS",
            "trial": "A",
            "outcome": "Outcome A",
            "cohort": "unspecified",
            "skipped": False,
            "error": None,
            "notes": "",
            "comparison": {
                "D1": False,
                "D2": True,
                "D3": True,
                "D4": True,
                "D5": True,
                "Overall": False,
            },
            "reference": {
                "D1": "Low",
                "D2": "Low",
                "D3": "Low",
                "D4": "Low",
                "D5": "Low",
                "Overall Risk": "Low",
            },
            "pipeline": {
                "domain_judgments": {
                    "D1": "High",
                    "D2": "Low",
                    "D3": "Low",
                    "D4": "Low",
                    "D5": "Low",
                },
                "overall_judgment": "High",
            },
            "audit_caught_mismatches": {"D1": True, "Overall": True},
        }
    ]
    summary = summarize_benchmark(results)

    write_benchmark_report(results, summary, tmp_path / "benchmark_report.md")

    report = (tmp_path / "benchmark_report.md").read_text(encoding="utf-8")
    assert "## Audit-Caught Mismatches" in report
    assert "| D1 | 100.0% (1/1) |" in report
    assert "| Overall | 100.0% (1/1) |" in report


def test_benchmark_report_emits_deterministic_mismatch_classification(tmp_path):
    results = [
        {
            "id": "A:OS",
            "trial": "A",
            "outcome": "Outcome A",
            "cohort": "unspecified",
            "skipped": False,
            "error": None,
            "notes": "",
            "comparison": {
                "D1": False,
                "D2": False,
                "D3": False,
                "D4": False,
                "D5": False,
                "Overall": False,
            },
            "reference": {
                "D1": "Low",
                "D2": "Low",
                "D3": "Low",
                "D4": "Low",
                "D5": "Low",
                "Overall Risk": "Low",
            },
            "pipeline": {
                "domain_judgments": {
                    "D1": "High",
                    "D2": "High",
                    "D3": "High",
                    "D4": "High",
                    "D5": "High",
                },
                "overall_judgment": "High",
                "sq_answers": {
                    "1.1": {"answer": "NI", "support_level": "weak", "quote": ""},
                    "2.1": {"answer": "Y", "support_level": "strong", "quote": "x"},
                    "3.1": {"answer": "Y", "support_level": "strong", "quote": "x"},
                    "4.1": {"answer": "Y", "support_level": "strong", "quote": "x"},
                    "5.1": {"answer": "Y", "support_level": "strong", "quote": "x"},
                },
            },
            "packet_quality": {
                "D2": {
                    "packet_grade": "insufficient",
                    "missing_evidence": ["allocation"],
                },
            },
            "schema_failures": [{"domain": "D1", "error": "schema"}],
            "audit_caught_mismatches": {"Overall": True},
            "mismatch_classification": {
                "D1": {"category": "parse", "signals": ["schema_failure"]},
                "D2": {"category": "packet", "signals": ["packet_grade:insufficient"]},
                "D3": {"category": "retrieval", "signals": ["quote_missing"]},
                "D4": {"category": "SQ", "signals": ["support_level:weak"]},
                "D5": {"category": "judge", "signals": ["judge_signal"]},
                "Overall": {
                    "category": "reference_ambiguity",
                    "signals": ["audit_caught_mismatch"],
                },
            },
        },
        {
            "id": "B:OS",
            "trial": "B",
            "outcome": "Outcome B",
            "cohort": "unspecified",
            "skipped": False,
            "error": "Required supplements not found",
            "notes": "Required supplements not found",
            "comparison": {},
            "mismatch_classification": {
                "Overall": {
                    "category": "blocked_incomplete",
                    "signals": ["assessment_error"],
                }
            },
        },
    ]
    summary = summarize_benchmark(results)

    write_benchmark_report(results, summary, tmp_path / "benchmark_report.md")

    benchmark_json = json.loads(
        (tmp_path / "benchmark_results.json").read_text(encoding="utf-8")
    )
    categories = benchmark_json["aggregate"]["mismatch_classification"]["categories"]
    assert set(categories) >= {
        "parse",
        "retrieval",
        "packet",
        "SQ",
        "judge",
        "reference_ambiguity",
        "blocked_incomplete",
    }
    assert benchmark_json["assessments"][0]["diagnostics"]["mismatch_classification"][
        "D1"
    ] == {"category": "parse", "signals": ["schema_failure"]}

    report = (tmp_path / "benchmark_report.md").read_text(encoding="utf-8")
    assert "## Mismatch Classification" in report
    assert "| parse | 1 |" in report


def test_classify_mismatches_uses_existing_audit_signals_without_diagnosis_agent():
    result = {
        "comparison": {
            "D1": False,
            "D2": False,
            "D3": False,
            "D4": False,
            "D5": False,
            "Overall": False,
        },
        "schema_failures": [{"domain": "D1", "error": "invalid xml"}],
        "packet_quality": {
            "D2": {"packet_grade": "insufficient"},
            "D3": {"retrieval_confidence": "low"},
        },
        "pipeline": {
            "sq_answers": {
                "4.1": {"answer": "PY", "support_level": "weak", "quote": "x"},
                "5.1": {"answer": "Y", "support_level": "strong", "quote": "x"},
            }
        },
        "audit_caught_mismatches": {"Overall": True},
    }

    classifications = classify_mismatches(result)

    assert classifications == {
        "D1": {"category": "parse", "signals": ["schema_failure"]},
        "D2": {"category": "packet", "signals": ["packet_grade:insufficient"]},
        "D3": {"category": "retrieval", "signals": ["retrieval_confidence:low"]},
        "D4": {"category": "SQ", "signals": ["support_level:weak"]},
        "D5": {"category": "judge", "signals": ["judgment_label_mismatch"]},
        "Overall": {
            "category": "judge",
            "signals": ["audit_caught_mismatch", "judgment_label_mismatch"],
        },
    }


def test_classify_mismatches_reserves_reference_ambiguity_for_inconsistent_reference():
    result = {
        "comparison": {"D3": False},
        "reference": {
            "D1": "Low",
            "D2": "Low",
            "D3": "Some concerns",
            "D4": "Low",
            "D5": "Low",
            "Overall Risk": "Low",
        },
        "pipeline": {
            "sq_answers": {
                "3.1": {"answer": "Y", "support_level": "strong", "quote": "x"}
            }
        },
        "audit_caught_mismatches": {"D3": True},
    }

    classifications = classify_mismatches(result)

    assert classifications["D3"] == {
        "category": "reference_ambiguity",
        "signals": [
            "audit_caught_mismatch",
            "reference_overall_lower_than_domain",
        ],
    }


def test_find_supplements_for_trial_handles_spaces_and_case(tmp_path):
    supplement_root = tmp_path / "supplement"
    trial_dir = supplement_root / "SWOG 1216"
    trial_dir.mkdir(parents=True)
    protocol = trial_dir / "protocol_jco.21.02517.pdf"
    dss = trial_dir / "dss_jco.21.02517.pdf"
    protocol.write_bytes(b"pdf")
    dss.write_bytes(b"pdf")

    result = find_supplements_for_trial(supplement_root, "swog 1216")

    assert result == [dss, protocol]


def test_run_benchmark_passes_discovered_supplements(tmp_path, monkeypatch):
    pdf_dir = tmp_path / "benchmark"
    pdf_dir.mkdir()
    (pdf_dir / "TITAN.pdf").write_bytes(b"pdf")
    supplement_root = pdf_dir / "supplement"
    trial_dir = supplement_root / "TITAN"
    trial_dir.mkdir(parents=True)
    protocol = trial_dir / "protocol.pdf"
    protocol.write_bytes(b"pdf")

    reference_csv = tmp_path / "ref.csv"
    reference_csv.write_text(
        "Trial,D1,D2,D3,D4,D5,Overall Risk\nTITAN,Low,Low,Low,Low,Low,Low\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    calls = []

    def fake_run_assessment(**kwargs):
        calls.append(kwargs)
        assessment_dir = Path(kwargs["output_dir"])
        assessment_dir.mkdir(parents=True)
        (assessment_dir / "TITAN_rob2_data.json").write_text(
            '{"domain_judgments": {}, "overall_judgment": ""}',
            encoding="utf-8",
        )

    monkeypatch.setattr("rob2_pipeline.benchmark.run_assessment", fake_run_assessment)

    results = run_benchmark(
        pdf_dir=pdf_dir,
        reference_csvs={"OS": reference_csv},
        outcome_map=[{"trial": "TITAN", "outcome_code": "OS", "cohort": "unspecified"}],
        output_dir=output_dir,
        supplement_dir=supplement_root,
        use_supplements=True,
    )

    assert calls[0]["supplementary_paths"] == [str(protocol)]
    assert results[0]["supplements_found"] == 1


def test_run_benchmark_required_supplements_errors_when_missing(tmp_path):
    pdf_dir = tmp_path / "benchmark"
    pdf_dir.mkdir()
    (pdf_dir / "TITAN.pdf").write_bytes(b"pdf")
    reference_csv = tmp_path / "ref.csv"
    reference_csv.write_text(
        "Trial,D1,D2,D3,D4,D5,Overall Risk\nTITAN,Low,Low,Low,Low,Low,Low\n",
        encoding="utf-8",
    )

    results = run_benchmark(
        pdf_dir=pdf_dir,
        reference_csvs={"OS": reference_csv},
        outcome_map=[{"trial": "TITAN", "outcome_code": "OS"}],
        output_dir=tmp_path / "out",
        supplement_dir=tmp_path / "missing",
        use_supplements=True,
        supplement_policy="required",
    )

    assert results[0]["skipped"] is False
    assert "Required supplements" in results[0]["error"]
    assert "Required supplements" in results[0]["notes"]
    assert results[0]["timing"]["trace_available"] is False
    assert results[0]["timing"]["trace_error"] == "assessment not run"


def test_run_benchmark_required_supplements_errors_on_parse_failure(
    tmp_path, monkeypatch
):
    pdf_dir = tmp_path / "benchmark"
    pdf_dir.mkdir()
    (pdf_dir / "TITAN.pdf").write_bytes(b"pdf")
    supplement_root = pdf_dir / "supplement"
    trial_dir = supplement_root / "TITAN"
    trial_dir.mkdir(parents=True)
    protocol = trial_dir / "protocol.pdf"
    protocol.write_bytes(b"pdf")

    reference_csv = tmp_path / "ref.csv"
    reference_csv.write_text(
        "Trial,D1,D2,D3,D4,D5,Overall Risk\nTITAN,Low,Low,Low,Low,Low,Low\n",
        encoding="utf-8",
    )

    def fake_run_assessment(**kwargs):
        assessment_dir = Path(kwargs["output_dir"])
        assessment_dir.mkdir(parents=True)
        (assessment_dir / "TITAN_rob2_data.json").write_text(
            """
            {
              "domain_judgments": {},
              "overall_judgment": "",
              "source_documents": [
                {"document_id": "primary", "document_name": "TITAN.pdf", "is_primary": true, "status": "parsed"},
                {"document_id": "supplement:001", "document_name": "protocol.pdf", "is_primary": false, "status": "failed"}
              ]
            }
            """,
            encoding="utf-8",
        )

    monkeypatch.setattr("rob2_pipeline.benchmark.run_assessment", fake_run_assessment)

    results = run_benchmark(
        pdf_dir=pdf_dir,
        reference_csvs={"OS": reference_csv},
        outcome_map=[{"trial": "TITAN", "outcome_code": "OS"}],
        output_dir=tmp_path / "out",
        supplement_dir=supplement_root,
        use_supplements=True,
        supplement_policy="required",
    )

    assert "Required supplement ingestion failed" in results[0]["error"]
    assert results[0]["comparison"] == {}


def test_required_supplement_failures_detects_requested_but_not_ingested():
    failures = _required_supplement_failures(
        [Path("inputs/benchmark/supplement/TITAN/protocol.pdf")],
        [
            {
                "document_id": "primary",
                "path": "inputs/benchmark/TITAN.pdf",
                "is_primary": True,
                "status": "parsed",
            }
        ],
    )

    assert failures == ["protocol.pdf (not ingested)"]


def test_required_supplement_failures_accepts_all_requested_parsed():
    failures = _required_supplement_failures(
        [Path("inputs/benchmark/supplement/TITAN/protocol.pdf")],
        [
            {
                "document_id": "supplement:001",
                "path": "inputs/benchmark/supplement/TITAN/protocol.pdf",
                "is_primary": False,
                "status": "parsed",
            }
        ],
    )

    assert failures == []


def test_required_supplement_failures_accepts_partial_with_window_warnings():
    failures = _required_supplement_failures(
        [Path("inputs/benchmark/supplement/TITAN/protocol.pdf")],
        [
            {
                "document_id": "supplement:001",
                "path": "inputs/benchmark/supplement/TITAN/protocol.pdf",
                "is_primary": False,
                "status": "partial",
                "error": "Supplement parser diagnostics recorded a partial parse",
            }
        ],
    )

    assert failures == []
