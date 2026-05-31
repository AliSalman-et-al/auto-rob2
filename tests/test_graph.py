from pathlib import Path
import json
from unittest.mock import Mock, patch

import pytest

from rob2_pipeline.graph import build_rob2_graph, timed_node
from rob2_pipeline.trace import get_current_trace, start_trace, end_trace
from rob2_pipeline.models import empty_paper_evidence
from rob2_pipeline.pdf_ingestion import DocumentRepr
from rob2_pipeline.pipeline import run_assessment
from rob2_pipeline.providers.base import LLMResponse


def _make_pdf(path: Path):
    path.write_bytes(b"%PDF-1.7\n% test fixture\n")


def _pdf_text() -> str:
    return "\n".join(
        [
            "Abstract",
            "This randomized controlled trial compared Drug A with placebo.",
            "Methods",
            "Participants were randomly assigned using a computer-generated sequence.",
            "Allocation was concealed centrally. The trial used intention-to-treat analysis.",
            "Blinding",
            "Participants and investigators were blinded.",
            "Outcomes",
            "The primary outcome was mortality.",
            "Results",
            "100 participants were randomized and all had outcome data.",
            "Trial registration",
            "ClinicalTrials.gov NCT00000000.",
        ]
    )


def _initial_state(pdf_path: str) -> dict:
    return {
        "pdf_path": pdf_path,
        "full_text": "",
        "evidence": empty_paper_evidence(),
        "is_rct": False,
        "rct_screen_evidence": "",
        "intervention": "Not reported",
        "comparator": "Not reported",
        "outcome": "",
        "outcome_type": "vital-status",
        "numerical_result": "Not reported",
        "effect_of_interest": "ITT",
        "registration_number": "Not reported",
        "registered_endpoint": "Not reported",
        "registered_analysis": "Not reported",
        "n_randomized": "Not reported",
        "sources_consulted": [],
        "sq_answers": {},
        "domain_judgments": {},
        "domain_rationales": {},
        "overall_judgment": "",
        "overall_rationale": "",
        "ni_count": 0,
        "high_uncertainty_sqs": [],
        "human_review_priority": "HIGH",
        "markdown_report": "",
        "errors": [],
        "llm_call_log": [],
    }


def _response_by_node(node_name: str):
    responses = {
        "paper_evidence_extraction": """
        <evidence>
          <abstract><text>This randomized controlled trial compared Drug A with placebo.</text><tables></tables></abstract>
          <methods><text>Participants were randomly assigned using a computer-generated sequence. Allocation was concealed centrally. The trial used intention-to-treat analysis.</text><tables></tables></methods>
          <results><text>100 participants were randomized and all had outcome data.</text><tables></tables></results>
          <d1_randomization><text>Participants were randomly assigned using a computer-generated sequence. Allocation was concealed centrally.</text><tables></tables></d1_randomization>
          <d2_blinding><text>Participants and investigators were blinded.</text><tables></tables></d2_blinding>
          <d3_missing_data><text>100 participants were randomized and all had outcome data.</text><tables></tables></d3_missing_data>
          <d4_outcome_meas><text>The primary outcome was mortality. The trial used intention-to-treat analysis.</text><tables></tables></d4_outcome_meas>
          <d5_registration><text>ClinicalTrials.gov NCT00000000.</text><tables></tables></d5_registration>
          <consort_flow><text>100 participants were randomized.</text><tables></tables></consort_flow>
          <baseline_table><text>baseline balanced</text><tables></tables></baseline_table>
        </evidence>
        """,
        "rct_screener": """
        <screening><is_rct>YES</is_rct><evidence>"randomly assigned"</evidence><study_design>RCT</study_design><note></note></screening>
        """,
        "preliminary_info": """
        <preliminary_info>
          <experimental_intervention><value>Drug A</value><quote>"Drug A" (Abstract)</quote></experimental_intervention>
          <comparator_intervention><value>Placebo</value><quote>"placebo" (Abstract)</quote></comparator_intervention>
          <outcome_assessed><value>mortality</value><quote>"mortality" (Outcomes)</quote><is_primary>YES</is_primary></outcome_assessed>
          <outcome_type>vital-status</outcome_type>
          <numerical_result><value>RR 0.90 (95% CI 0.70-1.10)</value><quote>"RR 0.90" (Results)</quote></numerical_result>
          <n_randomized><value>100</value><quote>"100 participants" (Results)</quote></n_randomized>
          <trial_registration><number>NCT00000000</number><registry>ClinicalTrials.gov</registry><quote>"NCT00000000" (Registration)</quote></trial_registration>
          <registered_primary_endpoint><value>mortality</value><quote>"mortality" (Registration)</quote></registered_primary_endpoint>
          <registered_analysis><value>ITT</value><quote>"intention-to-treat" (Methods)</quote></registered_analysis>
        </preliminary_info>
        """,
        "domain1_sq": """
        <domain1>
          <sq_1_1><answer>Y</answer><quote>"computer-generated sequence" (Methods)</quote><justification>Random sequence stated.</justification></sq_1_1>
          <sq_1_2><answer>Y</answer><quote>"concealed centrally" (Methods)</quote><justification>Central concealment stated.</justification></sq_1_2>
          <sq_1_3><answer>N</answer><quote>"baseline balanced" (Results)</quote><justification>No concerning imbalance.</justification></sq_1_3>
        </domain1>
        """,
        "domain2_sq12": """
        <domain2_part1>
          <sq_2_1><answer>N</answer><quote>"blinded" (Blinding)</quote><justification>Participants were blinded.</justification></sq_2_1>
          <sq_2_2><answer>N</answer><quote>"investigators were blinded" (Blinding)</quote><justification>Personnel were blinded.</justification></sq_2_2>
        </domain2_part1>
        """,
        "domain2_analysis": """
        <domain2_analysis>
          <sq_2_6><answer>Y</answer><quote>"intention-to-treat analysis" (Methods)</quote><justification>ITT analysis was used.</justification></sq_2_6>
          <sq_2_7><answer>NA</answer><quote>Not applicable</quote><justification>Not applicable</justification></sq_2_7>
        </domain2_analysis>
        """,
        "domain3_sq": """
        <domain3>
          <sq_3_1><answer>Y</answer><quote>"all had outcome data" (Results)</quote><completeness_calculation>100/100 = 100%</completeness_calculation><justification>Outcome data were complete.</justification></sq_3_1>
          <sq_3_2><answer>NA</answer><quote>Not applicable</quote><justification>Not applicable</justification></sq_3_2>
          <sq_3_3><answer>NA</answer><quote>Not applicable</quote><justification>Not applicable</justification><uncertainty_flag>NORMAL</uncertainty_flag></sq_3_3>
          <sq_3_4><answer>NA</answer><quote>Not applicable</quote><justification>Not applicable</justification><uncertainty_flag>NORMAL</uncertainty_flag></sq_3_4>
        </domain3>
        """,
        "domain4_sq": """
        <domain4>
          <sq_4_1><answer>N</answer><quote>"mortality" (Outcomes)</quote><justification>Mortality is objective.</justification></sq_4_1>
          <sq_4_2><answer>N</answer><quote>"primary outcome was mortality" (Outcomes)</quote><justification>Same method used.</justification></sq_4_2>
          <sq_4_3><answer>N</answer><auto_set_reason></auto_set_reason><quote>"blinded" (Blinding)</quote><justification>Assessors were blinded.</justification></sq_4_3>
          <sq_4_4><answer>NA</answer><quote>Not applicable</quote><justification>Not applicable</justification></sq_4_4>
          <sq_4_5><answer>NA</answer><quote>Not applicable</quote><justification>Not applicable</justification><uncertainty_flag>NORMAL</uncertainty_flag></sq_4_5>
        </domain4>
        """,
        "domain5_sq": """
        <domain5>
          <sq_5_1><answer>Y</answer><quote>"NCT00000000" (Registration)</quote><justification>Trial was registered.</justification><registration_comparison>No discrepancy.</registration_comparison></sq_5_1>
          <sq_5_2><answer>N</answer><quote>"primary outcome was mortality" (Outcomes)</quote><justification>No selective measurement evident.</justification></sq_5_2>
          <sq_5_3><answer>N</answer><quote>"intention-to-treat analysis" (Methods)</quote><justification>No selective analysis evident.</justification></sq_5_3>
        </domain5>
        """,
    }
    return responses[node_name]


def _node_from_prompt(prompt: str) -> str:
    if "<evidence>" in prompt and "d1_randomization" in prompt:
        return "paper_evidence_extraction"
    if "<screening>" in prompt:
        return "rct_screener"
    if "<preliminary_info>" in prompt:
        return "preliminary_info"
    if "<domain1>" in prompt:
        return "domain1_sq"
    if "<domain2_part1>" in prompt:
        return "domain2_sq12"
    if "<domain2_conditional>" in prompt:
        return "domain2_conditional"
    if "<domain2_analysis>" in prompt:
        return "domain2_analysis"
    if "<domain3>" in prompt:
        return "domain3_sq"
    if "<domain4>" in prompt:
        return "domain4_sq"
    if "<domain5>" in prompt:
        return "domain5_sq"
    raise KeyError("Unknown prompt")


class _FakeProvider:
    def __init__(self):
        self.complete = Mock(side_effect=self._complete)

    def _complete(self, system: str, user: str) -> LLMResponse:
        node_name = _node_from_prompt(user)
        return LLMResponse(_response_by_node(node_name), "test-model", 1, 1, 1.0)


def _pfs_response_by_node(node_name: str):
    responses = dict(
        paper_evidence_extraction="""
        <evidence>
          <abstract><text>Open-label randomized trial of docetaxel plus ADT versus ADT alone.</text><tables></tables></abstract>
          <methods><text>Patients were randomly assigned. Overall survival and progression outcomes were assessed.</text><tables></tables></methods>
          <results><text>Median time to biochemical, symptomatic, or radiographic progression was improved.</text><tables></tables></results>
          <d1_randomization><text>Patients were randomly assigned by a central statistical center.</text><tables></tables></d1_randomization>
          <d2_blinding><text>Open-label treatment assignment.</text><tables></tables></d2_blinding>
          <d3_missing_data><text>All randomly assigned patients were followed.</text><tables></tables></d3_missing_data>
          <d4_outcome_meas><text>Overall survival was time to death. Progression-free survival was biochemical, symptomatic, or radiographic progression.</text><tables></tables></d4_outcome_meas>
          <d5_registration><text>ClinicalTrials.gov NCT00309985.</text><tables></tables></d5_registration>
          <consort_flow><text>790 patients were randomized.</text><tables></tables></consort_flow>
          <baseline_table><text>Baseline characteristics were balanced.</text><tables></tables></baseline_table>
        </evidence>
        """,
        rct_screener="""
        <screening><is_rct>YES</is_rct><evidence>randomly assigned</evidence><study_design>RCT</study_design><note></note></screening>
        """,
        preliminary_info="""
        <preliminary_info>
          <experimental_intervention><value>Docetaxel + ADT</value></experimental_intervention>
          <comparator_intervention><value>ADT alone</value></comparator_intervention>
          <outcome_assessed><value>Overall Survival</value></outcome_assessed>
          <outcome_type>clinician-composite</outcome_type>
          <numerical_result><value>HR 0.61</value></numerical_result>
          <n_randomized><value>790</value></n_randomized>
          <trial_registration><number>NCT00309985</number></trial_registration>
          <registered_primary_endpoint><value>Overall Survival</value></registered_primary_endpoint>
          <registered_secondary_endpoints>Not reported</registered_secondary_endpoints>
          <registered_analysis><value>ITT</value></registered_analysis>
        </preliminary_info>
        """,
        domain1_sq="""
        <domain1>
          <sq_1_1><answer>PY</answer><quote>randomly assigned</quote><justification>Randomized.</justification></sq_1_1>
          <sq_1_2><answer>PY</answer><quote>central statistical center</quote><justification>Central randomization.</justification></sq_1_2>
          <sq_1_3><answer>N</answer><quote>balanced</quote><justification>No imbalance.</justification></sq_1_3>
        </domain1>
        """,
        domain2_sq12="""
        <domain2_part1>
          <sq_2_1><answer>Y</answer><quote>Open-label</quote><justification>Participants were aware.</justification></sq_2_1>
          <sq_2_2><answer>Y</answer><quote>Open-label</quote><justification>Carers were aware.</justification></sq_2_2>
        </domain2_part1>
        """,
        domain2_conditional="""
        <domain2_conditional>
          <sq_2_3><answer>N</answer><quote>No deviations</quote><justification>No deviations from intended interventions.</justification></sq_2_3>
          <sq_2_4><answer>NA</answer><quote>Not applicable</quote><justification>Not applicable</justification><uncertainty_flag>NORMAL</uncertainty_flag></sq_2_4>
          <sq_2_5><answer>NA</answer><quote>Not applicable</quote><justification>Not applicable</justification></sq_2_5>
        </domain2_conditional>
        """,
        domain2_analysis="""
        <domain2_analysis>
          <sq_2_6><answer>Y</answer><quote>ITT</quote><justification>ITT analysis was used.</justification></sq_2_6>
          <sq_2_7><answer>NA</answer><quote>Not applicable</quote><justification>Not applicable</justification></sq_2_7>
        </domain2_analysis>
        """,
        domain3_sq="""
        <domain3>
          <sq_3_1><answer>Y</answer><quote>All followed</quote><completeness_calculation>790/790</completeness_calculation><justification>Complete follow-up.</justification></sq_3_1>
          <sq_3_2><answer>NA</answer><quote>Not applicable</quote><justification>Not applicable</justification></sq_3_2>
          <sq_3_3><answer>NA</answer><quote>Not applicable</quote><justification>Not applicable</justification><uncertainty_flag>NORMAL</uncertainty_flag></sq_3_3>
          <sq_3_4><answer>NA</answer><quote>Not applicable</quote><justification>Not applicable</justification><uncertainty_flag>NORMAL</uncertainty_flag></sq_3_4>
        </domain3>
        """,
        domain4_sq="""
        <domain4>
          <sq_4_1><answer>N</answer><quote>Progression-free survival definition</quote><justification>Standard endpoint.</justification></sq_4_1>
          <sq_4_2><answer>N</answer><quote>Same definition</quote><justification>Same method.</justification></sq_4_2>
          <sq_4_3><answer>PY</answer><quote>Open-label</quote><justification>Assessors likely aware.</justification></sq_4_3>
          <sq_4_4><answer>PY</answer><quote>biochemical, symptomatic, or radiographic progression</quote><justification>Progression includes judgmental components.</justification></sq_4_4>
          <sq_4_5><answer>N</answer><quote>No evidence of actual influence</quote><justification>No direct evidence that assessment was influenced.</justification><uncertainty_flag>NORMAL</uncertainty_flag></sq_4_5>
        </domain4>
        """,
        domain5_sq="""
        <domain5>
          <sq_5_1><answer>Y</answer><quote>NCT00309985</quote><justification>Registered before analysis.</justification><registration_comparison>No discrepancy.</registration_comparison></sq_5_1>
          <sq_5_2><answer>N</answer><quote>Progression-free survival was registered</quote><justification>The composite endpoint was pre-specified, not selected post hoc.</justification></sq_5_2>
          <sq_5_3><answer>N</answer><quote>ITT analysis</quote><justification>No selective analysis evident.</justification></sq_5_3>
        </domain5>
        """,
    )
    return responses[node_name]


class _PfsProvider:
    def __init__(self):
        self.complete = Mock(side_effect=self._complete)

    def _complete(self, system: str, user: str) -> LLMResponse:
        node_name = _node_from_prompt(user)
        return LLMResponse(_pfs_response_by_node(node_name), "test-model", 1, 1, 1.0)


class _FakeConverter:
    def convert(self, _pdf_path):
        return type("ConversionResult", (), {"document": object()})()


def _patch_ingest_dependencies():
    return (
        patch(
            "rob2_pipeline.ingestion.assessment._get_docling_converter",
            return_value=_FakeConverter(),
        ),
        patch(
            "rob2_pipeline.ingestion.assessment.build_document_repr",
            return_value=DocumentRepr(blocks=[], full_text=_pdf_text()),
        ),
        patch(
            "rob2_pipeline.ingestion.assessment._build_docling_chunks", return_value=[]
        ),
    )


def test_timed_node_records_ok_span_and_returns_result():
    try:
        start_trace(trial="T", outcome="OS")

        def fake_node(state):
            assert state == {"value": 1}
            return {"value": 2}

        wrapped = timed_node("fake_node", fake_node)

        result = wrapped({"value": 1})

        trace = get_current_trace()
        assert result == {"value": 2}
        assert trace is not None
        assert len(trace.node_spans) == 1
        span = trace.node_spans[0]
        assert span.node == "fake_node"
        assert span.status == "ok"
        assert span.error is None
        assert span.timestamp_start
        assert span.timestamp_end
        assert span.duration_ms >= 0
    finally:
        end_trace()


def test_timed_node_records_error_span_and_reraises():
    try:
        start_trace(trial="T", outcome="OS")

        def fake_node(state):
            assert state == {"value": 1}
            raise RuntimeError("boom")

        wrapped = timed_node("fake_node", fake_node)

        with pytest.raises(RuntimeError, match="boom"):
            wrapped({"value": 1})

        trace = get_current_trace()
        assert trace is not None
        assert len(trace.node_spans) == 1
        span = trace.node_spans[0]
        assert span.node == "fake_node"
        assert span.status == "error"
        assert span.error == "boom"
        assert span.timestamp_start
        assert span.timestamp_end
        assert span.duration_ms >= 0
    finally:
        end_trace()


def test_graph_happy_path_with_mocked_llm(tmp_path):
    pdf_path = tmp_path / "trial.pdf"
    _make_pdf(pdf_path)

    provider = _FakeProvider()
    with (
        patch("rob2_pipeline.nodes.common.build_provider", return_value=provider),
        patch("rob2_pipeline.ingestion.evidence.build_provider", return_value=provider),
        patch("rob2_pipeline.registration_api.fetch_registration", return_value=None),
        patch(
            "rob2_pipeline.ingestion.assessment.extract_full_text",
            return_value=_pdf_text(),
        ),
        _patch_ingest_dependencies()[0],
        _patch_ingest_dependencies()[1],
        _patch_ingest_dependencies()[2],
    ):
        state = build_rob2_graph().invoke(_initial_state(str(pdf_path)))

    assert state["overall_judgment"] == "Low"
    assert state["domain_judgments"] == {
        "D1": "Low",
        "D2": "Low",
        "D3": "Low",
        "D4": "Low",
        "D5": "Low",
    }
    assert "1.1" in state["evidence_packets"]
    assert "1.1" in state["packet_grades"]
    assert "1.1" in state["evidence_facts"]
    assert "# RoB 2 Assessment" in state["markdown_report"]
    assert "## Verified evidence packets" in state["markdown_report"]
    assert len(state["llm_call_log"]) == 9
    assert provider.complete.call_count == 9


def test_graph_pfs_composite_endpoint_d4_some_concerns_d5_low(tmp_path):
    pdf_path = tmp_path / "trial.pdf"
    _make_pdf(pdf_path)
    fake_reg_data = {
        "protocolSection": {
            "designModule": {},
            "descriptionModule": {},
            "outcomesModule": {
                "primaryOutcomes": [{"measure": "Overall Survival"}],
                "secondaryOutcomes": [{"measure": "Progression-Free Survival"}],
                "otherOutcomes": [],
            },
        }
    }

    provider = _PfsProvider()
    state = _initial_state(str(pdf_path))
    state["outcome"] = "Progression-Free Survival"
    with (
        patch("rob2_pipeline.nodes.common.build_provider", return_value=provider),
        patch("rob2_pipeline.ingestion.evidence.build_provider", return_value=provider),
        patch(
            "rob2_pipeline.registration_api.fetch_registration",
            return_value=fake_reg_data,
        ),
        patch(
            "rob2_pipeline.ingestion.assessment.extract_full_text",
            return_value=_pdf_text(),
        ),
        _patch_ingest_dependencies()[0],
        _patch_ingest_dependencies()[1],
        _patch_ingest_dependencies()[2],
    ):
        result = build_rob2_graph().invoke(state)

    assert result["registered_endpoint"] == "Progression-Free Survival"
    assert result["domain_judgments"]["D4"] == "Some concerns"
    assert result["domain_judgments"]["D5"] == "Low"
    assert result["sq_answers"]["4.4"]["answer"] == "PY"
    assert result["sq_answers"]["5.2"]["answer"] == "N"


def test_graph_stops_for_non_rct(tmp_path):
    pdf_path = tmp_path / "cohort.pdf"
    _make_pdf(pdf_path)

    class _NonRctProvider:
        def complete(self, system: str, user: str) -> LLMResponse:
            assert _node_from_prompt(user) == "rct_screener"
            return LLMResponse(
                "<screening><is_rct>NO</is_rct><evidence>cohort</evidence><study_design>Cohort</study_design><note>Use ROBINS-I</note></screening>",
                "test-model",
                1,
                1,
                1.0,
            )

    provider = _NonRctProvider()
    with (
        patch("rob2_pipeline.nodes.common.build_provider", return_value=provider),
        patch(
            "rob2_pipeline.ingestion.evidence.build_provider",
            return_value=_FakeProvider(),
        ),
        patch("rob2_pipeline.registration_api.fetch_registration", return_value=None),
        patch(
            "rob2_pipeline.ingestion.assessment.extract_full_text",
            return_value=_pdf_text(),
        ),
        _patch_ingest_dependencies()[0],
        _patch_ingest_dependencies()[1],
        _patch_ingest_dependencies()[2],
    ):
        state = build_rob2_graph().invoke(_initial_state(str(pdf_path)))

    assert state["is_rct"] is False
    assert state["domain_judgments"] == {}
    assert state["markdown_report"] == ""
    assert state["errors"]


def test_rct_screener_prompt_includes_randomization_context(tmp_path):
    pdf_path = tmp_path / "trial.pdf"
    _make_pdf(pdf_path)
    captured = {}

    class _CaptureProvider:
        def complete(self, system: str, user: str) -> LLMResponse:
            node_name = _node_from_prompt(user)
            captured[node_name] = user
            return LLMResponse(_response_by_node(node_name), "test-model", 1, 1, 1.0)

    provider = _CaptureProvider()
    with (
        patch("rob2_pipeline.nodes.common.build_provider", return_value=provider),
        patch("rob2_pipeline.ingestion.evidence.build_provider", return_value=provider),
        patch("rob2_pipeline.registration_api.fetch_registration", return_value=None),
        patch(
            "rob2_pipeline.ingestion.assessment.extract_full_text",
            return_value=_pdf_text(),
        ),
        _patch_ingest_dependencies()[0],
        _patch_ingest_dependencies()[1],
        _patch_ingest_dependencies()[2],
    ):
        build_rob2_graph().invoke(_initial_state(str(pdf_path)))

    assert "randomized controlled trial" in captured["rct_screener"]
    assert "computer-generated sequence" in captured["rct_screener"]


def test_run_assessment_writes_outputs(tmp_path):
    pdf_path = tmp_path / "trial.pdf"
    output_dir = tmp_path / "outputs"
    _make_pdf(pdf_path)

    provider = _FakeProvider()
    with (
        patch("rob2_pipeline.nodes.common.build_provider", return_value=provider),
        patch("rob2_pipeline.ingestion.evidence.build_provider", return_value=provider),
        patch("rob2_pipeline.registration_api.fetch_registration", return_value=None),
        patch(
            "rob2_pipeline.ingestion.assessment.extract_full_text",
            return_value=_pdf_text(),
        ),
        _patch_ingest_dependencies()[0],
        _patch_ingest_dependencies()[1],
        _patch_ingest_dependencies()[2],
    ):
        state = run_assessment(str(pdf_path), output_dir=str(output_dir))

    assert state["overall_judgment"] == "Low"
    assert (output_dir / "trial_rob2_report.md").exists()
    assert (output_dir / "trial_rob2_data.json").exists()
    data = json.loads((output_dir / "trial_rob2_data.json").read_text(encoding="utf-8"))
    assert data["evidence"]["extraction_method"] == "docling_llm"
    assert "computer-generated sequence" in data["evidence"]["d1_randomization"]["text"]
    assert "rag_sources" in data
    assert "outcome_properties" in data
    assert "trial_facts" in data
    assert "evidence_packets" in data
    assert "packet_grades" in data
    assert "evidence_facts" in data
    assert "retrieval_grades" in data
    assert "evidence_validation_flags" in data
    assert data["overall_policy"] == "official_rob2"


def test_preliminary_node_populates_ctgov_fields(monkeypatch):
    """preliminary_info_node should populate CT.gov design, description, and flow fields."""
    import rob2_pipeline.registration_api as api_mod
    import rob2_pipeline.nodes.preliminary as preliminary_mod

    fake_reg_data = {
        "protocolSection": {
            "designModule": {
                "phases": ["PHASE3"],
                "designInfo": {
                    "allocationType": "RANDOMIZED",
                    "interventionModel": "PARALLEL",
                    "primaryPurpose": "TREATMENT",
                    "maskingInfo": {"masking": "NONE", "whoMasked": []},
                },
                "enrollmentInfo": {"count": 790},
            },
            "descriptionModule": {
                "briefSummary": "Phase III RCT.",
                "detailedDescription": "PRIMARY: OS.",
            },
            "oversightModule": {"oversightHasDmc": True},
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": "Test Network", "class": "NETWORK"}
            },
            "outcomesModule": {
                "primaryOutcomes": [{"measure": "Overall Survival"}],
                "secondaryOutcomes": [],
                "otherOutcomes": [],
            },
        },
        "resultsSection": {
            "participantFlowModule": {
                "groups": [{"id": "FG000", "title": "Drug A"}],
                "periods": [
                    {
                        "milestones": [
                            {
                                "title": "STARTED",
                                "achievements": [
                                    {"groupId": "FG000", "numSubjects": "790"}
                                ],
                            }
                        ]
                    }
                ],
            }
        },
    }
    response = """
    <preliminary_info>
      <experimental_intervention><value>Drug A</value></experimental_intervention>
      <comparator_intervention><value>Placebo</value></comparator_intervention>
      <outcome_assessed><value>mortality</value></outcome_assessed>
      <outcome_type>vital-status</outcome_type>
      <numerical_result><value>HR 0.90</value></numerical_result>
      <n_randomized><value>790</value></n_randomized>
      <trial_registration><number>NCT00309985</number></trial_registration>
      <registered_primary_endpoint><value>Not reported</value></registered_primary_endpoint>
      <registered_secondary_endpoints>Not reported</registered_secondary_endpoints>
      <registered_analysis><value>ITT</value></registered_analysis>
    </preliminary_info>
    """

    monkeypatch.setattr(
        api_mod, "fetch_registration", lambda nct_id, use_cache=True: fake_reg_data
    )
    monkeypatch.setattr(
        preliminary_mod,
        "call_node_llm",
        lambda state, prompt, node_name: (response, [], None),
    )

    result = preliminary_mod.preliminary_info_node(_initial_state("trial.pdf"))

    assert "RANDOMIZED" in result.get("ctgov_design", "")
    assert "PRIMARY" in result.get("ctgov_description", "")
    assert "STARTED" in result.get("ctgov_flow", "")


def test_preliminary_node_surfaces_matching_secondary_endpoint(monkeypatch):
    import rob2_pipeline.registration_api as api_mod
    import rob2_pipeline.nodes.preliminary as preliminary_mod

    fake_reg_data = {
        "protocolSection": {
            "designModule": {},
            "descriptionModule": {},
            "outcomesModule": {
                "primaryOutcomes": [{"measure": "Overall Survival"}],
                "secondaryOutcomes": [{"measure": "Progression-Free Survival"}],
                "otherOutcomes": [],
            },
        }
    }
    response = """
    <preliminary_info>
      <experimental_intervention><value>Docetaxel + ADT</value></experimental_intervention>
      <comparator_intervention><value>ADT alone</value></comparator_intervention>
      <outcome_assessed><value>Overall Survival</value></outcome_assessed>
      <outcome_type>clinician-composite</outcome_type>
      <numerical_result><value>HR 0.61</value></numerical_result>
      <n_randomized><value>790</value></n_randomized>
      <trial_registration><number>NCT00309985</number></trial_registration>
      <registered_primary_endpoint><value>Overall Survival</value></registered_primary_endpoint>
      <registered_secondary_endpoints>Not reported</registered_secondary_endpoints>
      <registered_analysis><value>ITT</value></registered_analysis>
    </preliminary_info>
    """
    state = _initial_state("trial.pdf")
    state["outcome"] = "Progression-Free Survival"

    monkeypatch.setattr(
        api_mod, "fetch_registration", lambda nct_id, use_cache=True: fake_reg_data
    )
    monkeypatch.setattr(
        preliminary_mod,
        "call_node_llm",
        lambda state, prompt, node_name: (response, [], None),
    )

    result = preliminary_mod.preliminary_info_node(state)

    assert result["outcome"] == "Progression-Free Survival"
    assert result["registered_endpoint"] == "Progression-Free Survival"
    assert result["registered_secondary_endpoints"] == "Progression-Free Survival"
