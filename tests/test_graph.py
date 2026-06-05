from pathlib import Path
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from rob2_pipeline.graph import build_rob2_graph, timed_node
from rob2_pipeline.trace import get_current_trace, start_trace, end_trace
from rob2_pipeline.models import empty_paper_evidence
from rob2_pipeline.ingestion.parse_artifacts import ParserProvenance, SourceParseArtifact
from rob2_pipeline.pipeline import run_assessment
from rob2_pipeline.providers.base import LLMResponse


def _core_contract_result(state, prompt, node_name, **kwargs):
    del prompt, kwargs
    artifacts = {
        "rct_screener": {
            "schema_version": "rct-screening-v1",
            "is_rct": True,
            "evidence": "randomly assigned",
            "study_design": "RCT",
            "note": "",
        },
        "preliminary_info": {
            "schema_version": "preliminary-info-v1",
            "intervention": "Drug A",
            "comparator": "Placebo",
            "assessed_outcome": "mortality",
            "outcome_type": "vital-status",
            "numerical_result": "RR 0.90 (95% CI 0.70-1.10)",
            "registration_number": "NCT00000000",
            "registered_primary_endpoint": "mortality",
            "registered_secondary_endpoints": "Not reported",
            "registered_analysis": "ITT",
            "n_randomized": "100",
        },
        "outcome_resolver": {
            "schema_version": "outcome-normalization-v1",
            "outcome_type": "vital-status",
            "normalized_definition": "Mortality.",
            "aliases": [],
            "outcome_properties": {
                "patient_reported": False,
                "safety_harm": False,
                "time_to_event": True,
                "composite": False,
                "lab_or_imaging_threshold": False,
                "blinded_adjudication": False,
                "objective_event": True,
                "clinician_judged": False,
            },
            "support": {
                "support_level": "strong",
                "support_rationale": "Mortality is a death-only endpoint.",
                "quotes": [
                    {
                        "quote": "The primary outcome was mortality.",
                        "source": "d4_outcome_meas",
                    }
                ],
                "constraints": [],
            },
            "uncertainty": False,
        },
        "paper_evidence_extraction": {
            "schema_version": "paper-evidence-extraction-v1",
            "abstract": {
                "text": "This randomized controlled trial compared Drug A with placebo.",
                "tables": [],
            },
            "methods": {
                "text": "Participants were randomly assigned using a computer-generated sequence. Allocation was concealed centrally. The trial used intention-to-treat analysis.",
                "tables": [],
            },
            "results": {
                "text": "100 participants were randomized and all had outcome data.",
                "tables": [],
            },
            "d1_randomization": {
                "text": "Participants were randomly assigned using a computer-generated sequence. Allocation was concealed centrally.",
                "tables": [],
            },
            "d2_blinding": {"text": "Participants and investigators were blinded.", "tables": []},
            "d3_missing_data": {"text": "100 participants were randomized and all had outcome data.", "tables": []},
            "d4_outcome_meas": {
                "text": "The primary outcome was mortality. The trial used intention-to-treat analysis.",
                "tables": [],
            },
            "d5_registration": {"text": "ClinicalTrials.gov NCT00000000.", "tables": []},
            "consort_flow": {"text": "100 participants were randomized.", "tables": []},
            "baseline_table": {"text": "baseline balanced", "tables": []},
        },
    }
    if node_name == "outcome_resolver" and state.get("outcome") == "Progression-Free Survival":
        artifacts["outcome_resolver"] = {
            **artifacts["outcome_resolver"],
            "outcome_type": "clinician-composite",
            "normalized_definition": "Progression-free survival.",
            "outcome_properties": {
                "patient_reported": False,
                "safety_harm": False,
                "time_to_event": True,
                "composite": True,
                "lab_or_imaging_threshold": True,
                "blinded_adjudication": False,
                "objective_event": False,
                "clinician_judged": True,
            },
            "support": {
                "support_level": "strong",
                "support_rationale": "PFS combines progression and death.",
                "quotes": [
                    {
                        "quote": "The primary outcome was mortality. The trial used intention-to-treat analysis.",
                        "source": "d4_outcome_meas",
                    }
                ],
                "constraints": [],
            },
        }
    return SimpleNamespace(
        artifact=artifacts[node_name],
        log=[{"node": node_name, "validation_status": "validated"}],
        status="validated",
        failure_reason=None,
    )


@pytest.fixture(autouse=True)
def _patch_core_json_contracts(monkeypatch):
    monkeypatch.setattr(
        "rob2_pipeline.nodes.ingest.call_json_contract_llm", _core_contract_result
    )
    monkeypatch.setattr(
        "rob2_pipeline.nodes.preliminary.call_json_contract_llm", _core_contract_result
    )
    monkeypatch.setattr(
        "rob2_pipeline.nodes.outcome_resolver.call_json_contract_llm",
        _core_contract_result,
    )
    monkeypatch.setattr(
        "rob2_pipeline.ingestion.evidence.call_json_contract_llm",
        _core_contract_result,
    )


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
        "outcome_resolver": """
        <outcome_resolution>
          <outcome_type>vital-status</outcome_type>
          <support_level>strong</support_level>
          <support_rationale>Mortality is defined as a death-only endpoint.</support_rationale>
          <properties>
            <patient_reported>false</patient_reported>
            <safety_harm>false</safety_harm>
            <time_to_event>true</time_to_event>
            <death_only_objective_event>true</death_only_objective_event>
            <composite>false</composite>
            <lab_or_imaging_threshold>false</lab_or_imaging_threshold>
            <blinded_adjudication>false</blinded_adjudication>
            <objective_event>true</objective_event>
            <clinician_judged>false</clinician_judged>
          </properties>
          <quotes>
            <quote source="d4_outcome_meas">The primary outcome was mortality.</quote>
          </quotes>
          <constraints></constraints>
        </outcome_resolution>
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
        "evidence_family_mining": json.dumps(
            {
                "facts": [
                    {
                        "artifact_id": "evidence-fact:d1:1.1:central-randomization",
                        "fact_type": "randomization_sequence",
                        "domain": "d1",
                        "sq_ids": ["1.1"],
                        "claim_type": "trial_method",
                        "claim": "Participants were randomly assigned centrally.",
                        "quote": "Participants were randomly assigned using a computer-generated sequence.",
                        "support_level": "strong",
                        "support_status": "supported",
                        "uncertainty": False,
                        "family": "randomization_allocation",
                        "family_fields": {
                            "method": "computer-generated sequence",
                            "allocation_concealment": "central concealment",
                            "unit_of_randomization": "participant",
                        },
                        "provenance": {
                            "document_id": "primary",
                            "document_name": "Primary paper",
                            "document_role": "primary",
                            "source_kind": "section_text",
                            "source_path": "unknown",
                            "source_section": "d1_randomization",
                            "page_numbers": [],
                        },
                    }
                ]
            }
        ),
    }
    return responses[node_name]


def _node_from_prompt(prompt: str) -> str:
    if "<evidence>" in prompt and "d1_randomization" in prompt:
        return "paper_evidence_extraction"
    if "<screening>" in prompt:
        return "rct_screener"
    if "<preliminary_info>" in prompt:
        return "preliminary_info"
    if "<outcome_resolution>" in prompt:
        return "outcome_resolver"
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
    if "Extract typed evidence facts from the bounded source zones only." in prompt:
        return "evidence_family_mining"
    raise KeyError("Unknown prompt")


def _family_response_from_prompt(prompt: str) -> str:
    sq_id = "1.1"
    for line in prompt.splitlines():
        if line.startswith("SQ ID: "):
            sq_id = line.removeprefix("SQ ID: ").strip()
            break
    domain = f"d{sq_id.split('.')[0]}"
    family_by_sq = {
        "1.1": "randomization_allocation",
        "1.2": "randomization_allocation",
        "2.1": "masking_awareness",
        "2.2": "masking_awareness",
        "2.3": "deviations_adherence",
        "2.4": "deviations_adherence",
        "2.5": "deviations_adherence",
        "2.6": "analysis_population",
        "2.7": "analysis_population",
        "3.1": "missing_outcome_data",
        "3.2": "missing_outcome_data",
        "3.3": "missing_outcome_data",
        "3.4": "missing_outcome_data",
        "4.1": "outcome_measurement",
        "4.2": "outcome_measurement",
        "4.3": "outcome_measurement",
        "4.4": "outcome_measurement",
        "4.5": "outcome_measurement",
        "5.1": "prespecification",
        "5.2": "result_reporting",
        "5.3": "result_reporting",
    }
    family = family_by_sq[sq_id]
    if family == "randomization_allocation":
        family_fields = {
            "method": "computer-generated sequence",
            "allocation_concealment": "central concealment",
            "unit_of_randomization": "participant",
        }
        claim_type = "trial_method"
    elif family == "masking_awareness":
        family_fields = {
            "participant_awareness": "participants were blinded",
            "personnel_awareness": "personnel were blinded",
            "masking_method": "matched placebo",
            "awareness_context": "double-blind trial conduct",
        }
        claim_type = "trial_method"
    elif family == "deviations_adherence":
        family_fields = {
            "awareness_status": "participants and personnel blinded",
            "deviation_description": "no important protocol deviations",
            "adherence_population": "all randomized participants",
            "analysis_population": "intention-to-treat",
            "outcome_impact": "no important impact on mortality",
        }
        claim_type = "trial_method"
    elif family == "analysis_population":
        family_fields = {
            "population_label": "intention-to-treat",
            "included_participants": "all randomized participants",
            "excluded_participants": "none reported",
            "analysis_principle": "analyzed as randomized",
            "exclusion_impact": "no important impact",
        }
        claim_type = "analysis"
    elif family == "missing_outcome_data":
        family_fields = {
            "randomized_count": "100",
            "outcome_data_count": "100",
            "missing_count": "0",
            "missing_reason": "no missing outcome data",
            "analysis_handling": "all randomized participants analyzed",
        }
        claim_type = "analysis"
    elif family == "outcome_measurement":
        family_fields = {
            "assessed_outcome": "mortality",
            "measurement_method": "death from any cause",
            "measurement_timing": "during follow-up",
            "assessor_awareness": "objective vital-status endpoint",
            "influence_risk": "not likely influenced by awareness",
        }
        claim_type = "outcome_measurement"
    elif family == "prespecification":
        family_fields = {
            "artifact_type": "registry",
            "identifier": "NCT00000000",
            "prespecified_outcome": "mortality",
            "prespecified_analysis": "intention-to-treat",
        }
        claim_type = "registry"
    else:
        family_fields = {
            "reported_outcome": "mortality",
            "reported_measurement": "death from any cause",
            "reported_analysis": "intention-to-treat",
            "result_metric": "risk ratio with confidence interval",
            "matches_prespecification": "matches the prespecified registry outcome",
        }
        claim_type = "result_reporting"
    return json.dumps(
        {
            "facts": [
                {
                    "artifact_id": f"evidence-fact:{domain}:{sq_id}:family",
                    "fact_type": f"{family}_fact",
                    "domain": domain,
                    "sq_ids": [sq_id],
                    "claim_type": claim_type,
                    "claim": "Selected source supports the family fact.",
                    "quote": "Selected source supports the family fact.",
                    "support_level": "strong",
                    "support_status": "supported",
                    "uncertainty": False,
                    "family": family,
                    "family_fields": family_fields,
                    "provenance": {
                        "document_id": "primary",
                        "document_name": "Primary paper",
                        "document_role": "primary",
                        "source_kind": "section_text",
                        "source_path": "unknown",
                        "source_section": "Methods",
                        "page_numbers": [],
                    },
                }
            ]
        }
    )


class _FakeProvider:
    def __init__(self):
        self.complete = Mock(side_effect=self._complete)

    def _complete(self, system: str, user: str) -> LLMResponse:
        node_name = _node_from_prompt(user)
        if node_name == "evidence_family_mining":
            return LLMResponse(
                _family_response_from_prompt(user), "test-model", 1, 1, 1.0
            )
        return LLMResponse(_response_by_node(node_name), "test-model", 1, 1, 1.0)


def _d1_contract_result(state, prompt, node_name, **kwargs):
    del state, prompt, kwargs
    return {
        "artifact": {
            "schema_version": "d1-sq-classifier-v1",
            "domain": "d1",
            "answers": [
                _d1_contract_answer("1.1", "Y", "computer-generated sequence"),
                _d1_contract_answer("1.2", "Y", "concealed centrally"),
                _d1_contract_answer("1.3", "N", "baseline balanced"),
            ],
        },
        "log": [
            {
                "node": node_name,
                "validation_status": "validated",
                "model": "test-model",
                "prompt_version": "d1-sq-classifier-prompt-v1",
                "schema_version": "d1-sq-classifier-v1",
                "attempts": [{"attempt": 1}],
            }
        ],
        "status": "validated",
    }


def _d1_contract_answer(sq_id: str, answer: str, quote: str) -> dict:
    return {
        "sq_id": sq_id,
        "answer": answer,
        "quote": quote,
        "justification": "Selected D1 packet evidence supports this answer.",
        "support_level": "strong",
        "support_rationale": "Supported by selected packet evidence.",
        "uncertainty": False,
        "packet_artifact_id": f"evidence-packet:d1:{sq_id}",
        "decision_table_artifact_id": f"decision-table:d1:{sq_id}",
        "supporting_fact_artifact_ids": [],
    }


def _domain_contract_result(state, prompt, node_name, **kwargs):
    del kwargs
    payload = json.loads(prompt.split("\n\n", 1)[1])
    domain = payload["domain"]
    stage = payload["stage"]
    packets = payload["evidence_packets"]
    answers = []
    for packet in packets:
        sq_id = packet["sq_id"]
        answer, quote = _contract_answer_for_sq(state, sq_id)
        answers.append(
            {
                "sq_id": sq_id,
                "answer": answer,
                "quote": quote,
                "justification": "Selected packet evidence supports this answer.",
                "support_level": "strong",
                "support_rationale": "Supported by selected packet evidence.",
                "uncertainty": False,
                "packet_artifact_id": f"evidence-packet:{domain}:{sq_id}",
                "decision_table_artifact_id": f"decision-table:{domain}:{sq_id}",
                "supporting_fact_artifact_ids": [],
            }
        )
    artifact = {
        "schema_version": f"{domain}-sq-classifier-v1",
        "domain": domain,
        "stage": stage,
        "branching": payload.get("branching", {}),
        "outcome_specific_concerns": payload.get("outcome_specific_concerns", []),
        "answers": answers,
    }
    return {
        "artifact": artifact,
        "log": [
            {
                "node": node_name,
                "validation_status": "validated",
                "model": "test-model",
                "prompt_version": f"{domain}-{stage}-sq-classifier-prompt-v1",
                "schema_version": f"{domain}-sq-classifier-v1",
                "attempts": [{"attempt": 1}],
            }
        ],
        "status": "validated",
    }


def _contract_answer_for_sq(state: dict, sq_id: str) -> tuple[str, str]:
    is_pfs = state.get("outcome") == "Progression-Free Survival"
    pfs_answers = {
        "2.1": ("Y", "Open-label treatment assignment"),
        "2.2": ("Y", "Open-label treatment assignment"),
        "2.3": ("N", "no important protocol deviations"),
        "2.4": ("NA", "Not applicable"),
        "2.5": ("NA", "Not applicable"),
        "2.6": ("Y", "intention-to-treat"),
        "2.7": ("NA", "Not applicable"),
        "3.1": ("Y", "All randomly assigned patients were followed"),
        "3.2": ("NA", "Not applicable"),
        "3.3": ("NA", "Not applicable"),
        "3.4": ("NA", "Not applicable"),
        "4.1": ("N", "Progression-free survival was biochemical, symptomatic, or radiographic progression"),
        "4.2": ("N", "Progression-free survival was biochemical, symptomatic, or radiographic progression"),
        "4.3": ("PY", "Open-label treatment assignment"),
        "4.4": ("PY", "Progression-free survival was biochemical, symptomatic, or radiographic progression"),
        "4.5": ("N", "no evidence that assessment was influenced"),
        "5.1": ("Y", "NCT00309985"),
        "5.2": ("N", "Progression-Free Survival"),
        "5.3": ("N", "intention-to-treat"),
    }
    os_answers = {
        "2.1": ("N", "Participants and investigators were blinded"),
        "2.2": ("N", "Participants and investigators were blinded"),
        "2.6": ("Y", "intention-to-treat analysis"),
        "2.7": ("NA", "Not applicable"),
        "3.1": ("Y", "100 participants were randomized and all had outcome data"),
        "3.2": ("NA", "Not applicable"),
        "3.3": ("NA", "Not applicable"),
        "3.4": ("NA", "Not applicable"),
        "4.1": ("N", "The primary outcome was mortality"),
        "4.2": ("N", "The primary outcome was mortality"),
        "4.3": ("N", "Participants and investigators were blinded"),
        "4.4": ("NA", "Not applicable"),
        "4.5": ("NA", "Not applicable"),
        "5.1": ("Y", "NCT00000000"),
        "5.2": ("N", "The primary outcome was mortality"),
        "5.3": ("N", "intention-to-treat analysis"),
    }
    return (pfs_answers if is_pfs else os_answers)[sq_id]


def _patch_ingest_dependencies():
    def parse_sources(sources):
        return [
            SourceParseArtifact(
                source_identity={**source, "status": "parsed"},
                pages=[{"page_number": 1, "text": _pdf_text()}],
                diagnostics=[],
                provenance=ParserProvenance(
                    parser_name="fake-liteparse",
                    parser_version="1.0.0",
                    adapter_name="fake",
                    artifact_schema_version="parse-artifact-v1",
                    config={},
                ),
            )
            for source in sources
        ]

    return (patch("rob2_pipeline.ingestion.assessment.parse_sources", parse_sources),)


def _fast_rag_retrieval_node(state):
    text = (
        state.get("evidence", {})
        .get("methods", {})
        .get("text", "Participants were randomized and blinded.")
    )
    metadata = {
        domain: [
            {
                "text": text,
                "section": "Methods",
                "page_numbers": [1],
                "score": 0.01,
                "document_id": "primary",
                "document_name": "Primary paper",
                "document_role": "primary",
                "source_kind": "rag_chunk",
                "source_path": state.get("pdf_path", ""),
            }
        ]
        for domain in ("d1", "d2", "d3", "d4", "d5")
    }
    return {
        "rag_contexts": {
            "d1": text,
            "d2_blinding": text,
            "d2_deviations": text,
            "d2_analysis": text,
            "d3": text,
            "d4_measurement": text,
            "d4_assessor": text,
            "d5": text,
        },
        "rag_chunk_metadata": metadata,
        "retrieval_grades": {
            domain: {
                "relevance": 1.0,
                "coverage": 1.0,
                "missing_evidence": [],
                "retry_recommended": False,
            }
            for domain in ("d1", "d2", "d3", "d4", "d5")
        },
        "trial_retrieval_indexes": {"index": "test-index", "filtered": {}},
    }


def _patch_fast_rag():
    return patch("rob2_pipeline.graph.rag_retrieval_node", _fast_rag_retrieval_node)


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
        patch(
            "rob2_pipeline.nodes.domain1.call_json_contract_llm",
            side_effect=_d1_contract_result,
        ),
        patch(
            "rob2_pipeline.nodes.domain_classifier.call_json_contract_llm",
            side_effect=_domain_contract_result,
        ),
        patch("rob2_pipeline.registration_api.fetch_registration", return_value=None),
        _patch_ingest_dependencies()[0],
        _patch_fast_rag(),
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
    assert state["evidence_store"]["supported_facts"]
    assert len(state["llm_call_log"]) == 10
    assert provider.complete.call_count == 15


def test_graph_finalization_waits_for_all_domain_judges(tmp_path):
    pdf_path = tmp_path / "trial.pdf"
    _make_pdf(pdf_path)

    provider = _FakeProvider()
    try:
        start_trace(trial="trial", outcome="mortality")
        with (
            patch("rob2_pipeline.nodes.common.build_provider", return_value=provider),
            patch(
                "rob2_pipeline.nodes.domain1.call_json_contract_llm",
                side_effect=_d1_contract_result,
            ),
            patch(
                "rob2_pipeline.nodes.domain_classifier.call_json_contract_llm",
                side_effect=_domain_contract_result,
            ),
            patch("rob2_pipeline.registration_api.fetch_registration", return_value=None),
            _patch_ingest_dependencies()[0],
            _patch_fast_rag(),
        ):
            state = build_rob2_graph().invoke(_initial_state(str(pdf_path)))

        trace = get_current_trace()
        assert state["overall_judgment"] == "Low"
        assert trace is not None
        span_names = [span.node for span in trace.node_spans]
        assert span_names.count("quote_verifier") == 1
        assert span_names.count("overall_judge") == 1
        assert span_names.count("report_formatter") == 1
        quote_index = span_names.index("quote_verifier")
        for judge_name in (
            "domain1_judge",
            "domain2_judge",
            "domain3_judge",
            "domain4_judge",
            "domain5_judge",
        ):
            assert span_names.index(judge_name) < quote_index
    finally:
        end_trace()


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
    def _non_rct_contract(state, prompt, node_name, **kwargs):
        del state, prompt, kwargs
        return SimpleNamespace(
            artifact={
                "schema_version": "rct-screening-v1",
                "is_rct": False,
                "evidence": "cohort",
                "study_design": "Cohort",
                "note": "Use ROBINS-I",
            },
            log=[{"node": node_name, "validation_status": "validated"}],
            status="validated",
            failure_reason=None,
        )

    with (
        patch("rob2_pipeline.nodes.ingest.call_json_contract_llm", _non_rct_contract),
        patch("rob2_pipeline.nodes.common.build_provider", return_value=provider),
        patch(
            "rob2_pipeline.nodes.domain1.call_json_contract_llm",
            side_effect=_d1_contract_result,
        ),
        patch(
            "rob2_pipeline.nodes.domain_classifier.call_json_contract_llm",
            side_effect=_domain_contract_result,
        ),
        patch("rob2_pipeline.registration_api.fetch_registration", return_value=None),
        _patch_ingest_dependencies()[0],
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
    def _capture_rct_contract(state, prompt, node_name, **kwargs):
        captured[node_name] = prompt
        return _core_contract_result(state, prompt, node_name, **kwargs)

    with (
        patch("rob2_pipeline.nodes.ingest.call_json_contract_llm", _capture_rct_contract),
        patch("rob2_pipeline.nodes.common.build_provider", return_value=provider),
        patch(
            "rob2_pipeline.nodes.domain1.call_json_contract_llm",
            side_effect=_d1_contract_result,
        ),
        patch(
            "rob2_pipeline.nodes.domain_classifier.call_json_contract_llm",
            side_effect=_domain_contract_result,
        ),
        patch("rob2_pipeline.registration_api.fetch_registration", return_value=None),
        _patch_ingest_dependencies()[0],
        _patch_fast_rag(),
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
        patch(
            "rob2_pipeline.nodes.domain1.call_json_contract_llm",
            side_effect=_d1_contract_result,
        ),
        patch(
            "rob2_pipeline.nodes.domain_classifier.call_json_contract_llm",
            side_effect=_domain_contract_result,
        ),
        patch("rob2_pipeline.registration_api.fetch_registration", return_value=None),
        _patch_ingest_dependencies()[0],
        _patch_fast_rag(),
    ):
        state = run_assessment(str(pdf_path), output_dir=str(output_dir))

    assert state["overall_judgment"] == "Low"
    assert (output_dir / "trial_rob2_report.md").exists()
    assert (output_dir / "trial_rob2_data.json").exists()
    assert (
        output_dir / "trial_trial_workspace" / "trial-workspace-manifest.json"
    ).exists()
    assert (
        output_dir / "trial_trial_workspace" / "diagnostics" / "primary.json"
    ).exists()
    evidence_jsonl = (
        output_dir / "trial_trial_workspace" / "evidence_store" / "facts.jsonl"
    )
    assert evidence_jsonl.exists()
    assert "search_text" in evidence_jsonl.read_text(encoding="utf-8")
    data = json.loads((output_dir / "trial_rob2_data.json").read_text(encoding="utf-8"))
    assert data["evidence"]["extraction_method"] == "json_contract"
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
    monkeypatch.setattr(
        api_mod, "fetch_registration", lambda nct_id, use_cache=True: fake_reg_data
    )
    monkeypatch.setattr(
        preliminary_mod,
        "call_json_contract_llm",
        lambda state, prompt, node_name, **kwargs: SimpleNamespace(
            artifact={
                "schema_version": "preliminary-info-v1",
                "intervention": "Drug A",
                "comparator": "Placebo",
                "assessed_outcome": "mortality",
                "outcome_type": "vital-status",
                "numerical_result": "HR 0.90",
                "n_randomized": "790",
                "registration_number": "NCT00309985",
                "registered_primary_endpoint": "Not reported",
                "registered_secondary_endpoints": "Not reported",
                "registered_analysis": "ITT",
            },
            log=[],
            status="validated",
            failure_reason=None,
        ),
    )

    result = preliminary_mod.preliminary_info_node(_initial_state("trial.pdf"))

    assert "RANDOMIZED" in result.get("ctgov_design", "")
    assert "PRIMARY" in result.get("ctgov_description", "")
    assert "STARTED" in result.get("ctgov_flow", "")
    registry_sources = [
        source
        for source in result.get("source_documents", [])
        if source.get("document_role") == "registry"
    ]
    assert registry_sources == [
        {
            "document_id": "registry:NCT00309985",
            "document_name": "ClinicalTrials.gov NCT00309985",
            "document_role": "registry",
            "source_kind": "ctgov",
            "path": "https://clinicaltrials.gov/study/NCT00309985",
            "is_primary": False,
            "status": "parsed",
            "retrieval_date": registry_sources[0]["retrieval_date"],
            "api_response_hash": registry_sources[0]["api_response_hash"],
        }
    ]
    assert registry_sources[0]["retrieval_date"]
    assert len(registry_sources[0]["api_response_hash"]) == 64


def test_preliminary_node_normalizes_embedded_nct_id(monkeypatch):
    import rob2_pipeline.registration_api as api_mod
    import rob2_pipeline.nodes.preliminary as preliminary_mod

    state = _initial_state("trial.pdf")
    state["evidence"] = empty_paper_evidence()
    state["evidence"]["d5_registration"]["text"] = (
        "ClinicalTrials.gov number, NCT00309985."
    )

    monkeypatch.setattr(
        api_mod,
        "fetch_registration",
        lambda nct_id, use_cache=True: {
            "protocolSection": {
                "outcomesModule": {"primaryOutcomes": [{"measure": "Overall Survival"}]},
                "designModule": {},
                "descriptionModule": {},
            }
        },
    )
    monkeypatch.setattr(
        preliminary_mod,
        "call_json_contract_llm",
        lambda state, prompt, node_name, **kwargs: SimpleNamespace(
            artifact={
                "schema_version": "preliminary-info-v1",
                "intervention": "Drug A",
                "comparator": "Placebo",
                "assessed_outcome": "Overall Survival",
                "outcome_type": "vital-status",
                "numerical_result": "HR 0.90",
                "n_randomized": "790",
                "registration_number": "ClinicalTrials.gov number, NCT00309985.",
                "registered_primary_endpoint": "Not reported",
                "registered_secondary_endpoints": "Not reported",
                "registered_analysis": "ITT",
            },
            log=[],
            status="validated",
            failure_reason=None,
        ),
    )

    result = preliminary_mod.preliminary_info_node(state)

    assert result["registration_number"] == "NCT00309985"
    assert result["ctgov_registry_document"]["document_id"] == "registry:NCT00309985"
    assert result["registered_endpoint"] == "Overall Survival"


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
    state = _initial_state("trial.pdf")
    state["outcome"] = "Progression-Free Survival"

    monkeypatch.setattr(
        api_mod, "fetch_registration", lambda nct_id, use_cache=True: fake_reg_data
    )
    monkeypatch.setattr(
        preliminary_mod,
        "call_json_contract_llm",
        lambda state, prompt, node_name, **kwargs: SimpleNamespace(
            artifact={
                "schema_version": "preliminary-info-v1",
                "intervention": "Docetaxel + ADT",
                "comparator": "ADT alone",
                "assessed_outcome": "Overall Survival",
                "outcome_type": "clinician-composite",
                "numerical_result": "HR 0.61",
                "n_randomized": "790",
                "registration_number": "NCT00309985",
                "registered_primary_endpoint": "Overall Survival",
                "registered_secondary_endpoints": "Not reported",
                "registered_analysis": "ITT",
            },
            log=[],
            status="validated",
            failure_reason=None,
        ),
    )

    result = preliminary_mod.preliminary_info_node(state)

    assert result["outcome"] == "Progression-Free Survival"
    assert result["registered_endpoint"] == "Progression-Free Survival"
    assert result["registered_secondary_endpoints"] == "Progression-Free Survival"
