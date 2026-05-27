from rob2_pipeline.nodes.outcome_resolver import (
    infer_outcome_properties,
    outcome_resolver_node,
    outcome_type_from_properties,
)


class _LLMResponse:
    def __init__(self, content: str):
        self.content = content
        self.model = "mock-model"
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached = False
        self.reasoning_content = None


class _LLMProvider:
    def __init__(self, content: str):
        self._content = content

    def complete(self, system: str, user: str):
        del system, user
        return _LLMResponse(self._content)


def _os_state():
    return {
        "outcome": "Overall survival",
        "outcome_code": "OS",
        "registered_endpoint": "Overall Survival",
        "evidence": {
            "abstract": {
                "text": (
                    "Overall survival was defined as time from randomization "
                    "to death from any cause."
                ),
                "tables": [],
                "source": "primary",
            },
            "methods": {
                "text": (
                    "Patients completed symptom questionnaires. Safety, adverse "
                    "events, and radiographic progression were also assessed."
                ),
                "tables": [],
                "source": "primary",
            },
        },
        "ctgov_outcomes": "PRIMARY: Overall Survival",
        "errors": [],
    }


def _pfs_state():
    return {
        "outcome": "Progression-free survival",
        "outcome_code": "PFS",
        "registered_endpoint": "Progression-free survival",
        "registered_secondary_endpoints": "Overall survival; response rate",
        "registered_analysis": "Prespecified analysis of progression-free survival.",
        "evidence": {
            "d5_registration": {
                "text": "The protocol prespecified progression-free survival as the primary endpoint.",
                "tables": [],
                "source": "protocol",
            },
            "methods": {
                "text": (
                    "Patients completed symptom questionnaires and reported nausea, "
                    "which were unrelated to the endpoint definition."
                ),
                "tables": [],
                "source": "primary",
            },
        },
        "ctgov_outcomes": "PRIMARY: Progression-free survival; SECONDARY: Overall survival",
        "errors": [],
    }


def _ae_state():
    return {
        "outcome": "Serious adverse events",
        "outcome_code": "AE",
        "registered_endpoint": "Serious adverse events",
        "registered_secondary_endpoints": "Quality of life",
        "registered_analysis": "Safety analysis according to the protocol.",
        "evidence": {
            "d5_registration": {
                "text": "The protocol prespecified serious adverse events as the safety endpoint.",
                "tables": [],
                "source": "protocol",
            },
            "methods": {
                "text": (
                    "Participants also completed symptom questionnaires and the paper "
                    "described pain scales, but these were not the assessed endpoint."
                ),
                "tables": [],
                "source": "primary",
            },
        },
        "ctgov_outcomes": "PRIMARY: Serious adverse events; SECONDARY: Quality of life",
        "errors": [],
    }


def test_mortality_endpoint_is_vital_status():
    props = infer_outcome_properties(
        "All-cause mortality",
        "The endpoint was death from any cause.",
    )

    assert props["objective_event"] is True
    assert props["composite"] is False
    assert outcome_type_from_properties(props) == "vital-status"


def test_composite_time_to_event_is_not_vital_status():
    props = infer_outcome_properties(
        "Event-free survival",
        "Event-free survival was time to relapse, hospitalization, treatment failure, or death.",
    )

    assert props["time_to_event"] is True
    assert props["composite"] is True
    assert outcome_type_from_properties(props) == "clinician-composite"


def test_patient_reported_outcome_takes_priority():
    props = infer_outcome_properties(
        "Pain severity",
        "Pain severity was self-reported using a questionnaire.",
    )

    assert props["patient_reported"] is True
    assert outcome_type_from_properties(props) == "patient-reported"


def test_safety_harm_outcome_is_clinician_graded():
    props = infer_outcome_properties(
        "Serious harms",
        "Serious adverse events and toxicity were graded by study clinicians.",
    )

    assert props["safety_harm"] is True
    assert outcome_type_from_properties(props) == "clinician-graded"


def test_lab_threshold_outcome_is_biomarker_when_not_composite():
    props = infer_outcome_properties(
        "Viral suppression",
        "Viral suppression was measured by laboratory assay below a prespecified threshold.",
    )

    assert props["lab_or_imaging_threshold"] is True
    assert outcome_type_from_properties(props) == "biomarker"


def test_os_scope_ignores_unrelated_paper_wide_harms_and_symptoms():
    result = outcome_resolver_node(_os_state())

    assert result["outcome_type"] == "vital-status"
    assert result["outcome_properties"]["objective_event"] is True
    assert result["outcome_properties"]["patient_reported"] is False
    assert result["outcome_properties"]["safety_harm"] is False
    assert result["outcome_resolution"]["support"][0]["source_priority"] == 1
    assert any(
        item["source_priority"] == 2 for item in result["outcome_resolution"]["support"]
    )


def test_pfs_scope_keeps_progression_free_survival_from_broad_symptom_text():
    result = outcome_resolver_node(_pfs_state())

    assert result["outcome_type"] == "clinician-composite"
    assert result["outcome_properties"]["time_to_event"] is True
    assert result["outcome_properties"]["composite"] is True
    assert result["outcome_properties"]["patient_reported"] is False
    assert result["outcome_properties"]["safety_harm"] is False
    assert result["outcome_resolution"]["support"][0]["source_priority"] == 1
    assert any(
        item["source_priority"] == 3 for item in result["outcome_resolution"]["support"]
    )


def test_ae_scope_keeps_safety_endpoint_from_patient_reported_noise():
    result = outcome_resolver_node(_ae_state())

    assert result["outcome_type"] == "clinician-graded"
    assert result["outcome_properties"]["safety_harm"] is True
    assert result["outcome_properties"]["patient_reported"] is False
    assert result["outcome_properties"]["objective_event"] is False
    assert result["outcome_resolution"]["support"][0]["source_priority"] == 1


def test_registry_support_does_not_override_primary_endpoint_definition():
    result = outcome_resolver_node(
        {
            "outcome": "Overall survival",
            "outcome_code": "OS",
            "evidence": {
                "abstract": {
                    "text": "Overall survival was defined as death from any cause.",
                    "tables": [],
                    "source": "primary",
                }
            },
            "ctgov_outcomes": "PRIMARY: Adverse events; SECONDARY: Overall Survival",
            "errors": [],
        }
    )

    support_priorities = [
        item["source_priority"] for item in result["outcome_resolution"]["support"]
    ]
    assert result["outcome_type"] == "vital-status"
    assert 2 in support_priorities
    assert 4 in support_priorities
    assert result["outcome_properties"]["safety_harm"] is False


def test_invalid_llm_output_falls_back_to_deterministic_resolution(monkeypatch):
    from rob2_pipeline.nodes import outcome_resolver as module

    monkeypatch.setattr(
        module, "build_provider", lambda: _LLMProvider("not json at all")
    )

    state = _os_state()
    state["enable_llm_outcome_resolution"] = True
    result = module.outcome_resolver_node(state)

    assert result["outcome_type"] == "vital-status"
    assert result["outcome_properties"]["objective_event"] is True
    assert any("invalid" in error.lower() for error in result["errors"])


def test_contradictory_llm_output_is_overruled_by_scope_guardrails(monkeypatch):
    from rob2_pipeline.nodes import outcome_resolver as module

    payload = """
    {
      "outcome_properties": {
        "objective_event": false,
        "clinician_judged": false,
        "patient_reported": true,
        "composite": false,
        "time_to_event": false,
        "safety_harm": true,
        "lab_or_imaging_threshold": false,
        "blinded_adjudication": false
      },
      "outcome_type": "patient-reported",
      "support": [],
      "warnings": ["LLM guessed from broad text."]
    }
    """
    monkeypatch.setattr(module, "build_provider", lambda: _LLMProvider(payload))

    state = _os_state()
    state["enable_llm_outcome_resolution"] = True
    result = module.outcome_resolver_node(state)

    assert result["outcome_type"] == "vital-status"
    assert result["outcome_properties"]["patient_reported"] is False
    assert result["outcome_properties"]["safety_harm"] is False
    assert any("contradicted" in error.lower() for error in result["errors"])


def test_pfs_scope_ignores_unrelated_patient_reported_and_safety_mentions():
    result = outcome_resolver_node(
        {
            "outcome": "Progression-free survival",
            "outcome_code": "PFS",
            "evidence": {
                "abstract": {
                    "text": (
                        "Radiographic progression-free survival was defined as "
                        "time to radiographic progression or death."
                    ),
                    "tables": [],
                    "source": "primary",
                },
                "methods": {
                    "text": (
                        "Quality of life questionnaires and treatment-emergent "
                        "adverse events were collected as separate outcomes."
                    ),
                    "tables": [],
                    "source": "primary",
                },
            },
            "errors": [],
        }
    )

    assert result["outcome_type"] == "clinician-composite"
    assert result["outcome_properties"]["composite"] is True
    assert result["outcome_properties"]["patient_reported"] is False
    assert result["outcome_properties"]["safety_harm"] is False


def test_ae_scope_ignores_unrelated_survival_and_imaging_mentions():
    result = outcome_resolver_node(
        {
            "outcome": "Adverse events",
            "outcome_code": "AE",
            "evidence": {
                "abstract": {
                    "text": ("Adverse events were graded according to CTCAE criteria."),
                    "tables": [],
                    "source": "primary",
                },
                "results": {
                    "text": (
                        "Overall survival and imaging-assessed progression-free "
                        "survival were also reported."
                    ),
                    "tables": [],
                    "source": "primary",
                },
            },
            "errors": [],
        }
    )

    assert result["outcome_type"] == "clinician-graded"
    assert result["outcome_properties"]["safety_harm"] is True
    assert result["outcome_properties"]["objective_event"] is False


def test_invalid_llm_resolution_falls_back_with_warning():
    result = outcome_resolver_node(
        {
            "outcome": "Overall survival",
            "outcome_code": "OS",
            "evidence": {
                "abstract": {
                    "text": "Overall survival was death from any cause.",
                    "tables": [],
                    "source": "primary",
                }
            },
            "outcome_resolution_llm_response": "not json",
            "errors": [],
        }
    )

    assert result["outcome_type"] == "vital-status"
    assert result["outcome_resolution"]["resolver"] == "deterministic_fallback"
    assert "deterministic fallback" in result["outcome_resolution"]["warnings"][0]


def test_contradictory_llm_resolution_is_guardrailed_to_death_scope():
    result = outcome_resolver_node(
        {
            "outcome": "Overall survival",
            "outcome_code": "OS",
            "evidence": {
                "abstract": {
                    "text": "Overall survival was defined as death from any cause.",
                    "tables": [],
                    "source": "primary",
                }
            },
            "outcome_resolution_llm_response": (
                '{"outcome_type":"patient-reported","properties":{'
                '"objective_event":false,'
                '"clinician_judged":true,'
                '"patient_reported":true,'
                '"composite":false,'
                '"time_to_event":false,'
                '"safety_harm":true,'
                '"lab_or_imaging_threshold":false,'
                '"blinded_adjudication":false'
                "}}"
            ),
            "errors": [],
        }
    )

    assert result["outcome_type"] == "vital-status"
    assert result["outcome_properties"]["patient_reported"] is False
    assert result["outcome_properties"]["safety_harm"] is False
    assert any(
        "Guardrail cleared" in warning
        for warning in result["outcome_resolution"]["warnings"]
    )
