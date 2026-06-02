from rob2_pipeline.models import empty_paper_evidence
from rob2_pipeline.nodes import outcome_resolver


def _state(outcome: str, measurement_text: str = ""):
    evidence = empty_paper_evidence()
    evidence["d4_outcome_meas"]["text"] = measurement_text
    evidence["results"]["text"] = (
        "Progression-free survival was a composite endpoint assessed by imaging. "
        "Patient-reported quality of life was collected as a secondary endpoint."
    )
    return {
        "outcome": outcome,
        "outcome_type": "clinician-composite",
        "numerical_result": "",
        "evidence": evidence,
        "support_constraints": [],
        "errors": [],
    }


def _mock_llm(response: str, calls: list[dict]):
    def fake_call(state, prompt, node_name):
        calls.append({"state": state, "prompt": prompt, "node_name": node_name})
        return (
            response,
            [
                {
                    "node": node_name,
                    "prompt_length_chars": len(prompt),
                    "response_length_chars": len(response),
                    "latency_ms": 0,
                    "cache_hit": False,
                }
            ],
            None,
        )

    return fake_call


def _response(
    outcome_type: str,
    support_level: str,
    rationale: str,
    props: dict[str, bool],
    quote: str,
):
    def flag(name: str) -> str:
        return str(props.get(name, False)).lower()

    return f"""
<outcome_resolution>
  <outcome_type>{outcome_type}</outcome_type>
  <support_level>{support_level}</support_level>
  <support_rationale>{rationale}</support_rationale>
  <properties>
    <patient_reported>{flag("patient_reported")}</patient_reported>
    <safety_harm>{flag("safety_harm")}</safety_harm>
    <time_to_event>{flag("time_to_event")}</time_to_event>
    <death_only_objective_event>{flag("death_only_objective_event")}</death_only_objective_event>
    <composite>{flag("composite")}</composite>
    <lab_or_imaging_threshold>{flag("lab_or_imaging_threshold")}</lab_or_imaging_threshold>
    <blinded_adjudication>{flag("blinded_adjudication")}</blinded_adjudication>
    <objective_event>{flag("objective_event")}</objective_event>
    <clinician_judged>{flag("clinician_judged")}</clinician_judged>
  </properties>
  <quotes>
    <quote source="d4_outcome_meas">{quote}</quote>
  </quotes>
  <constraints></constraints>
</outcome_resolution>
"""


def test_resolver_uses_assessed_outcome_bound_llm_evidence(monkeypatch):
    calls = []
    monkeypatch.setattr(
        outcome_resolver,
        "call_node_llm",
        _mock_llm(
            """
<outcome_resolution>
  <outcome_type>vital-status</outcome_type>
  <support_level>strong</support_level>
  <support_rationale>The assessed outcome is OS and the measurement quote defines it as death from any cause.</support_rationale>
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
    <quote source="d4_outcome_meas">Overall survival was defined as time from randomization to death from any cause.</quote>
  </quotes>
  <constraints></constraints>
</outcome_resolution>
""",
            calls,
        ),
    )

    result = outcome_resolver.outcome_resolver_node(
        _state(
            "Overall survival",
            "Overall survival was defined as time from randomization to death from any cause.",
        )
    )

    assert result["outcome_type"] == "vital-status"
    assert result["outcome_properties"]["objective_event"] is True
    assert result["outcome_properties"]["composite"] is False
    assert result["outcome_classification_support"]["support_level"] == "strong"
    assert result["outcome_classification_support"]["quotes"] == [
        {
            "quote": "Overall survival was defined as time from randomization to death from any cause.",
            "source": "d4_outcome_meas",
        }
    ]
    assert "Progression-free survival" in calls[0]["prompt"]
    assert calls[0]["node_name"] == "outcome_resolver"


def test_invalid_llm_output_falls_back_to_unsupported_without_regex_semantics(
    monkeypatch,
):
    monkeypatch.setattr(
        outcome_resolver,
        "call_node_llm",
        _mock_llm(
            "<outcome_resolution><outcome_type>vital-status</outcome_type></outcome_resolution>",
            [],
        ),
    )

    result = outcome_resolver.outcome_resolver_node(
        _state(
            "Overall survival",
            "Overall survival was defined as time from randomization to death from any cause.",
        )
    )

    assert result["outcome_type"] == "clinician-composite"
    assert result["outcome_properties"] == outcome_resolver.DEFAULT_OUTCOME_PROPERTIES
    assert result["outcome_classification_support"]["support_level"] == "unsupported"
    assert (
        result["support_constraints"][0]["constraint_type"]
        == "missing_required_evidence"
    )
    assert "outcome resolver output" in result["support_constraints"][0]["reason"]


def test_untraceable_quote_creates_constraint_and_unsupported_classification(
    monkeypatch,
):
    monkeypatch.setattr(
        outcome_resolver,
        "call_node_llm",
        _mock_llm(
            """
<outcome_resolution>
  <outcome_type>patient-reported</outcome_type>
  <support_level>strong</support_level>
  <support_rationale>Quote says participants reported symptoms.</support_rationale>
  <properties>
    <patient_reported>true</patient_reported>
    <safety_harm>false</safety_harm>
    <time_to_event>false</time_to_event>
    <death_only_objective_event>false</death_only_objective_event>
    <composite>false</composite>
    <lab_or_imaging_threshold>false</lab_or_imaging_threshold>
    <blinded_adjudication>false</blinded_adjudication>
    <objective_event>false</objective_event>
    <clinician_judged>false</clinician_judged>
  </properties>
  <quotes>
    <quote source="d4_outcome_meas">Participants self-reported symptom severity daily.</quote>
  </quotes>
  <constraints></constraints>
</outcome_resolution>
""",
            [],
        ),
    )

    result = outcome_resolver.outcome_resolver_node(
        _state("Symptom severity", "Clinicians graded symptom severity at each visit.")
    )

    assert result["outcome_type"] == "clinician-composite"
    assert result["outcome_classification_support"]["support_level"] == "unsupported"
    assert result["support_constraints"][0]["constraint_type"] == "quote_untraceable"


def test_clinician_composite_progression_outcome(monkeypatch):
    quote = "Progression-free survival was time from randomization to radiographic progression or death."
    monkeypatch.setattr(
        outcome_resolver,
        "call_node_llm",
        _mock_llm(
            _response(
                "clinician-composite",
                "strong",
                "The assessed outcome combines progression and death.",
                {
                    "time_to_event": True,
                    "composite": True,
                    "lab_or_imaging_threshold": True,
                    "clinician_judged": True,
                },
                quote,
            ),
            [],
        ),
    )

    result = outcome_resolver.outcome_resolver_node(
        _state("Progression-free survival", quote)
    )

    assert result["outcome_type"] == "clinician-composite"
    assert result["outcome_properties"]["composite"] is True
    assert result["outcome_properties"]["lab_or_imaging_threshold"] is True


def test_safety_outcome_resolves_to_clinician_graded(monkeypatch):
    quote = (
        "Serious adverse events were graded by study clinicians using CTCAE criteria."
    )
    monkeypatch.setattr(
        outcome_resolver,
        "call_node_llm",
        _mock_llm(
            _response(
                "clinician-graded",
                "strong",
                "The assessed outcome is clinician-graded safety harm.",
                {"safety_harm": True, "clinician_judged": True},
                quote,
            ),
            [],
        ),
    )

    result = outcome_resolver.outcome_resolver_node(
        _state("Serious adverse events", quote)
    )

    assert result["outcome_type"] == "clinician-graded"
    assert result["outcome_properties"]["safety_harm"] is True


def test_patient_reported_outcome_resolves_to_patient_reported(monkeypatch):
    quote = "Participants completed a quality-of-life questionnaire at each visit."
    monkeypatch.setattr(
        outcome_resolver,
        "call_node_llm",
        _mock_llm(
            _response(
                "patient-reported",
                "strong",
                "The measurement quote shows participant-reported questionnaire data.",
                {"patient_reported": True},
                quote,
            ),
            [],
        ),
    )

    result = outcome_resolver.outcome_resolver_node(_state("Quality of life", quote))

    assert result["outcome_type"] == "patient-reported"
    assert result["outcome_properties"]["patient_reported"] is True
