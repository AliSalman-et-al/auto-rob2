from types import SimpleNamespace

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


def _mock_llm(response: dict, calls: list[dict]):
    def fake_call(state, prompt, node_name, **_kwargs):
        calls.append({"state": state, "prompt": prompt, "node_name": node_name})
        artifact = response
        return SimpleNamespace(
            artifact=artifact,
            log=[
                {
                    "node": node_name,
                    "prompt_length_chars": len(prompt),
                    "response_length_chars": len(str(response)),
                    "latency_ms": 0,
                    "cache_hit": False,
                }
            ],
            status="validated",
        )

    return fake_call


def _response(
    outcome_type: str,
    support_level: str,
    rationale: str,
    props: dict[str, bool],
    quote: str,
    *,
    definition: str = "",
    aliases: list[str] | None = None,
    uncertainty: bool = False,
):
    outcome_properties = {
        field: props.get(field, False)
        for field in outcome_resolver.PROPERTY_FIELDS
    }
    return {
        "schema_version": "outcome-normalization-v1",
        "outcome_type": outcome_type,
        "normalized_definition": definition or "Assessed outcome definition.",
        "aliases": aliases or [],
        "outcome_properties": outcome_properties,
        "support": {
            "support_level": support_level,
            "support_rationale": rationale,
            "quotes": [{"quote": quote, "source": "d4_outcome_meas"}] if quote else [],
            "constraints": [],
        },
        "uncertainty": uncertainty,
    }


def test_resolver_uses_assessed_outcome_bound_llm_evidence(monkeypatch):
    calls = []
    monkeypatch.setattr(
        outcome_resolver,
        "call_json_contract_llm",
        _mock_llm(
            _response(
                "vital-status",
                "strong",
                "The assessed outcome is OS and the measurement quote defines it as death from any cause.",
                {
                    "time_to_event": True,
                    "objective_event": True,
                },
                "Overall survival was defined as time from randomization to death from any cause.",
                definition="Time from randomization to death from any cause.",
                aliases=["OS", "overall survival"],
            ),
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
    assert result["outcome_normalization_artifact"] == {
        "artifact_id": "outcome-normalization:Overall survival",
        "schema_version": "outcome-normalization-v1",
        "outcome": "Overall survival",
        "normalized_definition": "Time from randomization to death from any cause.",
        "aliases": ["OS", "overall survival"],
        "outcome_type": "vital-status",
        "outcome_properties": result["outcome_properties"],
        "binding_support": result["outcome_classification_support"],
        "auto_accept_blocked": False,
        "uncertainty": False,
    }
    assert "Progression-free survival" in calls[0]["prompt"]
    assert "normalized_definition" in calls[0]["prompt"]
    assert "aliases" in calls[0]["prompt"]
    assert "uncertainty" in calls[0]["prompt"]
    assert calls[0]["node_name"] == "outcome_resolver"


def test_invalid_llm_output_falls_back_to_unsupported_without_regex_semantics(
    monkeypatch,
):
    monkeypatch.setattr(
        outcome_resolver,
        "call_json_contract_llm",
        _mock_llm(
            {"outcome_type": "vital-status"},
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
    assert result["outcome_normalization_artifact"]["auto_accept_blocked"] is True
    assert result["outcome_normalization_artifact"]["uncertainty"] is True
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
        "call_json_contract_llm",
        _mock_llm(
            _response(
                "patient-reported",
                "strong",
                "Quote says participants reported symptoms.",
                {"patient_reported": True},
                "Participants self-reported symptom severity daily.",
            ),
            [],
        ),
    )

    result = outcome_resolver.outcome_resolver_node(
        _state("Symptom severity", "Clinicians graded symptom severity at each visit.")
    )

    assert result["outcome_type"] == "clinician-composite"
    assert result["outcome_classification_support"]["support_level"] == "unsupported"
    assert result["support_constraints"][0]["constraint_type"] == "quote_untraceable"
    assert result["outcome_normalization_artifact"]["auto_accept_blocked"] is True


def test_clinician_composite_progression_outcome(monkeypatch):
    quote = "Progression-free survival was time from randomization to radiographic progression or death."
    monkeypatch.setattr(
        outcome_resolver,
        "call_json_contract_llm",
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
        "call_json_contract_llm",
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
        "call_json_contract_llm",
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
