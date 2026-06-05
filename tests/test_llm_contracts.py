import json

from pydantic import BaseModel, Field

from rob2_pipeline.llm_contracts import call_json_contract_llm
from rob2_pipeline.nodes.domain_classifier import _answer_quote_is_packet_bound
from rob2_pipeline.providers.base import LLMResponse
from rob2_pipeline.trace import end_trace, get_current_trace, start_trace


class _ToyArtifact(BaseModel):
    schema_version: str
    answer: str = Field(min_length=1)


class _FakeProvider:
    def __init__(self, *responses: str):
        self._responses = list(responses)
        self.prompts: list[str] = []

    @property
    def model_id(self) -> str:
        return "fake-model"

    def complete(self, system: str, user: str) -> LLMResponse:
        self.prompts.append(user)
        return LLMResponse(
            content=self._responses.pop(0),
            model="fake-model",
            input_tokens=10,
            output_tokens=5,
            latency_ms=1.0,
        )


def test_json_contract_validates_against_local_schema(monkeypatch):
    provider = _FakeProvider(json.dumps({"schema_version": "toy-v1", "answer": "Y"}))
    monkeypatch.setattr("rob2_pipeline.llm_contracts.build_provider", lambda: provider)

    result = call_json_contract_llm(
        {},
        "Return a toy artifact.",
        "toy_node",
        schema_model=_ToyArtifact,
        schema_version="toy-v1",
        prompt_version="toy-prompt-v1",
        fallback_factory=lambda reason: {"schema_version": "toy-v1", "answer": "NI"},
    )

    assert result.artifact == {"schema_version": "toy-v1", "answer": "Y"}
    assert result.status == "validated"
    assert "JSON schema:" in provider.prompts[0]
    assert "Do not omit required fields" in provider.prompts[0]
    assert "Required schema_version: toy-v1" in provider.prompts[0]
    assert result.log[0]["validation_status"] == "validated"


def test_json_contract_retries_then_records_deterministic_fallback(monkeypatch):
    provider = _FakeProvider(
        json.dumps({"schema_version": "toy-v1"}),
        "still not json",
    )
    monkeypatch.setattr("rob2_pipeline.llm_contracts.build_provider", lambda: provider)

    result = call_json_contract_llm(
        {},
        "Return a toy artifact.",
        "toy_node",
        schema_model=_ToyArtifact,
        schema_version="toy-v1",
        prompt_version="toy-prompt-v1",
        fallback_factory=lambda reason: {
            "schema_version": "toy-v1",
            "answer": "NI",
            "fallback_reason": reason,
        },
    )

    assert len(provider.prompts) == 2
    assert "Your previous JSON response for toy_node was invalid" in provider.prompts[1]
    assert result.status == "fallback"
    assert result.artifact["answer"] == "NI"
    assert result.artifact["fallback_reason"]
    assert result.log[0]["validation_status"] == "fallback"
    assert result.log[0]["fallback_artifact"]["answer"] == "NI"


def test_json_contract_trace_records_contract_metadata(monkeypatch):
    provider = _FakeProvider(json.dumps({"schema_version": "toy-v1", "answer": "Y"}))
    monkeypatch.setattr("rob2_pipeline.llm_contracts.build_provider", lambda: provider)
    start_trace(trial="trial", outcome="outcome")

    try:
        call_json_contract_llm(
            {},
            "Return a toy artifact.",
            "toy_node",
            schema_model=_ToyArtifact,
            schema_version="toy-v1",
            prompt_version="toy-prompt-v1",
            fallback_factory=lambda reason: {
                "schema_version": "toy-v1",
                "answer": "NI",
            },
        )
        trace = get_current_trace()
    finally:
        end_trace()

    assert trace is not None
    call = trace.llm_calls[0]
    assert call.provider == "openrouter"
    assert call.model == "fake-model"
    assert call.prompt_version == "toy-prompt-v1"
    assert call.schema_version == "toy-v1"
    assert call.parse_status == "parsed"
    assert call.validation_status == "validated"
    assert call.input_tokens == 10
    assert call.output_tokens == 5


def test_packet_quote_validator_accepts_traceable_ellipsis_fragments():
    packet = {
        "sources": [
            {
                "text": (
                    "Registered outcomes from ClinicalTrials.gov: PRIMARY: "
                    "Radiographic Progression-Free Survival (rPFS) Based on "
                    "Independent Central Review. ITT population was randomized."
                )
            }
        ],
        "candidate_facts": [],
    }
    answer = {
        "quote": (
            "Registered outcomes from ClinicalTrials.gov: PRIMARY: "
            "Radiographic Progression-Free Survival (rPFS) ... ITT population"
        )
    }

    assert _answer_quote_is_packet_bound(answer, packet)
