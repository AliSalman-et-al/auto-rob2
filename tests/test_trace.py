import json
from pathlib import Path

import pytest

from rob2_pipeline import trace as trace_module
from rob2_pipeline.trace import (
    append_llm_call,
    end_trace,
    get_current_trace,
    record_node_span,
    start_trace,
)


@pytest.fixture(autouse=True)
def _reset_current_trace():
    trace_module._CURRENT_TRACE = None
    yield
    trace_module._CURRENT_TRACE = None


def test_start_trace_creates_active_trace():
    trace = start_trace(trial="CHAARTED", outcome="Overall Survival")
    assert trace is get_current_trace()
    assert trace.trial == "CHAARTED"
    assert trace.outcome == "Overall Survival"
    assert trace.timestamp_start
    assert trace.timestamp_end is None
    assert trace.llm_calls == []


def test_end_trace_stamps_end_and_clears_current():
    start_trace(trial="X", outcome=None)
    trace = end_trace()
    assert trace is not None
    assert trace.timestamp_end
    assert get_current_trace() is None


def test_end_trace_returns_none_when_no_active_trace():
    assert end_trace() is None


def test_append_llm_call_records_full_io():
    start_trace(trial="T", outcome="OS")
    append_llm_call(
        node="domain1",
        system_prompt="sys",
        user_prompt="user prompt",
        response="<answer>Y</answer>",
        model="claude-test",
        input_tokens=10,
        output_tokens=4,
        cached=False,
        latency_ms=123,
        cache_hit=False,
        parse_error=None,
        parsed_answers={"1.1": {"answer": "Y"}},
    )
    trace = get_current_trace()
    assert len(trace.llm_calls) == 1
    call = trace.llm_calls[0]
    assert call.node == "domain1"
    assert call.system_prompt == "sys"
    assert call.user_prompt == "user prompt"
    assert call.response == "<answer>Y</answer>"
    assert call.model == "claude-test"
    assert call.latency_ms == 123
    assert call.parsed_answers == {"1.1": {"answer": "Y"}}


def test_append_llm_call_silently_ignores_when_no_active_trace():
    append_llm_call(
        node="domain1",
        system_prompt="x",
        user_prompt="y",
        response="z",
        model=None,
        input_tokens=None,
        output_tokens=None,
        cached=None,
        latency_ms=0,
        cache_hit=False,
    )


def test_pipeline_trace_write_writes_named_file(tmp_path: Path):
    trace = start_trace(trial="PEACE-1", outcome="PFS")
    append_llm_call(
        node="rct_screener",
        system_prompt="sys",
        user_prompt="prompt",
        response="ok",
        model="m",
        input_tokens=1,
        output_tokens=1,
        cached=False,
        latency_ms=5,
        cache_hit=False,
    )
    out_path = trace.write(tmp_path)
    assert out_path == tmp_path / "PEACE-1_trace.json"

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["trial"] == "PEACE-1"
    assert data["outcome"] == "PFS"
    assert len(data["llm_calls"]) == 1
    assert data["llm_calls"][0]["node"] == "rct_screener"
    assert data["node_spans"] == []


def test_record_node_span_records_ok_and_error_spans():
    start_trace(trial="T", outcome="OS")

    with record_node_span("screen"):
        pass

    with pytest.raises(RuntimeError, match="boom"):
        with record_node_span("extract"):
            raise RuntimeError("boom")

    trace = get_current_trace()
    assert trace is not None
    assert len(trace.node_spans) == 2

    ok_span, error_span = trace.node_spans
    assert ok_span.node == "screen"
    assert ok_span.status == "ok"
    assert ok_span.error is None
    assert ok_span.duration_ms >= 0
    assert ok_span.timestamp_start
    assert ok_span.timestamp_end

    assert error_span.node == "extract"
    assert error_span.status == "error"
    assert error_span.error == "boom"
    assert error_span.duration_ms >= 0
    assert error_span.timestamp_start
    assert error_span.timestamp_end


def test_pipeline_trace_serialization_includes_node_spans():
    trace = start_trace(trial="SER", outcome=None)
    with record_node_span("screen"):
        pass

    data = json.loads(trace.to_json())
    assert "node_spans" in data
    assert len(data["node_spans"]) == 1
    assert data["node_spans"][0]["node"] == "screen"


def test_call_node_llm_appends_to_trace_on_cache_hit(monkeypatch):
    from rob2_pipeline.nodes import common as common_module

    monkeypatch.setattr(
        common_module, "read_cache", lambda node, prompt: "<cached>OK</cached>"
    )
    monkeypatch.setattr(
        common_module, "write_cache", lambda node, prompt, response: None
    )

    start_trace(trial="T", outcome="OS")
    common_module.call_node_llm(state={}, prompt="hi", node_name="screen")

    trace = get_current_trace()
    assert len(trace.llm_calls) == 1
    call = trace.llm_calls[0]
    assert call.node == "screen"
    assert call.user_prompt == "hi"
    assert call.response == "<cached>OK</cached>"
    assert call.cache_hit is True
    assert call.model is None


def test_call_node_llm_appends_to_trace_on_live_call(monkeypatch):
    from rob2_pipeline.nodes import common as common_module

    monkeypatch.setattr(common_module, "read_cache", lambda node, prompt: None)
    monkeypatch.setattr(
        common_module, "write_cache", lambda node, prompt, response: None
    )

    class FakeResponse:
        content = "<answer>Y</answer>"
        model = "fake-model"
        input_tokens = 12
        output_tokens = 4
        cached = False
        reasoning_content = None

    class FakeProvider:
        def complete(self, system, user):
            return FakeResponse()

    monkeypatch.setattr(common_module, "build_provider", lambda: FakeProvider())

    start_trace(trial="T", outcome="OS")
    common_module.call_node_llm(state={}, prompt="real prompt", node_name="domain1")

    trace = get_current_trace()
    assert len(trace.llm_calls) == 1
    call = trace.llm_calls[0]
    assert call.node == "domain1"
    assert call.user_prompt == "real prompt"
    assert call.response == "<answer>Y</answer>"
    assert call.model == "fake-model"
    assert call.input_tokens == 12
    assert call.output_tokens == 4
    assert call.cache_hit is False
    assert call.is_repair is False


def test_call_node_llm_traces_both_responses_on_parse_retry(monkeypatch):
    from rob2_pipeline.nodes import common as common_module

    monkeypatch.setattr(common_module, "read_cache", lambda node, prompt: None)
    monkeypatch.setattr(
        common_module, "write_cache", lambda node, prompt, response: None
    )

    responses = iter(
        [
            ("<malformed>broken xml without closing tag", "model-v1", 100, 50),
            (
                "<sq_1_1><answer>Y</answer><quote>q</quote><justification>j</justification></sq_1_1>",
                "model-v2",
                80,
                40,
            ),
        ]
    )

    class FakeResponse:
        def __init__(self, content, model, in_tok, out_tok):
            self.content = content
            self.model = model
            self.input_tokens = in_tok
            self.output_tokens = out_tok
            self.cached = False
            self.reasoning_content = None

    class FakeProvider:
        def complete(self, system, user):
            content, model, in_tok, out_tok = next(responses)
            return FakeResponse(content, model, in_tok, out_tok)

    monkeypatch.setattr(common_module, "build_provider", lambda: FakeProvider())

    parse_calls = []

    def fake_parse_fn(text, sq_ids):
        parse_calls.append(text)
        if "<malformed>" in text:
            raise ValueError("synthetic parse failure")
        return {
            "1.1": {
                "answer": "Y",
                "quote": "q",
                "justification": "j",
                "uncertainty_flag": "NORMAL",
            }
        }

    start_trace(trial="T", outcome="OS")
    response, log, parsed = common_module.call_node_llm(
        state={},
        prompt="domain prompt",
        node_name="domain1_sq11",
        parse_fn=fake_parse_fn,
        parse_sq_ids=["1.1"],
    )

    assert parsed is not None
    assert "<sq_1_1>" in response

    trace = get_current_trace()
    assert len(trace.llm_calls) == 2, (
        "both malformed-first and repair-success should be traced"
    )

    first, second = trace.llm_calls
    assert first.is_repair is False
    assert first.response == "<malformed>broken xml without closing tag"
    assert first.parse_error == "synthetic parse failure"
    assert first.parsed_answers is None
    assert first.model == "model-v1"

    assert second.is_repair is True
    assert "<sq_1_1>" in second.response
    assert second.parse_error is None
    assert second.parsed_answers == {
        "1.1": {
            "answer": "Y",
            "quote": "q",
            "justification": "j",
            "uncertainty_flag": "NORMAL",
        }
    }
    assert second.model == "model-v2"
    assert "previous response for domain1_sq11 was invalid" in second.user_prompt


def test_call_node_llm_returns_ni_fallback_when_repair_still_invalid(monkeypatch):
    from rob2_pipeline.nodes import common as common_module

    monkeypatch.setattr(common_module, "read_cache", lambda node, prompt: None)
    monkeypatch.setattr(
        common_module, "write_cache", lambda node, prompt, response: None
    )

    responses = iter(
        [
            ("", "model-v1", 100, 0),
            ("", "model-v2", 80, 0),
        ]
    )

    class FakeResponse:
        def __init__(self, content, model, in_tok, out_tok):
            self.content = content
            self.model = model
            self.input_tokens = in_tok
            self.output_tokens = out_tok
            self.cached = False
            self.reasoning_content = None

    class FakeProvider:
        def complete(self, system, user):
            content, model, in_tok, out_tok = next(responses)
            return FakeResponse(content, model, in_tok, out_tok)

    monkeypatch.setattr(common_module, "build_provider", lambda: FakeProvider())

    def parse_failure(_text, sq_ids):
        raise ValueError(f"Missing signaling question: {sq_ids[0]}")

    start_trace(trial="T", outcome="AE")
    response, log, parsed = common_module.call_node_llm(
        state={},
        prompt="domain2 analysis prompt",
        node_name="domain2_analysis",
        parse_fn=parse_failure,
        parse_sq_ids=["2.6", "2.7"],
    )

    assert response == ""
    assert parsed == {
        "2.6": {
            "answer": "NI",
            "quote": "No relevant text found",
            "justification": "LLM response could not be parsed after repair.",
            "uncertainty_flag": "HIGH",
            "support_level": "unsupported",
            "support_rationale": "Response could not be parsed.",
        },
        "2.7": {
            "answer": "NI",
            "quote": "No relevant text found",
            "justification": "LLM response could not be parsed after repair.",
            "uncertainty_flag": "HIGH",
            "support_level": "unsupported",
            "support_rationale": "Response could not be parsed.",
        },
    }
    assert log[0]["suspected_parse_failures"] == ["2.6", "2.7"]

    trace = get_current_trace()
    assert len(trace.llm_calls) == 2
    first, second = trace.llm_calls
    assert first.is_repair is False
    assert first.parse_error == "Missing signaling question: 2.6"
    assert first.parsed_answers is None
    assert second.is_repair is True
    assert second.parse_error == "Missing signaling question: 2.6"
    assert second.parsed_answers == parsed
