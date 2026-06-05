"""Per-run trace collector for diagnostic analysis (Option C: LLM I/O only).

Captures full LLM system/user prompts and responses so failures can be
categorized as RAG-miss vs LLM-miss without re-running the pipeline.
Chunk retrieval data lives in Ali's rag_chunk_metadata state channel
(emitted as rag_sources in the per-trial JSON output) and is read from
there at categorization time, so we do not duplicate it here.

Uses a module-level current-trace global because the pipeline runs
sequentially in a single process; threading a trace object through every
LangGraph node would require touching ~10 files for no functional gain.
"""

import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class LlmNodeTrace:
    node: str
    system_prompt: str
    user_prompt: str
    response: str
    provider: str | None
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    cached: bool | None
    latency_ms: int
    cache_hit: bool
    parse_error: str | None
    parsed_answers: dict[str, Any] | None
    is_repair: bool = False
    # Chain-of-thought from providers that emit it (e.g. gpt-oss via OpenRouter).
    # Used at categorization time to tell whether the model reasoned over the
    # retrieved chunks or ignored them. None when the provider does not emit one.
    reasoning_content: str | None = None
    prompt_version: str | None = None
    schema_version: str | None = None
    parse_status: str | None = None
    validation_status: str | None = None
    validation_error: str | None = None
    fallback_artifact: dict[str, Any] | None = None


@dataclass
class PipelineNodeSpan:
    node: str
    status: str
    timestamp_start: str
    timestamp_end: str | None = None
    duration_ms: int | None = None
    error: str | None = None


@dataclass
class PipelineTrace:
    trial: str
    outcome: str | None
    timestamp_start: str
    timestamp_end: str | None = None
    llm_calls: list[LlmNodeTrace] = field(default_factory=list)
    node_spans: list[PipelineNodeSpan] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str)

    def write(self, output_dir: Path | str) -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{self.trial}_trace.json"
        path.write_text(self.to_json(), encoding="utf-8")
        return path


_CURRENT_TRACE: PipelineTrace | None = None


def start_trace(trial: str, outcome: str | None) -> PipelineTrace:
    global _CURRENT_TRACE
    _CURRENT_TRACE = PipelineTrace(
        trial=trial,
        outcome=outcome,
        timestamp_start=datetime.now(timezone.utc).isoformat(),
    )
    return _CURRENT_TRACE


def end_trace() -> PipelineTrace | None:
    global _CURRENT_TRACE
    trace = _CURRENT_TRACE
    if trace is not None:
        trace.timestamp_end = datetime.now(timezone.utc).isoformat()
    _CURRENT_TRACE = None
    return trace


def get_current_trace() -> PipelineTrace | None:
    return _CURRENT_TRACE


def append_llm_call(
    node: str,
    system_prompt: str,
    user_prompt: str,
    response: str,
    model: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
    cached: bool | None,
    latency_ms: int,
    cache_hit: bool,
    parse_error: str | None = None,
    parsed_answers: dict[str, Any] | None = None,
    is_repair: bool = False,
    reasoning_content: str | None = None,
    provider: str | None = None,
    prompt_version: str | None = None,
    schema_version: str | None = None,
    parse_status: str | None = None,
    validation_status: str | None = None,
    validation_error: str | None = None,
    fallback_artifact: dict[str, Any] | None = None,
) -> None:
    entry = LlmNodeTrace(
        node=node,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response=response,
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached=cached,
        latency_ms=latency_ms,
        cache_hit=cache_hit,
        parse_error=parse_error,
        parsed_answers=parsed_answers,
        is_repair=is_repair,
        reasoning_content=reasoning_content,
        prompt_version=prompt_version,
        schema_version=schema_version,
        parse_status=parse_status,
        validation_status=validation_status,
        validation_error=validation_error,
        fallback_artifact=fallback_artifact,
    )
    _write_llm_tap(entry)
    if _CURRENT_TRACE is None:
        return
    _CURRENT_TRACE.llm_calls.append(entry)


def _write_llm_tap(entry: LlmNodeTrace) -> None:
    tap_dir_raw = os.environ.get("ROB2_LLM_TAP_DIR")
    if not tap_dir_raw:
        return
    tap_dir = Path(tap_dir_raw).expanduser()
    tap_dir.mkdir(parents=True, exist_ok=True)
    tap_path = tap_dir / "llm_calls.jsonl"
    with tap_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(entry), ensure_ascii=False, default=str) + "\n")


@contextmanager
def record_node_span(node_name: str):
    trace = _CURRENT_TRACE
    if trace is None:
        yield
        return

    start = datetime.now(timezone.utc)
    start_counter = time.perf_counter()
    span = PipelineNodeSpan(
        node=node_name,
        status="ok",
        timestamp_start=start.isoformat(),
    )
    trace.node_spans.append(span)
    try:
        yield
    except Exception as exc:
        span.status = "error"
        span.error = str(exc)
        raise
    finally:
        end = datetime.now(timezone.utc)
        span.timestamp_end = end.isoformat()
        span.duration_ms = int((time.perf_counter() - start_counter) * 1000)
