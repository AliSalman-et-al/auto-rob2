"""Schema-first LLM call contract helpers."""

import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from rob2_pipeline import config
from rob2_pipeline.config import build_provider
from rob2_pipeline.trace import append_llm_call
from rob2_pipeline.types import LLMCallLogEntry


JSON_SYSTEM_MESSAGE = (
    "You are an expert systematic reviewer applying the Cochrane Risk of Bias 2 "
    "tool. Respond only with JSON that matches the requested schema. Do not add "
    "preamble, explanation, or markdown code fences."
)


@dataclass(frozen=True)
class JsonContractResult:
    artifact: dict[str, Any]
    log: list[LLMCallLogEntry]
    status: str
    failure_reason: str | None = None


def call_json_contract_llm(
    state: dict,
    prompt: str,
    node_name: str,
    *,
    schema_model: type[BaseModel],
    schema_version: str,
    prompt_version: str,
    fallback_factory: Callable[[str], dict[str, Any]],
    max_attempts: int = 2,
) -> JsonContractResult:
    """Call an LLM and validate JSON locally against a Pydantic schema.

    Provider-native JSON/schema features may be added by adapters later, but
    this helper remains the local validation authority for accepted artifacts.
    """

    del state  # Reserved for future state-aware contract metadata.
    provider = build_provider()
    attempts: list[dict[str, Any]] = []
    current_prompt = prompt
    last_failure = "JSON contract validation did not run."

    for attempt_index in range(max(1, max_attempts)):
        started = time.perf_counter()
        response_obj = provider.complete(system=JSON_SYSTEM_MESSAGE, user=current_prompt)
        latency_ms = int((time.perf_counter() - started) * 1000)
        parse_status = "not_parsed"
        validation_status = "not_validated"
        parse_error = None
        validation_error = None
        artifact: dict[str, Any] | None = None

        try:
            payload = json.loads(response_obj.content)
            parse_status = "parsed"
        except json.JSONDecodeError as exc:
            payload = None
            parse_status = "parse_failed"
            parse_error = str(exc)
            last_failure = f"JSON parse failed: {exc}"

        if payload is not None:
            try:
                validated = schema_model.model_validate(payload)
                artifact = validated.model_dump(mode="json")
                validation_status = "validated"
            except ValidationError as exc:
                validation_status = "validation_failed"
                validation_error = str(exc)
                last_failure = f"Schema validation failed: {exc}"

        attempts.append(
            {
                "response_obj": response_obj,
                "prompt": current_prompt,
                "latency_ms": latency_ms,
                "parse_status": parse_status,
                "validation_status": validation_status,
                "parse_error": parse_error,
                "validation_error": validation_error,
                "artifact": artifact,
                "is_repair": attempt_index > 0,
            }
        )
        _append_contract_trace(
            node_name=node_name,
            prompt=current_prompt,
            response=response_obj.content,
            response_obj=response_obj,
            latency_ms=latency_ms,
            prompt_version=prompt_version,
            schema_version=schema_version,
            parse_status=parse_status,
            validation_status=validation_status,
            parse_error=parse_error,
            validation_error=validation_error,
            is_repair=attempt_index > 0,
        )

        if artifact is not None:
            return JsonContractResult(
                artifact=artifact,
                log=[
                    _contract_log_entry(
                        node_name=node_name,
                        prompt=prompt,
                        response=response_obj.content,
                        response_obj=response_obj,
                        latency_ms=sum(a["latency_ms"] for a in attempts),
                        prompt_version=prompt_version,
                        schema_version=schema_version,
                        parse_status=parse_status,
                        validation_status=validation_status,
                        attempts=attempts,
                    )
                ],
                status="validated",
            )

        current_prompt = _repair_prompt(
            node_name=node_name,
            original_prompt=prompt,
            schema_model=schema_model,
            failure_reason=last_failure,
        )

    fallback_artifact = fallback_factory(last_failure)
    return JsonContractResult(
        artifact=fallback_artifact,
        log=[
            _contract_log_entry(
                node_name=node_name,
                prompt=prompt,
                response=attempts[-1]["response_obj"].content,
                response_obj=attempts[-1]["response_obj"],
                latency_ms=sum(a["latency_ms"] for a in attempts),
                prompt_version=prompt_version,
                schema_version=schema_version,
                parse_status=attempts[-1]["parse_status"],
                validation_status="fallback",
                attempts=attempts,
                fallback_artifact=fallback_artifact,
                failure_reason=last_failure,
            )
        ],
        status="fallback",
        failure_reason=last_failure,
    )


def _repair_prompt(
    *,
    node_name: str,
    original_prompt: str,
    schema_model: type[BaseModel],
    failure_reason: str,
) -> str:
    return (
        f"Your previous JSON response for {node_name} was invalid: {failure_reason}.\n"
        "Return only JSON matching this local schema.\n\n"
        f"JSON schema:\n{json.dumps(schema_model.model_json_schema(), indent=2)}\n\n"
        f"Original prompt:\n{original_prompt}"
    )


def _append_contract_trace(
    *,
    node_name: str,
    prompt: str,
    response: str,
    response_obj: Any,
    latency_ms: int,
    prompt_version: str,
    schema_version: str,
    parse_status: str,
    validation_status: str,
    parse_error: str | None,
    validation_error: str | None,
    is_repair: bool,
) -> None:
    append_llm_call(
        node=node_name,
        system_prompt=JSON_SYSTEM_MESSAGE,
        user_prompt=prompt,
        response=response,
        model=response_obj.model,
        input_tokens=response_obj.input_tokens,
        output_tokens=response_obj.output_tokens,
        cached=response_obj.cached,
        latency_ms=latency_ms,
        cache_hit=False,
        parse_error=parse_error,
        parsed_answers=None,
        is_repair=is_repair,
        reasoning_content=response_obj.reasoning_content,
        provider=config.PROVIDER_NAME,
        prompt_version=prompt_version,
        schema_version=schema_version,
        parse_status=parse_status,
        validation_status=validation_status,
        validation_error=validation_error,
    )


def _contract_log_entry(
    *,
    node_name: str,
    prompt: str,
    response: str,
    response_obj: Any,
    latency_ms: int,
    prompt_version: str,
    schema_version: str,
    parse_status: str,
    validation_status: str,
    attempts: list[dict[str, Any]],
    fallback_artifact: dict[str, Any] | None = None,
    failure_reason: str | None = None,
) -> LLMCallLogEntry:
    entry: LLMCallLogEntry = {
        "node": node_name,
        "prompt_length_chars": len(prompt),
        "response_length_chars": len(response),
        "latency_ms": latency_ms,
        "cache_hit": False,
        "model": response_obj.model,
        "input_tokens": response_obj.input_tokens,
        "output_tokens": response_obj.output_tokens,
        "cached": response_obj.cached,
        "provider": config.PROVIDER_NAME,
        "prompt_version": prompt_version,
        "schema_version": schema_version,
        "parse_status": parse_status,
        "validation_status": validation_status,
        "attempts": [
            {
                "attempt": index + 1,
                "parse_status": attempt["parse_status"],
                "validation_status": attempt["validation_status"],
                "parse_error": attempt["parse_error"],
                "validation_error": attempt["validation_error"],
                "is_repair": attempt["is_repair"],
            }
            for index, attempt in enumerate(attempts)
        ],
    }
    if fallback_artifact is not None:
        entry["fallback_artifact"] = fallback_artifact
    if failure_reason:
        entry["failure_reason"] = failure_reason
    return entry
