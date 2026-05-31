import time
from inspect import signature
from typing import Callable, Optional

from rob2_pipeline.cache import read_cache, write_cache
from rob2_pipeline.config import build_provider
from rob2_pipeline.trace import append_llm_call
from rob2_pipeline.types import LLMCallLogEntry
from rob2_pipeline.xml_parser import validate_sq_answers


SYSTEM_MESSAGE = (
    "You are an expert systematic reviewer applying the Cochrane Risk of Bias 2 "
    "(RoB 2) tool. Respond only in the XML format specified in the prompt. "
    "Do not add preamble, explanation, or markdown code fences around your XML."
)

NA_ANSWER = {
    "answer": "NA",
    "quote": "Not applicable",
    "justification": "Not applicable",
    "uncertainty_flag": "NORMAL",
    "support_level": "unsupported",
    "support_rationale": "Not applicable",
}


def _parse_failure_fallback(parse_sq_ids: list[str]) -> dict[str, dict]:
    return {
        sq_id: {
            "answer": "NI",
            "quote": "No relevant text found",
            "justification": "LLM response could not be parsed after repair.",
            "uncertainty_flag": "HIGH",
            "support_level": "unsupported",
            "support_rationale": "Response could not be parsed.",
        }
        for sq_id in parse_sq_ids
    }


def call_node_llm(
    state: dict,
    prompt: str,
    node_name: str,
    parse_fn: Optional[Callable[[str, list[str]], dict[str, dict]]] = None,
    parse_sq_ids: Optional[list[str]] = None,
    chunk_sources: Optional[list[str]] = None,
) -> tuple[str, list[LLMCallLogEntry], Optional[dict[str, dict]]]:
    """Call LLM for a node with optional cache and parser validation."""

    def _parse_and_validate(raw: str) -> Optional[dict[str, dict]]:
        if not (parse_fn and parse_sq_ids):
            return None
        parsed_local = parse_fn(raw, parse_sq_ids)
        validate_sq_answers(parsed_local, parse_sq_ids)
        return parsed_local

    cached = read_cache(node_name, prompt)
    log: list[LLMCallLogEntry] = []

    if cached is not None:
        log_entry = {
            "node": node_name,
            "prompt_length_chars": len(prompt),
            "response_length_chars": len(cached),
            "latency_ms": 0,
            "cache_hit": True,
        }
        if chunk_sources:
            log_entry["chunk_sources"] = chunk_sources
        parsed = _parse_and_validate(cached)
        log.append(log_entry)
        # Cache hit: we stored only the content, so reasoning_content is not
        # available on replay. Leave it as None.
        append_llm_call(
            node=node_name,
            system_prompt=SYSTEM_MESSAGE,
            user_prompt=prompt,
            response=cached,
            model=None,
            input_tokens=None,
            output_tokens=None,
            cached=True,
            latency_ms=0,
            cache_hit=True,
            parse_error=None,
            parsed_answers=parsed,
            reasoning_content=None,
        )
        return cached, log, parsed

    provider = build_provider()
    first_start = time.perf_counter()
    response_obj = provider.complete(system=SYSTEM_MESSAGE, user=prompt)
    response = response_obj.content
    first_latency_ms = int((time.perf_counter() - first_start) * 1000)

    parsed = None
    trace_appended = False
    cacheable_response = True
    suspected_parse_failures: list[str] = []
    if parse_fn and parse_sq_ids:
        try:
            parsed = _parse_and_validate(response)
        except Exception as exc:  # noqa: BLE001
            append_llm_call(
                node=node_name,
                system_prompt=SYSTEM_MESSAGE,
                user_prompt=prompt,
                response=response,
                model=response_obj.model,
                input_tokens=response_obj.input_tokens,
                output_tokens=response_obj.output_tokens,
                cached=response_obj.cached,
                latency_ms=first_latency_ms,
                cache_hit=False,
                parse_error=str(exc),
                parsed_answers=None,
                is_repair=False,
                reasoning_content=response_obj.reasoning_content,
            )
            repair_prompt = (
                f"Your previous response for {node_name} was invalid: {exc}. "
                "Return only well-formed XML in exactly the requested schema.\n\n"
                f"Original prompt:\n{prompt}"
            )
            repair_start = time.perf_counter()
            response_obj = provider.complete(system=SYSTEM_MESSAGE, user=repair_prompt)
            response = response_obj.content
            repair_latency_ms = int((time.perf_counter() - repair_start) * 1000)
            try:
                parsed = _parse_and_validate(response)
                repair_parse_error = None
            except Exception as repair_exc:  # noqa: BLE001
                parsed = _parse_failure_fallback(parse_sq_ids)
                suspected_parse_failures = list(parse_sq_ids)
                repair_parse_error = str(repair_exc)
                cacheable_response = False
            append_llm_call(
                node=node_name,
                system_prompt=SYSTEM_MESSAGE,
                user_prompt=repair_prompt,
                response=response,
                model=response_obj.model,
                input_tokens=response_obj.input_tokens,
                output_tokens=response_obj.output_tokens,
                cached=response_obj.cached,
                latency_ms=repair_latency_ms,
                cache_hit=False,
                parse_error=repair_parse_error,
                parsed_answers=parsed,
                is_repair=True,
                reasoning_content=response_obj.reasoning_content,
            )
            trace_appended = True

    if not trace_appended:
        append_llm_call(
            node=node_name,
            system_prompt=SYSTEM_MESSAGE,
            user_prompt=prompt,
            response=response,
            model=response_obj.model,
            input_tokens=response_obj.input_tokens,
            output_tokens=response_obj.output_tokens,
            cached=response_obj.cached,
            latency_ms=first_latency_ms,
            cache_hit=False,
            parse_error=None,
            parsed_answers=parsed,
            is_repair=False,
            reasoning_content=response_obj.reasoning_content,
        )

    latency_ms = int((time.perf_counter() - first_start) * 1000)
    if cacheable_response:
        write_cache(node_name, prompt, response)
    log_entry = {
        "node": node_name,
        "prompt_length_chars": len(prompt),
        "response_length_chars": len(response),
        "latency_ms": latency_ms,
        "cache_hit": False,
        "model": response_obj.model,
        "input_tokens": response_obj.input_tokens,
        "output_tokens": response_obj.output_tokens,
        "cached": response_obj.cached,
    }
    if chunk_sources:
        log_entry["chunk_sources"] = chunk_sources
    if suspected_parse_failures:
        log_entry["suspected_parse_failures"] = suspected_parse_failures
    log.append(log_entry)
    return response, log, parsed


def merge_sq_answers(state: dict, parsed: dict[str, dict]) -> dict[str, dict]:
    sq_answers = dict(state.get("sq_answers", {}))
    sq_answers.update(parsed)
    return sq_answers


def set_na(sq_answers: dict[str, dict], *sq_ids: str) -> dict[str, dict]:
    updated = dict(sq_answers)
    for sq_id in sq_ids:
        updated[sq_id] = dict(NA_ANSWER)
    return updated


def format_chunk_sources(state: dict, domain: str, limit: int = 5) -> list[str]:
    metas = state.get("rag_chunk_metadata", {}).get(domain, [])
    sources: list[str] = []
    for meta in metas[:limit]:
        page_numbers = meta.get("page_numbers") or []
        page = page_numbers[0] if page_numbers else "?"
        section = meta.get("section") or "Unknown"
        sources.append(f"[page {page}, {section}]")
    return sources


def call_node_llm_with_sources(
    call_fn: Callable,
    state: dict,
    prompt: str,
    node_name: str,
    parse_fn: Callable[[str, list[str]], dict[str, dict]],
    parse_sq_ids: list[str],
    chunk_sources: list[str],
) -> tuple[str, list[LLMCallLogEntry], Optional[dict[str, dict]]]:
    if "chunk_sources" in signature(call_fn).parameters:
        return call_fn(
            state,
            prompt,
            node_name,
            parse_fn,
            parse_sq_ids,
            chunk_sources=chunk_sources,
        )
    return call_fn(state, prompt, node_name, parse_fn, parse_sq_ids)


def add_domain_judgment(
    state: dict, domain: str, judgment: str, rationale: str
) -> dict:
    domain_judgments = dict(state.get("domain_judgments", {}))
    domain_rationales = dict(state.get("domain_rationales", {}))
    domain_judgments[domain] = judgment
    domain_rationales[domain] = rationale
    return {
        "domain_judgments": domain_judgments,
        "domain_rationales": domain_rationales,
    }
