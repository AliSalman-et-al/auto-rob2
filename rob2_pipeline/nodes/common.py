import time
from inspect import signature
from typing import Callable, Optional

from rob2_pipeline.cache import read_cache, write_cache
from rob2_pipeline.config import build_provider
from rob2_pipeline.methodology import METHODOLOGIES
from rob2_pipeline.methodology.render import render_methodology
from rob2_pipeline.trace import append_llm_call
from rob2_pipeline.types import LLMCallLogEntry
from rob2_pipeline.xml_parser import validate_sq_answers


SYSTEM_MESSAGE = (
    "You are an expert systematic reviewer applying the Cochrane Risk of Bias 2 "
    "(RoB 2) tool. Respond only in the XML format specified in the prompt. "
    "Do not add preamble, explanation, or markdown code fences around your XML."
)
VALID_SQ_ANSWERS = ("Y", "PY", "PN", "N", "NI", "NA")

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


def add_domain_judgment_with_pivotality_tests(
    state: dict,
    domain: str,
    judgment: str,
    rationale: str,
    judge_fn: Callable[[dict], tuple[str, str]],
    sq_ids: tuple[str, ...],
) -> dict:
    state, judgment, rationale = _adjudicate_pivotal_sq_answers(
        state, domain, judgment, rationale, judge_fn, sq_ids
    )
    update = add_domain_judgment(state, domain, judgment, rationale)
    pivotality_tests = list(state.get("pivotality_tests", {}).get(domain, []))

    for sq_id in sq_ids:
        sq_answer = state.get("sq_answers", {}).get(sq_id)
        if not sq_answer:
            continue
        support_level = sq_answer.get("support_level", "").lower()
        if support_level not in {"weak", "unsupported"}:
            continue

        test_sq_answers = {
            key: dict(value) for key, value in state.get("sq_answers", {}).items()
        }
        test_sq_answers[sq_id] = dict(test_sq_answers[sq_id])
        test_sq_answers[sq_id]["answer"] = "NI"
        test_judgment, _ = judge_fn(test_sq_answers)

        pivotality_tests.append(
            {
                "sq_id": sq_id,
                "original_answer": sq_answer.get("answer", "NI"),
                "support_level": support_level,
                "conservative_test_answer": "NI",
                "original_domain_judgment": judgment,
                "test_domain_judgment": test_judgment,
                "pivotal": test_judgment != judgment,
            }
        )

    if pivotality_tests:
        all_tests = dict(state.get("pivotality_tests", {}))
        all_tests[domain] = pivotality_tests
        update["pivotality_tests"] = all_tests
    if state.get("sq_support_adjudications"):
        update["sq_support_adjudications"] = state["sq_support_adjudications"]
    if state.get("sq_answers"):
        update["sq_answers"] = state["sq_answers"]
    if state.get("_sq_adjudication_llm_call_log"):
        update["llm_call_log"] = state["_sq_adjudication_llm_call_log"]
    return update


def _adjudicate_pivotal_sq_answers(
    state: dict,
    domain: str,
    judgment: str,
    rationale: str,
    judge_fn: Callable[[dict], tuple[str, str]],
    sq_ids: tuple[str, ...],
) -> tuple[dict, str, str]:
    if not _has_adjudication_context(state):
        return state, judgment, rationale

    updated_state = dict(state)
    sq_answers = {
        key: dict(value) for key, value in state.get("sq_answers", {}).items()
    }
    adjudications = list(state.get("sq_support_adjudications", {}).get(domain, []))
    llm_log = []
    changed_any = False

    for sq_id in sq_ids:
        sq_answer = sq_answers.get(sq_id)
        if not sq_answer:
            continue
        if sq_answer.get("answer") == "NA":
            continue
        support_level = sq_answer.get("support_level", "").lower()
        if support_level not in {"weak", "unsupported"}:
            continue
        if (
            sq_answer.get("support_rationale")
            == "Support rationale was not provided by the legacy response."
        ):
            continue

        impact = _adjudication_domain_impact(sq_answers, sq_id, judgment, judge_fn)
        if impact is None:
            continue

        node_name = f"sq_support_adjudication_{domain}_{sq_id.replace('.', '_')}"
        _response, log, parsed = call_node_llm_with_sources(
            call_node_llm,
            updated_state,
            _build_sq_support_adjudication_prompt(
                updated_state,
                domain,
                sq_id,
                sq_answer,
                judgment,
                impact["test_answer"],
                impact["test_domain_judgment"],
            ),
            node_name,
            updated_state.get("adjudication_parse_fn", state.get("parse_fn", None))
            or _parse_adjudication_passthrough,
            [sq_id],
            format_chunk_sources(updated_state, _source_domain_for(domain)),
        )
        llm_log.extend(log)
        adjudicated = dict((parsed or {}).get(sq_id, sq_answer))
        adjudicated.setdefault(
            "residual_uncertainty",
            adjudicated.get("support_rationale", "No residual uncertainty reported."),
        )
        changed = _answer_or_support_changed(sq_answer, adjudicated)
        if changed:
            sq_answers[sq_id] = adjudicated
            updated_state["sq_answers"] = sq_answers
            judgment, rationale = judge_fn(sq_answers)
            changed_any = True

        adjudications.append(
            {
                "sq_id": sq_id,
                "initial_answer": sq_answer,
                "adjudicated_answer": adjudicated,
                "domain_impact": {
                    "original_domain_judgment": impact["original_domain_judgment"],
                    "test_answer": impact["test_answer"],
                    "test_domain_judgment": impact["test_domain_judgment"],
                },
                "changed": changed,
                "llm_node": node_name,
            }
        )

    if adjudications:
        all_adjudications = dict(state.get("sq_support_adjudications", {}))
        all_adjudications[domain] = adjudications
        updated_state["sq_support_adjudications"] = all_adjudications
    if llm_log:
        updated_state["_sq_adjudication_llm_call_log"] = llm_log
    if changed_any:
        updated_state["sq_answers"] = sq_answers
    return updated_state, judgment, rationale


def _has_adjudication_context(state: dict) -> bool:
    return bool(state.get("evidence_packets") and state.get("outcome"))


def _source_domain_for(domain: str) -> str:
    return domain.lower()


def _answer_or_support_changed(initial: dict, adjudicated: dict) -> bool:
    keys = ("answer", "support_level", "support_rationale", "quote", "justification")
    return any(initial.get(key) != adjudicated.get(key) for key in keys)


def _adjudication_domain_impact(
    sq_answers: dict,
    sq_id: str,
    judgment: str,
    judge_fn: Callable[[dict], tuple[str, str]],
) -> dict | None:
    for test_answer in VALID_SQ_ANSWERS:
        if test_answer == sq_answers[sq_id].get("answer"):
            continue
        test_sq_answers = {key: dict(value) for key, value in sq_answers.items()}
        test_sq_answers[sq_id] = dict(test_sq_answers[sq_id])
        test_sq_answers[sq_id]["answer"] = test_answer
        test_judgment, _ = judge_fn(test_sq_answers)
        if test_judgment != judgment:
            return {
                "original_domain_judgment": judgment,
                "test_answer": test_answer,
                "test_domain_judgment": test_judgment,
            }
    return None


def _methodology_key(domain: str) -> str:
    if domain == "D2":
        return "D2_ASSIGNMENT"
    return domain


def _build_sq_support_adjudication_prompt(
    state: dict,
    domain: str,
    sq_id: str,
    initial_answer: dict,
    judgment: str,
    test_answer: str,
    test_judgment: str,
) -> str:
    methodology = METHODOLOGIES[_methodology_key(domain)]
    packet = state.get("evidence_packets", {}).get(sq_id, {})
    sources = packet.get("sources", [])
    rendered_sources = "\n".join(
        f"- {source.get('section', 'Unknown section')} p.{source.get('page_numbers', ['?'])[0] if source.get('page_numbers') else '?'}: {source.get('text', '')}"
        for source in sources[:5]
    )
    if not rendered_sources:
        rendered_sources = "No selected packet sources were available."

    domain_marker = _prompt_marker_for_adjudication(domain, sq_id)

    return f"""Re-evaluate one RoB 2 signaling-question answer. Do not re-run a full domain.

<{domain_marker}></{domain_marker}>
Outcome: {state.get("outcome", "Not reported")}
Domain: {domain}

{render_methodology(methodology, [sq_id])}

Original answer metadata:
<answer>{initial_answer.get("answer", "NI")}</answer>
<quote>{initial_answer.get("quote", "No relevant text found")}</quote>
<justification>{initial_answer.get("justification", "")}</justification>
<support_level>{initial_answer.get("support_level", "unsupported")}</support_level>
<support_rationale>{initial_answer.get("support_rationale", "")}</support_rationale>

Selected evidence packet sources:
{rendered_sources}

Quote/provenance warnings:
{packet.get("missing_evidence", [])}

Domain-judgment impact:
Original domain judgment: {judgment}
Alternative SQ answer tested for impact: {test_answer}
Alternative-answer domain judgment: {test_judgment}

Return one adjudicated answer for SQ {sq_id}. Include answer code, quote, justification, support level, support rationale, and residual uncertainty.
Respond in this exact XML format:
<sq_{sq_id.replace(".", "_")}>
  <answer>Y/PY/PN/N/NI/NA</answer>
  <quote>exact quote or No relevant text found</quote>
  <justification>brief rationale</justification>
  <uncertainty_flag>NORMAL or HIGH</uncertainty_flag>
  <support_level>strong/moderate/weak/unsupported</support_level>
  <support_rationale>brief support rationale</support_rationale>
  <residual_uncertainty>brief residual uncertainty</residual_uncertainty>
</sq_{sq_id.replace(".", "_")}>"""


def _parse_adjudication_passthrough(raw: str, sq_ids: list[str]) -> dict[str, dict]:
    from rob2_pipeline.xml_parser import parse_sq_response

    return parse_sq_response(raw, sq_ids)


def _prompt_marker_for_adjudication(domain: str, sq_id: str) -> str:
    if domain == "D2":
        if sq_id in {"2.1", "2.2"}:
            return "domain2_part1"
        if sq_id in {"2.6", "2.7", "2.6a"}:
            return "domain2_analysis"
        return "domain2_conditional"
    return f"domain{domain[1:]}"
