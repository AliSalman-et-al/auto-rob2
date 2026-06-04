import time
from inspect import signature
from typing import Callable, Literal, Optional

from rob2_pipeline.cache import read_cache, write_cache
from rob2_pipeline.config import build_provider
from rob2_pipeline.llm_contracts import JsonContractResult, call_json_contract_llm
from rob2_pipeline.methodology import METHODOLOGIES
from rob2_pipeline.methodology.render import render_methodology
from rob2_pipeline.trace import append_llm_call
from rob2_pipeline.types import LLMCallLogEntry
from rob2_pipeline.xml_parser import validate_sq_answers
from pydantic import BaseModel, ConfigDict, Field


SYSTEM_MESSAGE = (
    "You are an expert systematic reviewer applying the Cochrane Risk of Bias 2 "
    "(RoB 2) tool. Respond only in the XML format specified in the prompt. "
    "Do not add preamble, explanation, or markdown code fences around your XML."
)
VALID_SQ_ANSWERS = ("Y", "PY", "PN", "N", "NI", "NA")
WEAK_SUPPORT_LEVELS = {"weak", "unsupported"}
LESS_CONFIDENT_ANSWERS = {
    "Y": "PY",
    "PY": "NI",
    "N": "PN",
    "PN": "NI",
}

NA_ANSWER = {
    "answer": "NA",
    "quote": "Not applicable",
    "justification": "Not applicable",
    "uncertainty_flag": "NORMAL",
    "support_level": "unsupported",
    "support_rationale": "Not applicable",
}


class SqSupportAdjudicationArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sq_id: str
    answer: Literal["Y", "PY", "PN", "N", "NI", "NA"]
    quote: str = Field(min_length=1)
    justification: str = Field(min_length=1)
    uncertainty_flag: Literal["NORMAL", "HIGH"]
    support_level: Literal["strong", "moderate", "weak", "unsupported"]
    support_rationale: str = Field(min_length=1)
    residual_uncertainty: str = Field(min_length=1)
    quote_traceability_status: Literal[
        "traceable",
        "untraceable",
        "traceability_not_assessed",
    ] = "traceability_not_assessed"


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
    initial_judgment = judgment
    initial_rationale = rationale
    initial_sq_answers = {
        key: dict(value) for key, value in state.get("sq_answers", {}).items()
    }
    state, judgment, rationale = _adjudicate_pivotal_sq_answers(
        state, domain, judgment, rationale, judge_fn, sq_ids
    )
    update = add_domain_judgment(state, domain, judgment, rationale)
    initial_domain_judgments = dict(state.get("initial_domain_judgments", {}))
    initial_domain_rationales = dict(state.get("initial_domain_rationales", {}))
    initial_domain_judgments[domain] = initial_judgment
    initial_domain_rationales[domain] = initial_rationale
    update["initial_domain_judgments"] = initial_domain_judgments
    update["initial_domain_rationales"] = initial_domain_rationales
    pivotality_tests = list(state.get("pivotality_tests", {}).get(domain, []))
    new_pivotality_tests_by_sq = {}

    for sq_id in sq_ids:
        sq_answer = initial_sq_answers.get(sq_id)
        if not sq_answer:
            continue
        support_level = sq_answer.get("support_level", "").lower()
        constraints = _constraints_for_sq(state, sq_id)
        if support_level not in WEAK_SUPPORT_LEVELS and not constraints:
            continue

        conservative_answer, test_judgment = _conservative_pivotality_test(
            initial_sq_answers,
            sq_id,
            initial_judgment,
            judge_fn,
        )
        pivotal = test_judgment != initial_judgment

        test_record = {
            "sq_id": sq_id,
            "original_answer": sq_answer.get("answer", "NI"),
            "support_level": support_level or "constrained",
            "conservative_test_answer": conservative_answer,
            "original_domain_judgment": initial_judgment,
            "test_domain_judgment": test_judgment,
            "pivotal": pivotal,
            "acceptance_status": _acceptance_status(
                pivotal,
                state.get("sq_answers", {}).get(sq_id, sq_answer),
                state,
                domain,
                sq_id,
            ),
        }
        if constraints:
            test_record["constraints"] = constraints
        pivotality_tests.append(test_record)
        new_pivotality_tests_by_sq[sq_id] = test_record

    if pivotality_tests:
        all_tests = dict(state.get("pivotality_tests", {}))
        all_tests[domain] = pivotality_tests
        update["pivotality_tests"] = all_tests
    routing_decisions = [
        _micro_agent_routing_decision(
            sq_id,
            initial_sq_answers[sq_id],
            _constraints_for_sq(state, sq_id),
            new_pivotality_tests_by_sq.get(sq_id),
        )
        for sq_id in sq_ids
        if sq_id in initial_sq_answers
    ]
    if routing_decisions:
        all_decisions = dict(state.get("micro_agent_routing_decisions", {}))
        all_decisions[domain] = routing_decisions
        update["micro_agent_routing_decisions"] = all_decisions
    if state.get("sq_support_adjudications"):
        update["sq_support_adjudications"] = state["sq_support_adjudications"]
    if state.get("sq_answers"):
        update["sq_answers"] = state["sq_answers"]
    if state.get("_sq_adjudication_llm_call_log"):
        update["llm_call_log"] = state["_sq_adjudication_llm_call_log"]
    return update


def _conservative_pivotality_test(
    sq_answers: dict,
    sq_id: str,
    initial_judgment: str,
    judge_fn: Callable[[dict], tuple[str, str]],
) -> tuple[str, str]:
    original_answer = sq_answers[sq_id].get("answer", "NI")
    fallback_answer = "NI"
    fallback_judgment = initial_judgment
    next_answer = LESS_CONFIDENT_ANSWERS.get(original_answer, "NI")
    while next_answer:
        test_sq_answers = {key: dict(value) for key, value in sq_answers.items()}
        test_sq_answers[sq_id] = dict(test_sq_answers[sq_id])
        test_sq_answers[sq_id]["answer"] = next_answer
        test_judgment, _ = judge_fn(test_sq_answers)
        fallback_answer = next_answer
        fallback_judgment = test_judgment
        if test_judgment != initial_judgment:
            return next_answer, test_judgment
        if next_answer == "NI":
            break
        next_answer = LESS_CONFIDENT_ANSWERS.get(next_answer)
    return fallback_answer, fallback_judgment


def _constraints_for_sq(state: dict, sq_id: str) -> list[dict]:
    return [
        constraint
        for constraint in state.get("support_constraints", [])
        if constraint.get("sq_id") == sq_id
    ]


def _micro_agent_routing_decision(
    sq_id: str,
    sq_answer: dict,
    constraints: list[dict],
    pivotality_test: dict | None,
) -> dict:
    support_level = str(sq_answer.get("support_level", "")).lower()
    trigger_conditions = _routing_trigger_conditions(
        support_level, constraints, pivotality_test
    )
    if not trigger_conditions:
        return {
            "sq_id": sq_id,
            "status": "no_escalation",
            "route": "none",
            "trigger_conditions": [],
            "reason": (
                f"Answer support is {support_level or 'unspecified'} and no support "
                "constraints were recorded."
            ),
        }

    if pivotality_test and pivotality_test.get("pivotal"):
        route = _route_for_constraints(constraints) or "sq_support_adjudication"
        status = str(pivotality_test.get("acceptance_status", "needs_adjudication"))
        return {
            "sq_id": sq_id,
            "status": status,
            "route": route,
            "trigger_conditions": trigger_conditions,
            "reason": _routing_reason(route, support_level, pivotal=True),
        }

    return {
        "sq_id": sq_id,
        "status": "accepted",
        "route": "none",
        "trigger_conditions": trigger_conditions,
        "reason": "Recorded audit trigger is non-pivotal, so no micro-agent is routed.",
    }


def _routing_trigger_conditions(
    support_level: str,
    constraints: list[dict],
    pivotality_test: dict | None,
) -> list[str]:
    conditions = []
    if support_level in WEAK_SUPPORT_LEVELS:
        conditions.append(f"support_level={support_level}")
    for constraint in constraints:
        constraint_type = constraint.get("constraint_type", "unknown")
        conditions.append(f"support_constraint={constraint_type}")
    if pivotality_test:
        pivotal_status = "pivotal" if pivotality_test.get("pivotal") else "non_pivotal"
        conditions.append(f"pivotality_test={pivotal_status}")
    return conditions


def _route_for_constraints(constraints: list[dict]) -> str | None:
    constraint_types = {constraint.get("constraint_type") for constraint in constraints}
    if "missing_required_evidence" in constraint_types:
        return "retrieval_repair"
    if "semantic_support_conflict" in constraint_types:
        return "contradiction_resolution"
    if "quote_untraceable" in constraint_types or "wrong_outcome_context" in constraint_types:
        return "sq_support_adjudication"
    return None


def _routing_reason(route: str, support_level: str, *, pivotal: bool) -> str:
    if route == "retrieval_repair":
        return "Pivotal answer is missing required evidence and should repair retrieval before acceptance."
    if route == "contradiction_resolution":
        return "Pivotal answer has a semantic support conflict and should resolve contradiction before acceptance."
    if pivotal:
        level = support_level or "constrained"
        return f"Pivotal {level} SQ answer requires targeted support adjudication."
    return "No escalation route selected."


def _acceptance_status(
    pivotal: bool, final_answer: dict, state: dict, domain: str, sq_id: str
) -> str:
    if not pivotal:
        return "accepted"
    adjudicated = _adjudication_for_sq(state, domain, sq_id)
    if not adjudicated:
        return "needs_adjudication"
    if final_answer.get("support_level", "").lower() in WEAK_SUPPORT_LEVELS:
        return "audit_limited"
    return "accepted"


def _adjudication_for_sq(state: dict, domain: str, sq_id: str) -> dict | None:
    for attempt in state.get("sq_support_adjudications", {}).get(domain, []):
        if attempt.get("sq_id") == sq_id:
            return attempt
    return None


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
        if sq_answer.get("classification_blocked"):
            continue
        if sq_answer.get("answer") == "NA":
            continue
        support_level = sq_answer.get("support_level", "").lower()
        constraints = _constraints_for_sq(updated_state, sq_id)
        if support_level not in {"weak", "unsupported"} and not constraints:
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
        contract_result = call_json_contract_llm(
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
            schema_model=SqSupportAdjudicationArtifact,
            schema_version="sq-support-adjudication.v1",
            prompt_version="sq-support-adjudication-json.v1",
            fallback_factory=lambda failure_reason, sq_id=sq_id: {
                "sq_id": sq_id,
                "answer": sq_answer.get("answer", "NI"),
                "quote": sq_answer.get("quote", "No relevant text found"),
                "justification": sq_answer.get("justification", ""),
                "uncertainty_flag": sq_answer.get("uncertainty_flag", "HIGH"),
                "support_level": sq_answer.get("support_level", "unsupported"),
                "support_rationale": sq_answer.get("support_rationale", ""),
                "residual_uncertainty": failure_reason,
                "quote_traceability_status": _traceability_status(sq_answer),
            },
        )
        log = _annotate_adjudication_log_sources(
            contract_result.log,
            format_chunk_sources(updated_state, _source_domain_for(domain)),
        )
        llm_log.extend(log)
        raw_adjudicated = _answer_from_adjudication_artifact(contract_result.artifact)
        validation_errors = _validate_adjudicated_answer(
            raw_adjudicated, sq_id, contract_result
        )
        validation_status = "rejected" if validation_errors else "accepted"
        adjudicated = dict(sq_answer if validation_errors else raw_adjudicated)
        adjudicated.setdefault(
            "residual_uncertainty",
            adjudicated.get("support_rationale", "No residual uncertainty reported."),
        )
        changed_answer = (
            not validation_errors
            and sq_answer.get("answer") != adjudicated.get("answer")
        )
        changed_support = (
            not validation_errors
            and _support_level(sq_answer) != _support_level(adjudicated)
        )
        changed = changed_answer or changed_support
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
                "changed_answer": changed_answer,
                "changed_support": changed_support,
                "validation_status": validation_status,
                "validation_errors": validation_errors,
                "semantic_support_decision": {
                    "support_level": _support_level(adjudicated) or "unsupported",
                    "rationale": adjudicated.get("support_rationale")
                    or adjudicated.get("justification", ""),
                    "residual_uncertainty": adjudicated.get(
                        "residual_uncertainty",
                        "No residual uncertainty reported.",
                    ),
                },
                "effect_on_sq_status": {
                    "initial_answer": sq_answer.get("answer", "NI"),
                    "adjudicated_answer": adjudicated.get("answer", "NI"),
                    "changed_answer": changed_answer,
                    "changed_support": changed_support,
                },
                "effect_on_packet_status": _adjudication_packet_status_effect(
                    updated_state, sq_id, adjudicated
                ),
                "traceability_status": {
                    "initial": _traceability_status(sq_answer),
                    "adjudicated": _traceability_status(adjudicated, sq_answer),
                },
                "rationale": adjudicated.get("support_rationale")
                or adjudicated.get("justification", ""),
                "constraints": constraints,
                "provenance": {
                    "llm_node": node_name,
                    "chunk_sources": format_chunk_sources(
                        updated_state, _source_domain_for(domain)
                    ),
                },
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


def _support_level(answer: dict) -> str:
    return str(answer.get("support_level", "")).lower()


def _traceability_status(answer: dict, fallback: dict | None = None) -> str:
    if (
        fallback
        and answer.get("quote_traceability_status") == "traceability_not_assessed"
        and fallback.get("quote_traceability_status")
    ):
        return str(fallback["quote_traceability_status"])
    return str(
        answer.get("quote_traceability_status")
        or (fallback or {}).get("quote_traceability_status")
        or "traceability_not_assessed"
    )


_ADJUDICATION_ALLOWED_KEYS = {
    "answer",
    "quote",
    "justification",
    "uncertainty_flag",
    "support_level",
    "support_rationale",
    "residual_uncertainty",
    "quote_traceability_status",
}
_ADJUDICATION_REQUIRED_KEYS = {
    "answer",
    "quote",
    "justification",
    "uncertainty_flag",
    "support_level",
    "support_rationale",
}
_VALID_SUPPORT_LEVELS = {"strong", "moderate", "weak", "unsupported"}
_VALID_UNCERTAINTY_FLAGS = {"NORMAL", "HIGH"}


def _answer_from_adjudication_artifact(artifact: dict) -> dict:
    return {
        "answer": artifact.get("answer", "NI"),
        "quote": artifact.get("quote", "No relevant text found"),
        "justification": artifact.get("justification", ""),
        "uncertainty_flag": artifact.get("uncertainty_flag", "HIGH"),
        "support_level": artifact.get("support_level", "unsupported"),
        "support_rationale": artifact.get("support_rationale", ""),
        "residual_uncertainty": artifact.get("residual_uncertainty", ""),
        "quote_traceability_status": artifact.get(
            "quote_traceability_status", "traceability_not_assessed"
        ),
    }


def _annotate_adjudication_log_sources(
    log: list[LLMCallLogEntry], chunk_sources: list[str]
) -> list[LLMCallLogEntry]:
    if not chunk_sources:
        return log
    annotated = []
    for entry in log:
        copied = dict(entry)
        copied["chunk_sources"] = chunk_sources
        annotated.append(copied)
    return annotated


def _validate_adjudicated_answer(
    answer: dict, sq_id: str, contract_result: JsonContractResult | None = None
) -> list[str]:
    errors = []
    if contract_result and contract_result.status != "validated":
        errors.append(
            contract_result.failure_reason
            or "JSON contract validation failed for SQ adjudication output"
        )
    artifact_sq_id = (
        contract_result.artifact.get("sq_id")
        if contract_result and isinstance(contract_result.artifact, dict)
        else None
    )
    if artifact_sq_id is not None and artifact_sq_id != sq_id:
        errors.append(f"sq_id: expected {sq_id}")
    extra_keys = sorted(set(answer) - _ADJUDICATION_ALLOWED_KEYS)
    for key in extra_keys:
        errors.append(f"{key}: field is not allowed in SQ adjudication output")
    missing_keys = sorted(_ADJUDICATION_REQUIRED_KEYS - set(answer))
    for key in missing_keys:
        errors.append(f"{key}: field is required")
    if answer.get("answer") not in VALID_SQ_ANSWERS:
        errors.append(
            f"answer: expected one of {', '.join(VALID_SQ_ANSWERS)} for SQ {sq_id}"
        )
    support_level = str(answer.get("support_level", "")).lower()
    if support_level not in _VALID_SUPPORT_LEVELS:
        errors.append("support_level: expected strong, moderate, weak, or unsupported")
    if answer.get("uncertainty_flag") not in _VALID_UNCERTAINTY_FLAGS:
        errors.append("uncertainty_flag: expected NORMAL or HIGH")
    for key in ("quote", "justification", "support_rationale"):
        if key in answer and not isinstance(answer.get(key), str):
            errors.append(f"{key}: expected string")
    if "residual_uncertainty" in answer and not isinstance(
        answer.get("residual_uncertainty"), str
    ):
        errors.append("residual_uncertainty: expected string")
    return errors


def _adjudication_packet_status_effect(
    state: dict, sq_id: str, adjudicated: dict
) -> dict:
    packet = state.get("evidence_packets", {}).get(sq_id, {})
    readiness = packet.get("packet_readiness", {})
    initial_status = readiness.get("status") or packet.get("status") or "unknown"
    adjudicated_status = (
        "audit_limited"
        if _support_level(adjudicated) in WEAK_SUPPORT_LEVELS
        else "accepted"
    )
    return {
        "initial_status": initial_status,
        "adjudicated_status": adjudicated_status,
    }


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
Return only JSON matching the adjudication contract. Do not include markdown fences.

<{domain_marker}></{domain_marker}>
Outcome: {state.get("outcome", "Not reported")}
Domain: {domain}

{render_methodology(methodology, [sq_id])}

Original answer metadata:
- answer: {initial_answer.get("answer", "NI")}
- quote: {initial_answer.get("quote", "No relevant text found")}
- justification: {initial_answer.get("justification", "")}
- support_level: {initial_answer.get("support_level", "unsupported")}
- support_rationale: {initial_answer.get("support_rationale", "")}

Selected evidence packet sources:
{rendered_sources}

Quote/provenance warnings:
{packet.get("missing_evidence", [])}

Support constraints:
{_render_support_constraints(_constraints_for_sq(state, sq_id))}

Return one adjudicated answer for SQ {sq_id}. Include answer code, quote, justification, support level, support rationale, residual uncertainty, and quote traceability status.
JSON fields:
- sq_id: "{sq_id}"
- answer: one of "Y", "PY", "PN", "N", "NI", "NA"
- quote: exact quote or "No relevant text found"
- justification: brief rationale
- uncertainty_flag: "NORMAL" or "HIGH"
- support_level: "strong", "moderate", "weak", or "unsupported"
- support_rationale: brief support rationale
- residual_uncertainty: brief residual uncertainty
- quote_traceability_status: "traceable", "untraceable", or "traceability_not_assessed" """


def _render_support_constraints(constraints: list[dict]) -> str:
    if not constraints:
        return "No support constraints were recorded for this SQ."
    rendered = []
    for constraint in constraints:
        parts = [
            f"type={constraint.get('constraint_type', 'unknown')}",
            f"reason={constraint.get('reason', 'No reason recorded')}",
        ]
        if constraint.get("evidence_label"):
            parts.append(f"evidence_label={constraint['evidence_label']}")
        if constraint.get("evidence"):
            parts.append(f"evidence={constraint['evidence']}")
        if constraint.get("provenance"):
            parts.append(f"provenance={constraint['provenance']}")
        rendered.append("- " + "; ".join(parts))
    return "\n".join(rendered)


def _prompt_marker_for_adjudication(domain: str, sq_id: str) -> str:
    if domain == "D2":
        if sq_id in {"2.1", "2.2"}:
            return "domain2_part1"
        if sq_id in {"2.6", "2.7", "2.6a"}:
            return "domain2_analysis"
        return "domain2_conditional"
    return f"domain{domain[1:]}"
