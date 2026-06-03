from collections.abc import Callable
from dataclasses import dataclass

from rob2_pipeline.nodes.common import (
    call_node_llm,
    call_node_llm_with_sources,
    format_chunk_sources,
    merge_sq_answers,
)
from rob2_pipeline.state import RoB2State
from rob2_pipeline.xml_parser import parse_sq_response


BLOCKING_PACKET_STATUSES = {
    "needs_retrieval_repair",
    "needs_contradiction_resolution",
    "needs_quote_adjudication",
}


@dataclass(frozen=True)
class DomainSqStage:
    node_name: str
    sq_ids: tuple[str, ...]
    source_domain: str
    build_prompt: Callable[[RoB2State], str]
    postprocess: Callable[[RoB2State, dict[str, dict]], dict[str, dict]] | None = None
    parse_fn: Callable = parse_sq_response


def run_domain_sq_stage(
    state: RoB2State,
    stage: DomainSqStage,
    *,
    call_fn: Callable | None = None,
) -> RoB2State:
    if call_fn is None:
        call_fn = call_node_llm
    blocked = _blocked_sq_answers(state, stage)
    active_sq_ids = [sq_id for sq_id in stage.sq_ids if sq_id not in blocked]
    if blocked and not active_sq_ids:
        return {"sq_answers": merge_sq_answers(state, blocked), "llm_call_log": []}
    prompt = stage.build_prompt(state)
    _response, log, parsed = call_node_llm_with_sources(
        call_fn,
        state,
        prompt,
        stage.node_name,
        stage.parse_fn,
        active_sq_ids,
        chunk_sources=format_chunk_sources(state, stage.source_domain),
    )
    sq_answers = merge_sq_answers(state, {**blocked, **(parsed or {})})
    if stage.postprocess is not None:
        sq_answers = stage.postprocess(state, sq_answers)
    return {"sq_answers": sq_answers, "llm_call_log": log}


def _blocked_sq_answers(state: RoB2State, stage: DomainSqStage) -> dict[str, dict]:
    blocked: dict[str, dict] = {}
    readiness_by_sq = state.get("packet_readiness") or {}
    if not readiness_by_sq:
        readiness_by_sq = {
            sq_id: packet.get("packet_readiness", {})
            for sq_id, packet in (state.get("evidence_packets") or {}).items()
        }
    for sq_id in stage.sq_ids:
        readiness = readiness_by_sq.get(sq_id) or {}
        status = readiness.get("status", "")
        if status not in BLOCKING_PACKET_STATUSES:
            continue
        blocked[sq_id] = {
            "answer": "NI",
            "quote": "",
            "justification": (
                "SQ classification was blocked because the evidence packet "
                f"status is {status}."
            ),
            "uncertainty": True,
            "support_level": "unsupported",
            "support_rationale": readiness.get("blocking_reason", ""),
            "classification_blocked": True,
            "packet_status": status,
            "packet_readiness": readiness,
        }
    return blocked


def call_domain_sq_prompt(
    state: RoB2State,
    prompt: str,
    *,
    node_name: str,
    sq_ids: list[str],
    source_domain: str,
    parse_fn: Callable = parse_sq_response,
) -> tuple[dict[str, dict], list[dict]]:
    result = run_domain_sq_stage(
        state,
        DomainSqStage(
            node_name=node_name,
            sq_ids=tuple(sq_ids),
            source_domain=source_domain,
            build_prompt=lambda _state: prompt,
            parse_fn=parse_fn,
        ),
    )
    return result["sq_answers"], result["llm_call_log"]
