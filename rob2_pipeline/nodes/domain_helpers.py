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
    prompt = stage.build_prompt(state)
    _response, log, parsed = call_node_llm_with_sources(
        call_fn,
        state,
        prompt,
        stage.node_name,
        stage.parse_fn,
        list(stage.sq_ids),
        chunk_sources=format_chunk_sources(state, stage.source_domain),
    )
    sq_answers = merge_sq_answers(state, parsed or {})
    if stage.postprocess is not None:
        sq_answers = stage.postprocess(state, sq_answers)
    return {"sq_answers": sq_answers, "llm_call_log": log}


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
