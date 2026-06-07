from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rob2_pipeline.llm_contracts import call_json_contract_llm
from rob2_pipeline.nodes.common import format_chunk_sources, merge_sq_answers
from rob2_pipeline.state import RoB2State


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


class DomainSqAnswerArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sq_id: str
    answer: Literal["Y", "PY", "PN", "N", "NI", "NA"]
    quote: str = Field(min_length=1)
    justification: str = Field(min_length=1)
    uncertainty_flag: Literal["NORMAL", "HIGH"]
    support_level: Literal["strong", "moderate", "weak", "unsupported"]
    support_rationale: str = Field(min_length=1)


class DomainSqStageArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answers: list[DomainSqAnswerArtifact] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_sq_ids(self) -> "DomainSqStageArtifact":
        sq_ids = [answer.sq_id for answer in self.answers]
        if len(sq_ids) != len(set(sq_ids)):
            raise ValueError("answers must contain unique sq_id values")
        return self


def run_domain_sq_stage(
    state: RoB2State,
    stage: DomainSqStage,
    *,
    call_fn: Callable | None = None,
) -> RoB2State:
    blocked = _blocked_sq_answers(state, stage)
    active_sq_ids = [sq_id for sq_id in stage.sq_ids if sq_id not in blocked]
    if blocked and not active_sq_ids:
        return {"sq_answers": merge_sq_answers(state, blocked), "llm_call_log": []}
    prompt = stage.build_prompt(state)
    contract_call = call_fn or call_json_contract_llm
    result = contract_call(
        state,
        _json_stage_prompt(prompt, active_sq_ids),
        stage.node_name,
        schema_model=DomainSqStageArtifact,
        schema_version="domain-sq-stage.v1",
        prompt_version=f"{stage.node_name}-json.v1",
        fallback_factory=lambda failure_reason: _stage_fallback_artifact(
            active_sq_ids, failure_reason
        ),
    )
    log = _annotate_log_sources(
        result.log, format_chunk_sources(state, stage.source_domain)
    )
    parsed = _answers_from_stage_artifact(result.artifact, active_sq_ids, result.status)
    sq_answers = merge_sq_answers(state, {**blocked, **(parsed or {})})
    if stage.postprocess is not None:
        sq_answers = stage.postprocess(state, sq_answers)
    return {"sq_answers": sq_answers, "llm_call_log": log}


def _json_stage_prompt(prompt: str, sq_ids: list[str]) -> str:
    return (
        f"{prompt}\n\n"
        "Return only JSON matching the local DomainSqStageArtifact schema. "
        "Do not include markdown fences. "
        f"Return exactly one answer object for each active SQ id: {', '.join(sq_ids)}. "
        "Each answer must include sq_id, answer, quote, justification, "
        "uncertainty_flag, support_level, and support_rationale."
    )


def _stage_fallback_artifact(sq_ids: list[str], failure_reason: str) -> dict:
    return {
        "answers": [
            {
                "sq_id": sq_id,
                "answer": "NI",
                "quote": "No relevant text found",
                "justification": "LLM response could not satisfy the JSON contract.",
                "uncertainty_flag": "HIGH",
                "support_level": "unsupported",
                "support_rationale": failure_reason,
            }
            for sq_id in sq_ids
        ]
    }


def _answers_from_stage_artifact(
    artifact: dict, sq_ids: list[str], validation_status: str
) -> dict[str, dict]:
    answers_by_sq = {answer.get("sq_id"): answer for answer in artifact.get("answers", [])}
    parsed = {}
    for sq_id in sq_ids:
        answer = dict(
            answers_by_sq.get(sq_id)
            or _stage_fallback_artifact([sq_id], f"Missing JSON answer for SQ {sq_id}")[
                "answers"
            ][0]
        )
        answer.pop("sq_id", None)
        if validation_status != "validated":
            answer["contract_validation_status"] = validation_status
        parsed[sq_id] = answer
    return parsed


def _annotate_log_sources(log: list[dict], chunk_sources: list[str]) -> list[dict]:
    if not chunk_sources:
        return log
    return [{**entry, "chunk_sources": chunk_sources} for entry in log]


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
) -> tuple[dict[str, dict], list[dict]]:
    result = run_domain_sq_stage(
        state,
        DomainSqStage(
            node_name=node_name,
            sq_ids=tuple(sq_ids),
            source_domain=source_domain,
            build_prompt=lambda _state: prompt,
        ),
    )
    return result["sq_answers"], result["llm_call_log"]
