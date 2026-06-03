import json
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from rob2_pipeline.judges.domain1 import judge_domain1
from rob2_pipeline.llm_contracts import call_json_contract_llm
from rob2_pipeline.nodes.common import (
    add_domain_judgment_with_pivotality_tests,
    merge_sq_answers,
)
from rob2_pipeline.nodes.domain_context import build_domain1_context
from rob2_pipeline.nodes.domain_helpers import DomainSqStage, run_domain_sq_stage
from rob2_pipeline.prompts import PROMPT_DOMAIN1
from rob2_pipeline.state import RoB2State
from rob2_pipeline.types import EvidencePacket


D1_CLASSIFIER_SCHEMA_VERSION = "d1-sq-classifier-v1"
D1_CLASSIFIER_PROMPT_VERSION = "d1-sq-classifier-prompt-v1"
D1_SQ_IDS = ("1.1", "1.2", "1.3")


class D1SqAnswerArtifact(BaseModel):
    sq_id: Literal["1.1", "1.2", "1.3"]
    answer: Literal["Y", "PY", "PN", "N", "NI"]
    quote: str = Field(min_length=1)
    justification: str = Field(min_length=1)
    support_level: Literal["strong", "moderate", "weak", "unsupported"]
    support_rationale: str = Field(min_length=1)
    uncertainty: bool
    packet_artifact_id: str = Field(min_length=1)
    decision_table_artifact_id: str = Field(min_length=1)
    supporting_fact_artifact_ids: list[str] = Field(default_factory=list)


class D1SqClassifierArtifact(BaseModel):
    schema_version: Literal["d1-sq-classifier-v1"]
    domain: Literal["d1"]
    answers: list[D1SqAnswerArtifact] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def require_each_d1_sq_once(self) -> "D1SqClassifierArtifact":
        sq_ids = [answer.sq_id for answer in self.answers]
        if set(sq_ids) != set(D1_SQ_IDS) or len(sq_ids) != len(set(sq_ids)):
            raise ValueError("D1 classifier artifact must include each D1 SQ exactly once")
        return self


def build_domain1_prompt(state: RoB2State) -> str:
    context = build_domain1_context(state)
    return PROMPT_DOMAIN1.format(
        intervention=state["intervention"],
        comparator=state["comparator"],
        outcome=state["outcome"],
        randomization_text=context.randomization_text,
        baseline_text=context.baseline_text,
        consort_text=context.consort_text,
        rag_text=context.rag_text,
        ctgov_design=context.ctgov_design,
    )


DOMAIN1_STAGE = DomainSqStage(
    node_name="domain1_sq",
    sq_ids=D1_SQ_IDS,
    source_domain="d1",
    build_prompt=build_domain1_prompt,
)


def domain1_sq_node(state: RoB2State) -> RoB2State:
    if _has_ready_d1_packets(state):
        return run_domain1_json_classifier(state)
    return run_domain_sq_stage(state, DOMAIN1_STAGE)


def run_domain1_json_classifier(state: RoB2State) -> RoB2State:
    prompt = build_domain1_classifier_prompt(state)
    result = call_json_contract_llm(
        state,
        prompt,
        "domain1_sq_json",
        schema_model=D1SqClassifierArtifact,
        schema_version=D1_CLASSIFIER_SCHEMA_VERSION,
        prompt_version=D1_CLASSIFIER_PROMPT_VERSION,
        fallback_factory=_d1_classifier_fallback,
    )
    artifact = result["artifact"] if isinstance(result, dict) else result.artifact
    artifact = _enforce_packet_bound_answers(state, artifact)
    log = result["log"] if isinstance(result, dict) else result.log
    parsed = _artifact_to_sq_answers(artifact)
    return {
        "sq_answers": merge_sq_answers(state, parsed),
        "llm_call_log": log,
        "d1_sq_classifier_artifact": artifact,
    }


def build_domain1_classifier_prompt(state: RoB2State) -> str:
    packets = _d1_packets(state)
    payload = {
        "schema_version": D1_CLASSIFIER_SCHEMA_VERSION,
        "trial": {
            "intervention": state.get("intervention", ""),
            "comparator": state.get("comparator", ""),
            "outcome": state.get("outcome", ""),
        },
        "task": (
            "Classify Domain 1 signaling questions using only the supplied "
            "evidence_packets and their decision_table rows. Do not introduce "
            "or infer evidence outside these packets. Use NI when packet "
            "evidence is insufficient for a non-NI answer."
        ),
        "required_output_fields": [
            "sq_id",
            "answer",
            "quote",
            "justification",
            "support_level",
            "support_rationale",
            "uncertainty",
            "packet_artifact_id",
            "decision_table_artifact_id",
            "supporting_fact_artifact_ids",
        ],
        "evidence_packets": [_prompt_packet(packet) for packet in packets.values()],
    }
    return (
        "Return JSON matching the local D1SqClassifierArtifact schema.\n"
        "The packet decision tables are the context authority.\n\n"
        f"{json.dumps(payload, indent=2)}"
    )


def _has_ready_d1_packets(state: RoB2State) -> bool:
    packets = _d1_packets(state)
    if set(packets) != set(D1_SQ_IDS):
        return False
    return all(
        (packet.get("packet_readiness") or {}).get("status", "ready") == "ready"
        for packet in packets.values()
    )


def _d1_packets(state: RoB2State) -> dict[str, EvidencePacket]:
    return {
        sq_id: packet
        for sq_id, packet in (state.get("evidence_packets") or {}).items()
        if sq_id in D1_SQ_IDS and packet.get("domain") == "d1"
    }


def _prompt_packet(packet: EvidencePacket) -> dict:
    return {
        "artifact_id": packet.get("artifact_id", ""),
        "schema_version": packet.get("schema_version", ""),
        "sq_id": packet.get("sq_id", ""),
        "required_evidence": packet.get("required_evidence", []),
        "sources": [
            {
                "text": source.get("text", ""),
                "section": source.get("section", ""),
                "page_numbers": source.get("page_numbers", []),
                "document_name": source.get("document_name", ""),
                "document_role": source.get("document_role", ""),
                "source_kind": source.get("source_kind", ""),
            }
            for source in packet.get("sources", [])
        ],
        "candidate_facts": packet.get("candidate_facts", []),
        "gaps": packet.get("gaps", []),
        "failed_claims": packet.get("failed_claims", []),
        "missing_evidence": packet.get("missing_evidence", []),
        "negative_flags": packet.get("negative_flags", []),
        "decision_table": packet.get("decision_table", {}),
        "packet_readiness": packet.get("packet_readiness", {}),
    }


def _artifact_to_sq_answers(artifact: dict) -> dict[str, dict]:
    answers = {}
    for answer in artifact.get("answers", []):
        answers[answer["sq_id"]] = {
            "answer": answer["answer"],
            "quote": answer["quote"],
            "justification": answer["justification"],
            "support_level": answer["support_level"],
            "support_rationale": answer["support_rationale"],
            "uncertainty": answer["uncertainty"],
            "packet_artifact_id": answer["packet_artifact_id"],
            "decision_table_artifact_id": answer["decision_table_artifact_id"],
            "supporting_fact_artifact_ids": answer.get("supporting_fact_artifact_ids", []),
            "classifier_schema_version": artifact.get("schema_version", ""),
        }
        if answer.get("outside_packet_evidence_rejected"):
            answers[answer["sq_id"]]["outside_packet_evidence_rejected"] = True
    return answers


def _enforce_packet_bound_answers(state: RoB2State, artifact: dict) -> dict:
    packets = _d1_packets(state)
    bounded_answers = []
    for answer in artifact.get("answers", []):
        packet = packets.get(answer.get("sq_id", ""))
        if packet is None or _answer_quote_is_packet_bound(answer, packet):
            bounded_answers.append(answer)
            continue
        bounded_answers.append(
            {
                **answer,
                "answer": "NI",
                "quote": "No relevant text found",
                "justification": (
                    "The classifier cited evidence that was not present in the "
                    "selected evidence packet."
                ),
                "support_level": "unsupported",
                "support_rationale": (
                    "Outside-packet evidence was rejected by local contract validation."
                ),
                "uncertainty": True,
                "supporting_fact_artifact_ids": [],
                "outside_packet_evidence_rejected": True,
            }
        )
    return {**artifact, "answers": bounded_answers}


def _answer_quote_is_packet_bound(answer: dict, packet: EvidencePacket) -> bool:
    quote = _normalize_quote(answer.get("quote", ""))
    if quote in {"", "no relevant text found", "not applicable"}:
        return True
    packet_texts = []
    packet_texts.extend(source.get("text", "") for source in packet.get("sources", []))
    packet_texts.extend(fact.get("quote", "") for fact in packet.get("candidate_facts", []))
    packet_texts.extend(fact.get("claim", "") for fact in packet.get("candidate_facts", []))
    return any(quote in _normalize_quote(text) for text in packet_texts)


def _normalize_quote(text: str) -> str:
    return " ".join(str(text).casefold().split())


def _d1_classifier_fallback(reason: str) -> dict:
    return {
        "schema_version": D1_CLASSIFIER_SCHEMA_VERSION,
        "domain": "d1",
        "answers": [
            {
                "sq_id": sq_id,
                "answer": "NI",
                "quote": "No relevant text found",
                "justification": f"JSON D1 classifier fallback was used: {reason}",
                "support_level": "unsupported",
                "support_rationale": reason,
                "uncertainty": True,
                "packet_artifact_id": f"evidence-packet:d1:{sq_id}",
                "decision_table_artifact_id": f"decision-table:d1:{sq_id}",
                "supporting_fact_artifact_ids": [],
            }
            for sq_id in D1_SQ_IDS
        ],
    }


def domain1_judge_node(state: RoB2State) -> RoB2State:
    judgment, rationale = judge_domain1(state["sq_answers"])
    return add_domain_judgment_with_pivotality_tests(
        state, "D1", judgment, rationale, judge_domain1, DOMAIN1_STAGE.sq_ids
    )
