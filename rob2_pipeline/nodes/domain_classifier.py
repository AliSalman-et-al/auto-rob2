import json
import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from rob2_pipeline.llm_contracts import call_json_contract_llm
from rob2_pipeline.nodes.common import merge_sq_answers
from rob2_pipeline.state import RoB2State
from rob2_pipeline.types import EvidencePacket


ANSWER_CODES = ("Y", "PY", "PN", "N", "NI", "NA")
SUPPORT_LEVELS = ("strong", "moderate", "weak", "unsupported")


class DomainSqAnswerArtifact(BaseModel):
    sq_id: str = Field(min_length=1)
    answer: Literal["Y", "PY", "PN", "N", "NI", "NA"]
    quote: str = Field(min_length=1)
    justification: str = Field(min_length=1)
    support_level: Literal["strong", "moderate", "weak", "unsupported"]
    support_rationale: str = Field(min_length=1)
    uncertainty: bool
    packet_artifact_id: str = Field(min_length=1)
    decision_table_artifact_id: str = Field(min_length=1)
    supporting_fact_artifact_ids: list[str] = Field(default_factory=list)


class OutcomeSpecificConcernArtifact(BaseModel):
    concern: str = Field(min_length=1)
    support_level: Literal["strong", "moderate", "weak", "unsupported"]
    rationale: str = Field(min_length=1)


class DomainSqClassifierArtifact(BaseModel):
    schema_version: str = Field(pattern=r"^d[1-5]-sq-classifier-v1$")
    domain: Literal["d1", "d2", "d3", "d4", "d5"]
    stage: str = Field(min_length=1)
    branching: dict = Field(default_factory=dict)
    outcome_specific_concerns: list[OutcomeSpecificConcernArtifact] = Field(
        default_factory=list
    )
    answers: list[DomainSqAnswerArtifact] = Field(min_length=1)

    @model_validator(mode="after")
    def require_schema_domain_alignment(self) -> "DomainSqClassifierArtifact":
        if not self.schema_version.startswith(f"{self.domain}-"):
            raise ValueError("classifier schema_version must match domain")
        return self


def run_json_sq_classifier(
    state: RoB2State,
    *,
    domain: str,
    stage: str,
    sq_ids: tuple[str, ...],
    node_name: str,
    artifact_key: str,
    branching: dict | None = None,
    outcome_specific_concerns: list[dict] | None = None,
    postprocess=None,
) -> RoB2State:
    prompt = build_classifier_prompt(
        state,
        domain=domain,
        stage=stage,
        sq_ids=sq_ids,
        branching=branching or {},
        outcome_specific_concerns=outcome_specific_concerns or [],
    )
    schema_version = f"{domain}-sq-classifier-v1"
    result = call_json_contract_llm(
        state,
        prompt,
        node_name,
        schema_model=DomainSqClassifierArtifact,
        schema_version=schema_version,
        prompt_version=f"{domain}-{stage}-sq-classifier-prompt-v1",
        fallback_factory=lambda reason: _classifier_fallback(
            reason,
            domain=domain,
            stage=stage,
            sq_ids=sq_ids,
            branching=branching or {},
            outcome_specific_concerns=outcome_specific_concerns or [],
        ),
    )
    artifact = result["artifact"] if isinstance(result, dict) else result.artifact
    artifact = _enforce_stage_contract(state, artifact, domain=domain, sq_ids=sq_ids)
    log = result["log"] if isinstance(result, dict) else result.log
    parsed = _artifact_to_sq_answers(artifact)
    sq_answers = merge_sq_answers(state, parsed)
    if postprocess is not None:
        sq_answers = postprocess(state, sq_answers)
    return {
        "sq_answers": sq_answers,
        "llm_call_log": log,
        artifact_key: artifact,
        "domain_sq_classifier_artifacts": {domain: {stage: artifact}},
    }


def has_ready_packets(state: RoB2State, *, domain: str, sq_ids: tuple[str, ...]) -> bool:
    packets = packets_for_stage(state, domain=domain, sq_ids=sq_ids)
    if set(packets) != set(sq_ids):
        return False
    return all(
        (packet.get("packet_readiness") or {}).get("status", "ready") == "ready"
        for packet in packets.values()
    )


def packets_for_stage(
    state: RoB2State, *, domain: str, sq_ids: tuple[str, ...]
) -> dict[str, EvidencePacket]:
    return {
        sq_id: packet
        for sq_id, packet in (state.get("evidence_packets") or {}).items()
        if sq_id in sq_ids and packet.get("domain") == domain
    }


def build_classifier_prompt(
    state: RoB2State,
    *,
    domain: str,
    stage: str,
    sq_ids: tuple[str, ...],
    branching: dict,
    outcome_specific_concerns: list[dict],
) -> str:
    payload = {
        "schema_version": f"{domain}-sq-classifier-v1",
        "domain": domain,
        "stage": stage,
        "trial": {
            "intervention": state.get("intervention", ""),
            "comparator": state.get("comparator", ""),
            "outcome": state.get("outcome", ""),
            "outcome_type": state.get("outcome_type", ""),
            "effect_of_interest": state.get("effect_of_interest", "ITT"),
        },
        "branching": branching,
        "outcome_specific_concerns": outcome_specific_concerns,
        "task": (
            "Classify these signaling questions using only the supplied "
            "evidence_packets and their decision_table rows. Do not introduce "
            "or infer evidence outside these packets. Include support_level, "
            "support_rationale, and uncertainty for every non-NA answer. Use "
            "NI when packet evidence is insufficient for a non-NI answer. Quotes "
            "must be exact contiguous source text whenever possible; if a long "
            "quote is shortened with ellipses, every retained fragment must appear "
            "verbatim in the selected packet."
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
        "evidence_packets": [
            _prompt_packet(packet)
            for packet in packets_for_stage(
                state, domain=domain, sq_ids=sq_ids
            ).values()
        ],
    }
    return (
        "Return JSON matching the local DomainSqClassifierArtifact schema.\n"
        "The packet decision tables are the context authority.\n\n"
        f"{json.dumps(payload, indent=2)}"
    )


def _prompt_packet(packet: EvidencePacket) -> dict:
    return {
        "artifact_id": packet.get("artifact_id", ""),
        "schema_version": packet.get("schema_version", ""),
        "sq_id": packet.get("sq_id", ""),
        "required_evidence": packet.get("required_evidence", []),
        "contract": packet.get("contract", {}),
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


def _enforce_stage_contract(
    state: RoB2State, artifact: dict, *, domain: str, sq_ids: tuple[str, ...]
) -> dict:
    packets = packets_for_stage(state, domain=domain, sq_ids=sq_ids)
    answers = artifact.get("answers", [])
    answer_ids = [answer.get("sq_id") for answer in answers]
    if set(answer_ids) != set(sq_ids) or len(answer_ids) != len(set(answer_ids)):
        return _classifier_fallback(
            "Classifier artifact did not include each stage SQ exactly once.",
            domain=domain,
            stage=str(artifact.get("stage", "")),
            sq_ids=sq_ids,
            branching=artifact.get("branching", {}),
            outcome_specific_concerns=artifact.get("outcome_specific_concerns", []),
        )
    bounded_answers = []
    for answer in answers:
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
    normalized_packet_texts = [_normalize_quote(text) for text in packet_texts]
    if any(quote in text for text in normalized_packet_texts):
        return True
    fragments = [
        fragment
        for fragment in _quote_fragments(answer.get("quote", ""))
        if len(fragment) >= 24
    ]
    return bool(fragments) and all(
        any(fragment in text for text in normalized_packet_texts)
        for fragment in fragments
    )


# Typographic glyph variants that PDF extraction and LLM output disagree on,
# which would otherwise make an identical quote fail the substring check.
# Dashes: hyphen, non-breaking hyphen, figure dash, en/em dash, horizontal bar,
# hyphen bullet, minus sign, small/fullwidth forms. Quotes: curly single and
# double. Plus the ellipsis character.
_DASH_VARIANTS_RE = re.compile(r"[‐‑‒–—―⁃−﹘﹣－]")
_SINGLE_QUOTE_RE = re.compile(r"[‘’‚‛]")
_DOUBLE_QUOTE_RE = re.compile(r"[“”„‟]")


def _normalize_quote(text: str) -> str:
    # Fold typographic glyph variants so a quote that reproduces the source text
    # with different glyphs (e.g. "Kaplan‑Meier" with a non-breaking hyphen, or
    # "patients’" with a curly apostrophe) still matches. This does not weaken
    # the guard's purpose of rejecting hallucinated or paraphrased quotes: only
    # punctuation glyphs and whitespace are normalized, never words.
    normalized = _DASH_VARIANTS_RE.sub("-", str(text))
    normalized = _SINGLE_QUOTE_RE.sub("'", normalized)
    normalized = _DOUBLE_QUOTE_RE.sub('"', normalized)
    normalized = normalized.replace("…", "...")
    return " ".join(normalized.casefold().split())


def _quote_fragments(text: str) -> list[str]:
    normalized = str(text).replace("…", "...")
    return [
        _normalize_quote(fragment)
        for fragment in normalized.split("...")
        if _normalize_quote(fragment)
    ]


def _classifier_fallback(
    reason: str,
    *,
    domain: str,
    stage: str,
    sq_ids: tuple[str, ...],
    branching: dict,
    outcome_specific_concerns: list[dict],
) -> dict:
    return {
        "schema_version": f"{domain}-sq-classifier-v1",
        "domain": domain,
        "stage": stage,
        "branching": branching,
        "outcome_specific_concerns": outcome_specific_concerns,
        "answers": [
            {
                "sq_id": sq_id,
                "answer": "NI",
                "quote": "No relevant text found",
                "justification": f"JSON {domain.upper()} classifier fallback was used: {reason}",
                "support_level": "unsupported",
                "support_rationale": reason,
                "uncertainty": True,
                "packet_artifact_id": f"evidence-packet:{domain}:{sq_id}",
                "decision_table_artifact_id": f"decision-table:{domain}:{sq_id}",
                "supporting_fact_artifact_ids": [],
            }
            for sq_id in sq_ids
        ],
    }
