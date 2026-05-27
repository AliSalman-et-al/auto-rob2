"""Build small, SQ-specific evidence packets from retrieved source chunks."""

from __future__ import annotations

from rob2_pipeline.nodes.evidence_contracts import CONTRACTS, EvidenceContract
from rob2_pipeline.nodes.evidence_packet_grading import (
    compact,
    confidence,
    grade_packet,
    missing_evidence,
    negative_flags,
    source_to_fact,
)
from rob2_pipeline.nodes.evidence_source_selection import candidate_sources
from rob2_pipeline.nodes.evidence_source_selection import aliases_for_outcome
from rob2_pipeline.nodes.evidence_source_selection import looks_like_wrong_outcome
from rob2_pipeline.nodes.evidence_source_selection import role_rank
from rob2_pipeline.state import RoB2State
from rob2_pipeline.types import EvidenceFact, EvidencePacket, RetrievalGrade


def evidence_packet_builder_node(state: RoB2State) -> RoB2State:
    return build_evidence_packets(state)


def build_evidence_packets(state: RoB2State) -> dict:
    packets: dict[str, EvidencePacket] = {}
    facts: dict[str, list[EvidenceFact]] = {}
    grades: dict[str, RetrievalGrade] = {}
    for sq_id, contract in CONTRACTS.items():
        packet = _build_packet_for_contract(state, contract)
        packets[sq_id] = packet
        facts[sq_id] = packet.get("candidate_facts", [])
        grades[sq_id] = packet.get(
            "packet_grade",
            grade_packet(
                0.0,
                packet.get("missing_evidence", []),
                packet.get("negative_flags", []),
            ),
        )
    return {
        "evidence_packets": packets,
        "evidence_facts": facts,
        "packet_grades": grades,
    }


def packet_block_for_domain(
    evidence_packets: dict[str, EvidencePacket], domain: str, max_chars: int = 6500
) -> str:
    parts: list[str] = []
    for sq_id in sorted(
        sq for sq, packet in evidence_packets.items() if packet.get("domain") == domain
    ):
        packet = evidence_packets[sq_id]
        sources = packet.get("sources", [])
        source_lines = []
        for source in sources[:3]:
            pages = source.get("page_numbers") or []
            page = f"page {pages[0]}" if pages else "no page"
            section = source.get("section") or "Unknown"
            document_name = source.get("document_name") or "Unknown document"
            document_role = (
                source.get("document_role") or source.get("source_kind") or "source"
            )
            text = compact(source.get("text", ""), 700)
            source_lines.append(
                f"- {document_role} ({document_name}), {page}, {section}: {text}"
            )
        missing = ", ".join(packet.get("missing_evidence", [])) or "none"
        flags = ", ".join(packet.get("negative_flags", [])) or "none"
        parts.append(
            "\n".join(
                [
                    f"SQ {sq_id} verified evidence packet",
                    f"Required evidence: {', '.join(packet.get('required_evidence', []))}",
                    f"Missing evidence: {missing}",
                    f"Negative flags: {flags}",
                    *source_lines,
                ]
            )
        )
    return compact("\n\n".join(parts), max_chars)


def _build_packet_for_contract(
    state: RoB2State, contract: EvidenceContract
) -> EvidencePacket:
    candidates = candidate_sources(state, contract)
    ranked_candidates = _outcome_scoped_candidates(state, contract, candidates)
    ranked = sorted(
        ranked_candidates,
        key=lambda source: (
            _outcome_rank(state, contract, source),
            -len(source.get("matched_terms", [])),
            role_rank(contract.domain, source.get("document_role", "")),
            source.get("score", 1e9),
        ),
    )
    selected = ranked[:3]
    text = "\n\n".join(
        source.get("text", "") for source in selected if source.get("text")
    )
    matched = {term for source in selected for term in source.get("matched_terms", [])}
    missing = missing_evidence(contract, text, matched)
    flags = negative_flags(state, contract, selected, text)
    retrieval_confidence = confidence(contract, selected, missing, flags)
    facts = [
        source_to_fact(contract, source, retrieval_confidence, state)
        for source in selected
        if source.get("text")
    ]
    return EvidencePacket(
        sq_id=contract.sq_id,
        domain=contract.domain,
        required_evidence=list(contract.required_evidence),
        sources=selected,
        candidate_facts=facts,
        text=text,
        retrieval_confidence=retrieval_confidence,
        missing_evidence=missing,
        negative_flags=flags,
        packet_grade=grade_packet(retrieval_confidence, missing, flags),
    )


def _outcome_scoped_candidates(
    state: RoB2State, contract: EvidenceContract, candidates: list
) -> list:
    if not contract.outcome_bound:
        return candidates
    on_scope = [
        source
        for source in candidates
        if _outcome_rank(state, contract, source) in (0, 1)
    ]
    return on_scope or candidates


def _outcome_rank(state: RoB2State, contract: EvidenceContract, source: dict) -> int:
    if not contract.outcome_bound:
        return 0
    outcome = str(state.get("outcome", "")).casefold().strip()
    text = str(source.get("text", "")).casefold()
    if outcome and any(alias in text for alias in aliases_for_outcome(outcome)):
        return 0
    if looks_like_wrong_outcome(outcome, text):
        return 2
    return 1
