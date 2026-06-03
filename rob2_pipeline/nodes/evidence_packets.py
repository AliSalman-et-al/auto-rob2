"""Build small, SQ-specific evidence packets from retrieved source chunks."""

from __future__ import annotations

from rob2_pipeline.nodes.evidence_contracts import CONTRACTS, EvidenceContract
from rob2_pipeline.nodes.evidence_packet_grading import (
    compact,
    confidence,
    contradictions_for_sources,
    grade_packet,
    missing_label_to_failed_claim,
    missing_label_to_gap,
    missing_evidence,
    negative_flags,
    source_to_fact,
)
from rob2_pipeline.nodes.evidence_source_selection import candidate_sources
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
    ranked = sorted(
        candidates,
        key=lambda source: (
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
        source_to_fact(contract, source, retrieval_confidence)
        for source in selected
        if source.get("text")
    ]
    gaps = [missing_label_to_gap(contract, label) for label in missing]
    failed_claims = [
        missing_label_to_failed_claim(
            contract,
            label,
            selected[0] if selected else None,
        )
        for label in missing
    ]
    contradictions = contradictions_for_sources(contract, selected)
    return EvidencePacket(
        artifact_id=f"evidence-packet:{contract.domain}:{contract.sq_id}",
        schema_version="1.0",
        sq_id=contract.sq_id,
        domain=contract.domain,
        outcome=str(state.get("outcome", "")),
        required_evidence=list(contract.required_evidence),
        sources=selected,
        candidate_facts=facts,
        gaps=gaps,
        failed_claims=failed_claims,
        contradictions=contradictions,
        text=text,
        retrieval_confidence=retrieval_confidence,
        missing_evidence=missing,
        negative_flags=flags,
        packet_grade=grade_packet(retrieval_confidence, missing, flags),
    )
