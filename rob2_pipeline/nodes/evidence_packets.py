"""Build small, SQ-specific evidence packets from retrieved source chunks."""

from __future__ import annotations

from rob2_pipeline.methodology import (
    DOMAIN1_METHODOLOGY,
    DOMAIN2_ASSIGNMENT_METHODOLOGY,
    DOMAIN3_METHODOLOGY,
    DOMAIN4_METHODOLOGY,
    DOMAIN5_METHODOLOGY,
)
from rob2_pipeline.methodology.types import DomainMethodology
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
from rob2_pipeline.types import DecisionTable, EvidenceFact, EvidencePacket, RetrievalGrade


METHODOLOGY_BY_DOMAIN: dict[str, DomainMethodology] = {
    "d1": DOMAIN1_METHODOLOGY,
    "d2": DOMAIN2_ASSIGNMENT_METHODOLOGY,
    "d3": DOMAIN3_METHODOLOGY,
    "d4": DOMAIN4_METHODOLOGY,
    "d5": DOMAIN5_METHODOLOGY,
}


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
        decision_table = _render_decision_table(packet.get("decision_table", {}))
        parts.append(
            "\n".join(
                [
                    f"SQ {sq_id} verified evidence packet",
                    f"Required evidence: {', '.join(packet.get('required_evidence', []))}",
                    f"Missing evidence: {missing}",
                    f"Negative flags: {flags}",
                    decision_table,
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
    decision_table = build_decision_table(
        contract=contract,
        facts=facts,
        gaps=gaps,
        missing=missing,
    )
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
        decision_table=decision_table,
        text=text,
        retrieval_confidence=retrieval_confidence,
        missing_evidence=missing,
        negative_flags=flags,
        packet_grade=grade_packet(retrieval_confidence, missing, flags),
    )


def build_decision_table(
    *,
    contract: EvidenceContract,
    facts: list[EvidenceFact],
    gaps: list[dict],
    missing: list[str],
) -> DecisionTable:
    methodology = METHODOLOGY_BY_DOMAIN[contract.domain]
    rule_card = methodology.rule_cards[contract.sq_id]
    allowed_answers = list(rule_card.response_rules)
    fact_summaries = [_fact_summary(fact) for fact in facts]
    gap_summaries = [_gap_summary(gap) for gap in gaps]
    rows = []
    for answer, rule in rule_card.response_rules.items():
        is_default = answer == "NI"
        has_support = bool(fact_summaries) and not missing
        rows.append(
            {
                "answer": answer,
                "rule": rule.guidance,
                "allowed_by_packet": is_default or has_support,
                "supporting_facts": [] if is_default else fact_summaries,
                "evidence_gaps": gap_summaries if is_default or missing else [],
                "insufficient_evidence_default": is_default,
            }
        )
    if "NI" not in allowed_answers:
        allowed_answers.append("NI")
        rows.append(
            {
                "answer": "NI",
                "rule": "Use when selected packet evidence is insufficient for the applicable non-NA answer options.",
                "allowed_by_packet": bool(missing or not fact_summaries),
                "supporting_facts": [],
                "evidence_gaps": gap_summaries,
                "insufficient_evidence_default": True,
            }
        )
    return DecisionTable(
        artifact_id=f"decision-table:{contract.domain}:{contract.sq_id}",
        schema_version="1.0",
        sq_id=contract.sq_id,
        allowed_answers=allowed_answers,
        rows=rows,
        default_insufficient_evidence_answer="NI",
        classifier_instruction=(
            "Choose only from selected packet evidence. Do not use outside "
            "knowledge or unstated context; when selected packet evidence is "
            "insufficient for a non-NA option, choose NI unless the RoB 2 "
            "branching rules make the SQ not applicable."
        ),
    )


def _fact_summary(fact: EvidenceFact) -> dict:
    return {
        "artifact_id": fact.get("artifact_id", ""),
        "fact_type": fact.get("fact_type", ""),
        "claim": fact.get("claim", ""),
        "support_level": fact.get("support_level", ""),
        "source": fact.get("document_name", "") or fact.get("source_section", ""),
    }


def _gap_summary(gap: dict) -> dict:
    return {
        "artifact_id": gap.get("artifact_id", ""),
        "missing_evidence": gap.get("missing_evidence", ""),
        "reason": gap.get("reason", ""),
    }


def _render_decision_table(decision_table: dict) -> str:
    if not decision_table:
        return ""
    lines = [
        "Mini decision table:",
        f"Classifier instruction: {decision_table.get('classifier_instruction', '')}",
    ]
    for row in decision_table.get("rows", []):
        answer = row.get("answer", "")
        if row.get("insufficient_evidence_default"):
            support = "Default when selected packet evidence is insufficient"
        else:
            facts = row.get("supporting_facts") or []
            if facts:
                support = "; ".join(
                    compact(fact.get("claim", ""), 180) for fact in facts[:2]
                )
            else:
                support = "No selected packet fact currently supports this option"
        gaps = row.get("evidence_gaps") or []
        if gaps:
            gap_text = "; gaps: " + ", ".join(
                gap.get("missing_evidence", "") for gap in gaps[:3]
            )
        else:
            gap_text = ""
        lines.append(f"- {answer}: {support}{gap_text}")
    return "\n".join(lines)
