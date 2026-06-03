"""Grade evidence packets and convert selected sources into facts."""

from __future__ import annotations

import re

from rob2_pipeline.nodes.evidence_contracts import (
    DENOMINATOR_RE,
    PRESPEC_TERMS,
    RESULT_STAT_RE,
    SENTENCE_SPLIT_RE,
    EvidenceContract,
)
from rob2_pipeline.nodes.evidence_source_selection import (
    contract_terms,
    looks_like_wrong_outcome,
)
from rob2_pipeline.state import RoB2State
from rob2_pipeline.types import EvidenceFact, PacketReadiness, PacketSource, RetrievalGrade
from rob2_pipeline.types import EvidenceGap


def missing_evidence(
    contract: EvidenceContract, text: str, matched: set[str]
) -> list[str]:
    missing: list[str] = []
    lowered = text.casefold()
    if not text.strip():
        missing.extend(contract.required_evidence)
    elif not matched:
        missing.extend(contract.required_evidence)
    if contract.needs_denominator and not DENOMINATOR_RE.search(text):
        missing.append("denominator_or_percentage")
    if contract.needs_prespecification and not any(
        term in lowered for term in PRESPEC_TERMS
    ):
        missing.append("protocol_or_registration")
    return list(dict.fromkeys(missing))


def negative_flags(
    state: RoB2State,
    contract: EvidenceContract,
    selected: list[PacketSource],
    text: str,
) -> list[str]:
    flags: list[str] = []
    if contract.outcome_bound and looks_like_wrong_outcome(
        state.get("outcome", ""), text
    ):
        flags.append("possible_wrong_outcome_context")
    # Only real RAG chunks need page numbers. Structured sources such as
    # section text and ClinicalTrials.gov have no page metadata by design.
    if any(
        source.get("source_kind", "rag_chunk") == "rag_chunk"
        and not source.get("page_numbers")
        for source in selected
    ):
        flags.append("missing_page_source")
    lowered = text.casefold()
    if (
        contract.domain == "d5"
        and RESULT_STAT_RE.search(text)
        and not any(term in lowered for term in PRESPEC_TERMS)
    ):
        flags.append("results_without_prespecification")
    if text and not selected:
        flags.append("generic_background_only")
    return list(dict.fromkeys(flags))


def confidence(
    contract: EvidenceContract,
    selected: list[PacketSource],
    missing: list[str],
    flags: list[str],
) -> float:
    if not selected:
        return 0.0
    matched = {term for source in selected for term in source.get("matched_terms", [])}
    term_score = min(1.0, len(matched) / max(1, min(4, len(contract_terms(contract)))))
    source_score = min(1.0, len(selected) / 2)
    penalty = 0.2 * len(missing) + 0.15 * len(flags)
    return round(
        max(0.0, min(1.0, (term_score * 0.7) + (source_score * 0.3) - penalty)), 3
    )


def grade_packet(
    confidence: float, missing: list[str], flags: list[str]
) -> RetrievalGrade:
    coverage = 0.0 if missing else 1.0
    return {
        "relevance": confidence,
        "coverage": coverage,
        "missing_evidence": missing,
        "retry_recommended": bool(missing or flags or confidence < 0.35),
    }


def packet_readiness(
    *,
    sq_id: str,
    missing: list[str],
    flags: list[str],
    contradictions: list[dict],
    facts: list[EvidenceFact],
    confidence: float,
) -> PacketReadiness:
    mechanical_status = "complete" if not missing and not flags else "incomplete"
    mechanical = {
        "status": mechanical_status,
        "missing_evidence": missing,
        "negative_flags": flags,
        "contradictions": [item.get("label", "contradiction") for item in contradictions],
    }
    support_levels = [fact.get("support_level", "unsupported") for fact in facts]
    adequate_support = {"strong", "moderate"}
    semantic_status = (
        "adequate"
        if support_levels and all(level in adequate_support for level in support_levels)
        else "limited"
    )
    semantic = {
        "status": semantic_status,
        "support_levels": support_levels,
        "confidence": confidence,
    }
    status = "ready"
    blocking_reason = ""
    if missing:
        status = "needs_retrieval_repair"
        blocking_reason = "Selected packet sources do not mechanically cover required evidence."
    elif contradictions:
        status = "needs_contradiction_resolution"
        blocking_reason = "Selected packet sources contain unresolved contradictory claims."
    elif any(flag in {"missing_page_source", "quote_untraceable"} for flag in flags):
        status = "needs_quote_adjudication"
        blocking_reason = "Selected packet sources need quote or provenance adjudication."
    elif flags:
        status = "needs_retrieval_repair"
        blocking_reason = "Selected packet sources have mechanical negative flags."
    elif semantic_status != "adequate":
        status = "audit_limited"
        blocking_reason = "Selected packet facts are semantically limited."
    return {
        "artifact_id": f"packet-readiness:{sq_id}",
        "schema_version": "1.0",
        "sq_id": sq_id,
        "status": status,
        "mechanical_completeness": mechanical,
        "semantic_adequacy": semantic,
        "blocking_reason": blocking_reason,
    }


def source_to_fact(
    contract: EvidenceContract, source: PacketSource, confidence: float
) -> EvidenceFact:
    quote = best_sentence(source.get("text", ""), source.get("matched_terms", []))
    support_level = support_level_for_confidence(confidence, bool(quote))
    provenance = {
        "document_id": source.get("document_id", "primary"),
        "document_name": source.get("document_name", "Primary paper"),
        "document_role": source.get("document_role", "primary"),
        "source_kind": source.get("source_kind", "rag_chunk"),
        "source_path": source.get("source_path", ""),
        "source_section": source.get("section", ""),
        "page_numbers": source.get("page_numbers", []),
        "retrieval_date": source.get("retrieval_date", ""),
        "api_response_hash": source.get("api_response_hash", ""),
    }
    return EvidenceFact(
        artifact_id=f"evidence-fact:{contract.domain}:{contract.sq_id}:{slug(contract.required_evidence[0] if contract.required_evidence else 'evidence')}",
        fact_type=contract.required_evidence[0]
        if contract.required_evidence
        else "evidence",
        domain=contract.domain,
        sq_ids=[contract.sq_id],
        claim_type=claim_type_for_contract(contract),
        claim=compact(quote, 240),
        quote=quote,
        source_section=source.get("section", ""),
        page_numbers=source.get("page_numbers", []),
        confidence=confidence,
        support_level=support_level,
        support_status="supported" if quote else "missing",
        uncertainty=support_level in {"weak", "unsupported"},
        document_id=source.get("document_id", ""),
        document_name=source.get("document_name", ""),
        document_role=source.get("document_role", ""),
        source_kind=source.get("source_kind", ""),
        source_path=source.get("source_path", ""),
        retrieval_date=source.get("retrieval_date", ""),
        api_response_hash=source.get("api_response_hash", ""),
        provenance=provenance,
    )


def missing_label_to_gap(contract: EvidenceContract, label: str) -> EvidenceGap:
    return EvidenceGap(
        artifact_id=f"evidence-gap:{contract.domain}:{contract.sq_id}:{slug(label)}",
        domain=contract.domain,
        sq_ids=[contract.sq_id],
        missing_evidence=label,
        reason=f"No selected source supported required evidence label {label}.",
    )


def missing_label_to_failed_claim(
    contract: EvidenceContract,
    label: str,
    source: PacketSource | None,
) -> EvidenceFact:
    provenance = provenance_for_source(source or {})
    return EvidenceFact(
        artifact_id=f"evidence-fact:{contract.domain}:{contract.sq_id}:{slug(label)}:failed",
        fact_type=label,
        domain=contract.domain,
        sq_ids=[contract.sq_id],
        claim_type=claim_type_for_contract(contract),
        claim=f"Required evidence label {label} was not supported by selected sources.",
        quote="",
        source_section=provenance["source_section"],
        page_numbers=provenance["page_numbers"],
        confidence=0.0,
        support_level="unsupported",
        support_status="failed",
        uncertainty=True,
        failure_reason=f"No selected source supported required evidence label {label}.",
        document_id=provenance["document_id"],
        document_name=provenance["document_name"],
        document_role=provenance["document_role"],
        source_kind=provenance["source_kind"],
        source_path=provenance["source_path"],
        retrieval_date=provenance["retrieval_date"],
        api_response_hash=provenance["api_response_hash"],
        provenance=provenance,
    )


def provenance_for_source(source: PacketSource) -> dict:
    return {
        "document_id": source.get("document_id") or "unknown",
        "document_name": source.get("document_name") or "Unknown document",
        "document_role": source.get("document_role") or "unknown",
        "source_kind": source.get("source_kind") or "unknown",
        "source_path": source.get("source_path") or "unknown",
        "source_section": source.get("section") or "Unknown",
        "page_numbers": source.get("page_numbers", []),
        "retrieval_date": source.get("retrieval_date", ""),
        "api_response_hash": source.get("api_response_hash", ""),
    }


def contradictions_for_sources(
    contract: EvidenceContract, selected: list[PacketSource]
) -> list[dict]:
    if contract.sq_id != "1.2":
        return []

    positive: PacketSource | None = None
    negative: PacketSource | None = None
    for source in selected:
        text = source.get("text", "")
        if _has_negated_concealment(text):
            negative = negative or source
            continue
        if _has_positive_concealment(text):
            positive = positive or source

    if not positive or not negative:
        return []

    return [
        {
            "artifact_id": f"evidence-contradiction:{contract.domain}:{contract.sq_id}:allocation-concealment",
            "domain": contract.domain,
            "sq_ids": [contract.sq_id],
            "label": "allocation_concealment",
            "reason": "Selected sources make conflicting claims about allocation concealment.",
            "dominant_source": _source_summary(positive),
            "conflicting_source": _source_summary(negative),
        }
    ]


def _has_positive_concealment(text: str) -> bool:
    lowered = text.casefold()
    return "conceal" in lowered and not _has_negated_concealment(text)


def _has_negated_concealment(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:not|no|without|unconcealed|open)\b.{0,40}\bconceal",
            text,
            re.I,
        )
        or re.search(r"\bconceal\w*\b.{0,40}\b(?:not|no|without)\b", text, re.I)
    )


def _source_summary(source: PacketSource) -> dict:
    return {
        "document_id": source.get("document_id", ""),
        "document_name": source.get("document_name", ""),
        "document_role": source.get("document_role", ""),
        "source_kind": source.get("source_kind", ""),
        "source_path": source.get("source_path", ""),
        "section": source.get("section", ""),
        "page_numbers": source.get("page_numbers", []),
        "quote": compact(source.get("text", ""), 240),
    }


def best_sentence(text: str, terms: list[str]) -> str:
    sentences = SENTENCE_SPLIT_RE.split(text.strip())
    for sentence in sentences:
        if any(term.casefold() in sentence.casefold() for term in terms):
            return sentence.strip()
    return sentences[0].strip() if sentences and sentences[0].strip() else ""


def compact(text: str, max_chars: int) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def support_level_for_confidence(confidence: float, has_quote: bool) -> str:
    if not has_quote:
        return "unsupported"
    if confidence >= 0.75:
        return "strong"
    if confidence >= 0.45:
        return "moderate"
    return "weak"


def claim_type_for_contract(contract: EvidenceContract) -> str:
    if contract.domain in {"d1", "d2"}:
        return "trial_method"
    if contract.domain in {"d3", "d4"}:
        return "outcome_measurement"
    if contract.domain == "d5":
        return "result_reporting"
    return "other"


def slug(text: str) -> str:
    slugged = re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-")
    return slugged or "evidence"
