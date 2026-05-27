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
from rob2_pipeline.nodes.verification import classify_evidence_support
from rob2_pipeline.state import RoB2State
from rob2_pipeline.types import EvidenceFact, PacketSource, RetrievalGrade


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


def source_to_fact(
    contract: EvidenceContract,
    source: PacketSource,
    confidence: float,
    state: RoB2State | None = None,
) -> EvidenceFact:
    quote = best_sentence(source.get("text", ""), source.get("matched_terms", []))
    support = classify_evidence_support(quote, source.get("text", ""))
    support_status = support["status"]
    if (
        state is not None
        and contract.outcome_bound
        and looks_like_wrong_outcome(state.get("outcome", ""), quote)
    ):
        support_status = "source-mismatched"
    return EvidenceFact(
        fact_type=contract.required_evidence[0]
        if contract.required_evidence
        else "evidence",
        domain=contract.domain,
        sq_ids=[contract.sq_id],
        claim=compact(quote, 240),
        quote=quote,
        source_section=source.get("section", ""),
        page_numbers=source.get("page_numbers", []),
        confidence=confidence,
        support_status=support_status if quote else "unsupported",
        document_id=source.get("document_id", ""),
        document_name=source.get("document_name", ""),
        document_role=source.get("document_role", ""),
        source_kind=source.get("source_kind", ""),
        source_path=source.get("source_path", ""),
    )


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
