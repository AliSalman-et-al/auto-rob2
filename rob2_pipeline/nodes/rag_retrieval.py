from rob2_pipeline.models import PaperEvidence, format_evidence
from rob2_pipeline.ingestion.evidence import extract_censoring_context
from rob2_pipeline.rag import (
    DOMAIN_SECTION_FILTERS,
    build_filtered_index,
    build_index,
    grade_retrieved_context,
    retrieve_adaptive,
)
from rob2_pipeline.rag_queries import domain_queries
from rob2_pipeline.state import RoB2State

_DOMAINS = ["d1", "d2", "d3", "d4", "d5"]


def _sections_fallback(evidence: PaperEvidence) -> dict[str, str]:
    return {
        "d1": "\n\n".join(
            part
            for part in [
                format_evidence(evidence["d1_randomization"]),
                format_evidence(evidence["baseline_table"]),
                format_evidence(evidence["consort_flow"]),
            ]
            if part
        ),
        "d2_blinding": format_evidence(evidence["d2_blinding"]),
        "d2_deviations": "\n\n".join(
            part
            for part in [
                format_evidence(evidence["d2_blinding"]),
                format_evidence(evidence["methods"]),
            ]
            if part
        ),
        "d2_analysis": "\n\n".join(
            part
            for part in [
                format_evidence(evidence["d4_outcome_meas"]),
                format_evidence(evidence["results"]),
            ]
            if part
        ),
        "d3": "\n\n".join(
            part
            for part in [
                format_evidence(evidence["d3_missing_data"]),
                format_evidence(evidence["consort_flow"]),
            ]
            if part
        ),
        "d4_measurement": format_evidence(evidence["d4_outcome_meas"]),
        "d4_assessor": format_evidence(evidence["d2_blinding"]),
        "d5": "\n\n".join(
            part
            for part in [
                format_evidence(evidence["d5_registration"]),
                format_evidence(evidence["results"]),
            ]
            if part
        ),
    }


def _empty_metadata() -> dict[str, list[dict]]:
    return {domain: [] for domain in _DOMAINS}


def _compat_contexts(domain_contexts: dict[str, str]) -> dict[str, str]:
    return {
        "d1": domain_contexts.get("d1", ""),
        "d2_blinding": domain_contexts.get("d2", ""),
        "d2_deviations": domain_contexts.get("d2", ""),
        "d2_analysis": domain_contexts.get("d2", ""),
        "d3": domain_contexts.get("d3", ""),
        "d4_measurement": domain_contexts.get("d4", ""),
        "d4_assessor": domain_contexts.get("d4", ""),
        "d5": domain_contexts.get("d5", ""),
    }


def rag_retrieval_node(state: RoB2State) -> dict:
    chunks = state.get("docling_chunks") or []
    rag_chunk_metadata = _empty_metadata()
    retrieval_grades = {}
    trial_indexes = state.get("trial_retrieval_indexes") or {}

    if not chunks:
        rag_contexts = _sections_fallback(state["evidence"])
        retrieval_grades = {
            domain: grade_retrieved_context(domain, rag_contexts.get(domain, ""), [])
            for domain in _DOMAINS
        }
    else:
        try:
            if not trial_indexes:
                trial_indexes = {
                    "index": build_index(chunks),
                    "filtered": {
                        domain: build_filtered_index(chunks, keywords)
                        for domain, keywords in DOMAIN_SECTION_FILTERS.items()
                    },
                }
            index = trial_indexes.get("index") or build_index(chunks)
            filtered_indexes = trial_indexes.get("filtered") or {}
            domain_contexts: dict[str, str] = {}
            for domain in _DOMAINS:
                filtered_index = filtered_indexes.get(domain)
                if domain not in filtered_indexes:
                    filtered_index = build_filtered_index(
                        chunks, DOMAIN_SECTION_FILTERS.get(domain, [])
                    )
                text, metas = retrieve_adaptive(
                    index, filtered_index, domain_queries(domain)
                )
                domain_contexts[domain] = text
                rag_chunk_metadata[domain] = [dict(meta) for meta in metas]
                retrieval_grades[domain] = grade_retrieved_context(
                    domain, text, rag_chunk_metadata[domain]
                )
            rag_contexts = _compat_contexts(domain_contexts)
        except Exception as error:  # noqa: BLE001
            rag_contexts = _sections_fallback(state["evidence"])
            retrieval_grades = {
                domain: grade_retrieved_context(domain, rag_contexts.get(domain, ""), [])
                for domain in _DOMAINS
            }
            retrieval_mode_warning = (
                "RAG retrieval failed; used section evidence fallback: "
                f"{error}"
            )
            state["evidence"]["warnings"].append(retrieval_mode_warning)

    censoring = extract_censoring_context(
        state.get("full_text", ""), state.get("outcome", "")
    )
    if censoring:
        rag_contexts["d3"] = (rag_contexts.get("d3", "") + "\n\n" + censoring).strip()
    return {
        "rag_contexts": rag_contexts,
        "rag_chunk_metadata": rag_chunk_metadata,
        "retrieval_grades": retrieval_grades,
        "trial_retrieval_indexes": trial_indexes if chunks else {},
    }
