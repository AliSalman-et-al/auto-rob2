from rob2_pipeline.models import empty_paper_evidence
from rob2_pipeline.nodes.evidence_packets import (
    build_evidence_packets,
    packet_block_for_domain,
)


def test_evidence_packets_module_keeps_stable_public_api():
    from rob2_pipeline.nodes import evidence_packets

    assert callable(evidence_packets.evidence_packet_builder_node)
    assert callable(evidence_packets.build_evidence_packets)
    assert callable(evidence_packets.packet_block_for_domain)


def _state_with_chunks(
    domain: str, chunks: list[dict], outcome: str = "Progression-Free Survival"
) -> dict:
    evidence = empty_paper_evidence("test")
    return {
        "outcome": outcome,
        "evidence": evidence,
        "rag_chunk_metadata": {
            "d1": [],
            "d2": [],
            "d3": [],
            "d4": [],
            "d5": [],
            domain: chunks,
        },
        "retrieval_grades": {},
    }


def test_builds_sq_specific_packet_for_allocation_concealment():
    state = _state_with_chunks(
        "d1",
        [
            {
                "text": "Allocation was concealed through a central web randomization system before enrolment.",
                "section": "Methods",
                "page_numbers": [3],
                "score": 0.1,
            }
        ],
    )

    result = build_evidence_packets(state)

    packet = result["evidence_packets"]["1.2"]
    assert packet["domain"] == "d1"
    assert "conceal" not in packet["missing_evidence"]
    assert packet["sources"][0]["page_numbers"] == [3]
    assert packet["retrieval_confidence"] > 0


def test_d3_completeness_packet_flags_missing_denominator():
    state = _state_with_chunks(
        "d3",
        [
            {
                "text": "The analysis used available participants and reported missing outcome data were uncommon.",
                "section": "Results",
                "page_numbers": [8],
                "score": 0.2,
            }
        ],
    )

    result = build_evidence_packets(state)

    packet = result["evidence_packets"]["3.1"]
    assert "denominator_or_percentage" in packet["missing_evidence"]
    assert packet["packet_grade"]["retry_recommended"] is True


def test_packet_builder_flags_wrong_outcome_context():
    state = _state_with_chunks(
        "d5",
        [
            {
                "text": "Overall survival was the primary endpoint and HR 0.82 was reported.",
                "section": "Results",
                "page_numbers": [9],
                "score": 0.1,
            }
        ],
        outcome="Progression-Free Survival",
    )

    result = build_evidence_packets(state)

    packet = result["evidence_packets"]["5.2"]
    assert "possible_wrong_outcome_context" in packet["negative_flags"]
    assert packet["packet_grade"]["retry_recommended"] is True


def test_d4_pfs_packet_prefers_progression_definition_over_os_and_toxicity_text():
    state = _state_with_chunks(
        "d4",
        [
            {
                "text": "Overall survival was defined as time from randomization to death from any cause.",
                "section": "Endpoints",
                "page_numbers": [5],
                "score": 0.1,
            },
            {
                "text": "Safety outcomes included grade 3 or higher adverse events and treatment-emergent toxicity.",
                "section": "Safety",
                "page_numbers": [9],
                "score": 0.1,
            },
            {
                "text": "Progression-free survival was investigator-assessed radiographic progression or death using RECIST criteria.",
                "section": "Endpoints",
                "page_numbers": [6],
                "score": 0.3,
            },
        ],
        outcome="Progression-Free Survival",
    )

    result = build_evidence_packets(state)

    packet = result["evidence_packets"]["4.1"]
    assert packet["sources"][0]["text"].startswith("Progression-free survival")
    assert "possible_wrong_outcome_context" not in packet["negative_flags"]


def test_d5_packet_flags_results_without_prespecification_evidence():
    state = _state_with_chunks(
        "d5",
        [
            {
                "text": "Progression-free survival improved with HR 0.70 and p=0.01.",
                "section": "Results",
                "page_numbers": [10],
                "score": 0.1,
            }
        ],
        outcome="Progression-Free Survival",
    )

    result = build_evidence_packets(state)

    packet = result["evidence_packets"]["5.1"]
    assert "results_without_prespecification" in packet["negative_flags"]
    assert "protocol_or_registration" in packet["missing_evidence"]


def test_packet_block_for_domain_is_compact_and_sq_labeled():
    state = _state_with_chunks(
        "d1",
        [
            {
                "text": "Participants were randomized using permuted blocks and allocation was concealed centrally.",
                "section": "Methods",
                "page_numbers": [2],
                "score": 0.1,
            }
        ],
    )
    result = build_evidence_packets(state)

    block = packet_block_for_domain(result["evidence_packets"], "d1")

    assert "SQ 1.1" in block
    assert "SQ 1.2" in block
    assert "page 2" in block


def test_d5_packet_includes_ctgov_source_without_page_numbers():
    evidence = empty_paper_evidence("test")
    state = {
        "outcome": "Overall Survival",
        "evidence": evidence,
        "ctgov_outcomes": (
            "Primary Outcome: Overall Survival. Time Frame: from randomization to death."
        ),
        "registered_endpoint": "Overall Survival",
        "registered_secondary_endpoints": "Progression-Free Survival",
        "registered_analysis": "Cox proportional hazards model",
        "rag_chunk_metadata": {"d1": [], "d2": [], "d3": [], "d4": [], "d5": []},
        "retrieval_grades": {},
    }

    result = build_evidence_packets(state)

    sources = result["evidence_packets"]["5.1"]["sources"]
    assert any(source.get("source_kind") == "ctgov" for source in sources)
    ctgov = [source for source in sources if source.get("source_kind") == "ctgov"][0]
    assert ctgov["document_name"] == "ClinicalTrials.gov"
    assert ctgov["document_role"] == "registry"
    assert ctgov["page_numbers"] == []
    assert (
        "missing_page_source" not in result["evidence_packets"]["5.1"]["negative_flags"]
    )


def test_d5_packet_prefers_protocol_over_primary_result_when_terms_match():
    state = _state_with_chunks(
        "d5",
        [
            {
                "text": "Published results report progression-free survival HR 0.70.",
                "section": "Results",
                "page_numbers": [8],
                "score": 0.1,
                "document_id": "primary",
                "document_name": "paper.pdf",
                "document_role": "primary",
                "source_kind": "rag_chunk",
                "source_path": "paper.pdf",
            },
            {
                "text": "The protocol prespecified progression-free survival as a secondary endpoint.",
                "section": "Endpoints",
                "page_numbers": [12],
                "score": 0.2,
                "document_id": "supplement:001",
                "document_name": "protocol.pdf",
                "document_role": "protocol",
                "source_kind": "rag_chunk",
                "source_path": "protocol.pdf",
            },
        ],
        outcome="Progression-Free Survival",
    )

    result = build_evidence_packets(state)

    first = result["evidence_packets"]["5.2"]["sources"][0]
    assert first["document_role"] == "protocol"


def test_d3_packet_uses_ctgov_flow_only_after_primary_completeness_evidence():
    evidence = empty_paper_evidence("test")
    evidence["d3_missing_data"]["text"] = (
        "Outcome data were available for 196 of 200 randomized participants."
    )
    state = {
        "outcome": "Overall Survival",
        "evidence": evidence,
        "ctgov_flow": (
            "Participant flow (from ClinicalTrials.gov posted results): "
            "STARTED: 220; COMPLETED: 180."
        ),
        "rag_chunk_metadata": {"d1": [], "d2": [], "d3": [], "d4": [], "d5": []},
        "retrieval_grades": {},
    }

    result = build_evidence_packets(state)

    sources = result["evidence_packets"]["3.1"]["sources"]
    assert sources[0]["document_role"] == "primary"
    assert any(source.get("source_kind") == "ctgov" for source in sources)


def test_d4_packet_uses_registry_outcome_definition_as_lower_priority_fallback():
    state = _state_with_chunks(
        "d4",
        [
            {
                "text": "The protocol defined progression-free survival using RECIST 1.1 criteria.",
                "section": "Protocol endpoints",
                "page_numbers": [13],
                "score": 0.3,
                "document_id": "supplement:protocol",
                "document_name": "protocol.pdf",
                "document_role": "protocol",
                "source_kind": "rag_chunk",
                "source_path": "protocol.pdf",
            }
        ],
        outcome="Progression-Free Survival",
    )
    state["ctgov_outcomes"] = (
        "Primary Outcome: Progression-Free Survival. Time Frame: from randomization "
        "to radiographic progression or death."
    )

    result = build_evidence_packets(state)

    sources = result["evidence_packets"]["4.1"]["sources"]
    assert sources[0]["document_role"] == "protocol"
    assert any(source.get("document_role") == "registry" for source in sources)


def test_d2_packet_prefers_evidence_derived_deviation_concern_over_trial_name():
    state = _state_with_chunks(
        "d2",
        [
            {
                "text": "PEACE-1 was an open-label randomized trial.",
                "section": "Methods",
                "page_numbers": [4],
                "score": 0.1,
                "document_role": "primary",
            },
            {
                "text": (
                    "Protocol deviations included delayed radiotherapy and differential "
                    "co-interventions; adherence to assigned treatment was lower in one arm."
                ),
                "section": "Protocol deviations",
                "page_numbers": [8],
                "score": 0.2,
                "document_role": "primary",
            },
        ],
        outcome="Adverse Events",
    )

    result = build_evidence_packets(state)

    first_source = result["evidence_packets"]["2.3"]["sources"][0]
    assert "Protocol deviations" in first_source["text"]
    assert "co-interventions" in first_source["text"]


def test_packet_block_renders_document_name_and_role():
    state = _state_with_chunks(
        "d5",
        [
            {
                "text": "The protocol prespecified overall survival as the primary endpoint.",
                "section": "Endpoints",
                "page_numbers": [12],
                "score": 0.1,
                "document_id": "supplement:001",
                "document_name": "protocol.pdf",
                "document_role": "protocol",
                "source_kind": "rag_chunk",
                "source_path": "protocol.pdf",
            }
        ],
        outcome="Overall Survival",
    )
    result = build_evidence_packets(state)

    block = packet_block_for_domain(result["evidence_packets"], "d5")

    assert "protocol.pdf" in block
    assert "protocol" in block
    assert "page 12" in block


def test_section_text_sources_carry_source_kind_tag():
    """Section-text fallback sources must be tagged source_kind="section_text"
    so downstream code can distinguish them from real RAG chunks."""
    evidence = empty_paper_evidence("test")
    evidence["d1_randomization"]["text"] = (
        "Patients were randomized 1:1 using a centralized interactive web response system."
    )
    state = {
        "outcome": "Overall Survival",
        "evidence": evidence,
        "rag_chunk_metadata": {
            "d1": [],
            "d2": [],
            "d3": [],
            "d4": [],
            "d5": [],
        },
        "retrieval_grades": {},
    }

    result = build_evidence_packets(state)

    sources = result["evidence_packets"]["1.1"]["sources"]
    section_text_sources = [
        s for s in sources if s.get("section") == "d1_randomization"
    ]
    assert section_text_sources, "expected at least one section-text source for SQ 1.1"
    for source in section_text_sources:
        assert source.get("source_kind") == "section_text", (
            f"section-text source missing source_kind tag: {source}"
        )


def test_section_text_sources_always_present_with_rag_chunks():
    """Section-text fallback is unconditional: even when RAG returned chunks,
    section-text sources should still be added to the candidate pool as
    supplementary context."""
    evidence = empty_paper_evidence("test")
    evidence["d1_randomization"]["text"] = (
        "Patients were randomized 1:1 using a centralized interactive web response system."
    )
    state = {
        "outcome": "Overall Survival",
        "evidence": evidence,
        "rag_chunk_metadata": {
            "d1": [
                {
                    "text": "Randomization used permuted blocks stratified by site.",
                    "section": "Methods",
                    "page_numbers": [4],
                    "score": 0.5,
                }
            ],
            "d2": [],
            "d3": [],
            "d4": [],
            "d5": [],
        },
        "retrieval_grades": {},
    }

    result = build_evidence_packets(state)

    # At least one SQ in d1 should have a section-text source even though RAG
    # returned a real chunk for the domain.
    found_section_text = False
    for sq_id in ("1.1", "1.2", "1.3"):
        sources = result["evidence_packets"][sq_id]["sources"]
        if any(s.get("source_kind") == "section_text" for s in sources):
            found_section_text = True
            break
    assert found_section_text, (
        "section-text sources should still appear in the candidate pool when "
        "RAG returned chunks, since the fallback is unconditional"
    )


def test_verifier_does_not_flag_section_text_for_missing_page_numbers():
    """A section-text source has no page metadata by design. The verifier
    must not raise missing_page_source on a packet that contains a real chunk
    with page numbers plus a section-text source without page numbers."""
    evidence = empty_paper_evidence("test")
    evidence["d1_randomization"]["text"] = (
        "Patients were randomized 1:1 using a centralized interactive web response system."
    )
    state = {
        "outcome": "Overall Survival",
        "evidence": evidence,
        "rag_chunk_metadata": {
            "d1": [
                {
                    "text": "Randomization used permuted blocks stratified by site.",
                    "section": "Methods",
                    "page_numbers": [4],
                    "score": 0.1,
                }
            ],
            "d2": [],
            "d3": [],
            "d4": [],
            "d5": [],
        },
        "retrieval_grades": {},
    }

    result = build_evidence_packets(state)

    # At least one SQ in d1 should have both a chunk source (with pages) and a
    # section-text source (without pages). missing_page_source must not fire.
    for sq_id in ("1.1", "1.2", "1.3"):
        packet = result["evidence_packets"][sq_id]
        kinds = {s.get("source_kind") for s in packet["sources"]}
        if {"rag_chunk", "section_text"}.issubset(kinds):
            assert "missing_page_source" not in packet["negative_flags"], (
                f"SQ {sq_id} should not be flagged missing_page_source: the "
                f"only source with empty page_numbers is a section-text source"
            )
            return
    raise AssertionError(
        "test setup did not produce any packet with both rag_chunk and "
        "section_text sources"
    )


def test_verifier_still_flags_chunk_source_with_empty_page_numbers():
    """A real RAG chunk that is missing page numbers is still a defect and
    must still trigger missing_page_source. Only section-text sources get a
    pass."""
    evidence = empty_paper_evidence("test")
    state = {
        "outcome": "Overall Survival",
        "evidence": evidence,
        "rag_chunk_metadata": {
            "d1": [
                {
                    "text": "Randomization used permuted blocks stratified by site.",
                    "section": "Methods",
                    "page_numbers": [],
                    "score": 0.1,
                }
            ],
            "d2": [],
            "d3": [],
            "d4": [],
            "d5": [],
        },
        "retrieval_grades": {},
    }

    result = build_evidence_packets(state)

    packet = result["evidence_packets"]["1.1"]
    rag_sources = [s for s in packet["sources"] if s.get("source_kind") == "rag_chunk"]
    assert rag_sources, "expected at least one RAG chunk source"
    assert any(not s.get("page_numbers") for s in rag_sources), (
        "test setup should have a chunk source with empty page_numbers"
    )
    assert "missing_page_source" in packet["negative_flags"], (
        "missing_page_source should still fire for a real RAG chunk that "
        "has empty page_numbers"
    )
