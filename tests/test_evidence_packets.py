from rob2_pipeline.models import empty_paper_evidence
from rob2_pipeline.evidence_store import EvidenceFactRecord
from rob2_pipeline.evidence_store import EvidencePacketRecord
from rob2_pipeline.nodes.evidence_packets import (
    build_evidence_packets,
    packet_block_for_domain,
)
from rob2_pipeline.nodes.evidence_packet_grading import packet_readiness
from rob2_pipeline.nodes.evidence_packet_grading import resolve_source_conflict


class _RecordingSupplementIndex:
    def __init__(self, segments: list[dict]):
        self.segments = segments
        self.calls: list[dict] = []

    def retrieve(self, query: str, *, domain: str, top_k: int = 5) -> dict:
        self.calls.append({"query": query, "domain": domain, "top_k": top_k})
        return {"segments": self.segments[:top_k], "best_score": 0.7}


def test_evidence_packets_module_keeps_stable_public_api():
    from rob2_pipeline.nodes import evidence_packets

    assert callable(evidence_packets.evidence_packet_builder_node)
    assert callable(evidence_packets.build_evidence_packets)
    assert callable(evidence_packets.packet_block_for_domain)


def _state_with_chunks(
    domain: str, chunks: list[dict], outcome: str = "Progression-Free Survival"
) -> dict:
    evidence = empty_paper_evidence("test")
    supplement_chunks = [
        {
            "source_kind": "supplement_segment",
            "document_id": chunk.get("document_id", "supplement:test"),
            "document_name": chunk.get("document_name", "supplement.pdf"),
            "document_role": chunk.get("document_role", "protocol"),
            "source_path": chunk.get("source_path", "supplement.pdf"),
            **chunk,
        }
        for chunk in chunks
    ]
    return {
        "outcome": outcome,
        "evidence": evidence,
        "supplement_indexes": {"supplement:test": _RecordingSupplementIndex(supplement_chunks)},
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


def test_packet_builder_retrieves_supplement_candidates_from_contract_terms():
    evidence = empty_paper_evidence("test")
    supplement_index = _RecordingSupplementIndex(
        [
            {
                "text": "The protocol used a central web allocation system before enrolment.",
                "section": "Allocation",
                "page_numbers": [11],
                "score": 0.7,
                "source_kind": "supplement_segment",
                "document_id": "supplement:protocol",
                "document_name": "protocol.pdf",
                "document_role": "protocol",
                "source_path": "protocol.pdf",
            }
        ]
    )
    state = {
        "outcome": "Overall Survival",
        "evidence": evidence,
        "supplement_indexes": {"supplement:protocol": supplement_index},
    }

    result = build_evidence_packets(state)

    packet = result["evidence_packets"]["1.2"]
    assert packet["sources"][0]["source_kind"] == "supplement_segment"
    assert packet["sources"][0]["document_name"] == "protocol.pdf"
    d1_calls = [call for call in supplement_index.calls if call["domain"] == "d1"]
    assert d1_calls
    allocation_query = next(
        call["query"] for call in d1_calls if "allocation_concealment" in call["query"]
    )
    assert "allocation_concealment" in allocation_query
    assert "enrolment_timing" in allocation_query
    assert "conceal" in allocation_query
    assert "allocation concealment" not in allocation_query


def test_d5_packet_merges_supplement_registry_and_section_text_with_provenance():
    evidence = empty_paper_evidence("test")
    evidence["d5_registration"]["text"] = (
        "The primary report states overall survival was the primary endpoint."
    )
    supplement_index = _RecordingSupplementIndex(
        [
            {
                "text": (
                    "The statistical analysis plan prespecified overall survival "
                    "and the Cox proportional hazards analysis."
                ),
                "section": "SAP Analysis",
                "page_numbers": [17],
                "score": 0.8,
                "source_kind": "supplement_segment",
                "document_id": "supplement:sap",
                "document_name": "sap.pdf",
                "document_role": "sap",
                "source_path": "sap.pdf",
            }
        ]
    )
    state = {
        "outcome": "Overall Survival",
        "evidence": evidence,
        "registered_endpoint": "Overall Survival",
        "registered_analysis": "Cox proportional hazards model",
        "ctgov_outcomes": "Primary Outcome: Overall Survival.",
        "supplement_indexes": {"supplement:sap": supplement_index},
    }

    result = build_evidence_packets(state)

    packet = result["evidence_packets"]["5.1"]
    source_kinds = {source.get("source_kind") for source in packet["sources"]}
    assert {"supplement_segment", "ctgov", "section_text"}.issubset(source_kinds)
    assert packet["sources"][0]["document_role"] == "sap"
    assert "results_without_prespecification" not in packet["negative_flags"]

    block = packet_block_for_domain(result["evidence_packets"], "d5")
    assert "sap (sap.pdf), page 17, SAP Analysis" in block
    assert "registry (ClinicalTrials.gov), no page, ClinicalTrials.gov" in block
    assert "primary (Primary paper evidence), no page, d5_registration" in block


def test_packet_readiness_separates_mechanical_completeness_from_semantic_adequacy():
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
    readiness = packet["packet_readiness"]
    assert readiness["mechanical_completeness"]["status"] == "incomplete"
    assert "denominator_or_percentage" in readiness["mechanical_completeness"]["missing_evidence"]
    assert readiness["semantic_adequacy"]["status"] in {"adequate", "limited"}
    assert readiness["status"] == "needs_retrieval_repair"


def test_packet_readiness_can_emit_all_review_statuses():
    ready = packet_readiness(
        sq_id="1.1",
        missing=[],
        flags=[],
        contradictions=[],
        facts=[{"support_level": "moderate"}],
        confidence=0.6,
    )
    contradiction = packet_readiness(
        sq_id="1.2",
        missing=[],
        flags=[],
        contradictions=[{"label": "allocation_concealment"}],
        facts=[{"support_level": "strong"}],
        confidence=0.9,
    )
    quote = packet_readiness(
        sq_id="1.1",
        missing=[],
        flags=["quote_untraceable"],
        contradictions=[],
        facts=[{"support_level": "strong"}],
        confidence=0.9,
    )
    audit_limited = packet_readiness(
        sq_id="4.4",
        missing=[],
        flags=[],
        contradictions=[],
        facts=[{"support_level": "weak"}],
        confidence=0.3,
    )

    assert ready["status"] == "ready"
    assert contradiction["status"] == "needs_contradiction_resolution"
    assert quote["status"] == "needs_quote_adjudication"
    assert audit_limited["status"] == "audit_limited"


def test_d1_packet_schema_validates_required_artifact_fields():
    state = _state_with_chunks(
        "d1",
        [
            {
                "text": "Participants were randomized by a computer-generated sequence.",
                "section": "Methods",
                "page_numbers": [2],
                "score": 0.1,
                "document_id": "primary:TITAN",
                "document_name": "TITAN primary report",
                "document_role": "primary",
                "source_kind": "rag_chunk",
                "source_path": "inputs/benchmark/TITAN.pdf",
            }
        ],
    )

    result = build_evidence_packets(state)

    packet = EvidencePacketRecord.model_validate(result["evidence_packets"]["1.1"])
    assert packet.artifact_id == "evidence-packet:d1:1.1"
    assert packet.schema_version == "1.0"
    assert packet.outcome == "Progression-Free Survival"


def test_packet_includes_decision_table_with_default_insufficient_evidence_row():
    state = _state_with_chunks(
        "d1",
        [
            {
                "text": "Participants were randomized by a computer-generated sequence.",
                "section": "Methods",
                "page_numbers": [2],
                "score": 0.1,
                "document_id": "primary:TITAN",
                "document_name": "TITAN primary report",
                "document_role": "primary",
                "source_kind": "rag_chunk",
                "source_path": "inputs/benchmark/TITAN.pdf",
            }
        ],
    )

    result = build_evidence_packets(state)

    packet = result["evidence_packets"]["1.1"]
    table = packet["decision_table"]
    assert table["sq_id"] == "1.1"
    assert table["default_insufficient_evidence_answer"] == "NI"
    assert "selected packet evidence" in table["classifier_instruction"]
    assert {row["answer"] for row in table["rows"]} >= {"Y", "PY", "NI"}
    y_row = next(row for row in table["rows"] if row["answer"] == "Y")
    assert y_row["supporting_facts"]
    assert y_row["evidence_gaps"] == []
    ni_row = next(row for row in table["rows"] if row["answer"] == "NI")
    assert ni_row["insufficient_evidence_default"] is True


def test_d2_d5_packets_expose_contract_schema_and_source_policy():
    state = _state_with_chunks(
        "d5",
        [
            {
                "text": "The protocol prespecified progression-free survival as a secondary endpoint.",
                "section": "Endpoints",
                "page_numbers": [12],
                "score": 0.1,
                "document_id": "supplement:protocol",
                "document_name": "protocol.pdf",
                "document_role": "protocol",
                "source_kind": "rag_chunk",
                "source_path": "protocol.pdf",
            }
        ],
        outcome="Progression-Free Survival",
    )

    result = build_evidence_packets(state)

    for sq_id, packet in result["evidence_packets"].items():
        if packet["domain"] not in {"d2", "d3", "d4", "d5"}:
            continue
        contract = packet["contract"]
        assert contract["sq_id"] == sq_id
        assert contract["required_evidence"]
        assert contract["allowed_answers"] == packet["decision_table"]["allowed_answers"]
        assert contract["outcome_binding_status"] in {
            "outcome_bound",
            "trial_level",
        }
        if packet["domain"] in {"d4", "d5"}:
            assert contract["outcome_binding_status"] == "outcome_bound"

    d5_contract = result["evidence_packets"]["5.1"]["contract"]
    assert d5_contract["source_hierarchy"] == [
        "protocol",
        "sap",
        "registry",
        "primary",
        "appendix",
    ]


def test_d5_packet_record_validates_contract_artifact():
    state = _state_with_chunks(
        "d5",
        [
            {
                "text": "The protocol prespecified overall survival as the primary endpoint.",
                "section": "Endpoints",
                "page_numbers": [12],
                "score": 0.1,
                "document_id": "supplement:protocol",
                "document_name": "protocol.pdf",
                "document_role": "protocol",
                "source_kind": "rag_chunk",
                "source_path": "protocol.pdf",
            }
        ],
        outcome="Overall Survival",
    )

    result = build_evidence_packets(state)

    packet = EvidencePacketRecord.model_validate(result["evidence_packets"]["5.1"])
    assert packet.contract.artifact_id == "packet-contract:d5:5.1"
    assert packet.contract.allowed_answers == packet.decision_table.allowed_answers
    assert packet.contract.outcome_binding_status == "outcome_bound"


def test_packet_block_renders_decision_table_classifier_constraint():
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

    block = packet_block_for_domain(result["evidence_packets"], "d3")

    assert "Mini decision table:" in block
    assert "- Y:" in block
    assert "- NI: Default when selected packet evidence is insufficient" in block
    assert "Choose only from selected packet evidence" in block


def test_unsupported_d1_claims_appear_as_gaps_and_failed_claims():
    state = _state_with_chunks(
        "d1",
        [
            {
                "text": "The study describes eligibility criteria and clinic visits.",
                "section": "Methods",
                "page_numbers": [2],
                "score": 0.1,
                "document_id": "primary:TITAN",
                "document_name": "TITAN primary report",
                "document_role": "primary",
                "source_kind": "rag_chunk",
                "source_path": "inputs/benchmark/TITAN.pdf",
            }
        ],
    )

    result = build_evidence_packets(state)

    packet = result["evidence_packets"]["1.2"]
    assert {gap["missing_evidence"] for gap in packet["gaps"]} == {
        "allocation_concealment",
        "enrolment_timing",
    }
    assert {claim["fact_type"] for claim in packet["failed_claims"]} == {
        "allocation_concealment",
        "enrolment_timing",
    }
    assert all(claim["support_status"] == "failed" for claim in packet["failed_claims"])


def test_d1_contradictions_remain_visible_when_dominant_source_is_selected():
    state = _state_with_chunks(
        "d1",
        [
            {
                "text": "Allocation was concealed through a central web randomization system before enrolment.",
                "section": "Methods",
                "page_numbers": [3],
                "score": 0.1,
                "document_id": "primary:TITAN",
                "document_name": "TITAN primary report",
                "document_role": "primary",
                "source_kind": "rag_chunk",
                "source_path": "inputs/benchmark/TITAN.pdf",
            },
            {
                "text": "Allocation was not concealed before participants were assigned.",
                "section": "Protocol",
                "page_numbers": [12],
                "score": 0.2,
                "document_id": "supplement:protocol",
                "document_name": "TITAN protocol",
                "document_role": "protocol",
                "source_kind": "rag_chunk",
                "source_path": "inputs/benchmark/supplement/TITAN/protocol.pdf",
            },
        ],
    )

    result = build_evidence_packets(state)

    packet = result["evidence_packets"]["1.2"]
    assert packet["sources"][0]["document_role"] == "primary"
    assert packet["contradictions"]
    contradiction = packet["contradictions"][0]
    assert contradiction["label"] == "allocation_concealment"
    assert contradiction["source_hierarchy"] == ["primary", "protocol", "appendix"]
    assert contradiction["source_roles"] == ["primary", "protocol"]
    assert contradiction["support_levels"] == {
        "dominant": "strong",
        "conflicting": "strong",
    }
    assert contradiction["dominant_source"]["document_role"] == "primary"
    assert contradiction["conflicting_source"]["document_role"] == "protocol"
    assert contradiction["dominant_claim"] == "Allocation concealment was reported."
    assert contradiction["rationale"]
    assert contradiction["hierarchy_override"] is False
    assert packet["packet_readiness"]["status"] == "needs_contradiction_resolution"


def test_source_conflict_can_override_hierarchy_only_with_rationale_and_quote_support():
    protocol = {
        "text": "The protocol says allocation was concealed by central randomization.",
        "document_role": "protocol",
        "document_name": "Protocol",
    }
    primary = {
        "text": "The primary paper states allocation was not concealed.",
        "document_role": "primary",
        "document_name": "Primary paper",
    }

    default = resolve_source_conflict(
        domain="d1",
        sq_id="1.2",
        label="allocation_concealment",
        positive=protocol,
        negative=primary,
    )
    unsupported_override = resolve_source_conflict(
        domain="d1",
        sq_id="1.2",
        label="allocation_concealment",
        positive=protocol,
        negative=primary,
        override_source=protocol,
        override_rationale="Protocol quote is more specific to allocation setup.",
    )
    supported_override = resolve_source_conflict(
        domain="d1",
        sq_id="1.2",
        label="allocation_concealment",
        positive=protocol,
        negative=primary,
        override_source=protocol,
        override_rationale="Protocol quote is more specific to allocation setup.",
        override_quote="allocation was concealed by central randomization",
    )

    assert default["dominant_source"]["document_role"] == "primary"
    assert unsupported_override["dominant_source"]["document_role"] == "primary"
    assert unsupported_override["override_rejected_reason"]
    assert supported_override["dominant_source"]["document_role"] == "protocol"
    assert supported_override["hierarchy_override"] is True
    assert supported_override["override_rationale"] == (
        "Protocol quote is more specific to allocation setup."
    )


def test_packet_candidate_facts_validate_against_base_evidence_fact_contract():
    state = _state_with_chunks(
        "d1",
        [
            {
                "text": "Allocation was concealed through a central web randomization system before enrolment.",
                "section": "Methods",
                "page_numbers": [3],
                "score": 0.1,
                "document_id": "primary:TITAN",
                "document_name": "TITAN primary report",
                "document_role": "primary",
                "source_kind": "rag_chunk",
                "source_path": "inputs/benchmark/TITAN.pdf",
            }
        ],
    )

    result = build_evidence_packets(state)
    fact = result["evidence_packets"]["1.2"]["candidate_facts"][0]

    validated = EvidenceFactRecord.model_validate(fact)

    assert validated.artifact_id.startswith("evidence-fact:d1:1.2:")
    assert validated.provenance.document_id == "primary:TITAN"


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


def test_d3_completeness_packet_accepts_count_with_all_outcome_data():
    state = _state_with_chunks(
        "d3",
        [
            {
                "text": "100 participants were randomized and all had outcome data.",
                "section": "Results",
                "page_numbers": [8],
                "score": 0.2,
            }
        ],
    )

    result = build_evidence_packets(state)

    packet = result["evidence_packets"]["3.1"]
    assert "denominator_or_percentage" not in packet["missing_evidence"]


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


def test_d5_packet_carries_registry_snapshot_provenance():
    evidence = empty_paper_evidence("test")
    state = {
        "outcome": "Overall Survival",
        "evidence": evidence,
        "registration_number": "NCT00309985",
        "ctgov_outcomes": "PRIMARY: Overall Survival",
        "registered_endpoint": "Overall Survival",
        "registered_secondary_endpoints": "",
        "registered_analysis": "Cox proportional hazards model",
        "ctgov_registry_document": {
            "document_id": "registry:NCT00309985",
            "document_name": "ClinicalTrials.gov NCT00309985",
            "document_role": "registry",
            "source_kind": "ctgov",
            "path": "https://clinicaltrials.gov/study/NCT00309985",
            "retrieval_date": "2026-06-03",
            "api_response_hash": "a" * 64,
        },
        "rag_chunk_metadata": {"d1": [], "d2": [], "d3": [], "d4": [], "d5": []},
        "retrieval_grades": {},
    }

    result = build_evidence_packets(state)

    ctgov = [
        source
        for source in result["evidence_packets"]["5.1"]["sources"]
        if source.get("source_kind") == "ctgov"
    ][0]
    assert ctgov["document_id"] == "registry:NCT00309985"
    assert ctgov["source_path"] == "https://clinicaltrials.gov/study/NCT00309985"
    assert ctgov["retrieval_date"] == "2026-06-03"
    assert ctgov["api_response_hash"] == "a" * 64
    fact = result["evidence_packets"]["5.1"]["candidate_facts"][0]
    assert fact["provenance"]["retrieval_date"] == "2026-06-03"
    assert fact["provenance"]["api_response_hash"] == "a" * 64


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
    state = _state_with_chunks(
        "d1",
        [
            {
                "text": "Randomization used permuted blocks stratified by site.",
                "section": "Methods",
                "page_numbers": [4],
                "score": 0.1,
            }
        ],
        outcome="Overall Survival",
    )
    state["evidence"] = evidence

    result = build_evidence_packets(state)

    # At least one SQ in d1 should have both a supplement segment with pages and
    # a section-text source without pages. missing_page_source must not fire.
    for sq_id in ("1.1", "1.2", "1.3"):
        packet = result["evidence_packets"][sq_id]
        kinds = {s.get("source_kind") for s in packet["sources"]}
        if {"supplement_segment", "section_text"}.issubset(kinds):
            assert "missing_page_source" not in packet["negative_flags"], (
                f"SQ {sq_id} should not be flagged missing_page_source: the "
                f"only source with empty page_numbers is a section-text source"
            )
            return
    raise AssertionError(
        "test setup did not produce any packet with both supplement_segment and "
        "section_text sources"
    )


def test_supplement_segment_with_empty_page_numbers_is_provenance_warning():
    """A retrieved supplement segment missing page numbers is a provenance
    warning, not a fatal packet defect."""
    evidence = empty_paper_evidence("test")
    state = _state_with_chunks(
        "d1",
        [
            {
                "text": "Randomization used permuted blocks stratified by site.",
                "section": "Methods",
                "page_numbers": [],
                "score": 0.1,
            }
        ],
        outcome="Overall Survival",
    )
    state["evidence"] = evidence

    result = build_evidence_packets(state)

    packet = result["evidence_packets"]["1.1"]
    supplement_sources = [
        s for s in packet["sources"] if s.get("source_kind") == "supplement_segment"
    ]
    assert supplement_sources, "expected at least one supplement segment source"
    assert any(not s.get("page_numbers") for s in supplement_sources), (
        "test setup should have a supplement segment source with empty page_numbers"
    )
    assert "missing_page_source" not in packet["negative_flags"]
    assert "missing_supplement_page_numbers" in packet["provenance_warnings"]
    assert packet["packet_grade"]["retry_recommended"] is False
    assert packet["packet_readiness"]["status"] == "ready"
