import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from rob2_pipeline.evidence_store import (
    EvidenceFactRecord,
    EvidenceStore,
    mine_evidence_families,
)


def _valid_fact(**overrides):
    fact = {
        "artifact_id": "evidence-fact:d1:1.1:central-randomization",
        "fact_type": "randomization_sequence",
        "domain": "d1",
        "sq_ids": ["1.1"],
        "claim_type": "trial_method",
        "claim": "Participants were randomly assigned centrally.",
        "quote": "Participants were randomly assigned centrally.",
        "support_level": "strong",
        "support_status": "supported",
        "uncertainty": False,
        "provenance": {
            "document_id": "primary:TITAN",
            "document_name": "TITAN primary report",
            "document_role": "primary",
            "source_kind": "rag_chunk",
            "source_path": "inputs/benchmark/TITAN.pdf",
            "source_section": "Methods",
            "page_numbers": [4],
        },
    }
    fact.update(overrides)
    return fact


def test_supported_evidence_fact_validates_required_provenance_and_support():
    fact = EvidenceFactRecord.model_validate(_valid_fact())

    assert fact.support_status == "supported"
    assert fact.provenance.document_role == "primary"


def test_evidence_fact_accepts_only_canonical_support_levels():
    for level in ["strong", "moderate", "weak"]:
        assert EvidenceFactRecord.model_validate(
            _valid_fact(support_level=level)
        ).support_level == level
    assert (
        EvidenceFactRecord.model_validate(
            _valid_fact(
                support_level="unsupported",
                support_status="failed",
                quote="",
                failure_reason="Selected source does not support the claim.",
            )
        ).support_level
        == "unsupported"
    )

    with pytest.raises(ValidationError):
        EvidenceFactRecord.model_validate(_valid_fact(support_level="low"))


def test_unsupported_claims_cannot_be_supported_facts_but_weak_claims_can():
    weak = EvidenceFactRecord.model_validate(_valid_fact(support_level="weak"))

    assert weak.support_status == "supported"

    with pytest.raises(ValidationError):
        EvidenceFactRecord.model_validate(_valid_fact(support_level="unsupported"))


def test_family_specific_facts_require_fields_needed_for_packet_construction():
    fact = EvidenceFactRecord.model_validate(
        _valid_fact(
            fact_type="randomization_sequence",
            family="randomization_allocation",
            family_fields={
                "method": "central randomization",
                "allocation_concealment": "central office concealed assignment",
                "unit_of_randomization": "participant",
            },
        )
    )

    assert fact.family == "randomization_allocation"
    assert fact.family_fields.method == "central randomization"

    with pytest.raises(ValidationError):
        EvidenceFactRecord.model_validate(
            _valid_fact(
                fact_type="randomization_sequence",
                family="randomization_allocation",
                family_fields={"method": "central randomization"},
            )
        )


def test_prespecification_facts_require_structured_artifact_and_analysis_fields():
    fact = EvidenceFactRecord.model_validate(
        _valid_fact(
            artifact_id="evidence-fact:d5:5.1:nct-prespecified-os",
            fact_type="prespecified_analysis",
            domain="d5",
            sq_ids=["5.1"],
            claim_type="registry",
            claim="Overall survival was prespecified in the registry.",
            quote="Overall survival was prespecified in the registry.",
            family="prespecification",
            family_fields={
                "artifact_type": "registry",
                "identifier": "NCT01234567",
                "prespecified_outcome": "overall survival",
                "prespecified_analysis": "time-to-event comparison",
            },
        )
    )

    assert fact.family_fields.artifact_type == "registry"
    assert fact.family_fields.prespecified_analysis == "time-to-event comparison"

    with pytest.raises(ValidationError):
        EvidenceFactRecord.model_validate(
            _valid_fact(
                artifact_id="evidence-fact:d5:5.1:nct-prespecified-os",
                fact_type="prespecified_analysis",
                domain="d5",
                sq_ids=["5.1"],
                claim_type="registry",
                family="prespecification",
                family_fields={
                    "artifact_type": "registry",
                    "identifier": "NCT01234567",
                    "prespecified_outcome": "overall survival",
                },
            )
        )


def test_registry_prespecification_fact_accepts_snapshot_provenance():
    fact = EvidenceFactRecord.model_validate(
        _valid_fact(
            artifact_id="evidence-fact:d5:5.1:nct-prespecified-os",
            fact_type="prespecified_analysis",
            domain="d5",
            sq_ids=["5.1"],
            claim_type="registry",
            claim="Overall survival was prespecified in the registry.",
            quote="PRIMARY: Overall Survival",
            family="prespecification",
            family_fields={
                "artifact_type": "registry",
                "identifier": "NCT00309985",
                "prespecified_outcome": "Overall Survival",
                "prespecified_analysis": "Cox proportional hazards model",
            },
            provenance={
                "document_id": "registry:NCT00309985",
                "document_name": "ClinicalTrials.gov NCT00309985",
                "document_role": "registry",
                "source_kind": "ctgov",
                "source_path": "https://clinicaltrials.gov/study/NCT00309985",
                "source_section": "ClinicalTrials.gov",
                "page_numbers": [],
                "retrieval_date": "2026-06-03",
                "api_response_hash": "a" * 64,
            },
        )
    )

    assert fact.provenance.retrieval_date == "2026-06-03"
    assert fact.provenance.api_response_hash == "a" * 64


def test_supported_evidence_fact_rejects_missing_quote_or_provenance():
    with pytest.raises(ValidationError):
        EvidenceFactRecord.model_validate(_valid_fact(quote=""))

    invalid = _valid_fact()
    invalid["provenance"].pop("document_id")
    with pytest.raises(ValidationError):
        EvidenceFactRecord.model_validate(invalid)


def test_evidence_store_keeps_supported_facts_failed_claims_and_gaps_separate():
    supported = _valid_fact()
    failed = _valid_fact(
        artifact_id="evidence-fact:d1:1.1:untraceable",
        support_level="unsupported",
        support_status="failed",
        quote="",
        failure_reason="Quote could not be traced to a source document.",
    )

    store = EvidenceStore.model_validate(
        {
            "artifact_id": "evidence-store:TITAN:overall-survival",
            "schema_version": "1.0",
            "supported_facts": [supported],
            "failed_claims": [failed],
            "gaps": [
                {
                    "artifact_id": "evidence-gap:d3:3.1:denominator",
                    "domain": "d3",
                    "sq_ids": ["3.1"],
                    "missing_evidence": "denominator_or_percentage",
                    "reason": "No analyzable-participant denominator was found.",
                }
            ],
        }
    )

    assert len(store.supported_facts) == 1
    assert len(store.failed_claims) == 1
    assert len(store.gaps) == 1
    assert store.supported_facts[0].support_level == "strong"
    assert store.failed_claims[0].support_status == "failed"


def test_minimal_evidence_store_golden_fixture_is_valid():
    fixture = Path("tests/fixtures/minimal_evidence_store.json")
    store = EvidenceStore.model_validate_json(fixture.read_text())

    assert store.artifact_id == "evidence-store:minimal"
    assert json.loads(store.model_dump_json())["supported_facts"][0]["artifact_id"]


def test_mine_evidence_families_bounds_llm_prompt_to_selected_packet_sources():
    prompts = []

    def fake_call(state, prompt, node_name):
        prompts.append(prompt)
        return (
            json.dumps(
                {
                    "facts": [
                        {
                            "artifact_id": "evidence-fact:d1:1.1:central-randomization",
                            "fact_type": "randomization_sequence",
                            "domain": "d1",
                            "sq_ids": ["1.1"],
                            "claim_type": "trial_method",
                            "claim": "Participants were assigned centrally.",
                            "quote": "Participants were assigned centrally.",
                            "support_level": "strong",
                            "support_status": "supported",
                            "uncertainty": False,
                            "family": "randomization_allocation",
                            "family_fields": {
                                "method": "central randomization",
                                "allocation_concealment": "central office",
                                "unit_of_randomization": "participant",
                            },
                            "provenance": {
                                "document_id": "primary:TITAN",
                                "document_name": "TITAN primary report",
                                "document_role": "primary",
                                "source_kind": "rag_chunk",
                                "source_path": "inputs/benchmark/TITAN.pdf",
                                "source_section": "Methods",
                                "page_numbers": [4],
                            },
                        }
                    ]
                }
            ),
            [{"node": node_name, "cache_hit": False}],
            None,
        )

    state = {
        "pdf_path": "inputs/benchmark/TITAN.pdf",
        "outcome": "Overall Survival",
        "evidence_packets": {
            "1.1": {
                "sq_id": "1.1",
                "domain": "d1",
                "sources": [
                    {
                        "text": "Participants were assigned centrally.",
                        "section": "Methods",
                        "page_numbers": [4],
                        "document_id": "primary:TITAN",
                        "document_name": "TITAN primary report",
                        "document_role": "primary",
                        "source_kind": "rag_chunk",
                        "source_path": "inputs/benchmark/TITAN.pdf",
                    }
                ],
            }
        },
        "full_text": "This full text must not be sent to the family miner.",
    }

    update = mine_evidence_families(state, call_fn=fake_call)

    assert update["evidence_store"]["supported_facts"][0]["family"] == (
        "randomization_allocation"
    )
    assert "Participants were assigned centrally." in prompts[0]
    assert "full text must not be sent" not in prompts[0]


def test_mine_evidence_families_keeps_unsupported_claims_visible_but_unselected():
    def fake_call(state, prompt, node_name):
        return (
            json.dumps(
                {
                    "facts": [
                        {
                            "artifact_id": "evidence-fact:d1:1.1:weak-randomization",
                            "fact_type": "randomization_sequence",
                            "domain": "d1",
                            "sq_ids": ["1.1"],
                            "claim_type": "trial_method",
                            "claim": "Participants were assigned by an unclear random method.",
                            "quote": "Participants were assigned by an unclear random method.",
                            "support_level": "weak",
                            "support_status": "supported",
                            "uncertainty": True,
                            "family": "randomization_allocation",
                            "family_fields": {
                                "method": "unclear random method",
                                "allocation_concealment": "not reported",
                                "unit_of_randomization": "participant",
                            },
                            "provenance": {
                                "document_id": "primary:TITAN",
                                "document_name": "TITAN primary report",
                                "document_role": "primary",
                                "source_kind": "rag_chunk",
                                "source_path": "inputs/benchmark/TITAN.pdf",
                                "source_section": "Methods",
                                "page_numbers": [4],
                            },
                        },
                        {
                            "artifact_id": "evidence-fact:d1:1.1:unsupported-allocation",
                            "fact_type": "allocation_concealment",
                            "domain": "d1",
                            "sq_ids": ["1.1"],
                            "claim_type": "trial_method",
                            "claim": "Allocation concealment was adequate.",
                            "quote": "",
                            "support_level": "unsupported",
                            "support_status": "failed",
                            "uncertainty": True,
                            "provenance": {
                                "document_id": "primary:TITAN",
                                "document_name": "TITAN primary report",
                                "document_role": "primary",
                                "source_kind": "rag_chunk",
                                "source_path": "inputs/benchmark/TITAN.pdf",
                                "source_section": "Methods",
                                "page_numbers": [4],
                            },
                            "failure_reason": "Selected source does not support this claim.",
                        },
                    ]
                }
            ),
            [{"node": node_name, "cache_hit": False}],
            None,
        )

    state = {
        "pdf_path": "inputs/benchmark/TITAN.pdf",
        "outcome": "Overall Survival",
        "evidence_packets": {
            "1.1": {
                "sq_id": "1.1",
                "domain": "d1",
                "sources": [
                    {
                        "text": "Participants were assigned by an unclear random method.",
                        "section": "Methods",
                        "page_numbers": [4],
                        "document_id": "primary:TITAN",
                        "document_name": "TITAN primary report",
                        "document_role": "primary",
                        "source_kind": "rag_chunk",
                        "source_path": "inputs/benchmark/TITAN.pdf",
                    }
                ],
            }
        },
    }

    store = mine_evidence_families(state, call_fn=fake_call)["evidence_store"]

    assert store["supported_facts"][0]["support_level"] == "weak"
    assert store["supported_facts"][0]["support_status"] == "supported"
    assert store["failed_claims"][0]["support_level"] == "unsupported"
    assert store["failed_claims"][0]["support_status"] == "failed"


def test_mine_evidence_families_retries_then_records_failed_claim_on_bad_schema():
    calls = []

    def fake_call(state, prompt, node_name):
        calls.append(prompt)
        return (
            json.dumps(
                {
                    "facts": [
                        {
                            "artifact_id": "evidence-fact:d1:1.1:bad",
                            "fact_type": "randomization_sequence",
                            "domain": "d1",
                            "sq_ids": ["1.1"],
                            "claim_type": "trial_method",
                            "claim": "Randomized centrally.",
                            "quote": "Randomized centrally.",
                            "support_level": "strong",
                            "support_status": "supported",
                            "uncertainty": False,
                            "family": "randomization_allocation",
                            "family_fields": {"method": "central"},
                            "provenance": {
                                "document_id": "primary:TITAN",
                                "document_name": "TITAN primary report",
                                "document_role": "primary",
                                "source_kind": "rag_chunk",
                                "source_path": "inputs/benchmark/TITAN.pdf",
                                "source_section": "Methods",
                                "page_numbers": [4],
                            },
                        }
                    ]
                }
            ),
            [{"node": node_name, "cache_hit": False}],
            None,
        )

    state = {
        "pdf_path": "inputs/benchmark/TITAN.pdf",
        "outcome": "Overall Survival",
        "evidence_packets": {
            "1.1": {
                "sq_id": "1.1",
                "domain": "d1",
                "sources": [
                    {
                        "text": "Randomized centrally.",
                        "section": "Methods",
                        "page_numbers": [4],
                        "document_id": "primary:TITAN",
                        "document_name": "TITAN primary report",
                        "document_role": "primary",
                        "source_kind": "rag_chunk",
                        "source_path": "inputs/benchmark/TITAN.pdf",
                    }
                ],
            }
        },
    }

    update = mine_evidence_families(state, call_fn=fake_call)

    assert len(calls) == 2
    assert "Your previous evidence-family extraction was invalid" in calls[1]
    store = update["evidence_store"]
    assert store["supported_facts"] == []
    assert store["failed_claims"][0]["support_status"] == "failed"
    assert "validation failed" in store["failed_claims"][0]["failure_reason"]
