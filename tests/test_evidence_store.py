import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from rob2_pipeline.evidence_store import EvidenceFactRecord, EvidenceStore


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
