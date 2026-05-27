from rob2_pipeline.models import empty_paper_evidence
from rob2_pipeline.nodes.sq_control import apply_domain4_control
from rob2_pipeline.nodes.trial_facts import extract_trial_facts
from rob2_pipeline.pipeline import _assessment_json


def test_masking_facts_capture_open_label_awareness():
    evidence = empty_paper_evidence()
    evidence["d2_blinding"]["text"] = (
        "This was an open-label trial. Participants and investigators were aware "
        "of the assigned treatment."
    )

    facts = extract_trial_facts({"evidence": evidence})["masking_facts"]

    assert facts["participant_awareness"]["status"] == "aware"
    assert facts["personnel_awareness"]["status"] == "aware"
    assert facts["outcome_assessor_awareness"]["status"] == "aware"
    assert facts["participant_awareness"]["source_strength"] == "primary"
    assert "open-label" in facts["participant_awareness"]["quotes"][0]["quote"]


def test_masking_facts_capture_blinded_and_unclear_statuses():
    blinded_evidence = empty_paper_evidence()
    blinded_evidence["d2_blinding"]["text"] = (
        "Participants, investigators, and outcome assessors were masked to "
        "treatment assignment."
    )

    blinded = extract_trial_facts({"evidence": blinded_evidence})["masking_facts"]

    assert blinded["participant_awareness"]["status"] == "unaware"
    assert blinded["personnel_awareness"]["status"] == "unaware"
    assert blinded["outcome_assessor_awareness"]["status"] == "unaware"

    unclear = extract_trial_facts({"evidence": empty_paper_evidence()})["masking_facts"]

    assert unclear["participant_awareness"]["status"] == "unclear"
    assert unclear["personnel_awareness"]["status"] == "unclear"
    assert unclear["outcome_assessor_awareness"]["status"] == "unclear"


def test_masking_facts_capture_blinded_adjudication_from_registry_design():
    facts = extract_trial_facts(
        {
            "evidence": empty_paper_evidence(),
            "ctgov_design": "Masking: None (Open Label). Primary Purpose: Treatment.",
            "rag_contexts": {
                "d4_assessor": (
                    "Progression was reviewed by a blinded independent central "
                    "adjudication committee."
                )
            },
        }
    )["masking_facts"]

    assert facts["participant_awareness"]["status"] == "aware"
    assert facts["personnel_awareness"]["status"] == "aware"
    assert facts["blinded_adjudication"]["status"] == "present"
    assert facts["blinded_adjudication"]["quotes"][0]["source_kind"] == "source_text"


def test_assessment_json_exposes_masking_facts():
    state = {
        "pdf_path": "paper.pdf",
        "rag_chunk_metadata": {},
        "masking_facts": {
            "participant_awareness": {"status": "aware", "quotes": []},
        },
    }

    data = _assessment_json(state)

    assert data["masking_facts"]["participant_awareness"]["status"] == "aware"


def test_domain4_control_uses_masking_facts_without_d2_sq_answers():
    sq_answers = {
        "4.1": {"answer": "N", "quote": "Progression was assessed by RECIST."},
        "4.2": {"answer": "N", "quote": "Same schedule in both groups."},
        "4.3": {"answer": "NI", "quote": "No relevant text found"},
    }
    state = {
        "outcome": "Progression-free survival",
        "outcome_type": "clinician-composite",
        "outcome_properties": {},
        "masking_facts": {
            "participant_awareness": {"status": "aware"},
            "personnel_awareness": {"status": "aware"},
            "outcome_assessor_awareness": {"status": "aware"},
            "blinded_adjudication": {"status": "absent"},
        },
        "rag_contexts": {
            "d4_measurement": "Progression-free survival was assessed by investigators using RECIST.",
        },
        "evidence": empty_paper_evidence(),
        "sq_answers": {},
    }

    controlled = apply_domain4_control(state, sq_answers)

    assert controlled["4.3"]["answer"] == "PY"
    assert controlled["4.4"]["answer"] == "PY"
    assert controlled["4.5"]["answer"] == "PN"
