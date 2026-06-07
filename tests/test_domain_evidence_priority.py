from rob2_pipeline.models import empty_paper_evidence
from rob2_pipeline.nodes.domain1 import build_domain1_prompt


def test_domain1_keeps_structured_evidence_when_rag_context_exists():
    evidence = empty_paper_evidence()
    evidence["d1_randomization"]["text"] = (
        "Allocation managed by the ECOG-ACRIN Statistical Center."
    )
    evidence["baseline_table"]["text"] = "Baseline characteristics were well balanced."
    evidence["consort_flow"]["text"] = "All randomized patients were included."
    state = {
        "intervention": "Docetaxel + ADT",
        "comparator": "ADT alone",
        "outcome": "Overall Survival",
        "evidence": evidence,
        "rag_contexts": {
            "d1": "Patients were assigned to ADT alone or ADT plus docetaxel."
        },
        "evidence_packets": {
            "1.1": {
                "domain": "d1",
                "required_evidence": ["sequence_generation"],
                "missing_evidence": [],
                "negative_flags": [],
                "sources": [],
            }
        },
        "sq_answers": {},
    }

    prompt = build_domain1_prompt(state)

    assert (
        "Allocation managed by the ECOG-ACRIN Statistical Center" in prompt
    )
    assert "Baseline characteristics were well balanced" in prompt
    assert "Patients were assigned to ADT alone" in prompt
    assert "SQ 1.1" not in prompt or "verified evidence packet" in prompt
