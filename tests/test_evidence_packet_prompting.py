from rob2_pipeline.models import empty_paper_evidence
from rob2_pipeline.nodes.domain1 import build_domain1_prompt


def test_domain_prompt_includes_verified_evidence_packet():
    evidence = empty_paper_evidence("test")
    evidence["d1_randomization"]["text"] = "Participants were randomized centrally."
    state = {
        "evidence": evidence,
        "intervention": "Drug A",
        "comparator": "Placebo",
        "outcome": "Overall Survival",
        "ctgov_design": "",
        "rag_contexts": {"d1": "generic randomization context"},
        "rag_chunk_metadata": {},
        "trial_facts": {},
        "sq_answers": {},
        "evidence_packets": {
            "1.1": {
                "sq_id": "1.1",
                "domain": "d1",
                "required_evidence": ["sequence_generation"],
                "missing_evidence": [],
                "negative_flags": [],
                "sources": [
                    {
                        "text": "Computer-generated random allocation sequence.",
                        "section": "Methods",
                        "page_numbers": [2],
                        "score": 0.1,
                    }
                ],
            }
        },
    }

    prompt = build_domain1_prompt(state)

    assert "SQ 1.1 verified evidence packet" in prompt
    assert "Computer-generated random allocation sequence" in prompt
