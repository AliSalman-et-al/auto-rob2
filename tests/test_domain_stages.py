from rob2_pipeline.nodes.domain_helpers import DomainSqStage, run_domain_sq_stage


def test_domain_sq_stage_runs_llm_and_postprocesses_answers():
    calls = []

    def fake_call_fn(state, prompt, node_name, parse_fn, parse_sq_ids, chunk_sources=None):
        calls.append(
            {
                "prompt": prompt,
                "node_name": node_name,
                "parse_sq_ids": parse_sq_ids,
                "chunk_sources": chunk_sources,
            }
        )
        return (
            "",
            [{"node": node_name, "cache_hit": False}],
            {"9.1": {"answer": "Y"}},
        )

    def build_prompt(state):
        return f"Outcome: {state['outcome']}"

    def postprocess(state, sq_answers):
        updated = dict(sq_answers)
        updated["9.2"] = {"answer": "NA"}
        return updated

    state = {
        "outcome": "Overall Survival",
        "sq_answers": {"existing": {"answer": "N"}},
        "rag_chunk_metadata": {
            "d9": [{"section": "Methods", "page_numbers": [4]}],
        },
    }
    stage = DomainSqStage(
        node_name="domain9_sq",
        sq_ids=("9.1",),
        source_domain="d9",
        build_prompt=build_prompt,
        postprocess=postprocess,
    )

    result = run_domain_sq_stage(state, stage, call_fn=fake_call_fn)

    assert calls == [
        {
            "prompt": "Outcome: Overall Survival",
            "node_name": "domain9_sq",
            "parse_sq_ids": ["9.1"],
            "chunk_sources": ["[page 4, Methods]"],
        }
    ]
    assert result == {
        "sq_answers": {
            "existing": {"answer": "N"},
            "9.1": {"answer": "Y"},
            "9.2": {"answer": "NA"},
        },
        "llm_call_log": [{"node": "domain9_sq", "cache_hit": False}],
    }
