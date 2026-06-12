from rob2_pipeline.nodes.domain_helpers import DomainSqStage, run_domain_sq_stage
from rob2_pipeline.llm_contracts import JsonContractResult


def test_domain_sq_stage_runs_llm_and_postprocesses_answers():
    calls = []

    def fake_call_fn(
        state,
        prompt,
        node_name,
        *,
        schema_model,
        schema_version,
        prompt_version,
        fallback_factory,
    ):
        calls.append(
            {
                "prompt": prompt,
                "node_name": node_name,
                "schema_model": schema_model.__name__,
                "schema_version": schema_version,
                "prompt_version": prompt_version,
            }
        )
        return JsonContractResult(
            artifact={
                "answers": [
                    {
                        "sq_id": "9.1",
                        "answer": "Y",
                        "quote": "Randomized",
                        "justification": "Reported randomization.",
                        "uncertainty_flag": "NORMAL",
                        "support_level": "strong",
                        "support_rationale": "Directly supported.",
                    }
                ]
            },
            log=[{"node": node_name, "cache_hit": False}],
            status="validated",
            failure_reason=None,
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
        "evidence_packets": {
            "9.1": {
                "domain": "d9",
                "sources": [{"section": "Methods", "page_numbers": [4]}],
            }
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

    assert calls[0]["prompt"].startswith("Outcome: Overall Survival")
    assert "Return only JSON" in calls[0]["prompt"]
    assert "9.1" in calls[0]["prompt"]
    assert calls[0] == {
        "prompt": calls[0]["prompt"],
        "node_name": "domain9_sq",
        "schema_model": "DomainSqStageArtifact",
        "schema_version": "domain-sq-stage.v1",
        "prompt_version": "domain9_sq-json.v1",
    }
    assert result == {
        "sq_answers": {
            "existing": {"answer": "N"},
            "9.1": {
                "answer": "Y",
                "quote": "Randomized",
                "justification": "Reported randomization.",
                "uncertainty_flag": "NORMAL",
                "support_level": "strong",
                "support_rationale": "Directly supported.",
            },
            "9.2": {"answer": "NA"},
        },
        "llm_call_log": [
            {
                "node": "domain9_sq",
                "cache_hit": False,
                "chunk_sources": ["[page 4, Methods]"],
            }
        ],
    }


def test_domain_sq_stage_blocks_classifier_when_packet_needs_repair():
    calls = []

    def fake_call_fn(state, prompt, node_name, **kwargs):
        calls.append(node_name)
        return JsonContractResult(
            artifact={"answers": []},
            status="validated",
            failure_reason=None,
            log=[],
        )

    state = {
        "outcome": "Overall Survival",
        "sq_answers": {},
        "packet_readiness": {
            "3.1": {
                "status": "needs_retrieval_repair",
                "blocking_reason": "Missing denominator evidence.",
            }
        },
    }
    stage = DomainSqStage(
        node_name="domain3_sq",
        sq_ids=("3.1",),
        source_domain="d3",
        build_prompt=lambda _state: "classify SQ 3.1",
    )

    result = run_domain_sq_stage(state, stage, call_fn=fake_call_fn)

    assert calls == []
    assert result["llm_call_log"] == []
    assert result["sq_answers"]["3.1"]["answer"] == "NI"
    assert result["sq_answers"]["3.1"]["classification_blocked"] is True
    assert result["sq_answers"]["3.1"]["packet_status"] == "needs_retrieval_repair"


def test_domain_sq_stage_still_classifies_unblocked_sqs_in_mixed_stage():
    calls = []

    def fake_call_fn(state, prompt, node_name, **kwargs):
        calls.append(prompt)
        return JsonContractResult(
            artifact={
                "answers": [
                    {
                        "sq_id": "4.4",
                        "answer": "PY",
                        "quote": "Blinded review",
                        "justification": "Outcome assessors were blinded.",
                        "uncertainty_flag": "NORMAL",
                        "support_level": "moderate",
                        "support_rationale": "Packet supports blinding.",
                    }
                ]
            },
            status="validated",
            failure_reason=None,
            log=[],
        )

    state = {
        "outcome": "Progression-Free Survival",
        "sq_answers": {},
        "packet_readiness": {
            "4.1": {
                "status": "needs_quote_adjudication",
                "blocking_reason": "Missing page provenance.",
            },
            "4.4": {"status": "ready"},
        },
    }
    stage = DomainSqStage(
        node_name="domain4_sq",
        sq_ids=("4.1", "4.4"),
        source_domain="d4",
        build_prompt=lambda _state: "classify domain 4",
    )

    result = run_domain_sq_stage(state, stage, call_fn=fake_call_fn)

    assert len(calls) == 1
    assert "4.4" in calls[0]
    assert "4.1" not in calls[0]
    assert result["sq_answers"]["4.1"]["classification_blocked"] is True
    assert result["sq_answers"]["4.4"]["answer"] == "PY"


def test_domain_sq_stage_uses_deterministic_json_fallback_when_contract_fails():
    def fake_call_fn(state, prompt, node_name, *, fallback_factory, **kwargs):
        return JsonContractResult(
            artifact=fallback_factory("invalid json"),
            status="fallback",
            failure_reason="invalid json",
            log=[{"node": node_name, "validation_status": "fallback"}],
        )

    result = run_domain_sq_stage(
        {"sq_answers": {}},
        DomainSqStage(
            node_name="domain3_sq",
            sq_ids=("3.1",),
            source_domain="d3",
            build_prompt=lambda _state: "classify domain 3",
        ),
        call_fn=fake_call_fn,
    )

    assert result["sq_answers"]["3.1"]["answer"] == "NI"
    assert result["sq_answers"]["3.1"]["contract_validation_status"] == "fallback"
    assert result["sq_answers"]["3.1"]["support_rationale"] == "invalid json"


def test_domain2_conditional_legacy_branch_has_call_function(monkeypatch):
    import rob2_pipeline.nodes.domain2 as domain2

    captured = {}

    def fake_has_ready_packets(state, *, domain, sq_ids):
        captured["packet_check"] = (domain, sq_ids)
        return False

    def fake_run_domain_sq_stage(state, stage, call_fn=None):
        captured["stage"] = stage.node_name
        captured["call_fn"] = call_fn
        return {"sq_answers": {"2.3": {"answer": "NI"}}}

    monkeypatch.setattr(domain2, "has_ready_packets", fake_has_ready_packets)
    monkeypatch.setattr(domain2, "run_domain_sq_stage", fake_run_domain_sq_stage)

    result = domain2.domain2_conditional_node({"sq_answers": {}})

    assert captured["packet_check"] == ("d2", ("2.3", "2.4", "2.5"))
    assert captured["stage"] == "domain2_conditional"
    assert captured["call_fn"] is None
    assert result["sq_answers"]["2.3"]["answer"] == "NI"
