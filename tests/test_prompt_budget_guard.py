from rob2_pipeline.llm_contracts import enforce_prompt_budget


def test_under_budget_prompt_is_unchanged():
    prompt = "small prompt"
    out, dropped = enforce_prompt_budget(prompt, budget_tokens=1000, node="domain1_sq")
    assert out == prompt
    assert dropped == 0


def test_over_budget_prompt_is_trimmed_to_budget():
    prompt = "x" * 600000  # ~200k tokens at 3 chars/token
    out, dropped = enforce_prompt_budget(prompt, budget_tokens=115000, node="domain4_sq")
    assert len(out) <= 115000 * 3 + 80  # within budget (+marker)
    assert dropped > 0
