OVERALL_JUDGE_VERSION = "overall-judge-v1"
OVERALL_RULE_TABLE_VERSION = "rob2-overall-rule-table-v1"


def judge_overall(domain_judgments: dict) -> tuple[str, str]:
    """
    Low: Low in ALL domains.
    Some concerns: Some concerns in >=1 domain, High in none.
    High: High in >=1 domain. Multiple Some concerns should be escalated only
    when they substantially lower confidence in the result; this implementation
    flags that case for review rather than applying a blind count threshold.
    """
    values = list(domain_judgments.values())

    if any(v == "High" for v in values):
        high_domains = [k for k, v in domain_judgments.items() if v == "High"]
        return "High", f"High in: {', '.join(high_domains)}"

    if values and all(v == "Low" for v in values):
        return "Low", "Low in all 5 domains"

    some_concerns_domains = [
        k for k, v in domain_judgments.items() if v == "Some concerns"
    ]
    n_sc = len(some_concerns_domains)

    if n_sc == 2:
        return "Some concerns", (
            f"Some concerns in 2 domains: {', '.join(some_concerns_domains)}. "
            "2 domains with Some concerns. "
            "Skill guidance: present both Some concerns and High as plausible if the "
            "concerns are complementary and together substantially lower confidence. "
            "FLAG FOR HUMAN REVIEW."
        )
    if n_sc >= 3:
        return "Some concerns", (
            f"Some concerns in {n_sc} domains ({', '.join(some_concerns_domains)}). "
            "Skill guidance: 3 or more domains with Some concerns is very likely an "
            "overall High judgment if the concerns substantially lower confidence. "
            "Probable High; FLAG FOR HUMAN REVIEW/CONFIRMATION."
        )

    return (
        "Some concerns",
        f"Some concerns in {n_sc} domain(s): {', '.join(some_concerns_domains)}",
    )


def judge_overall_artifact(
    domain_judgments: dict, policy: str = "official_rob2"
) -> dict:
    """Apply the deterministic overall policy and return an audit artifact."""
    label, rationale = judge_overall(domain_judgments)
    rule_path = _overall_rule_path(domain_judgments, label)
    if policy == "benchmark_reference" and _benchmark_reference_low(domain_judgments):
        label = "Low"
        rationale = "Benchmark-reference policy: Low when at most one domain has Some concerns and none are High."
        rule_path = "overall:benchmark-reference-at-most-one-some-concern"
    return {
        "artifact_id": "overall-judgment",
        "schema_version": "overall-judgment-v1",
        "judge_version": OVERALL_JUDGE_VERSION,
        "rule_table_version": OVERALL_RULE_TABLE_VERSION,
        "policy": policy,
        "input_domain_judgments": dict(domain_judgments),
        "applied_rule_path": rule_path,
        "label": label,
        "rationale": rationale,
    }


def _overall_rule_path(domain_judgments: dict, label: str) -> str:
    values = list(domain_judgments.values())
    if any(value == "High" for value in values):
        return "overall:any-high"
    if values and all(value == "Low" for value in values):
        return "overall:all-low"
    some_concerns_count = sum(1 for value in values if value == "Some concerns")
    if some_concerns_count >= 3:
        return "overall:three-or-more-some-concerns"
    if some_concerns_count == 2:
        return "overall:two-some-concerns"
    return f"overall:{label.lower().replace(' ', '-')}"


def _benchmark_reference_low(domain_judgments: dict) -> bool:
    values = list(domain_judgments.values())
    return (
        bool(values)
        and not any(value == "High" for value in values)
        and sum(1 for value in values if value == "Some concerns") <= 1
    )
