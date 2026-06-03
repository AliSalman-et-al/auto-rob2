DOMAIN5_JUDGE_VERSION = "d5-judge-v1"
DOMAIN5_RULE_TABLE_VERSION = "rob2-d5-rule-table-v1"
DOMAIN5_SQ_IDS = ("5.1", "5.2", "5.3")


def judge_domain5(sq: dict) -> tuple[str, str]:
    """Implements the RoB 2 Domain 5 decision table."""
    s51 = sq.get("5.1", {}).get("answer", "NI")
    s52 = sq.get("5.2", {}).get("answer", "NI")
    s53 = sq.get("5.3", {}).get("answer", "NI")

    if s52 in ("Y", "PY") or s53 in ("Y", "PY"):
        return "High", "5.2 or 5.3 = Y/PY (selective result reporting) -> High"
    if s51 in ("Y", "PY") and s52 in ("N", "PN") and s53 in ("N", "PN"):
        return "Low", "5.1=Y/PY and 5.2=5.3=N/PN -> Low"
    return "Some concerns", f"5.1={s51} 5.2={s52} 5.3={s53} -> Some concerns"


def judge_domain5_artifact(sq: dict) -> dict:
    """Apply the deterministic D5 table and return an audit artifact."""
    label, rationale = judge_domain5(sq)
    return {
        "artifact_id": "d5-judgment",
        "schema_version": "d5-judgment-v1",
        "domain": "d5",
        "judge_version": DOMAIN5_JUDGE_VERSION,
        "rule_table_version": DOMAIN5_RULE_TABLE_VERSION,
        "input_sq_answers": {sq_id: dict(sq.get(sq_id, {})) for sq_id in DOMAIN5_SQ_IDS},
        "applied_rule_path": _rule_path(label, rationale),
        "label": label,
        "rationale": rationale,
    }


def _rule_path(label: str, rationale: str) -> str:
    if rationale.startswith("5.2 or 5.3 = Y/PY"):
        return "d5:selective-result-reporting"
    if rationale.startswith("5.1=Y/PY and 5.2=5.3=N/PN"):
        return "d5:prespecified-and-not-selective"
    return f"d5:{label.lower().replace(' ', '-')}"
