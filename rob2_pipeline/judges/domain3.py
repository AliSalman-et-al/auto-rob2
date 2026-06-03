DOMAIN3_JUDGE_VERSION = "d3-judge-v1"
DOMAIN3_RULE_TABLE_VERSION = "rob2-d3-rule-table-v1"
DOMAIN3_SQ_IDS = ("3.1", "3.2", "3.3", "3.4")


def judge_domain3(sq: dict) -> tuple[str, str]:
    """
    3.1=Y/PY -> Low
    3.1=N/PN/NI | 3.2=Y/PY -> Low
    3.1=N/PN/NI | 3.2=N/PN | 3.3=N/PN -> Low
    3.1=N/PN/NI | 3.2=N/PN | 3.3=Y/PY/NI | 3.4=N/PN -> Some concerns
    3.1=N/PN/NI | 3.2=N/PN | 3.3=Y/PY/NI | 3.4=Y/PY/NI -> High
    """
    s31 = sq.get("3.1", {}).get("answer", "NI")
    s32 = sq.get("3.2", {}).get("answer", "NA")
    s33 = sq.get("3.3", {}).get("answer", "NA")
    s34 = sq.get("3.4", {}).get("answer", "NA")

    if s31 in ("Y", "PY"):
        return "Low", "3.1=Y/PY (nearly complete data) -> Low"
    if s32 in ("Y", "PY"):
        return "Low", "3.2=Y/PY (evidence of no bias from missing data) -> Low"
    if s33 in ("N", "PN"):
        return "Low", "3.3=N/PN (missingness cannot depend on true value) -> Low"
    if s33 in ("Y", "PY", "NI") and s34 in ("Y", "PY", "NI"):
        return "High", "3.3=Y/PY/NI and 3.4=Y/PY/NI -> High"
    if s33 in ("Y", "PY", "NI") and s34 in ("N", "PN"):
        return "Some concerns", "3.3=Y/PY/NI and 3.4=N/PN -> Some concerns"
    return (
        "Some concerns",
        f"Unresolved D3 answers: 3.1={s31} 3.2={s32} 3.3={s33} 3.4={s34}",
    )


def judge_domain3_artifact(sq: dict) -> dict:
    """Apply the deterministic D3 table and return an audit artifact."""
    label, rationale = judge_domain3(sq)
    return {
        "artifact_id": "d3-judgment",
        "schema_version": "d3-judgment-v1",
        "domain": "d3",
        "judge_version": DOMAIN3_JUDGE_VERSION,
        "rule_table_version": DOMAIN3_RULE_TABLE_VERSION,
        "input_sq_answers": {sq_id: dict(sq.get(sq_id, {})) for sq_id in DOMAIN3_SQ_IDS},
        "applied_rule_path": _rule_path(label, rationale),
        "label": label,
        "rationale": rationale,
    }


def _rule_path(label: str, rationale: str) -> str:
    if rationale.startswith("3.1=Y/PY"):
        return "d3:nearly-complete-data"
    if rationale.startswith("3.2=Y/PY"):
        return "d3:no-bias-from-missing-data"
    if rationale.startswith("3.3=N/PN"):
        return "d3:missingness-not-dependent-on-true-value"
    if rationale.startswith("3.3=Y/PY/NI and 3.4=Y/PY/NI"):
        return "d3:missingness-likely-dependent-on-true-value"
    if rationale.startswith("3.3=Y/PY/NI and 3.4=N/PN"):
        return "d3:missingness-may-depend-on-true-value"
    if rationale.startswith("Unresolved D3 answers"):
        return "d3:unresolved"
    return f"d3:{label.lower().replace(' ', '-')}"
