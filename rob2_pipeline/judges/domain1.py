DOMAIN1_JUDGE_VERSION = "d1-judge-v1"
DOMAIN1_RULE_TABLE_VERSION = "rob2-d1-rule-table-v1"


def judge_domain1(sq: dict) -> tuple[str, str]:
    """
    Decision table (from rob2-algorithm.md):
    1.1        | 1.2       | 1.3       | Judgment
    Y/PY/NI    | Y/PY      | NI/N/PN   | Low
    Y/PY       | Y/PY      | Y/PY      | Some concerns
    N/PN/NI    | Y/PY      | Y/PY      | Some concerns
    Any        | NI        | N/PN/NI   | Some concerns
    Any        | NI        | Y/PY      | High
    Any        | N/PN      | Any       | High
    """
    artifact = judge_domain1_artifact(sq)
    return artifact["label"], artifact["rationale"]


def judge_domain1_artifact(sq: dict) -> dict:
    """Apply the deterministic D1 table and return an audit artifact."""

    s11 = sq.get("1.1", {}).get("answer", "NI")
    s12 = sq.get("1.2", {}).get("answer", "NI")
    s13 = sq.get("1.3", {}).get("answer", "NI")

    if s12 in ("N", "PN"):
        return _artifact(
            sq,
            "High",
            "Row: Any / N-PN / Any -> High (allocation not concealed)",
            "d1-row-6:any/n-pn/any",
        )
    if s12 == "NI" and s13 in ("Y", "PY"):
        return _artifact(
            sq,
            "High",
            "Row: Any / NI / Y-PY -> High (no concealment info + baseline imbalance)",
            "d1-row-5:any/ni/y-py",
        )
    if s12 == "NI" and s13 in ("N", "PN", "NI"):
        return _artifact(
            sq,
            "Some concerns",
            "Row: Any / NI / N-PN-NI -> Some concerns (concealment unclear)",
            "d1-row-4:any/ni/n-pn-ni",
        )
    if s11 in ("Y", "PY", "NI") and s12 in ("Y", "PY") and s13 in ("NI", "N", "PN"):
        return _artifact(
            sq,
            "Low",
            "Row: Y-PY-NI / Y-PY / NI-N-PN -> Low",
            "d1-row-1:y-py-ni/y-py/ni-n-pn",
        )
    if s12 in ("Y", "PY") and s13 in ("Y", "PY"):
        return _artifact(
            sq,
            "Some concerns",
            "Row: Any / Y-PY / Y-PY -> Some concerns (baseline imbalance)",
            "d1-row-2-3:any/y-py/y-py",
        )
    return _artifact(
        sq,
        "Some concerns",
        f"No exact row match for 1.1={s11}, 1.2={s12}, 1.3={s13}; defaulting to Some concerns",
        "d1-default:no-exact-row-match",
    )


def _artifact(sq: dict, label: str, rationale: str, rule_path: str) -> dict:
    return {
        "artifact_id": "d1-judgment",
        "schema_version": "d1-judgment-v1",
        "domain": "d1",
        "judge_version": DOMAIN1_JUDGE_VERSION,
        "rule_table_version": DOMAIN1_RULE_TABLE_VERSION,
        "input_sq_answers": {
            sq_id: dict(sq.get(sq_id, {})) for sq_id in ("1.1", "1.2", "1.3")
        },
        "applied_rule_path": rule_path,
        "label": label,
        "rationale": rationale,
    }
