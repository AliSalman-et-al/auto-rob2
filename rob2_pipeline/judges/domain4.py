DOMAIN4_JUDGE_VERSION = "d4-judge-v1"
DOMAIN4_RULE_TABLE_VERSION = "rob2-d4-rule-table-v1"
DOMAIN4_SQ_IDS = ("4.1", "4.2", "4.3", "4.4", "4.5")


def judge_domain4(sq: dict) -> tuple[str, str]:
    """Implements the RoB 2 Domain 4 decision table."""
    s41 = sq.get("4.1", {}).get("answer", "NI")
    s42 = sq.get("4.2", {}).get("answer", "NI")
    s43 = sq.get("4.3", {}).get("answer", "NA")
    s44 = sq.get("4.4", {}).get("answer", "NA")
    s45 = sq.get("4.5", {}).get("answer", "NA")

    if s41 in ("Y", "PY"):
        return "High", "4.1=Y/PY (inappropriate measurement method) -> High"
    if s42 in ("Y", "PY"):
        return "High", "4.2=Y/PY (differential measurement between groups) -> High"
    if s45 in ("Y", "PY"):
        return (
            "High",
            "4.5=Y/PY (assessment likely influenced by intervention knowledge) -> High",
        )

    if s41 in ("N", "PN", "NI") and s42 in ("N", "PN") and s43 in ("N", "PN"):
        return "Low", "4.1=N/PN/NI, 4.2=N/PN, and 4.3=N/PN -> Low"
    if (
        s41 in ("N", "PN", "NI")
        and s42 in ("N", "PN")
        and s43 in ("Y", "PY", "NI")
        and s44 in ("N", "PN")
    ):
        return "Low", "4.1=N/PN/NI, 4.2=N/PN, 4.3=Y/PY/NI, and 4.4=N/PN -> Low"

    if s41 in ("N", "PN", "NI") and s42 == "NI" and s43 in ("N", "PN"):
        return "Some concerns", "4.2=NI and 4.3=N/PN -> Some concerns"
    if (
        s41 in ("N", "PN", "NI")
        and s42 == "NI"
        and s43 in ("Y", "PY", "NI")
        and s44 in ("N", "PN")
    ):
        return "Some concerns", "4.2=NI, 4.3=Y/PY/NI, and 4.4=N/PN -> Some concerns"
    if (
        s41 in ("N", "PN", "NI")
        and s42 in ("N", "PN", "NI")
        and s43 in ("Y", "PY", "NI")
        and s44 in ("Y", "PY", "NI")
        and s45 in ("N", "PN")
    ):
        return "Some concerns", "4.4=Y/PY/NI and 4.5=N/PN -> Some concerns"
    if (
        s41 in ("N", "PN", "NI")
        and s42 in ("N", "PN", "NI")
        and s43 in ("Y", "PY", "NI")
        and s44 in ("Y", "PY", "NI")
        and s45 == "NI"
    ):
        return "High", "4.4=Y/PY/NI and 4.5=NI -> High"

    return (
        "Some concerns",
        f"Unresolved D4 answers: 4.1={s41} 4.2={s42} 4.3={s43} 4.4={s44} 4.5={s45}",
    )


def judge_domain4_artifact(sq: dict) -> dict:
    """Apply the deterministic D4 table and return an audit artifact."""
    label, rationale = judge_domain4(sq)
    return {
        "artifact_id": "d4-judgment",
        "schema_version": "d4-judgment-v1",
        "domain": "d4",
        "judge_version": DOMAIN4_JUDGE_VERSION,
        "rule_table_version": DOMAIN4_RULE_TABLE_VERSION,
        "input_sq_answers": {sq_id: dict(sq.get(sq_id, {})) for sq_id in DOMAIN4_SQ_IDS},
        "applied_rule_path": _rule_path(label, rationale),
        "label": label,
        "rationale": rationale,
    }


def _rule_path(label: str, rationale: str) -> str:
    if rationale.startswith("4.1=Y/PY"):
        return "d4:inappropriate-measurement-method"
    if rationale.startswith("4.2=Y/PY"):
        return "d4:differential-measurement"
    if rationale.startswith("4.5=Y/PY"):
        return "d4:assessment-influenced-by-intervention-knowledge"
    if "4.3=N/PN -> Low" in rationale:
        return "d4:no-assessor-awareness"
    if "4.4=N/PN -> Low" in rationale:
        return "d4:objective-or-uninfluenced-assessment"
    if "4.2=NI" in rationale:
        return "d4:differential-measurement-unclear"
    if "4.4=Y/PY/NI and 4.5=N/PN" in rationale:
        return "d4:assessor-awareness-not-influential"
    if "4.4=Y/PY/NI and 4.5=NI" in rationale:
        return "d4:assessor-influence-unclear"
    if rationale.startswith("Unresolved D4 answers"):
        return "d4:unresolved"
    return f"d4:{label.lower().replace(' ', '-')}"
