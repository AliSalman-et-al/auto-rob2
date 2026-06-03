DOMAIN2_JUDGE_VERSION = "d2-judge-v1"
DOMAIN2_ASSIGNMENT_RULE_TABLE_VERSION = "rob2-d2-assignment-rule-table-v1"
DOMAIN2_ADHERING_RULE_TABLE_VERSION = "rob2-d2-adhering-rule-table-v1"
DOMAIN2_SQ_IDS = ("2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7")


def judge_domain2(sq: dict, effect_of_interest: str = "ITT") -> tuple[str, str]:
    """
    Implements the RoB 2 Domain 2 algorithms.

    Version A assesses the effect of assignment to intervention (ITT).
    Version B assesses the effect of adhering to intervention (per-protocol).
    """
    s21 = sq.get("2.1", {}).get("answer", "NI")
    s22 = sq.get("2.2", {}).get("answer", "NI")
    s23 = sq.get("2.3", {}).get("answer", "NA")
    s24 = sq.get("2.4", {}).get("answer", "NA")
    s25 = sq.get("2.5", {}).get("answer", "NA")
    s26 = sq.get("2.6", {}).get("answer", "NI")
    s27 = sq.get("2.7", {}).get("answer", "NA")

    if effect_of_interest.lower() == "per-protocol":
        aware = s21 in ("Y", "PY", "NI") or s22 in ("Y", "PY", "NI")
        non_protocol_balanced_or_na = s23 in ("Y", "PY", "NA")
        non_protocol_unbalanced_or_unknown = aware and s23 in ("N", "PN", "NI")
        implementation_problem = s24 in ("Y", "PY", "NI")
        adherence_problem = s25 in ("Y", "PY", "NI")

        no_biasing_deviations = (
            (not aware or non_protocol_balanced_or_na)
            and s24 in ("N", "PN", "NA")
            and s25 in ("N", "PN", "NA")
        )
        if no_biasing_deviations:
            return (
                "Low",
                "Version B Low: no important non-protocol imbalance, implementation failure, or non-adherence affecting the outcome",
            )

        deviation_concern = (
            non_protocol_unbalanced_or_unknown
            or implementation_problem
            or adherence_problem
        )
        if deviation_concern and s26 in ("Y", "PY"):
            return (
                "Some concerns",
                "Version B Some concerns: deviation concern addressed by appropriate adherence-effect analysis",
            )
        if deviation_concern and s26 in ("N", "PN", "NI"):
            return (
                "High",
                "Version B High: deviation concern without appropriate adherence-effect analysis",
            )
        return "Some concerns", (
            f"Version B unresolved official-table path: 2.1={s21}, 2.2={s22}, "
            f"2.3={s23}, 2.4={s24}, 2.5={s25}, 2.6={s26}"
        )

    def _part1():
        if s21 in ("N", "PN") and s22 in ("N", "PN"):
            return (
                "Low",
                "Part1 Low: participants, carers, and intervention deliverers were unaware",
            )
        if (s21 in ("Y", "PY", "NI") or s22 in ("Y", "PY", "NI")) and s23 in (
            "N",
            "PN",
        ):
            return (
                "Low",
                "Part1 Low: awareness present/unclear but no trial-context deviations",
            )
        if (s21 in ("Y", "PY", "NI") or s22 in ("Y", "PY", "NI")) and s23 == "NI":
            return (
                "Some concerns",
                "Part1 Some concerns: no information on trial-context deviations",
            )
        if (
            (s21 in ("Y", "PY", "NI") or s22 in ("Y", "PY", "NI"))
            and s23 in ("Y", "PY")
            and s24 in ("N", "PN")
        ):
            return (
                "Some concerns",
                "Part1 Some concerns: deviations were not likely to affect the outcome",
            )
        if (
            (s21 in ("Y", "PY", "NI") or s22 in ("Y", "PY", "NI"))
            and s23 in ("Y", "PY")
            and s24 in ("Y", "PY", "NI")
            and s25 in ("Y", "PY")
        ):
            return (
                "Some concerns",
                "Part1 Some concerns: outcome-affecting deviations were balanced",
            )
        if (
            (s21 in ("Y", "PY", "NI") or s22 in ("Y", "PY", "NI"))
            and s23 in ("Y", "PY")
            and s24 in ("Y", "PY", "NI")
            and s25 in ("N", "PN", "NI")
        ):
            return (
                "High",
                "Part1 High: outcome-affecting deviations were not balanced or balance was unknown",
            )
        return "Some concerns", (
            f"Part1 unresolved official-table path: 2.1={s21}, 2.2={s22}, "
            f"2.3={s23}, 2.4={s24}, 2.5={s25}"
        )

    def _part2():
        if s26 in ("Y", "PY"):
            return "Low", "2.6=Y/PY (appropriate ITT analysis) -> Part2=Low"
        if s26 in ("N", "PN", "NI") and s27 in ("N", "PN"):
            return "Some concerns", "2.6=N/PN/NI and 2.7=N/PN -> Part2=Some concerns"
        if s26 in ("N", "PN", "NI") and s27 in ("Y", "PY", "NI"):
            return "High", "2.6=N/PN/NI and 2.7=Y/PY/NI -> Part2=High"
        return (
            "Some concerns",
            f"Part2 unresolved official-table path: 2.6={s26}, 2.7={s27}",
        )

    p1_j, p1_r = _part1()
    p2_j, p2_r = _part2()

    if p1_j == "High" or p2_j == "High":
        return "High", f"Part1={p1_j} ({p1_r}); Part2={p2_j} ({p2_r})"
    if p1_j == "Some concerns" or p2_j == "Some concerns":
        return "Some concerns", f"Part1={p1_j} ({p1_r}); Part2={p2_j} ({p2_r})"
    return "Low", f"Part1=Low ({p1_r}); Part2=Low ({p2_r})"


def judge_domain2_artifact(sq: dict, effect_of_interest: str = "ITT") -> dict:
    """Apply the deterministic D2 table and return an audit artifact."""
    label, rationale = judge_domain2(sq, effect_of_interest)
    mode = (
        "adhering"
        if effect_of_interest.lower() == "per-protocol"
        else "assignment"
    )
    return {
        "artifact_id": "d2-judgment",
        "schema_version": "d2-judgment-v1",
        "domain": "d2",
        "judge_version": DOMAIN2_JUDGE_VERSION,
        "rule_table_version": (
            DOMAIN2_ADHERING_RULE_TABLE_VERSION
            if mode == "adhering"
            else DOMAIN2_ASSIGNMENT_RULE_TABLE_VERSION
        ),
        "effect_of_interest": effect_of_interest,
        "input_sq_answers": {sq_id: dict(sq.get(sq_id, {})) for sq_id in DOMAIN2_SQ_IDS},
        "applied_rule_path": _rule_path(mode, label, rationale),
        "label": label,
        "rationale": rationale,
    }


def _rule_path(mode: str, label: str, rationale: str) -> str:
    normalized = label.lower().replace(" ", "-")
    if mode == "assignment":
        part1 = _extract_part_label(rationale, "Part1")
        part2 = _extract_part_label(rationale, "Part2")
        if part1 and part2:
            return f"d2-assignment:part1-{part1}+part2-{part2}"
    if rationale.startswith("Version B Low"):
        return "d2-adhering:no-biasing-deviations"
    if rationale.startswith("Version B Some concerns"):
        return "d2-adhering:deviation-concern+appropriate-analysis"
    if rationale.startswith("Version B High"):
        return "d2-adhering:deviation-concern+inappropriate-analysis"
    if "unresolved official-table path" in rationale:
        return f"d2-{mode}:unresolved"
    return f"d2-{mode}:{normalized}"


def _extract_part_label(rationale: str, part: str) -> str:
    marker = f"{part}="
    if marker not in rationale:
        return ""
    value = rationale.split(marker, 1)[1].split(" ", 1)[0]
    return value.lower().replace(" ", "-")
