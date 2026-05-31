from rob2_pipeline.state import RoB2State


DOMAIN_TITLES = {
    "D1": "Bias arising from the randomization process",
    "D2": "Bias due to deviations from intended interventions",
    "D3": "Bias due to missing outcome data",
    "D4": "Bias in measurement of the outcome",
    "D5": "Bias in selection of the reported result",
}

QUESTIONS = {
    "1.1": "Was the allocation sequence random?",
    "1.2": "Was the allocation sequence concealed?",
    "1.3": "Did baseline differences suggest a problem?",
    "2.1": "Were participants aware of assigned intervention?",
    "2.2": "Were carers and people delivering interventions aware?",
    "2.3": "Were there trial-context deviations?",
    "2.4": "Were deviations likely to affect the outcome?",
    "2.5": "Were deviations balanced between groups?",
    "2.6": "Was an appropriate ITT analysis used?",
    "2.7": "Was there potential for substantial impact?",
    "3.1": "Were outcome data available for nearly all participants?",
    "3.2": "Is there evidence the result was not biased by missing data?",
    "3.3": "Could missingness depend on its true value?",
    "3.4": "Is it likely missingness depended on its true value?",
    "4.1": "Was the outcome measurement method inappropriate?",
    "4.2": "Could measurement have differed between groups?",
    "4.3": "Were outcome assessors aware of intervention received?",
    "4.4": "Could knowledge of intervention influence assessment?",
    "4.5": "Was assessment likely influenced by knowledge?",
    "5.1": "Was analysis in accordance with a pre-specified plan?",
    "5.2": "Was result selected from multiple outcome measurements?",
    "5.3": "Was result selected from multiple analyses?",
}

DOMAIN_SQS = {
    "D1": ["1.1", "1.2", "1.3"],
    "D2": ["2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7"],
    "D3": ["3.1", "3.2", "3.3", "3.4"],
    "D4": ["4.1", "4.2", "4.3", "4.4", "4.5"],
    "D5": ["5.1", "5.2", "5.3"],
}


def _clean_cell(value: str) -> str:
    cleaned = value or "No relevant text found"
    if cleaned.startswith("Auto-set:"):
        cleaned = "No relevant text found"
    return cleaned.replace("\n", " ").replace("|", "\\|")


def _support_cell(answer: dict) -> str:
    level = _clean_cell(answer.get("support_level", "unsupported")).title()
    rationale = _clean_cell(answer.get("support_rationale", "No relevant text found"))
    return f"**{level}**: {rationale}"


def _effect_label(state: RoB2State) -> str:
    effect = state.get("effect_of_interest", "ITT")
    if str(effect).lower() == "per-protocol":
        return "Effect of adhering to intervention (per-protocol)"
    return "Effect of assignment to intervention (intention-to-treat)"


def _domain_table(state: RoB2State, domain: str) -> str:
    sq_answers = state.get("sq_answers", {})
    rows = [
        f"## Domain {domain[-1]}: {DOMAIN_TITLES[domain]}",
        "",
        "| Question | Answer | Evidence support | Supporting quote | Justification |",
        "|----------|--------|------------------|-----------------|---------------|",
    ]
    for sq_id in DOMAIN_SQS[domain]:
        answer = sq_answers.get(sq_id, {})
        rows.append(
            "| "
            f"{sq_id} {QUESTIONS[sq_id]} | "
            f"{_clean_cell(answer.get('answer', 'NI'))} | "
            f"{_support_cell(answer)} | "
            f"{_clean_cell(answer.get('quote', 'No relevant text found'))} | "
            f"{_clean_cell(answer.get('justification', 'No relevant text found'))} |"
        )
    rows.extend(
        [
            "",
            f"**Domain {domain[-1]} judgment: {state.get('domain_judgments', {}).get(domain, 'Not assessed')}**",
            f"**Algorithm rationale:** {state.get('domain_rationales', {}).get(domain, 'Not assessed')}",
            "",
        ]
    )
    return "\n".join(rows)


def _packet_quality_section(state: RoB2State) -> str:
    packet_grades = state.get("packet_grades", {}) or {}
    actions = state.get("verification_actions", []) or []
    retry_sqs = [
        sq_id
        for sq_id, grade in sorted(packet_grades.items())
        if grade.get("retry_recommended")
    ]
    retry_text = ", ".join(retry_sqs) if retry_sqs else "None"
    action_text = (
        "; ".join(
            f"{action.get('sq_id', '?')}: {action.get('action', 'review')}"
            for action in actions[:8]
        )
        or "None"
    )
    return "\n".join(
        [
            "## Verified evidence packets",
            "",
            f"- Packets built: {len(packet_grades)}",
            f"- Packets requiring retry/escalation: {retry_text}",
            f"- Verification actions: {action_text}",
            "",
        ]
    )


def _title_level(value: str) -> str:
    return _clean_cell(value).title()


def _answer_delta(initial: dict, adjudicated: dict) -> str:
    initial_answer = _clean_cell(initial.get("answer", "NI"))
    final_answer = _clean_cell(adjudicated.get("answer", "NI"))
    initial_support = _title_level(initial.get("support_level", "unsupported"))
    final_support = _title_level(adjudicated.get("support_level", "unsupported"))
    return f"Answer {initial_answer} -> {final_answer}; support {initial_support} -> {final_support}"


def _adjudication_status(attempt: dict) -> str:
    adjudicated = attempt.get("adjudicated_answer", {}) or {}
    support_level = str(adjudicated.get("support_level", "")).lower()
    if support_level in {"weak", "unsupported"} or adjudicated.get("residual_uncertainty"):
        return "uncertainty remains"
    return "changed" if attempt.get("changed") else "unchanged"


def _adjudication_summary_section(state: RoB2State) -> str:
    adjudications = state.get("sq_support_adjudications", {}) or {}
    if not adjudications:
        return ""

    rows = ["## SQ support adjudication", ""]
    for domain in ["D1", "D2", "D3", "D4", "D5"]:
        for attempt in adjudications.get(domain, []):
            sq_id = _clean_cell(attempt.get("sq_id", "?"))
            initial = attempt.get("initial_answer", {}) or {}
            adjudicated = attempt.get("adjudicated_answer", {}) or {}
            status = _adjudication_status(attempt)
            rows.append(
                f"- {domain} SQ {sq_id} triggered adjudication ({status}): "
                f"{_answer_delta(initial, adjudicated)}."
            )
            rationale = _clean_cell(
                adjudicated.get("support_rationale")
                or adjudicated.get("justification")
                or "No relevant text found"
            )
            rows.append(f"  Final support: {rationale}")
            if adjudicated.get("residual_uncertainty"):
                rows.append(
                    f"  Residual uncertainty: {_clean_cell(adjudicated.get('residual_uncertainty'))}"
                )
    rows.append("")
    return "\n".join(rows)


def report_formatter_node(state: RoB2State) -> RoB2State:
    high_uncertainty = state.get("high_uncertainty_sqs", [])
    high_uncertainty_text = ", ".join(high_uncertainty) if high_uncertainty else "None"
    sources = ", ".join(state.get("sources_consulted", [])) or "Not reported"
    limitations = (
        "This is an automated first-pass assessment for human review. Human verification is required, "
        f"especially for {state.get('ni_count', 0)} NI answer(s), high-uncertainty signaling questions "
        f"({high_uncertainty_text}), and any overall-judgment escalation flagged in the rationale."
    )
    parts = [
        "# RoB 2 Assessment",
        "",
        "## Trial information",
        f"- **Trial:** {state.get('intervention', 'Not reported')} vs {state.get('comparator', 'Not reported')}",
        f"- **Experimental intervention:** {state.get('intervention', 'Not reported')}",
        f"- **Comparator:** {state.get('comparator', 'Not reported')}",
        f"- **Outcome assessed:** {state.get('outcome', 'Not reported')}",
        f"- **Numerical result:** {state.get('numerical_result', 'Not reported')}",
        f"- **Effect of interest:** {_effect_label(state)}",
        f"- **Sources consulted:** {sources}",
        "",
    ]
    for domain in ["D1", "D2", "D3", "D4", "D5"]:
        parts.append(_domain_table(state, domain))
    adjudication_summary = _adjudication_summary_section(state)
    if adjudication_summary:
        parts.append(adjudication_summary)
    parts.append(_packet_quality_section(state))
    parts.extend(
        [
            "## Overall risk of bias",
            "",
            f"**Overall judgment: {state.get('overall_judgment', 'Not assessed')}**",
            "",
            f"**Rationale:** {state.get('overall_rationale', 'Not assessed')}",
            "",
            "## Limitations of this assessment",
            limitations,
            "",
            "## Quality flags",
            f"- NI answers: {state.get('ni_count', 0)}",
            f"- High-uncertainty signaling questions: {high_uncertainty_text}",
            f"- Human review priority: {state.get('human_review_priority', 'HIGH')}",
        ]
    )
    markdown_report = "\n".join(parts)
    return {"markdown_report": markdown_report}
