import re
from typing import Literal

from rob2_pipeline.nodes.common import set_na
from rob2_pipeline.state import RoB2State


_TTE_DIRECT_MISSINGNESS_RE = re.compile(
    r"\b("
    r"missing\s+outcome\s+data|"
    r"outcome\s+data\s+(?:were\s+)?(?:available|complete)|"
    r"complete\s+(?:outcome\s+)?(?:data|follow-up|followup)|"
    r"all\s+(?:patients|participants).{0,40}(?:followed|outcome\s+data)|"
    r"(?:lost|loss)\s+to\s+follow[- ]?up|"
    r"no\s+(?:patients|participants).{0,40}(?:lost|missing)|"
    r"ascertain(?:ed|ment).{0,40}(?:complete|status|outcome|survival|event)|"
    r"(?:vital|survival|mortality)\s+status.{0,40}(?:available|ascertained|complete)"
    r")\b",
    re.IGNORECASE,
)


_TTE_ANALYSIS_ONLY_RE = re.compile(
    r"\b("
    r"intention[- ]to[- ]treat|itt|"
    r"kaplan[- ]meier|cox|hazard\s+ratio|"
    r"censor(?:ed|ing)?|last\s+follow[- ]?up|analysis\s+population"
    r")\b",
    re.IGNORECASE,
)


_D2_AFFIRMATIVE_NO_DEVIATIONS_RE = re.compile(
    r"\b("
    r"no\s+(?:(?:major|important|substantial)\s+)?(?:protocol\s+)?"
    r"deviations?\s+(?:occurred|were\s+reported|were\s+observed|were\s+recorded)|"
    r"there\s+were\s+no\s+(?:major\s+|important\s+|substantial\s+|protocol\s+)?"
    r"(?:protocol\s+)?deviations?|"
    r"no\s+participants?\s+(?:crossed\s+over|received\s+non[- ]protocol|switched\s+treatment)|"
    r"adherence\s+(?:was\s+)?(?:high|similar).{0,60}(?:between|across)\s+(?:groups|arms)|"
    r"(?:protocol\s+)?deviations?\s+(?:were\s+)?(?:balanced|similar)\s+"
    r"(?:between|across)\s+(?:groups|arms)"
    r")\b",
    re.IGNORECASE,
)


_D2_DEVIATIONS_PRESENT_RE = re.compile(
    r"\b("
    r"protocol\s+deviations?\s+included|"
    r"(?:\d+|[1-9]\d*%|several|some)\s+major\s+protocol\s+deviations?|"
    r"non[- ]?adheren(?:ce|t)|poor\s+adherence|treatment\s+discontinuation|"
    r"dose\s+(?:interruptions?|reductions?|modifications?)|"
    r"crossed\s+over|cross[- ]?over|crossover|switched\s+(?:to|from)\s+(?:the\s+)?(?:active|control|placebo|treatment)|"
    r"contamination|non[- ]protocol\s+(?:intervention|treatment|therapy)|"
    r"co[- ]interventions?|concomitant\s+(?:therapy|treatment)|"
    r"rescue\s+(?:therapy|treatment|medication)|"
    r"imbalanc(?:e|ed).{0,80}(?:deviation|adherence|discontinuation|interruption|rescue|co[- ]intervention)"
    r")\b",
    re.IGNORECASE,
)


_D2_GENERIC_SILENCE_RE = re.compile(
    r"\b("
    r"no\s+relevant\s+text\s+found|not\s+reported|not\s+mentioned|"
    r"did\s+not\s+mention|absence\s+of\s+reported|"
    r"methods\s+were\s+described|results\s+were\s+summari[sz]ed|"
    r"intention[- ]to[- ]treat|itt|all\s+randomi[sz]ed"
    r")\b",
    re.IGNORECASE,
)


def classify_d2_deviation_evidence(
    state: RoB2State, text: str
) -> dict[str, object]:
    """Classify whether SQ 2.3 has direct trial-context deviation support."""
    evidence_text = " ".join([_domain2_deviation_text(state), str(text or "")])
    has_affirmative_no = bool(_D2_AFFIRMATIVE_NO_DEVIATIONS_RE.search(evidence_text))
    has_deviations = bool(_D2_DEVIATIONS_PRESENT_RE.search(evidence_text))
    explicit_contradiction = has_affirmative_no and bool(
        re.search(
            r"\b(crossed\s+over|cross[- ]?over|crossover|non[- ]protocol\s+(?:intervention|treatment|therapy)|rescue\s+(?:therapy|treatment|medication))\b",
            evidence_text,
            re.I,
        )
    )

    if has_deviations and explicit_contradiction:
        return {
            "classification": "contradictory",
            "reason": "Evidence contains both no-deviation language and deviation signals.",
        }
    if has_deviations:
        return {
            "classification": "deviations_present",
            "reason": "Evidence reports protocol deviations, non-adherence, cross-over, co-interventions, rescue therapy, or arm imbalance.",
        }
    if has_affirmative_no:
        return {
            "classification": "affirmative_no_deviations",
            "reason": "Evidence affirmatively states no relevant deviations or balanced deviation/adherence patterns.",
        }
    if _D2_GENERIC_SILENCE_RE.search(evidence_text) or not evidence_text.strip():
        return {
            "classification": "insufficient",
            "reason": "Generic methods/results text, ITT language, or lack of reported deviations is not affirmative no-deviation evidence.",
        }
    return {
        "classification": "insufficient",
        "reason": "Direct evidence about trial-context deviations is missing.",
    }


def apply_domain2_sq12_control(
    state: RoB2State, sq_answers: dict[str, dict]
) -> dict[str, dict]:
    s21 = sq_answers.get("2.1", {}).get("answer", "NI")
    s22 = sq_answers.get("2.2", {}).get("answer", "NI")
    if (
        state.get("effect_of_interest", "ITT").lower() != "per-protocol"
        and s21 in ("N", "PN")
        and s22 in ("N", "PN")
    ):
        return set_na(sq_answers, "2.3", "2.4", "2.5")
    return sq_answers


def next_domain2_stage(state: RoB2State) -> Literal["conditional", "analysis"]:
    if state.get("effect_of_interest", "ITT").lower() == "per-protocol":
        return "conditional"
    s21 = state["sq_answers"].get("2.1", {}).get("answer", "NI")
    s22 = state["sq_answers"].get("2.2", {}).get("answer", "NI")
    if s21 in ("N", "PN") and s22 in ("N", "PN"):
        return "analysis"
    return "conditional"


def apply_domain2_conditional_control(
    state: RoB2State, sq_answers: dict[str, dict]
) -> dict[str, dict]:
    if state.get("effect_of_interest", "ITT").lower() == "per-protocol":
        return sq_answers
    sq_answers = _apply_d2_deviation_gate(state, sq_answers)
    s23 = sq_answers.get("2.3", {}).get("answer", "NI")
    s24 = sq_answers.get("2.4", {}).get("answer", "NI")
    if s23 in ("N", "PN", "NI"):
        return set_na(sq_answers, "2.4", "2.5")
    if s24 in ("N", "PN", "NA"):
        return set_na(sq_answers, "2.5")
    return sq_answers


def _apply_d2_deviation_gate(
    state: RoB2State, sq_answers: dict[str, dict]
) -> dict[str, dict]:
    sq23 = sq_answers.get("2.3", {})
    if sq23.get("answer") not in ("N", "PN"):
        return sq_answers
    if not _domain2_participants_or_personnel_aware(state):
        return sq_answers

    quote_text = " ".join(
        str(sq23.get(field, "")) for field in ("quote", "justification")
    )
    support = classify_d2_deviation_evidence(state, quote_text)
    classification = support["classification"]
    if classification == "affirmative_no_deviations":
        return sq_answers

    updated = dict(sq_answers)
    if classification == "deviations_present":
        answer = "Y"
        note = "D2 deviation evidence gate classified 2.3 support as deviations present"
    else:
        answer = "NI"
        note = (
            "D2 deviation evidence gate classified 2.3 support as "
            f"{classification}"
        )
    updated["2.3"] = {
        **sq23,
        "answer": answer,
        "justification": (
            f"{sq23.get('justification', '').strip()} "
            f"{note}: {support['reason']}"
        ).strip(),
        "uncertainty_flag": sq23.get("uncertainty_flag", "HIGH"),
        "deviation_evidence_support": support,
    }
    for sq_id in ("2.4", "2.5"):
        updated.setdefault(
            sq_id,
            {
                "answer": "NI",
                "quote": "No relevant text found",
                "justification": "D2 remains applicable after deviation evidence gating.",
                "uncertainty_flag": "HIGH",
            },
        )
    return updated


def _domain2_participants_or_personnel_aware(state: RoB2State) -> bool:
    sq_answers = state.get("sq_answers") or {}
    s21 = sq_answers.get("2.1", {}).get("answer", "NI")
    s22 = sq_answers.get("2.2", {}).get("answer", "NI")
    if s21 in ("Y", "PY", "NI") or s22 in ("Y", "PY", "NI"):
        return True
    masking_facts = state.get("masking_facts") or {}
    for fact_name in ("participant_awareness", "personnel_awareness"):
        status = (masking_facts.get(fact_name) or {}).get("status")
        if status == "aware":
            return True
        if status == "unaware":
            continue
    return bool(
        re.search(
            r"\b(open-label|open label|unblinded|not blinded|not masked)\b",
            _domain2_deviation_text(state),
            re.I,
        )
    )


def _domain2_deviation_text(state: RoB2State) -> str:
    evidence = state.get("evidence") or {}
    parts = [
        _section_text(evidence.get("d2_deviations", {})),
        _section_text(evidence.get("results", {})),
        _section_text(evidence.get("methods", {})),
        str((state.get("rag_contexts") or {}).get("d2_deviations", "")),
        str((state.get("trial_facts") or {}).get("protocol_deviations", "")),
    ]
    return "\n".join(part for part in parts if part)


def apply_domain2_analysis_control(
    state: RoB2State, sq_answers: dict[str, dict]
) -> dict[str, dict]:
    if state.get(
        "effect_of_interest", "ITT"
    ).lower() == "per-protocol" or sq_answers.get("2.6", {}).get("answer", "NI") in (
        "Y",
        "PY",
    ):
        return set_na(sq_answers, "2.7")
    return sq_answers


def apply_domain3_control(
    state: RoB2State, sq_answers: dict[str, dict]
) -> dict[str, dict]:
    sq_answers = _apply_d3_completeness_gate(state, sq_answers)
    sq_answers = _calibrate_time_to_event_d3_1(state, sq_answers)
    s31 = sq_answers.get("3.1", {}).get("answer", "NI")
    s32 = sq_answers.get("3.2", {}).get("answer", "NA")
    s33 = sq_answers.get("3.3", {}).get("answer", "NA")
    if s31 in ("Y", "PY"):
        return set_na(sq_answers, "3.2", "3.3", "3.4")
    if s32 in ("Y", "PY"):
        return set_na(sq_answers, "3.3", "3.4")
    if s33 in ("N", "PN"):
        return set_na(sq_answers, "3.4")
    return sq_answers


_DIRECT_COMPLETENESS_RE = re.compile(
    r"\b("
    r"outcome\s+data\s+(?:were\s+)?(?:available|complete|obtained|collected)|"
    r"(?:complete|completed)\s+(?:outcome\s+)?(?:data|follow-up|followup)|"
    r"(?:vital|survival|mortality|event)\s+status.{0,60}(?:available|ascertained|complete)|"
    r"ascertain(?:ed|ment).{0,60}(?:complete|status|outcome|event|survival)|"
    r"(?:lost|loss)\s+to\s+follow[- ]?up|"
    r"missing\s+outcome\s+data|"
    r"no\s+(?:patients|participants).{0,50}(?:lost|missing)"
    r")\b",
    re.IGNORECASE,
)


_ANALYSIS_ONLY_RE = re.compile(
    r"\b("
    r"intention[- ]to[- ]treat|itt|modified\s+intention|mitt|"
    r"kaplan[- ]meier|cox|hazard\s+ratio|survival\s+model|"
    r"censor(?:ed|ing)?|last\s+follow[- ]?up|"
    r"analysis\s+population|full\s+analysis\s+set"
    r")\b",
    re.IGNORECASE,
)


_SAFETY_POPULATION_RE = re.compile(
    r"\b("
    r"safety\s+(?:analysis\s+)?population|safety\s+set|"
    r"received\s+at\s+least\s+one\s+dose|treated\s+population"
    r")\b",
    re.IGNORECASE,
)


_RANDOMIZED_DENOMINATOR_RE = re.compile(
    r"\b(?:randomi[sz]ed|randomly\s+assigned|enrolled)\D{0,40}(\d{2,5})\b|"
    r"\b(\d{2,5})\s+(?:patients|participants)\s+(?:were\s+)?(?:randomi[sz]ed|randomly\s+assigned)\b",
    re.IGNORECASE,
)


_OUTCOME_DENOMINATOR_RE = re.compile(
    r"\b(?:outcome\s+data|primary\s+outcome|vital\s+status|survival\s+status|event\s+status|"
    r"follow[- ]?up|analysis)\D{0,80}?(\d{2,5})\s+(?:of|/)\s+(\d{2,5})\b|"
    r"\b(\d{2,5})\s+(?:of|/)\s+(\d{2,5})\D{0,80}?"
    r"(?:outcome\s+data|primary\s+outcome|vital\s+status|survival\s+status|event\s+status|follow[- ]?up|analysis)\b|"
    r"\b(\d{2,5})\s+(?:patients|participants)\D{0,60}?"
    r"(?:had|with|included in|available for)\D{0,40}?"
    r"(?:outcome\s+data|primary\s+outcome|vital\s+status|survival\s+status|event\s+status|follow[- ]?up)\b",
    re.IGNORECASE,
)


def classify_d3_completeness_support(state: RoB2State, text: str) -> dict[str, object]:
    """Classify whether D3 3.1 has direct completeness support."""
    evidence_text = " ".join([_domain3_missingness_text(state), str(text or "")])
    randomized = _randomized_denominators(evidence_text)
    outcome_counts = _outcome_counts(evidence_text)
    direct = bool(_DIRECT_COMPLETENESS_RE.search(evidence_text))
    analysis_only = bool(_ANALYSIS_ONLY_RE.search(evidence_text))
    safety_only = bool(_SAFETY_POPULATION_RE.search(evidence_text)) and not direct

    if randomized and outcome_counts:
        largest_randomized = max(randomized)
        best_observed = max(count for count, _total in outcome_counts)
        best_total = max(total for _count, total in outcome_counts)
        if best_total and best_total != largest_randomized and best_observed < 0.95 * largest_randomized:
            return {
                "classification": "contradictory",
                "reason": "Outcome-data denominators conflict with the randomized denominator.",
                "randomized_denominators": randomized,
                "outcome_denominators": outcome_counts,
            }
        if direct and best_observed / max(best_total, 1) >= 0.95:
            return {
                "classification": "sufficient",
                "reason": "Direct outcome-data denominator support shows nearly complete data.",
                "randomized_denominators": randomized,
                "outcome_denominators": outcome_counts,
            }

    if safety_only or _SAFETY_POPULATION_RE.search(evidence_text):
        return {
            "classification": "insufficient",
            "reason": "Safety population exclusions do not establish complete randomized-participant outcome data.",
            "randomized_denominators": randomized,
            "outcome_denominators": outcome_counts,
        }
    if analysis_only and not direct:
        return {
            "classification": "insufficient",
            "reason": "Analysis-population, model, or censoring language is not direct completeness evidence.",
            "randomized_denominators": randomized,
            "outcome_denominators": outcome_counts,
        }
    if direct and outcome_counts:
        count, total = max(outcome_counts, key=lambda pair: pair[1])
        if count / max(total, 1) >= 0.95:
            return {
                "classification": "sufficient",
                "reason": "Direct outcome-data denominator support shows nearly complete data.",
                "randomized_denominators": randomized,
                "outcome_denominators": outcome_counts,
            }
    return {
        "classification": "insufficient",
        "reason": "Direct denominator or percentage support for assessed-outcome completeness is missing.",
        "randomized_denominators": randomized,
        "outcome_denominators": outcome_counts,
    }


def _apply_d3_completeness_gate(
    state: RoB2State, sq_answers: dict[str, dict]
) -> dict[str, dict]:
    sq31 = sq_answers.get("3.1", {})
    if sq31.get("answer") not in ("Y", "PY"):
        return sq_answers
    quote_text = " ".join(
        str(sq31.get(field, ""))
        for field in ("quote", "completeness_calculation", "justification")
    )
    support = classify_d3_completeness_support(state, quote_text)
    if support["classification"] == "sufficient":
        return sq_answers

    updated = dict(sq_answers)
    updated["3.1"] = {
        **sq31,
        "answer": "NI",
        "justification": (
            f"{sq31.get('justification', '').strip()} "
            f"D3 completeness evidence gate classified 3.1 support as {support['classification']}: {support['reason']}"
        ).strip(),
        "uncertainty_flag": sq31.get("uncertainty_flag", "HIGH"),
        "completeness_support": support,
    }
    for sq_id in ("3.2", "3.3", "3.4"):
        updated.setdefault(
            sq_id,
            {
                "answer": "NI",
                "quote": "No relevant text found",
                "justification": "D3 remains applicable after completeness evidence gating.",
                "uncertainty_flag": "HIGH",
            },
        )
    return updated


def _randomized_denominators(text: str) -> list[int]:
    values: list[int] = []
    for match in _RANDOMIZED_DENOMINATOR_RE.finditer(text):
        for group in match.groups():
            if group:
                values.append(int(group))
                break
    return values


def _outcome_counts(text: str) -> list[tuple[int, int]]:
    counts: list[tuple[int, int]] = []
    for match in _OUTCOME_DENOMINATOR_RE.finditer(text):
        groups = [int(group) for group in match.groups() if group]
        if len(groups) >= 2:
            counts.append((groups[0], groups[1]))
        elif len(groups) == 1:
            counts.append((groups[0], groups[0]))
    return counts


def _calibrate_time_to_event_d3_1(
    state: RoB2State, sq_answers: dict[str, dict]
) -> dict[str, dict]:
    if not state.get("outcome_properties", {}).get("time_to_event"):
        return sq_answers
    sq31 = sq_answers.get("3.1", {})
    if sq31.get("answer") not in ("Y", "PY"):
        return sq_answers

    evidence_text = _domain3_missingness_text(state)
    quote_text = " ".join(
        str(sq31.get(field, ""))
        for field in ("quote", "completeness_calculation", "justification")
    )
    text = " ".join([evidence_text, quote_text])
    if _TTE_DIRECT_MISSINGNESS_RE.search(text):
        return sq_answers
    if not _TTE_ANALYSIS_ONLY_RE.search(text):
        return sq_answers

    updated = dict(sq_answers)
    updated["3.1"] = {
        **sq31,
        "answer": "NI",
        "justification": (
            f"{sq31.get('justification', '').strip()} "
            "For time-to-event outcomes, ITT, survival-model, Kaplan-Meier, or censoring language alone does not directly establish negligible missing outcome data."
        ).strip(),
        "uncertainty_flag": sq31.get("uncertainty_flag", "HIGH"),
    }
    for sq_id in ("3.2", "3.3", "3.4"):
        updated.setdefault(
            sq_id,
            {
                "answer": "NI",
                "quote": "No relevant text found",
                "justification": "D3 remains applicable after time-to-event missingness calibration.",
                "uncertainty_flag": "HIGH",
            },
        )
    return updated


def _domain3_missingness_text(state: RoB2State) -> str:
    evidence = state.get("evidence", {})
    parts = [
        _section_text(evidence.get("d3_missing_data", {})),
        _section_text(evidence.get("consort_flow", {})),
        _section_text(evidence.get("results", {})),
        state.get("rag_contexts", {}).get("d3", ""),
        state.get("ctgov_flow", ""),
    ]
    return "\n".join(str(part) for part in parts if part)


def _section_text(section: object) -> str:
    if isinstance(section, dict):
        return "\n".join(
            str(section.get(key, "")) for key in ("text", "tables") if section.get(key)
        )
    return str(section or "")


def apply_domain4_control(
    state: RoB2State, sq_answers: dict[str, dict]
) -> dict[str, dict]:
    updated = dict(sq_answers)
    outcome_type = _effective_domain4_outcome_type(state)
    assessor_is_aware = _domain4_assessor_is_aware(state, outcome_type)

    has_blinded_adjudication = _has_blinded_adjudication(state)
    influence_potential = _domain4_influence_potential(
        state, outcome_type, assessor_is_aware, has_blinded_adjudication
    )
    pfs_open_label_concern = (
        assessor_is_aware
        and _is_pfs_outcome(state)
        and not has_blinded_adjudication
        and _progression_uses_clinician_or_investigator_assessment(state)
    )

    if assessor_is_aware and outcome_type == "patient-reported":
        existing_quote = updated.get("4.3", {}).get("quote") or ""
        quote = (
            existing_quote
            if existing_quote and not existing_quote.startswith("Auto-set:")
            else _domain4_awareness_quote(state)
        )
        updated["4.3"] = {
            "answer": "Y",
            "quote": quote or "No relevant text found",
            "justification": "Participant is the assessor; cannot be blinded to own treatment.",
            "uncertainty_flag": "NORMAL",
        }
    elif (
        assessor_is_aware
        and not has_blinded_adjudication
        and outcome_type
        in (
            "clinician-graded",
            "clinician-composite",
        )
    ):
        existing_quote = updated.get("4.3", {}).get("quote") or ""
        quote = (
            existing_quote
            if existing_quote and not existing_quote.startswith("Auto-set:")
            else _domain4_awareness_quote(state)
        )
        updated["4.3"] = {
            "answer": "PY",
            "quote": quote or "No relevant text found",
            "justification": "In an open-label trial, the clinician grading or adjudicating the outcome is likely aware of treatment assignment.",
            "uncertainty_flag": "NORMAL",
        }
    if influence_potential == "likely":
        updated["4.4"] = {
            "answer": "Y",
            "quote": _domain4_quote(state, updated),
            "justification": "Direct evidence indicates that knowledge of intervention assignment could influence outcome assessment.",
            "uncertainty_flag": "NORMAL",
        }
        updated["4.5"] = {
            "answer": "PY",
            "quote": _domain4_quote(state, updated),
            "justification": "Direct evidence beyond open-label status indicates that assessment was likely influenced.",
            "uncertainty_flag": "NORMAL",
        }
    elif influence_potential == "plausible" or pfs_open_label_concern:
        updated["4.4"] = {
            "answer": "PY",
            "quote": _domain4_quote(state, updated),
            "justification": "Outcome assessment involves judgment without blinded adjudication, so intervention knowledge could plausibly influence assessment.",
            "uncertainty_flag": "NORMAL",
        }
        updated["4.5"] = {
            "answer": "PN",
            "quote": _domain4_quote(state, updated),
            "justification": "Open-label status shows plausible influence, but direct evidence does not establish that assessment was likely influenced.",
            "uncertainty_flag": "NORMAL",
        }
    elif outcome_type in ("vital-status", "biomarker"):
        s41 = updated.get("4.1", {}).get("answer", "NI")
        s42 = updated.get("4.2", {}).get("answer", "NI")
        s43 = updated.get("4.3", {}).get("answer", "NI")
        s44 = updated.get("4.4", {}).get("answer", "NI")
        if s41 in ("N", "PN", "NI") and s42 in ("N", "PN") and s43 == "NA":
            updated["4.3"] = {
                "answer": "NI",
                "quote": updated.get("4.3", {}).get("quote")
                or "No relevant text found",
                "justification": "Assessor awareness is not reported; NA is not applicable when 4.1 and 4.2 do not indicate measurement problems.",
                "uncertainty_flag": "NORMAL",
            }
            s43 = "NI"
        if (
            s41 in ("N", "PN", "NI")
            and s42 in ("N", "PN")
            and s43 in ("Y", "PY", "NI")
            and s44 in ("NI", "NA")
        ):
            updated["4.4"] = {
                "answer": "N",
                "quote": updated.get("4.1", {}).get("quote")
                or updated.get("4.2", {}).get("quote")
                or "No relevant text found",
                "justification": "The outcome is inherently objective, so knowledge of intervention assignment is unlikely to influence assessment.",
                "uncertainty_flag": "NORMAL",
            }
            s44 = "N"
        if (
            s41 in ("N", "PN", "NI")
            and s42 in ("N", "PN")
            and s43 in ("Y", "PY", "NI")
            and s44 in ("N", "PN")
        ):
            updated["4.5"] = {
                "answer": "NA",
                "quote": "Not applicable",
                "justification": "Not applicable",
                "uncertainty_flag": "NORMAL",
            }

    s41 = updated.get("4.1", {}).get("answer", "NI")
    s42 = updated.get("4.2", {}).get("answer", "NI")
    s43 = updated.get("4.3", {}).get("answer", "NA")
    s44 = updated.get("4.4", {}).get("answer", "NA")
    if s41 in ("Y", "PY") or s42 in ("Y", "PY"):
        return set_na(updated, "4.3", "4.4", "4.5")
    if s43 in ("N", "PN"):
        return set_na(updated, "4.4", "4.5")
    if s44 in ("N", "PN"):
        return set_na(updated, "4.5")
    return updated


def _domain4_influence_potential(
    state: RoB2State,
    outcome_type: str,
    assessor_is_aware: bool,
    has_blinded_adjudication: bool,
) -> str:
    if outcome_type in ("vital-status", "biomarker") or has_blinded_adjudication:
        return "low"
    if not assessor_is_aware:
        return "unknown"
    if _domain4_text_indicates_likely_influence(state):
        return "likely"
    if outcome_type in ("patient-reported", "clinician-graded", "clinician-composite"):
        return "plausible"
    return "unknown"


def _is_pfs_outcome(state: RoB2State) -> bool:
    if str(state.get("outcome_code", "")).casefold() == "pfs":
        return True
    outcome = str(state.get("outcome", "")).casefold()
    return (
        "progression-free survival" in outcome or "progression free survival" in outcome
    )


def _effective_domain4_outcome_type(state: RoB2State) -> str:
    if _is_objective_os_outcome(state):
        return "vital-status"
    return str(state.get("outcome_type", "clinician-composite"))


def _is_objective_os_outcome(state: RoB2State) -> bool:
    if str(state.get("outcome_code", "")).casefold() == "os":
        return True
    outcome = str(state.get("outcome", "")).casefold()
    if "overall survival" in outcome or "death from any cause" in outcome:
        return True
    props = state.get("outcome_properties") or {}
    return bool(
        props.get("objective_event")
        and not props.get("patient_reported")
        and not props.get("safety_harm")
    )


def _has_blinded_adjudication(state: RoB2State) -> bool:
    masking_facts = state.get("masking_facts") or {}
    blinded_adjudication = masking_facts.get("blinded_adjudication") or {}
    if blinded_adjudication.get("status") == "present":
        return True
    if blinded_adjudication.get("status") == "absent":
        return False
    props = state.get("outcome_properties") or {}
    if props.get("blinded_adjudication"):
        return True
    text = _domain4_text(state)
    return bool(
        re.search(
            r"\b(blinded|masked|independent|central).{0,80}\b(adjudication|committee|review|assessor)\b",
            text,
            re.I,
        )
    )


def _domain4_assessor_is_aware(state: RoB2State, outcome_type: str) -> bool:
    masking_facts = state.get("masking_facts") or {}
    fact_name = (
        "participant_awareness"
        if outcome_type == "patient-reported"
        else "outcome_assessor_awareness"
    )
    status = (masking_facts.get(fact_name) or {}).get("status")
    if status == "aware":
        return True
    if status == "unaware":
        return False
    return _domain4_text_indicates_assessor_awareness(state)


def _domain4_text_indicates_assessor_awareness(state: RoB2State) -> bool:
    text = _domain4_text(state)
    return bool(
        re.search(
            r"\b(open-label|open label|unblinded|not blinded|not masked)\b",
            text,
            re.I,
        )
    )


def _domain4_awareness_quote(state: RoB2State) -> str:
    masking_facts = state.get("masking_facts") or {}
    for fact_name in ("outcome_assessor_awareness", "participant_awareness"):
        fact = masking_facts.get(fact_name) or {}
        for quote in fact.get("quotes") or []:
            text = quote.get("quote")
            if text:
                return str(text)
    return _domain4_quote(state, {})


def _progression_uses_clinician_or_investigator_assessment(state: RoB2State) -> bool:
    text = _domain4_text(state)
    if not re.search(r"\bprogression\b", text, re.I):
        return False
    return bool(
        re.search(
            r"\b(investigator|clinician|physician|radiographic|imaging|recist|assessment|assessed)\b",
            text,
            re.I,
        )
    )


def _domain4_text_indicates_likely_influence(state: RoB2State) -> bool:
    text = _domain4_text(state)
    return bool(
        re.search(
            r"\b(treating|delivering|delivered|provided|provider|physiotherapist|therapist).{0,120}\b(assess|assessed|assessment|rated|graded|evaluated)\b",
            text,
            re.I,
        )
        or re.search(
            r"\b(assess|assessed|assessment|rated|graded|evaluated).{0,120}\b(treating|delivering|delivered|provided|provider|physiotherapist|therapist)\b",
            text,
            re.I,
        )
        or re.search(
            r"\b(strong belief|expectation|preferred treatment|desired treatment|vested interest)\b",
            text,
            re.I,
        )
    )


def _domain4_text(state: RoB2State) -> str:
    parts = [
        str(state.get("outcome", "")),
        str((state.get("rag_contexts") or {}).get("d4_measurement", "")),
        str((state.get("rag_contexts") or {}).get("d4_assessor", "")),
    ]
    evidence = state.get("evidence") or {}
    d4_evidence = evidence.get("d4_outcome_meas") or {}
    if isinstance(d4_evidence, dict):
        parts.append(str(d4_evidence.get("text", "")))
    return "\n".join(part for part in parts if part)


def _domain4_quote(state: RoB2State, sq_answers: dict[str, dict]) -> str:
    return (
        sq_answers.get("4.1", {}).get("quote")
        or sq_answers.get("4.2", {}).get("quote")
        or (state.get("rag_contexts") or {}).get("d4_measurement")
        or "No relevant text found"
    )
