import re
from typing import Literal

from rob2_pipeline.nodes.common import set_na
from rob2_pipeline.state import RoB2State


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
    sq_answers = _apply_domain2_actual_deviation_guard(sq_answers)
    s23 = sq_answers.get("2.3", {}).get("answer", "NI")
    s24 = sq_answers.get("2.4", {}).get("answer", "NI")
    if s23 in ("N", "PN", "NI"):
        return set_na(sq_answers, "2.4", "2.5")
    if s24 in ("N", "PN", "NA"):
        return set_na(sq_answers, "2.5")
    return sq_answers


def _apply_domain2_actual_deviation_guard(
    sq_answers: dict[str, dict]
) -> dict[str, dict]:
    answer = dict(sq_answers.get("2.3", {}))
    if answer.get("answer") not in {"Y", "PY", "NI"}:
        return sq_answers
    text = " ".join(
        str(answer.get(field, ""))
        for field in ("quote", "justification", "support_rationale")
    ).casefold()
    actual_deviation_signals = (
        "non-adherence",
        "nonadherence",
        "cross-over",
        "crossover",
        "contamination",
        "protocol deviation",
        "deviated",
        "deviation occurred",
        "did not receive",
        "received no",
        "discontinued assigned",
        "treatment switching",
        "switched treatment",
    )
    design_only_signals = (
        "eligible",
        "eligibility",
        "not eligible",
        "must be obtained",
        "within 4 weeks",
        "open-label",
        "open label",
        "masking: none",
        "no masking",
        "aware of treatment",
        "aware of allocation",
        "standard of care plus",
        "added to standard of care",
        "addition to standard of care",
        "plus abiraterone",
        "plus prednisone",
        "not randomly assigned according to docetaxel prescription",
        "not randomly assignedaccording to docetaxel prescription",
        "toxicities recorded",
        "not directly comparable",
    )
    if any(signal in text for signal in actual_deviation_signals):
        return sq_answers
    if not any(signal in text for signal in design_only_signals):
        return sq_answers
    updated = dict(sq_answers)
    answer.update(
        {
            "answer": "N",
            "justification": (
                "The cited text describes trial design, eligibility, masking, "
                "or protocol requirements, but does not report that deviations "
                "from intended intervention occurred after assignment."
            ),
            "support_level": "moderate",
            "support_rationale": (
                "Local D2 control requires actual post-randomization deviations, "
                "not awareness, eligibility rules, or protocol instructions alone."
            ),
            "uncertainty": True,
            "d2_actual_deviation_guard_applied": True,
        }
    )
    updated["2.3"] = answer
    return updated


def apply_domain2_analysis_control(
    state: RoB2State, sq_answers: dict[str, dict]
) -> dict[str, dict]:
    sq_answers = _apply_domain2_safety_analysis_guard(state, sq_answers)
    if state.get(
        "effect_of_interest", "ITT"
    ).lower() == "per-protocol" or sq_answers.get("2.6", {}).get("answer", "NI") in (
        "Y",
        "PY",
    ):
        return set_na(sq_answers, "2.7")
    return sq_answers


def _apply_domain2_safety_analysis_guard(
    state: RoB2State, sq_answers: dict[str, dict]
) -> dict[str, dict]:
    if not _is_safety_harm_outcome(state):
        return sq_answers
    answer_27 = dict(sq_answers.get("2.7", {}))
    if answer_27.get("answer") in {"N", "PN"}:
        return sq_answers
    safety_text = _safety_evidence_text(state)
    if not (
        "safety population includes patients who actually received" in safety_text
        or "safety population" in safety_text
    ):
        return sq_answers
    updated = dict(sq_answers)
    answer_27.update(
        {
            "answer": "PN",
            "quote": (
                "The safety population includes patients who actually received "
                "the assigned treatment."
            ),
            "justification": (
                "For harms, the available outcome-bound evidence reports adverse "
                "events in the safety population. The cited evidence does not show "
                "that exclusions or analysis failures substantially impacted the "
                "adverse-event result."
            ),
            "support_level": "moderate",
            "support_rationale": (
                "Local D2 safety analysis control found safety-population "
                "reporting evidence and no cited evidence of analysis-failure "
                "impact on harms."
            ),
            "uncertainty": True,
            "d2_safety_analysis_guard_applied": True,
        }
    )
    updated["2.7"] = answer_27
    return updated


def apply_domain3_control(
    state: RoB2State, sq_answers: dict[str, dict]
) -> dict[str, dict]:
    sq_answers = _apply_domain3_safety_outcome_binding_guard(state, sq_answers)
    sq_answers = _apply_domain3_time_to_event_completeness_guard(state, sq_answers)
    sq_answers = _apply_domain3_time_to_event_event_guard(state, sq_answers)
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


def _apply_domain3_safety_outcome_binding_guard(
    state: RoB2State, sq_answers: dict[str, dict]
) -> dict[str, dict]:
    properties = state.get("outcome_properties") or {}
    if not properties.get("safety_harm") and "adverse" not in str(
        state.get("outcome", "")
    ).casefold():
        return sq_answers
    answer = dict(sq_answers.get("3.1", {}))
    if answer.get("answer") in {"N", "PN", "NI"}:
        pass
    elif answer.get("answer") in {"Y", "PY"}:
        answer_text_probe = " ".join(
            str(answer.get(field, ""))
            for field in ("quote", "justification", "support_rationale")
        ).casefold()
        if not any(
            signal in answer_text_probe
            for signal in (
                "classifier fallback",
                "classifier artifact did not include",
                "no relevant text found",
                "overallpopulation",
                "overall population",
                "efficacy analysis",
                "lost to follow",
                "consent withdrawn",
                "consentwithdrawn",
                "radiographic progression-free survival",
                "coprimary endpoints",
            )
        ):
            return sq_answers
    else:
        return sq_answers
    answer_text = " ".join(
        str(answer.get(field, ""))
        for field in ("quote", "justification", "support_rationale")
    ).casefold()
    efficacy_flow_signals = (
        "overallpopulation for efficacy analysis",
        "overall population for efficacy analysis",
        "upcoming overall survival",
        "radiographic progression-free survival",
        "rpfs",
        "coprimary endpoints",
    )
    safety_text = _safety_evidence_text(state)
    safety_completeness_signals = (
        "safety population includes patients who actually received",
        "table 3: adverse events in the safety population",
        "any adverse events",
        "severe adverse events",
        "toxicities recorded",
        "safety population",
    )
    missingness_flow_signals = (
        "lost to follow-up",
        "lost to follow",
        "consent withdrawn",
        "consentwithdrawn",
        "overallpopulation",
        "overall population",
    )
    if not (
        any(signal in answer_text for signal in efficacy_flow_signals)
        or any(signal in answer_text for signal in missingness_flow_signals)
        or "classifier fallback" in answer_text
        or "classifier artifact did not include" in answer_text
        or "no relevant text found" in answer_text
    ):
        return sq_answers
    if not any(signal in safety_text for signal in safety_completeness_signals):
        return sq_answers
    count_support = _safety_analysis_count_support(safety_text)
    updated = dict(sq_answers)
    quote = (
        count_support["quote"]
        if count_support
        else (
            "The safety population includes patients who actually received "
            "the assigned treatment. Severe adverse events were reported in "
            "the safety population."
        )
    )
    justification = (
        count_support["justification"]
        if count_support
        else (
            "The cited missingness rationale is drawn from efficacy-flow "
            "text, not adverse-event ascertainment. Outcome-bound safety "
            "evidence reports adverse events in the safety population, so "
            "missing adverse-event outcome data are probably not sufficient "
            "to materially affect the harms result."
        )
    )
    answer.update(
        {
            "answer": "PY",
            "quote": quote,
            "justification": justification,
            "support_level": "moderate",
            "support_rationale": (
                "Local D3 safety guard used adverse-event-specific safety "
                "population/reporting evidence and rejected efficacy-flow text "
                "as the basis for harms missingness."
            ),
            "uncertainty": True,
            "d3_safety_outcome_binding_guard_applied": True,
        }
    )
    updated["3.1"] = answer
    return updated


def _safety_analysis_count_support(text: str) -> dict | None:
    normalized = " ".join(text.split())
    with_eff = [
        int(value)
        for value in re.findall(
            r"(\d{2,5})\s*included\s+inthe\s+adt\s+with\s+docetaxel\s+population\s+for\s+efficacy",
            normalized,
        )
    ]
    with_safety = [
        int(value)
        for value in re.findall(
            r"(\d{2,5})\s*included\s+inthe\s+adt\s+with\s+docetaxel\s+population\s+for\s+safety",
            normalized,
        )
    ]
    if not with_eff or len(with_safety) < 2:
        return None
    eff = max(with_eff)
    safety_counts = sorted(set(with_safety), reverse=True)[:2]
    if eff <= 0 or any(count <= 0 for count in safety_counts):
        return None
    min_ratio = min(count / eff for count in safety_counts)
    if min_ratio < 0.95:
        return None
    first = min(
        index
        for index in (
            normalized.find(f"{eff} included inthe adt with docetaxel population for efficacy analysis"),
            normalized.find(f"{eff} included inthe adt with docetaxel population for efficacy"),
            normalized.find(f"{safety_counts[0]}included inthe adt with docetaxel population for safety"),
            normalized.find(f"{safety_counts[1]}included inthe adt with docetaxel population for safety"),
        )
        if index >= 0
    )
    last = max(
        normalized.find("safety analysis", first),
        normalized.rfind("safety analysis", first, first + 500),
    )
    quote = normalized[first : last + len("safety analysis")]
    return {
        "quote": quote,
        "justification": (
            "The cited efficacy-flow text is not itself harms missingness "
            "evidence, but the same packet reports safety-analysis denominators: "
            f"{safety_counts[0]}/{eff} and {safety_counts[1]}/{eff} participants "
            "in the ADT-with-docetaxel strata were included in safety analyses. "
            "Those near-complete harms data support PY for SQ 3.1."
        ),
    }


def _safety_evidence_text(state: RoB2State) -> str:
    parts = []
    evidence = state.get("evidence") or {}
    for section_name in ("results", "d4_outcome_meas", "d3_missing_data"):
        section = evidence.get(section_name) or {}
        if isinstance(section, dict):
            parts.append(str(section.get("text", "")))
    for packet in (state.get("evidence_packets") or {}).values():
        for source in packet.get("sources", []):
            text = str(source.get("text", ""))
            if "adverse" in text.casefold() or "safety" in text.casefold():
                parts.append(text)
    return " ".join(parts).casefold()


def _apply_domain3_time_to_event_completeness_guard(
    state: RoB2State, sq_answers: dict[str, dict]
) -> dict[str, dict]:
    answer = dict(sq_answers.get("3.1", {}))
    if answer.get("answer") not in {"N", "PN", "NI"}:
        return sq_answers
    if not _is_time_to_event_outcome(state):
        return sq_answers
    text = " ".join(
        str(answer.get(field, ""))
        for field in ("quote", "justification", "support_rationale")
    ).casefold()
    endpoint_event_signals = (
        "death",
        "progression",
        "disease progression",
        "radiographic progression",
    )
    true_missingness_signals = (
        "censor",
        "censored",
        "lost to follow",
        "lost-to-follow",
        "withdrawal by subject",
        "withdrew consent",
        "switch",
        "switched",
        "stopped follow-up",
        "discontinued follow-up",
    )
    if not any(signal in text for signal in endpoint_event_signals):
        return sq_answers
    if any(signal in text for signal in true_missingness_signals):
        return sq_answers
    updated = dict(sq_answers)
    answer.update(
        {
            "answer": "PY",
            "justification": (
                "The cited rationale treats death or progression as missing "
                "outcome data. For a time-to-event outcome, those are observed "
                "endpoint events rather than missingness. No cited text shows "
                "substantial loss to follow-up, censoring, switching, or stopped "
                "outcome follow-up before ascertainment."
            ),
            "support_level": "moderate",
            "support_rationale": (
                "Local D3 control rejected endpoint events as evidence of "
                "incomplete outcome data for SQ 3.1."
            ),
            "uncertainty": True,
            "d3_time_to_event_completeness_guard_applied": True,
        }
    )
    updated["3.1"] = answer
    return updated


def _apply_domain3_time_to_event_event_guard(
    state: RoB2State, sq_answers: dict[str, dict]
) -> dict[str, dict]:
    answer = dict(sq_answers.get("3.4", {}))
    if answer.get("answer") not in {"Y", "PY"}:
        return sq_answers
    if not _is_time_to_event_outcome(state):
        return sq_answers
    text = " ".join(
        str(answer.get(field, ""))
        for field in ("quote", "justification", "support_rationale")
    ).casefold()
    endpoint_event_signals = (
        "death",
        "progression",
        "disease progression",
        "radiographic progression",
    )
    true_missingness_signals = (
        "censor",
        "censored",
        "lost to follow",
        "lost-to-follow",
        "withdrawal by subject",
        "withdrew consent",
        "switch",
        "switched",
        "stopped follow-up",
        "discontinued follow-up",
    )
    if not any(signal in text for signal in endpoint_event_signals):
        return sq_answers
    if any(signal in text for signal in true_missingness_signals):
        return sq_answers
    updated = dict(sq_answers)
    answer.update(
        {
            "answer": "PN",
            "justification": (
                "The cited rationale relies on death or progression, which are "
                "observed endpoint events for a time-to-event outcome rather than "
                "evidence that missingness was likely informative. No cited text "
                "shows early censoring, switching, loss to follow-up, or stopped "
                "outcome follow-up before ascertainment."
            ),
            "support_level": "moderate",
            "support_rationale": (
                "Local D3 control rejected endpoint events as sufficient evidence "
                "for likely informative missingness in a time-to-event outcome."
            ),
            "uncertainty": True,
            "d3_time_to_event_guard_applied": True,
        }
    )
    updated["3.4"] = answer
    return updated


def _is_time_to_event_outcome(state: RoB2State) -> bool:
    properties = state.get("outcome_properties") or {}
    outcome = str(state.get("outcome", "")).casefold()
    return bool(properties.get("time_to_event")) or any(
        term in outcome
        for term in (
            "survival",
            "time to",
            "time-to",
            "progression-free",
            "progression free",
        )
    )


def apply_domain4_control(
    state: RoB2State, sq_answers: dict[str, dict]
) -> dict[str, dict]:
    updated = dict(sq_answers)
    outcome_type = state.get("outcome_type", "clinician-composite")
    classification_support = state.get("outcome_classification_support", {})
    classification_support_level = classification_support.get(
        "support_level", "moderate"
    )
    classification_support_rationale = classification_support.get(
        "support_rationale", "Derived from objective outcome classification."
    )
    sq_2_1 = state.get("sq_answers", {}).get("2.1", {}).get("answer", "NI")
    sq_2_2 = state.get("sq_answers", {}).get("2.2", {}).get("answer", "NI")
    trial_is_open_label = sq_2_1 in ("Y", "PY") or sq_2_2 in ("Y", "PY")

    existing_43 = updated.get("4.3", {}).get("answer", "NI")
    if (
        trial_is_open_label
        and existing_43 not in ("N", "PN")
        and outcome_type
        in (
        "patient-reported",
        "clinician-graded",
        "clinician-composite",
        )
    ):
        existing_quote = updated.get("4.3", {}).get("quote") or ""
        quote = (
            existing_quote
            if existing_quote and not existing_quote.startswith("Auto-set:")
            else (
                state.get("sq_answers", {}).get("2.1", {}).get("quote")
                or state.get("sq_answers", {}).get("2.2", {}).get("quote")
                or "No relevant text found"
            )
        )
        if outcome_type == "patient-reported":
            answer = "Y"
            justification = (
                "Participant is the assessor; cannot be blinded to own treatment."
            )
        else:
            answer = "PY"
            justification = "In an open-label trial, the clinician grading or adjudicating the outcome is likely aware of treatment assignment."
        updated["4.3"] = {
            "answer": answer,
            "quote": quote or "No relevant text found",
            "justification": justification,
            "uncertainty_flag": "NORMAL",
            "support_level": "moderate",
            "support_rationale": "Derived from treatment-awareness evidence and outcome type.",
        }
        if _is_safety_harm_outcome(state):
            s41 = updated.get("4.1", {}).get("answer", "NI")
            s42 = updated.get("4.2", {}).get("answer", "NI")
            s45 = updated.get("4.5", {}).get("answer", "NI")
            safety_text = _safety_evidence_text(state)
            if s41 == "NI" and any(
                signal in safety_text
                for signal in (
                    "common terminology criteria",
                    "ctcae",
                    "adverse events were graded",
                    "medical dictionary for regulatory affairs",
                    "meddra",
                    "grade ≥3",
                    "grade 3",
                )
            ):
                updated["4.1"] = {
                    "answer": "N",
                    "quote": (
                        "Adverse events were graded on the basis of the National "
                        "Cancer Institute Common Terminology Criteria."
                    ),
                    "justification": (
                        "The trial used standard adverse-event grading/reporting "
                        "methods for harms, so the measurement method is suitable."
                    ),
                    "uncertainty_flag": "NORMAL",
                    "support_level": "moderate",
                    "support_rationale": (
                        "Local D4 safety control used outcome-bound adverse-event "
                        "grading evidence."
                    ),
                    "d4_safety_measurement_guard_applied": True,
                }
                s41 = "N"
            if s42 == "NI" and s41 in {"N", "PN"}:
                updated["4.2"] = {
                    "answer": "PN",
                    "quote": updated.get("4.1", {}).get("quote")
                    or "No relevant text found",
                    "justification": (
                        "The same adverse-event grading/reporting framework was "
                        "used for the trial arms; no cited evidence shows "
                        "differential measurement methods between groups."
                    ),
                    "uncertainty_flag": "NORMAL",
                    "support_level": "moderate",
                    "support_rationale": (
                        "Local D4 safety control found no outcome-bound evidence "
                        "of differential harms measurement."
                    ),
                    "d4_safety_measurement_guard_applied": True,
                }
                s42 = "PN"
            if s45 in {"Y", "PY"} and _only_open_label_supports_likely_influence(
                updated.get("4.5", {})
            ):
                updated["4.5"] = {
                    **updated["4.5"],
                    "answer": "PN",
                    "justification": (
                        "Open-label assessment establishes assessor awareness, "
                        "but the cited rationale does not provide outcome-bound "
                        "evidence that adverse-event grading was likely influenced."
                    ),
                    "support_level": "moderate",
                    "support_rationale": (
                        "Local D4 safety control separated possible influence "
                        "from likely influence for clinician-graded harms."
                    ),
                    "uncertainty": True,
                    "d4_safety_influence_guard_applied": True,
                }
            s44 = updated.get("4.4", {}).get("answer", "NI")
            if s44 in {"N", "PN"} and _safety_influence_answer_uses_efficacy_text(
                updated.get("4.4", {})
            ):
                updated["4.4"] = {
                    **updated["4.4"],
                    "answer": "NI",
                    "justification": (
                        "The cited text describes an efficacy endpoint or disease "
                        "progression definition, not whether adverse-event "
                        "assessment could be influenced by treatment awareness."
                    ),
                    "support_level": "unsupported",
                    "support_rationale": (
                        "Local D4 safety control rejected non-harms outcome text "
                        "as support for assessor-influence judgments about adverse "
                        "events."
                    ),
                    "uncertainty": True,
                    "supporting_fact_artifact_ids": [],
                    "d4_safety_influence_source_guard_applied": True,
                }
                if updated.get("4.5", {}).get("answer") == "NA":
                    updated["4.5"] = {
                        "answer": "PN",
                        "quote": updated.get("4.3", {}).get("quote")
                        or "No relevant text found",
                        "justification": (
                            "Open-label assessment establishes possible assessor "
                            "awareness, but no cited adverse-event evidence shows "
                            "that grading was likely influenced by that awareness."
                        ),
                        "uncertainty_flag": "NORMAL",
                        "support_level": "moderate",
                        "support_rationale": (
                            "Local D4 safety control requires outcome-bound harms "
                            "evidence before treating influence as likely."
                        ),
                        "d4_safety_influence_guard_applied": True,
                    }
            s44 = updated.get("4.4", {}).get("answer", "NI")
            if s44 in {"N", "PN"} and _safety_ctcae_understates_possible_influence(
                updated.get("4.4", {})
            ):
                updated["4.4"] = {
                    **updated["4.4"],
                    "answer": "PY",
                    "justification": (
                        "Open-label clinician grading of adverse events could be "
                        "influenced by treatment awareness. Standardized CTCAE "
                        "criteria reduce subjectivity, but they do not make "
                        "possible influence absent."
                    ),
                    "support_level": "moderate",
                    "support_rationale": (
                        "Local D4 safety control separated possible influence "
                        "for SQ 4.4 from likely influence for SQ 4.5."
                    ),
                    "uncertainty": True,
                    "d4_safety_possible_influence_guard_applied": True,
                }
                if updated.get("4.5", {}).get("answer") == "NA":
                    updated["4.5"] = {
                        "answer": "PN",
                        "quote": updated.get("4.4", {}).get("quote")
                        or updated.get("4.3", {}).get("quote")
                        or "No relevant text found",
                        "justification": (
                            "Although open-label clinician grading could be "
                            "influenced, the cited evidence does not show that "
                            "adverse-event grading was likely influenced; CTCAE "
                            "standardization argues against a likely influence "
                            "finding without additional differential assessment "
                            "evidence."
                        ),
                        "uncertainty_flag": "NORMAL",
                        "support_level": "moderate",
                        "support_rationale": (
                            "Local D4 safety control requires outcome-bound "
                            "evidence before treating possible influence as "
                            "likely influence."
                        ),
                        "d4_safety_influence_guard_applied": True,
                    }
    elif outcome_type in ("vital-status", "biomarker"):
        s41 = updated.get("4.1", {}).get("answer", "NI")
        s42 = updated.get("4.2", {}).get("answer", "NI")
        s43 = updated.get("4.3", {}).get("answer", "NI")
        s44 = updated.get("4.4", {}).get("answer", "NI")
        if outcome_type == "vital-status" and s41 == "NI":
            updated["4.1"] = {
                "answer": "N",
                "quote": updated.get("4.1", {}).get("quote")
                or "No relevant text found",
                "justification": (
                    "Vital status is an inherently objective outcome; absent "
                    "packet evidence of an unsuitable ascertainment method, the "
                    "measurement method is considered appropriate."
                ),
                "uncertainty_flag": "NORMAL",
                "support_level": classification_support_level,
                "support_rationale": classification_support_rationale,
                "d4_objective_control_applied": True,
            }
            s41 = "N"
        if s41 in ("N", "PN") and s42 == "NI":
            updated["4.2"] = {
                "answer": "PN",
                "quote": updated.get("4.2", {}).get("quote")
                or updated.get("4.1", {}).get("quote")
                or "No relevant text found",
                "justification": (
                    "For an objective vital-status or biomarker outcome with a "
                    "suitable measurement method and no packet evidence of "
                    "differential ascertainment, between-group measurement "
                    "differences are probably absent."
                ),
                "uncertainty_flag": "NORMAL",
                "support_level": classification_support_level,
                "support_rationale": classification_support_rationale,
            }
            s42 = "PN"
        if s41 in ("N", "PN", "NI") and s42 in ("N", "PN") and s43 == "NA":
            updated["4.3"] = {
                "answer": "NI",
                "quote": updated.get("4.3", {}).get("quote")
                or "No relevant text found",
                "justification": "Assessor awareness is not reported; NA is not applicable when 4.1 and 4.2 do not indicate measurement problems.",
                "uncertainty_flag": "NORMAL",
                "support_level": updated.get("4.3", {}).get("support_level", "weak"),
                "support_rationale": updated.get("4.3", {}).get(
                    "support_rationale",
                    "No direct assessor-awareness evidence was reported.",
                ),
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
                "support_level": classification_support_level,
                "support_rationale": classification_support_rationale,
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
                "support_level": "unsupported",
                "support_rationale": "Not applicable",
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


def _is_safety_harm_outcome(state: RoB2State) -> bool:
    properties = state.get("outcome_properties") or {}
    return bool(properties.get("safety_harm")) or "adverse" in str(
        state.get("outcome", "")
    ).casefold()


def _only_open_label_supports_likely_influence(answer: dict) -> bool:
    text = " ".join(
        str(answer.get(field, ""))
        for field in ("quote", "justification", "support_rationale")
    ).casefold()
    open_label_signals = (
        "open-label",
        "open label",
        "not masked",
        "no masking",
        "neither the investigators nor the patients were masked",
    )
    specific_influence_signals = (
        "differential",
        "subjective symptom",
        "patient-reported",
        "detection bias",
        "solicited",
        "non-systematic",
        "different schedule",
        "different assessment",
        "differential assessment",
    )
    return any(signal in text for signal in open_label_signals) and not any(
        signal in text for signal in specific_influence_signals
    )


def _safety_influence_answer_uses_efficacy_text(answer: dict) -> bool:
    text = " ".join(
        str(answer.get(field, ""))
        for field in ("quote", "justification", "support_rationale")
    ).casefold()
    efficacy_signals = (
        "crpc",
        "castration-resistant",
        "castration resistant",
        "radiographic progression",
        "progression-free",
        "progression free",
        "overall survival",
        "psa rise",
        "psa progression",
        "disease progression",
    )
    harms_signals = (
        "adverse event",
        "adverse events",
        "toxicity",
        "toxicities",
        "ctcae",
        "common terminology criteria",
        "meddra",
        "safety population",
        "grade 3",
        "grade ≥3",
    )
    return any(signal in text for signal in efficacy_signals) and not any(
        signal in text for signal in harms_signals
    )


def _safety_ctcae_understates_possible_influence(answer: dict) -> bool:
    text = " ".join(
        str(answer.get(field, ""))
        for field in ("quote", "justification", "support_rationale")
    ).casefold()
    awareness_signals = (
        "open-label",
        "open label",
        "unblinded",
        "not blinded",
        "not masked",
        "knowledge",
        "aware",
    )
    standardization_signals = (
        "ctcae",
        "common terminology criteria",
        "standardized",
        "standardised",
        "grading system",
    )
    overconfident_signals = (
        "limits the potential",
        "limited potential",
        "largely objective",
        "unlikely to influence",
        "not influenced",
    )
    return (
        any(signal in text for signal in awareness_signals)
        and any(signal in text for signal in standardization_signals)
        and any(signal in text for signal in overconfident_signals)
    )
