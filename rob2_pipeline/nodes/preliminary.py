import re

from lxml import etree

from rob2_pipeline.models import EVIDENCE_SECTION_FIELDS, format_evidence
from rob2_pipeline.nodes.common import call_node_llm
from rob2_pipeline.prompts import PROMPT_PRELIMINARY_INFO
from rob2_pipeline.state import RoB2State
from rob2_pipeline.xml_parser import sanitize_stray_lt, extract_tag


def _nested_text(
    xml_string: str, parent: str, child: str, default: str = "Not reported"
) -> str:
    try:
        sanitized = sanitize_stray_lt(xml_string)
        parser = etree.XMLParser(recover=True)
        root = etree.fromstring(f"<root>{sanitized}</root>".encode(), parser=parser)
        value = root.findtext(f".//{parent}/{child}")
        return value.strip() if value and value.strip() else default
    except Exception:
        return default


def _first_match(text: str, patterns: list[str], default: str = "Not reported") -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return default


def _prefer_extracted(value: str, fallback: str) -> str:
    return fallback if value == "Not reported" and fallback != "Not reported" else value


def _normalize_endpoint_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _endpoint_matches(assessed_outcome: str, endpoint: str) -> bool:
    assessed = _normalize_endpoint_name(assessed_outcome)
    candidate = _normalize_endpoint_name(endpoint)
    if not assessed or not candidate or candidate == "not reported":
        return False
    return assessed == candidate or assessed in candidate or candidate in assessed


def _split_endpoint_list(value: str) -> list[str]:
    if not value or value == "Not reported":
        return []
    return [part.strip() for part in re.split(r";|\n", value) if part.strip()]


def _ctgov_outcome_candidates(ctgov_outcomes: str) -> list[str]:
    candidates = []
    for line in ctgov_outcomes.splitlines():
        if ":" not in line:
            continue
        _, values = line.split(":", 1)
        candidates.extend(part.strip() for part in values.split(";") if part.strip())
    return candidates


def _matching_registered_endpoint(state: RoB2State) -> str | None:
    assessed_outcome = state.get("outcome", "")
    registered_endpoint = state.get("registered_endpoint", "")
    if _endpoint_matches(assessed_outcome, registered_endpoint):
        return None

    candidates = _split_endpoint_list(state.get("registered_secondary_endpoints", ""))
    candidates.extend(_ctgov_outcome_candidates(state.get("ctgov_outcomes", "")))
    for candidate in candidates:
        if _endpoint_matches(assessed_outcome, candidate):
            return candidate
    return None


def preliminary_info_node(state: RoB2State) -> RoB2State:
    evidence = state["evidence"]
    prompt = PROMPT_PRELIMINARY_INFO.format(
        abstract_text=format_evidence(evidence["abstract"]),
        methods_text=format_evidence(evidence["methods"]),
        results_text=format_evidence(evidence["results"]),
        registration_text=format_evidence(evidence["d5_registration"]),
        consort_text=format_evidence(evidence["consort_flow"]),
    )
    response, log, _ = call_node_llm(state, prompt, "preliminary_info")
    requested_outcome = state.get("outcome", "")
    extracted_outcome = _nested_text(response, "outcome_assessed", "value")
    evidence_text = "\n".join(
        [
            state.get("full_text", ""),
            *(format_evidence(evidence[field]) for field in EVIDENCE_SECTION_FIELDS),
        ]
    )
    registration_number = _prefer_extracted(
        _nested_text(response, "trial_registration", "number"),
        _first_match(evidence_text, [r"\b(NCT\d{8})\b", r"\b(ISRCTN\d+)\b"]),
    )
    n_randomized = _prefer_extracted(
        _nested_text(response, "n_randomized", "value"),
        _first_match(
            evidence_text,
            [r"(\d{2,6})\s+(?:patients|participants)\s+(?:were\s+)?randomi[sz]ed"],
        ),
    )
    registered_analysis = _prefer_extracted(
        _nested_text(response, "registered_analysis", "value"),
        "ITT"
        if re.search(
            r"intention[- ]to[- ]treat|\bITT\b", evidence_text, flags=re.IGNORECASE
        )
        else "Not reported",
    )
    updated_state = {
        "intervention": _nested_text(response, "experimental_intervention", "value"),
        "comparator": _nested_text(response, "comparator_intervention", "value"),
        "outcome": requested_outcome or extracted_outcome,
        "outcome_type": (
            extract_tag(response, "outcome_type") or "clinician-composite"
        ).strip(),
        "numerical_result": _nested_text(response, "numerical_result", "value"),
        "effect_of_interest": state.get("effect_of_interest", "ITT"),
        "registration_number": registration_number,
        "n_randomized": n_randomized,
        "registered_endpoint": _nested_text(
            response, "registered_primary_endpoint", "value"
        ),
        "registered_secondary_endpoints": (
            extract_tag(response, "registered_secondary_endpoints") or "Not reported"
        ).strip(),
        "registered_analysis": registered_analysis,
        "sources_consulted": [
            field
            for field in EVIDENCE_SECTION_FIELDS
            if format_evidence(evidence[field])
        ],
        "llm_call_log": log,
    }

    state = dict(state)
    state.update(updated_state)

    # Fetch registration data from ClinicalTrials.gov
    import os
    from rob2_pipeline.registration_api import (
        extract_description,
        extract_design_info,
        extract_outcomes,
        extract_participant_flow,
        fetch_registration,
        format_description_for_prompt,
        format_design_for_prompt,
        format_flow_for_prompt,
        format_outcomes_for_prompt,
    )

    _nct_id = state.get("registration_number", "")
    _use_cache = os.getenv("ROB2_CTGOV_CACHE", "1") != "0"
    if _nct_id.startswith("NCT"):
        _reg_data = fetch_registration(_nct_id, use_cache=_use_cache)
        if _reg_data:
            _outcomes = extract_outcomes(_reg_data)
            state["ctgov_outcomes"] = format_outcomes_for_prompt(_outcomes)
            state["ctgov_design"] = format_design_for_prompt(
                extract_design_info(_reg_data)
            )
            state["ctgov_description"] = format_description_for_prompt(
                extract_description(_reg_data)
            )
            state["ctgov_flow"] = format_flow_for_prompt(
                extract_participant_flow(_reg_data)
            )
            # Override secondary endpoints with API data (more reliable than text extraction)
            if _outcomes["secondary"]:
                state["registered_secondary_endpoints"] = "; ".join(
                    _outcomes["secondary"]
                )
            # Fill primary endpoint if not extracted from paper
            if _outcomes["primary"] and state.get("registered_endpoint") in (
                "Not reported",
                "",
                None,
            ):
                state["registered_endpoint"] = "; ".join(_outcomes["primary"])
        else:
            _unavailable = "(ClinicalTrials.gov data not available for this trial)"
            state["ctgov_outcomes"] = _unavailable
            state["ctgov_design"] = _unavailable
            state["ctgov_description"] = _unavailable
            state["ctgov_flow"] = _unavailable
    else:
        _skipped = "(No NCT registration number - ClinicalTrials.gov lookup skipped)"
        state["ctgov_outcomes"] = _skipped
        state["ctgov_design"] = _skipped
        state["ctgov_description"] = _skipped
        state["ctgov_flow"] = _skipped

    _matched_registered_endpoint = _matching_registered_endpoint(state)
    if _matched_registered_endpoint:
        state["registered_endpoint"] = _matched_registered_endpoint

    return state
