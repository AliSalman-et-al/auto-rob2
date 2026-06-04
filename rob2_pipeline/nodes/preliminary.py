import re
from typing import Literal

from pydantic import BaseModel, Field

from rob2_pipeline.llm_contracts import call_json_contract_llm
from rob2_pipeline.models import EVIDENCE_SECTION_FIELDS, format_evidence
from rob2_pipeline.prompts import PROMPT_PRELIMINARY_INFO
from rob2_pipeline.state import RoB2State


class PreliminaryInfoArtifact(BaseModel):
    schema_version: Literal["preliminary-info-v1"]
    intervention: str = Field(default="Not reported", min_length=1)
    comparator: str = Field(default="Not reported", min_length=1)
    assessed_outcome: str = Field(default="Not reported", min_length=1)
    outcome_type: str = Field(default="clinician-composite", min_length=1)
    numerical_result: str = Field(default="Not reported", min_length=1)
    registration_number: str = Field(default="Not reported", min_length=1)
    registered_primary_endpoint: str = Field(default="Not reported", min_length=1)
    registered_secondary_endpoints: str = Field(default="Not reported", min_length=1)
    registered_analysis: str = Field(default="Not reported", min_length=1)
    n_randomized: str = Field(default="Not reported", min_length=1)


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
    prompt = (
        "Return JSON matching PreliminaryInfoArtifact. Preserve exact reported "
        "wording where possible and use 'Not reported' for missing fields.\n\n"
        f"{prompt}"
    )
    result = call_json_contract_llm(
        state,
        prompt,
        "preliminary_info",
        schema_model=PreliminaryInfoArtifact,
        schema_version="preliminary-info-v1",
        prompt_version="preliminary-info-prompt-v1",
        fallback_factory=lambda reason: {
            "schema_version": "preliminary-info-v1",
            "intervention": "Not reported",
            "comparator": "Not reported",
            "assessed_outcome": state.get("outcome", "") or "Not reported",
            "outcome_type": "clinician-composite",
            "numerical_result": "Not reported",
            "registration_number": "Not reported",
            "registered_primary_endpoint": "Not reported",
            "registered_secondary_endpoints": "Not reported",
            "registered_analysis": "Not reported",
            "n_randomized": "Not reported",
            "fallback_reason": reason,
        },
    )
    artifact = result.artifact
    requested_outcome = state.get("outcome", "")
    extracted_outcome = artifact.get("assessed_outcome", "Not reported")
    evidence_text = "\n".join(
        [
            state.get("full_text", ""),
            *(format_evidence(evidence[field]) for field in EVIDENCE_SECTION_FIELDS),
        ]
    )
    registration_number = _prefer_extracted(
        artifact.get("registration_number", "Not reported"),
        _first_match(evidence_text, [r"\b(NCT\d{8})\b", r"\b(ISRCTN\d+)\b"]),
    )
    n_randomized = _prefer_extracted(
        artifact.get("n_randomized", "Not reported"),
        _first_match(
            evidence_text,
            [r"(\d{2,6})\s+(?:patients|participants)\s+(?:were\s+)?randomi[sz]ed"],
        ),
    )
    registered_analysis = _prefer_extracted(
        artifact.get("registered_analysis", "Not reported"),
        "ITT"
        if re.search(
            r"intention[- ]to[- ]treat|\bITT\b", evidence_text, flags=re.IGNORECASE
        )
        else "Not reported",
    )
    updated_state = {
        "intervention": artifact.get("intervention", "Not reported"),
        "comparator": artifact.get("comparator", "Not reported"),
        "outcome": requested_outcome or extracted_outcome,
        "outcome_type": artifact.get("outcome_type", "clinician-composite").strip(),
        "numerical_result": artifact.get("numerical_result", "Not reported"),
        "effect_of_interest": state.get("effect_of_interest", "ITT"),
        "registration_number": registration_number,
        "n_randomized": n_randomized,
        "registered_endpoint": artifact.get(
            "registered_primary_endpoint", "Not reported"
        ),
        "registered_secondary_endpoints": (
            artifact.get("registered_secondary_endpoints", "Not reported")
        ).strip(),
        "registered_analysis": registered_analysis,
        "sources_consulted": [
            field
            for field in EVIDENCE_SECTION_FIELDS
            if format_evidence(evidence[field])
        ],
        "llm_call_log": result.log,
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
        registry_source_document,
    )

    _nct_id = state.get("registration_number", "")
    _use_cache = os.getenv("ROB2_CTGOV_CACHE", "1") != "0"
    if _nct_id.startswith("NCT"):
        _reg_data = fetch_registration(_nct_id, use_cache=_use_cache)
        if _reg_data:
            registry_document = registry_source_document(_nct_id, _reg_data)
            state["ctgov_registry_document"] = registry_document
            source_documents = list(state.get("source_documents") or [])
            source_documents = [
                source
                for source in source_documents
                if source.get("document_id") != registry_document["document_id"]
            ]
            state["source_documents"] = [*source_documents, registry_document]
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

    # Auto-detect safety outcomes and override effect_of_interest
    _safety_keywords = {
        "adverse",
        "toxicity",
        "toxic",
        "safety",
        "tolerability",
        "harm",
        "side effect",
        "side-effect",
        "harms",
    }
    _outcome_lower = state.get("outcome", "").lower()
    _outcome_type = state.get("outcome_type", "")
    _user_set_effect = os.getenv("ROB2_EFFECT_OF_INTEREST", "ITT")

    if (
        _outcome_type == "clinician-graded"
        and any(kw in _outcome_lower for kw in _safety_keywords)
        and _user_set_effect == "ITT"
    ):  # only override if user didn't explicitly set it
        state["effect_of_interest"] = "per-protocol"
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(
            "INFO: effect_of_interest auto-set to 'per-protocol' because outcome '"
            + state.get("outcome", "")
            + "' was classified as a safety endpoint "
            "(outcome_type=clinician-graded with safety keywords). Domain 2 will use "
            "the per-protocol algorithm. Override with ROB2_EFFECT_OF_INTEREST=ITT "
            "environment variable if this is incorrect."
        )

    return state
