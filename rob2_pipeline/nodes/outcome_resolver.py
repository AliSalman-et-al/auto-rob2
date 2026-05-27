from __future__ import annotations

import json
import re

from rob2_pipeline.config import build_provider
from rob2_pipeline.models import format_evidence
from rob2_pipeline.state import RoB2State
from rob2_pipeline.state_factory import DEFAULT_OUTCOME_PROPERTIES


_PATTERNS = {
    "patient_reported": re.compile(
        r"\b(patient[- ]reported|self[- ]reported|questionnaire|quality of life|pain score|symptom score)\b",
        re.I,
    ),
    "safety_harm": re.compile(
        r"\b(adverse event|serious adverse|toxicity|harm|side effect|safety|tolerability)\b",
        re.I,
    ),
    "time_to_event": re.compile(
        r"\b(time to|survival|hazard ratio|kaplan[- ]meier|censor|event[- ]free)\b",
        re.I,
    ),
    "death_only": re.compile(
        r"\b(overall survival|all[- ]cause mortality|death from any cause|mortality|vital status)\b",
        re.I,
    ),
    "composite": re.compile(
        r"\b(composite|progression|relapse|recurrence|hospitali[sz]ation|treatment failure|event[- ]free|or death)\b",
        re.I,
    ),
    "lab_or_imaging": re.compile(
        r"\b(biomarker|laboratory|lab |blood|serum|imaging|radiographic|mri|ct scan|recist|threshold|assay)\b",
        re.I,
    ),
    "blinded_adjudication": re.compile(
        r"\b(blinded|masked|independent|central).{0,80}\b(adjudication|committee|review|assessor)\b",
        re.I,
    ),
}

_ALLOWED_OUTCOME_TYPES = {
    "patient-reported",
    "clinician-graded",
    "biomarker",
    "vital-status",
    "clinician-composite",
}

_FORCED_SCOPE_PROPERTIES: dict[str, dict[str, bool]] = {
    "OS": {
        "objective_event": True,
        "clinician_judged": False,
        "patient_reported": False,
        "composite": False,
        "time_to_event": True,
        "safety_harm": False,
        "lab_or_imaging_threshold": False,
    },
    "PFS": {
        "objective_event": False,
        "clinician_judged": True,
        "patient_reported": False,
        "composite": True,
        "time_to_event": True,
        "safety_harm": False,
    },
    "AE": {
        "objective_event": False,
        "clinician_judged": True,
        "patient_reported": False,
        "composite": False,
        "time_to_event": False,
        "safety_harm": True,
    },
}


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _normalise_outcome_scope(outcome: str, benchmark_code: str = "") -> str:
    candidates = [benchmark_code, outcome]
    for candidate in candidates:
        normalized = _normalize(candidate or "")
        if not normalized:
            continue
        if normalized in {
            "os",
            "overall survival",
            "death from any cause",
            "all cause mortality",
            "mortality",
            "vital status",
        }:
            return "OS"
        if normalized in {
            "pfs",
            "progression free survival",
            "event free survival",
            "disease free survival",
        }:
            return "PFS"
        if normalized in {
            "ae",
            "adverse events",
            "adverse event",
            "harms",
            "safety",
            "toxicity",
            "tolerability",
        }:
            return "AE"
    return ""


def _quote_from_text(text: str, scope_code: str) -> str:
    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        return ""

    if scope_code == "OS":
        terms = (
            "overall survival",
            "death from any cause",
            "all-cause mortality",
            "mortality",
            "vital status",
        )
    elif scope_code == "PFS":
        terms = (
            "progression-free",
            "progression free",
            "progression",
            "event-free",
            "relapse",
            "recurrence",
            "death",
        )
    elif scope_code == "AE":
        terms = (
            "adverse event",
            "serious adverse",
            "toxicity",
            "safety",
            "harm",
            "side effect",
            "tolerability",
        )
    else:
        terms = (
            "outcome",
            "endpoint",
            "measurement",
            "adjudication",
            "blinded",
            "assessor",
            "survival",
            "progression",
            "adverse event",
            "toxicity",
        )

    for sentence in re.split(r"(?<=[.!?])\s+", cleaned):
        lowered = sentence.casefold()
        if any(term in lowered for term in terms):
            return sentence.strip()[:220]
    return cleaned[:220]


def _scope_terms(outcome: str, benchmark_code: str) -> tuple[str, ...]:
    scope_code = _normalise_outcome_scope(outcome, benchmark_code)
    terms = [outcome, benchmark_code]
    if scope_code == "OS":
        terms.extend(
            [
                "overall survival",
                "death from any cause",
                "all-cause mortality",
                "mortality",
                "vital status",
            ]
        )
    elif scope_code == "PFS":
        terms.extend(
            [
                "progression-free",
                "progression free",
                "radiographic progression-free",
                "event-free",
            ]
        )
    elif scope_code == "AE":
        terms.extend(
            [
                "adverse event",
                "serious adverse",
                "toxicity",
                "safety endpoint",
            ]
        )
    return tuple(term.casefold() for term in terms if term)


def _is_outcome_local(text: str, outcome: str, benchmark_code: str) -> bool:
    lowered = text.casefold()
    return any(term in lowered for term in _scope_terms(outcome, benchmark_code))


def _section_text(state: RoB2State, field: str) -> str:
    evidence = state.get("evidence", {})
    section = evidence.get(field) if evidence else None
    return format_evidence(section) if section else ""


def _requested_scope_support(
    outcome: str, benchmark_code: str
) -> dict[str, str] | None:
    anchor = benchmark_code or outcome
    if not anchor:
        return None
    return {
        "source_priority": 1,
        "source_label": "requested outcome / benchmark code",
        "field": "outcome",
        "quote": anchor,
    }


def _source_candidates(state: RoB2State) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    rag_contexts = state.get("rag_contexts", {}) or {}
    outcome = state.get("outcome", "")
    benchmark_code = str(state.get("outcome_code", "") or "")

    for field in ("abstract", "methods", "results"):
        text = _section_text(state, field)
        if text and _is_outcome_local(text, outcome, benchmark_code):
            candidates.append(
                {
                    "source_priority": 2,
                    "source_label": "primary endpoint definition",
                    "field": field,
                    "text": text,
                }
            )

    primary_endpoint = state.get("registered_endpoint", "")
    if primary_endpoint and primary_endpoint.casefold() not in {"not reported", ""}:
        candidates.append(
            {
                "source_priority": 4,
                "source_label": "registry matched endpoint",
                "field": "registered_endpoint",
                "text": primary_endpoint,
            }
        )

    secondary = state.get("registered_secondary_endpoints", "")
    if secondary and secondary.casefold() not in {"not reported", ""}:
        candidates.append(
            {
                "source_priority": 4,
                "source_label": "registry matched endpoint",
                "field": "registered_secondary_endpoints",
                "text": secondary,
            }
        )

    protocol_text = _section_text(state, "d5_registration")
    if protocol_text:
        candidates.append(
            {
                "source_priority": 3,
                "source_label": "protocol / sap definition",
                "field": "d5_registration",
                "text": protocol_text,
            }
        )

    registered_analysis = state.get("registered_analysis", "")
    if registered_analysis and registered_analysis.casefold() not in {
        "not reported",
        "",
    }:
        candidates.append(
            {
                "source_priority": 3,
                "source_label": "protocol / sap definition",
                "field": "registered_analysis",
                "text": registered_analysis,
            }
        )

    ctgov_outcomes = state.get("ctgov_outcomes", "")
    if ctgov_outcomes and "not yet retrieved" not in ctgov_outcomes.casefold():
        candidates.append(
            {
                "source_priority": 4,
                "source_label": "registry matched endpoint",
                "field": "ctgov_outcomes",
                "text": ctgov_outcomes,
            }
        )

    d4_measurement = _section_text(state, "d4_outcome_meas")
    if d4_measurement:
        candidates.append(
            {
                "source_priority": 5,
                "source_label": "d4 outcome measurement",
                "field": "d4_outcome_meas",
                "text": d4_measurement,
            }
        )

    if rag_contexts.get("d4_measurement"):
        candidates.append(
            {
                "source_priority": 5,
                "source_label": "d4 measurement context",
                "field": "rag_contexts.d4_measurement",
                "text": str(rag_contexts.get("d4_measurement", "")),
            }
        )
    if rag_contexts.get("d4_assessor"):
        candidates.append(
            {
                "source_priority": 5,
                "source_label": "d4 assessor context",
                "field": "rag_contexts.d4_assessor",
                "text": str(rag_contexts.get("d4_assessor", "")),
            }
        )

    # Keep only meaningful, non-duplicated texts while preserving priority order.
    deduped: list[dict[str, str]] = []
    seen: set[tuple[int, str, str]] = set()
    for candidate in candidates:
        text = " ".join(candidate["text"].split()).strip()
        if not text:
            continue
        key = (candidate["source_priority"], candidate["field"], text.casefold())
        if key in seen:
            continue
        seen.add(key)
        candidate = dict(candidate)
        candidate["text"] = text
        deduped.append(candidate)
    return deduped


def _pattern_matches(text: str, scope_code: str) -> dict[str, bool]:
    lower = text.casefold()
    return {
        "patient_reported": bool(_PATTERNS["patient_reported"].search(lower)),
        "safety_harm": bool(_PATTERNS["safety_harm"].search(lower)),
        "time_to_event": bool(_PATTERNS["time_to_event"].search(lower)),
        "composite": bool(_PATTERNS["composite"].search(lower)),
        "lab_or_imaging_threshold": bool(_PATTERNS["lab_or_imaging"].search(lower)),
        "blinded_adjudication": bool(_PATTERNS["blinded_adjudication"].search(lower)),
        "objective_event": bool(_PATTERNS["death_only"].search(lower)),
    }


def _apply_guardrails(props: dict[str, bool], scope_code: str) -> dict[str, bool]:
    guarded = dict(props)
    for key, forced_value in _FORCED_SCOPE_PROPERTIES.get(scope_code, {}).items():
        guarded[key] = forced_value
    guarded["clinician_judged"] = (
        not guarded["patient_reported"] and not guarded["objective_event"]
    )
    if guarded["safety_harm"]:
        guarded["clinician_judged"] = True
    return guarded


def _update_properties_from_source(
    props: dict[str, bool],
    source_text: str,
    scope_code: str,
    source_priority: int,
    assignments: dict[str, int],
) -> None:
    matches = _pattern_matches(source_text, scope_code)
    for key in (
        "patient_reported",
        "safety_harm",
        "time_to_event",
        "composite",
        "lab_or_imaging_threshold",
        "blinded_adjudication",
        "objective_event",
    ):
        if not matches[key]:
            continue
        if key in _FORCED_SCOPE_PROPERTIES.get(scope_code, {}):
            continue
        current_priority = assignments.get(key)
        if current_priority is None or source_priority < current_priority:
            props[key] = True
            assignments[key] = source_priority


def infer_outcome_properties(
    outcome: str,
    evidence_text: str,
    *,
    benchmark_code: str = "",
) -> dict[str, bool]:
    scope_code = _normalise_outcome_scope(outcome, benchmark_code)
    text = "\n".join([outcome, benchmark_code, evidence_text])
    props = dict(DEFAULT_OUTCOME_PROPERTIES)
    props["patient_reported"] = bool(_PATTERNS["patient_reported"].search(text))
    props["safety_harm"] = bool(_PATTERNS["safety_harm"].search(text))
    props["time_to_event"] = bool(_PATTERNS["time_to_event"].search(text))
    props["composite"] = (
        bool(_PATTERNS["composite"].search(text)) and scope_code != "OS"
    )
    props["lab_or_imaging_threshold"] = bool(_PATTERNS["lab_or_imaging"].search(text))
    props["blinded_adjudication"] = bool(_PATTERNS["blinded_adjudication"].search(text))
    props["objective_event"] = bool(_PATTERNS["death_only"].search(text)) or (
        props["lab_or_imaging_threshold"]
        and not props["patient_reported"]
        and not props["composite"]
    )
    props["clinician_judged"] = (
        not props["patient_reported"] and not props["objective_event"]
    )
    if props["safety_harm"]:
        props["clinician_judged"] = True
    if scope_code in _FORCED_SCOPE_PROPERTIES:
        props = _apply_guardrails(props, scope_code)
    return props


def outcome_type_from_properties(props: dict[str, bool]) -> str:
    if props.get("patient_reported"):
        return "patient-reported"
    if props.get("safety_harm"):
        return "clinician-graded"
    if props.get("lab_or_imaging_threshold") and not props.get("composite"):
        return "biomarker"
    if props.get("objective_event") and not props.get("composite"):
        return "vital-status"
    return "clinician-composite"


def _deterministic_resolution(state: RoB2State) -> dict:
    outcome = state.get("outcome", "")
    benchmark_code = str(state.get("outcome_code", "") or "")
    scope_code = _normalise_outcome_scope(outcome, benchmark_code)
    requested_support = _requested_scope_support(outcome, benchmark_code)
    sources = _source_candidates(state)

    props = dict(DEFAULT_OUTCOME_PROPERTIES)
    assignments: dict[str, int] = {}
    if scope_code in _FORCED_SCOPE_PROPERTIES:
        props = _apply_guardrails(props, scope_code)
        for key in _FORCED_SCOPE_PROPERTIES[scope_code]:
            assignments[key] = 0
    elif outcome:
        outcome_text = outcome if not benchmark_code else f"{outcome}\n{benchmark_code}"
        scoped_props = infer_outcome_properties(
            outcome,
            outcome_text,
            benchmark_code=benchmark_code,
        )
        props.update(scoped_props)

    for source in sources:
        _update_properties_from_source(
            props,
            source["text"],
            scope_code,
            source["source_priority"],
            assignments,
        )

    if scope_code in _FORCED_SCOPE_PROPERTIES:
        props = _apply_guardrails(props, scope_code)
    else:
        props["clinician_judged"] = (
            not props["patient_reported"] and not props["objective_event"]
        )
        if props["safety_harm"]:
            props["clinician_judged"] = True

    support: list[dict[str, str]] = []
    if requested_support:
        support.append(requested_support)
    for source in sources:
        quote = _quote_from_text(source["text"], scope_code)
        if not quote:
            continue
        support.append(
            {
                "source_priority": source["source_priority"],
                "source_label": source["source_label"],
                "field": source["field"],
                "quote": quote,
            }
        )

    deduped_support: list[dict[str, str]] = []
    seen_support: set[tuple[int, str, str]] = set()
    for item in support:
        key = (
            int(item["source_priority"]),
            item["source_label"],
            item["quote"].casefold(),
        )
        if key in seen_support:
            continue
        seen_support.add(key)
        deduped_support.append(item)

    warnings: list[str] = []
    if scope_code in {"OS", "PFS", "AE"}:
        warnings.append(
            f"Outcome scope guardrail applied for {scope_code}; unrelated paper-wide mentions were ignored."
        )
    if not sources:
        warnings.append(
            "No outcome-local endpoint definition was available; used requested outcome and generic scoped inference."
        )

    outcome_type = outcome_type_from_properties(props)
    audit = {
        "requested_outcome": outcome,
        "benchmark_code": benchmark_code or scope_code or "",
        "outcome_type": outcome_type,
        "resolution_method": "deterministic",
        "resolver": "deterministic_fallback",
        "source_priority_order": [
            "requested outcome / benchmark code",
            "primary endpoint definition",
            "protocol / sap definition",
            "registry matched endpoint",
            "d4 outcome measurement",
        ],
        "support": deduped_support,
        "warnings": warnings,
    }
    return {
        "outcome_properties": props,
        "outcome_type": outcome_type,
        "outcome_resolution": audit,
        "warnings": warnings,
    }


def _extract_json_object(response: str) -> dict | None:
    cleaned = response.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(cleaned[start : end + 1])
    except Exception:  # noqa: BLE001
        return None


def _merge_llm_resolution(
    deterministic: dict, llm_payload: dict, scope_code: str
) -> tuple[dict, list[str]]:
    props = dict(deterministic["outcome_properties"])
    audit = dict(deterministic["outcome_resolution"])
    warnings = list(deterministic["warnings"])
    raw_props = llm_payload.get("outcome_properties") or llm_payload.get("properties")
    contradiction_keys: list[str] = []
    if isinstance(raw_props, dict):
        for key in DEFAULT_OUTCOME_PROPERTIES:
            value = raw_props.get(key)
            if isinstance(value, bool):
                props[key] = value
                forced = _FORCED_SCOPE_PROPERTIES.get(scope_code, {}).get(key)
                if forced is not None and value != forced:
                    contradiction_keys.append(key)
    if isinstance(llm_payload.get("warnings"), list):
        warnings.extend(
            str(item) for item in llm_payload["warnings"] if str(item).strip()
        )

    props = _apply_guardrails(
        {key: bool(props.get(key, False)) for key in DEFAULT_OUTCOME_PROPERTIES},
        scope_code,
    )
    if contradiction_keys:
        warnings.append(
            f"Guardrail cleared LLM contradictions for {scope_code or 'scoped'} "
            "because LLM contradicted scoped properties: "
            + ", ".join(sorted(set(contradiction_keys)))
        )
    outcome_type = outcome_type_from_properties(props)
    llm_type = str(llm_payload.get("outcome_type", "") or "").strip()
    if llm_type and llm_type not in _ALLOWED_OUTCOME_TYPES:
        warnings.append(
            f"LLM outcome_type {llm_type!r} is not a valid RoB 2 outcome type."
        )
    if llm_type and llm_type != outcome_type:
        warnings.append(
            f"LLM outcome_type {llm_type!r} was normalized to {outcome_type!r}."
        )

    if isinstance(llm_payload.get("support"), list):
        for item in llm_payload["support"]:
            if not isinstance(item, dict):
                continue
            if not item.get("quote"):
                continue
            support_item = {
                "source_priority": int(item.get("source_priority", 0) or 0),
                "source_label": str(item.get("source_label", "llm")),
                "field": str(item.get("field", "llm")),
                "quote": str(item.get("quote", "")).strip()[:220],
            }
            if support_item["source_priority"] > 0:
                audit["support"].append(support_item)

    seen_support: set[tuple[int, str, str]] = set()
    deduped_support: list[dict[str, str]] = []
    for item in audit["support"]:
        key = (
            int(item["source_priority"]),
            item["source_label"],
            item["quote"].casefold(),
        )
        if key in seen_support:
            continue
        seen_support.add(key)
        deduped_support.append(item)

    audit["support"] = deduped_support
    audit["outcome_type"] = outcome_type
    audit["resolution_method"] = "llm"
    audit["resolver"] = "llm"
    audit["warnings"] = warnings
    result = {
        "outcome_properties": props,
        "outcome_type": outcome_type,
        "outcome_resolution": audit,
        "warnings": warnings,
    }
    return result, warnings


def _llm_resolution(state: RoB2State, deterministic: dict) -> tuple[dict, list[str]]:
    outcome = state.get("outcome", "")
    benchmark_code = str(state.get("outcome_code", "") or "")
    scope_code = _normalise_outcome_scope(outcome, benchmark_code)
    raw_response = state.get("outcome_resolution_llm_response")
    if isinstance(raw_response, str) and raw_response.strip():
        parsed = _extract_json_object(raw_response)
        if not parsed:
            warnings = list(deterministic["warnings"])
            warnings.insert(
                0, "LLM outcome resolution was invalid; used deterministic fallback."
            )
            deterministic["warnings"] = warnings
            deterministic["outcome_resolution"]["warnings"] = warnings
            return deterministic, warnings
        merged, warnings = _merge_llm_resolution(deterministic, parsed, scope_code)
        merged["outcome_resolution"]["warnings"] = warnings
        return merged, warnings

    if not state.get("enable_llm_outcome_resolution"):
        return deterministic, list(deterministic["warnings"])

    payload = {
        "requested_outcome": outcome,
        "benchmark_code": benchmark_code or scope_code,
        "source_priority_order": [
            "requested outcome / benchmark code",
            "primary endpoint definition",
            "protocol / sap definition",
            "registry matched endpoint",
            "d4 outcome measurement",
        ],
        "scoped_evidence": deterministic["outcome_resolution"]["support"],
        "instructions": (
            "Classify the assessed endpoint using only the scoped evidence. "
            "Do not let paper-wide symptom, toxicity, or adverse-event mentions "
            "change an OS or death-from-any-cause outcome into patient-reported "
            "or safety-harm."
        ),
    }
    prompt = (
        "Return JSON only with keys outcome_properties, outcome_type, support, "
        "and warnings.\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )

    try:
        provider = build_provider()
        response_obj = provider.complete(
            system=(
                "You are resolving RoB 2 outcome scope. Return JSON only. "
                "Do not add markdown fences or extra commentary."
            ),
            user=prompt,
        )
    except Exception as exc:  # noqa: BLE001
        warnings = list(deterministic["warnings"])
        warnings.append(
            f"LLM outcome resolution unavailable; used deterministic fallback ({exc})."
        )
        deterministic["warnings"] = warnings
        deterministic["outcome_resolution"]["warnings"] = warnings
        return deterministic, warnings

    parsed = _extract_json_object(response_obj.content)
    if not parsed:
        warnings = list(deterministic["warnings"])
        warnings.insert(
            0,
            "LLM outcome resolution was invalid; used deterministic fallback.",
        )
        deterministic["warnings"] = warnings
        deterministic["outcome_resolution"]["warnings"] = warnings
        return deterministic, warnings

    merged, warnings = _merge_llm_resolution(deterministic, parsed, scope_code)
    merged["outcome_resolution"]["warnings"] = warnings
    return merged, warnings


def outcome_resolver_node(state: RoB2State) -> RoB2State:
    deterministic = _deterministic_resolution(state)
    resolved_state, warnings = _llm_resolution(state, deterministic)

    errors = list(state.get("errors", []))
    previous_type = state.get("outcome_type", "")
    resolved_type = resolved_state["outcome_type"]
    if previous_type and previous_type != resolved_type:
        errors.append(
            "INFO: outcome_type normalized from "
            f"{previous_type!r} to {resolved_type!r} using outcome-scoped resolution."
        )
    for warning in warnings:
        warning_lower = warning.lower()
        should_surface_as_error = (
            "llm" in warning_lower
            or "invalid" in warning_lower
            or "contradicted" in warning_lower
        )
        if should_surface_as_error and warning not in errors:
            errors.append(warning)

    return {
        "outcome_properties": resolved_state["outcome_properties"],
        "outcome_type": resolved_type,
        "outcome_resolution": resolved_state["outcome_resolution"],
        "errors": errors,
    }
