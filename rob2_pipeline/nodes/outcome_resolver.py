from __future__ import annotations

import xml.etree.ElementTree as ET

from rob2_pipeline.models import EVIDENCE_SECTION_FIELDS, format_evidence
from rob2_pipeline.nodes.common import call_node_llm
from rob2_pipeline.state import RoB2State
from rob2_pipeline.state_factory import DEFAULT_OUTCOME_PROPERTIES


VALID_OUTCOME_TYPES = {
    "patient-reported",
    "clinician-graded",
    "biomarker",
    "vital-status",
    "clinician-composite",
}
VALID_SUPPORT_LEVELS = {"strong", "moderate", "weak", "unsupported"}
PROPERTY_FIELDS = tuple(DEFAULT_OUTCOME_PROPERTIES)
LLM_PROPERTY_FIELDS = PROPERTY_FIELDS + ("death_only_objective_event",)


def _evidence_sections(state: RoB2State) -> dict[str, str]:
    evidence = state.get("evidence", {})
    sections: dict[str, str] = {}
    for field in EVIDENCE_SECTION_FIELDS:
        section = evidence.get(field) if evidence else None
        rendered = format_evidence(section) if section else ""
        if rendered.strip():
            sections[field] = rendered.strip()
    if state.get("numerical_result"):
        sections["numerical_result"] = state["numerical_result"]
    return sections


def _evidence_text(state: RoB2State) -> str:
    sections = _evidence_sections(state)
    parts = [state.get("outcome", "")]
    parts.extend(f"[{label}]\n{text}" for label, text in sections.items())
    return "\n\n".join(part for part in parts if part)


def _build_prompt(state: RoB2State) -> str:
    return f"""Resolve the assessed outcome's RoB 2 outcome classification using only evidence bound to the assessed outcome.

Assessed outcome: {state.get("outcome", "Not reported")}

Evidence sections:
{_evidence_text(state) or "No outcome-bound evidence was available."}

Ignore trial-wide mentions of other endpoint families unless the text ties them to the assessed outcome.
Return XML only:
<outcome_resolution>
  <outcome_type>patient-reported|clinician-graded|biomarker|vital-status|clinician-composite</outcome_type>
  <normalized_definition>concise assessed-outcome definition, including timepoint or measurement basis when available</normalized_definition>
  <aliases>
    <alias>abbreviation or synonym used for this assessed outcome</alias>
  </aliases>
  <support_level>strong|moderate|weak|unsupported</support_level>
  <support_rationale>one sentence explaining outcome-bound support</support_rationale>
  <uncertainty>true|false</uncertainty>
  <properties>
    <patient_reported>true|false</patient_reported>
    <safety_harm>true|false</safety_harm>
    <time_to_event>true|false</time_to_event>
    <death_only_objective_event>true|false</death_only_objective_event>
    <composite>true|false</composite>
    <lab_or_imaging_threshold>true|false</lab_or_imaging_threshold>
    <blinded_adjudication>true|false</blinded_adjudication>
    <objective_event>true|false</objective_event>
    <clinician_judged>true|false</clinician_judged>
  </properties>
  <quotes>
    <quote source="evidence_section_name">exact quote from the evidence</quote>
  </quotes>
  <constraints>
    <constraint type="missing_required_evidence|wrong_outcome_context|semantic_support_conflict">reason</constraint>
  </constraints>
</outcome_resolution>"""


def _parse_bool(value: str | None, field: str) -> bool:
    normalized = (value or "").strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Invalid boolean for {field}")


def _parse_resolution(raw: str) -> dict:
    root = ET.fromstring(raw.strip())
    if root.tag != "outcome_resolution":
        raise ValueError("Expected <outcome_resolution> root")
    outcome_type = (root.findtext("outcome_type") or "").strip()
    normalized_definition = (root.findtext("normalized_definition") or "").strip()
    support_level = (root.findtext("support_level") or "").strip().lower()
    support_rationale = (root.findtext("support_rationale") or "").strip()
    uncertainty_text = root.findtext("uncertainty")
    uncertainty = (
        _parse_bool(uncertainty_text, "uncertainty")
        if uncertainty_text is not None
        else True
    )
    if outcome_type not in VALID_OUTCOME_TYPES:
        raise ValueError("Invalid outcome_type")
    if support_level not in VALID_SUPPORT_LEVELS:
        raise ValueError("Invalid support_level")
    if not support_rationale:
        raise ValueError("Missing support_rationale")

    properties_el = root.find("properties")
    if properties_el is None:
        raise ValueError("Missing properties")
    llm_props = {
        field: _parse_bool(properties_el.findtext(field), field)
        for field in LLM_PROPERTY_FIELDS
    }
    props = {field: llm_props[field] for field in PROPERTY_FIELDS}

    quotes = []
    for quote_el in root.findall("./quotes/quote"):
        quote = (quote_el.text or "").strip()
        if quote:
            quotes.append({"quote": quote, "source": quote_el.attrib.get("source", "")})
    if support_level != "unsupported" and not quotes:
        raise ValueError("Missing quotes for supported outcome classification")

    constraints = []
    for constraint_el in root.findall("./constraints/constraint"):
        constraints.append(
            {
                "constraint_type": constraint_el.attrib.get(
                    "type", "semantic_support_conflict"
                ),
                "reason": (constraint_el.text or "").strip()
                or "LLM reported a support constraint.",
            }
        )
    return {
        "outcome_type": outcome_type,
        "normalized_definition": normalized_definition,
        "aliases": [
            (alias_el.text or "").strip()
            for alias_el in root.findall("./aliases/alias")
            if (alias_el.text or "").strip()
        ],
        "outcome_properties": props,
        "support": {
            "support_level": support_level,
            "support_rationale": support_rationale,
            "quotes": quotes,
            "constraints": constraints,
        },
        "uncertainty": uncertainty,
    }


def _quote_is_traceable(quote: str, sections: dict[str, str]) -> bool:
    normalized_quote = " ".join(quote.casefold().split())
    return any(
        normalized_quote in " ".join(text.casefold().split())
        for text in sections.values()
    )


def _unsupported(
    reason: str, constraint_type: str = "missing_required_evidence"
) -> dict:
    return {
        "outcome_type": "clinician-composite",
        "normalized_definition": "",
        "aliases": [],
        "outcome_properties": dict(DEFAULT_OUTCOME_PROPERTIES),
        "outcome_classification_support": {
            "support_level": "unsupported",
            "support_rationale": reason,
            "quotes": [],
            "constraints": [
                {
                    "constraint_type": constraint_type,
                    "reason": reason,
                }
            ],
        },
        "uncertainty": True,
    }


def _normalization_artifact(state: RoB2State, resolution: dict) -> dict:
    support = resolution["outcome_classification_support"]
    return {
        "artifact_id": f"outcome-normalization:{state.get('outcome', '')}",
        "schema_version": "outcome-normalization-v1",
        "outcome": state.get("outcome", ""),
        "normalized_definition": resolution.get("normalized_definition", ""),
        "aliases": resolution.get("aliases", []),
        "outcome_type": resolution["outcome_type"],
        "outcome_properties": resolution["outcome_properties"],
        "binding_support": support,
        "auto_accept_blocked": support.get("support_level") in {
            "weak",
            "unsupported",
        },
        "uncertainty": bool(resolution.get("uncertainty", False))
        or support.get("support_level") in {"weak", "unsupported"},
    }


def _validate_resolution(resolution: dict, sections: dict[str, str]) -> dict:
    constraints = list(resolution["support"].get("constraints", []))
    for quote in resolution["support"].get("quotes", []):
        if not _quote_is_traceable(quote["quote"], sections):
            constraints.append(
                {
                    "constraint_type": "quote_untraceable",
                    "reason": "Outcome resolver quote was not traceable to source evidence.",
                    "evidence": quote["quote"],
                    "provenance": {"source": quote.get("source", "")},
                }
            )
    if constraints:
        fallback = _unsupported(
            constraints[0]["reason"], constraints[0]["constraint_type"]
        )
        fallback["outcome_classification_support"]["constraints"] = constraints
        return fallback
    resolution["outcome_classification_support"] = resolution.pop("support")
    return resolution


def outcome_resolver_node(state: RoB2State) -> RoB2State:
    sections = _evidence_sections(state)
    response, log, _parsed = call_node_llm(
        state,
        _build_prompt(state),
        "outcome_resolver",
    )
    try:
        resolution = _validate_resolution(_parse_resolution(response), sections)
    except Exception as exc:  # noqa: BLE001
        resolution = _unsupported(f"Invalid outcome resolver output: {exc}")

    support_constraints = list(state.get("support_constraints", []))
    for constraint in resolution["outcome_classification_support"].get(
        "constraints", []
    ):
        support_constraints.append(
            {
                **constraint,
                "sq_id": "outcome_classification",
                "claim": {
                    "outcome": state.get("outcome", ""),
                    "outcome_type": resolution["outcome_type"],
                },
            }
        )

    errors = list(state.get("errors", []))
    previous_type = state.get("outcome_type", "")
    if previous_type and previous_type != resolution["outcome_type"]:
        errors.append(
            "INFO: outcome_type normalized from "
            f"{previous_type!r} to {resolution['outcome_type']!r} using outcome-bound LLM resolution."
        )

    return {
        "outcome_properties": resolution["outcome_properties"],
        "outcome_type": resolution["outcome_type"],
        "outcome_classification_support": resolution["outcome_classification_support"],
        "outcome_normalization_artifact": _normalization_artifact(state, resolution),
        "support_constraints": support_constraints,
        "errors": errors,
        "llm_call_log": log,
    }
