from __future__ import annotations
from typing import Literal

from pydantic import BaseModel, Field

from rob2_pipeline.llm_contracts import call_json_contract_llm
from rob2_pipeline.models import EVIDENCE_SECTION_FIELDS, format_evidence
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


class OutcomeQuoteArtifact(BaseModel):
    quote: str = Field(min_length=1)
    source: str = ""


class OutcomeConstraintArtifact(BaseModel):
    constraint_type: Literal[
        "missing_required_evidence",
        "wrong_outcome_context",
        "semantic_support_conflict",
        "quote_untraceable",
    ]
    reason: str = Field(min_length=1)


class OutcomeSupportArtifact(BaseModel):
    support_level: Literal["strong", "moderate", "weak", "unsupported"]
    support_rationale: str = Field(min_length=1)
    quotes: list[OutcomeQuoteArtifact] = Field(default_factory=list)
    constraints: list[OutcomeConstraintArtifact] = Field(default_factory=list)


class OutcomePropertiesArtifact(BaseModel):
    objective_event: bool = False
    clinician_judged: bool = False
    patient_reported: bool = False
    composite: bool = False
    time_to_event: bool = False
    safety_harm: bool = False
    lab_or_imaging_threshold: bool = False
    blinded_adjudication: bool = False
    death_only_objective_event: bool = False


class OutcomeNormalizationArtifact(BaseModel):
    schema_version: Literal["outcome-normalization-v1"]
    outcome_type: Literal[
        "patient-reported",
        "clinician-graded",
        "biomarker",
        "vital-status",
        "clinician-composite",
    ]
    normalized_definition: str = ""
    aliases: list[str] = Field(default_factory=list)
    outcome_properties: OutcomePropertiesArtifact
    support: OutcomeSupportArtifact
    uncertainty: bool


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
Return JSON matching OutcomeNormalizationArtifact. Include outcome_type,
normalized_definition, aliases, outcome_properties, support, uncertainty, quotes,
constraints, and uncertainty. outcome_properties must include objective_event,
clinician_judged, patient_reported, composite, time_to_event, safety_harm,
lab_or_imaging_threshold, blinded_adjudication, and death_only_objective_event.
Use only exact quotes copied from the evidence sections; if no exact definition
quote exists, quote the closest exact outcome-bound result text and explain the
remaining uncertainty."""


def _quote_is_traceable(quote: str, sections: dict[str, str]) -> bool:
    normalized_quote = " ".join(quote.casefold().split())
    return any(
        normalized_quote in " ".join(text.casefold().split())
        for text in sections.values()
    )


def _unsupported(
    reason: str,
    constraint_type: str = "missing_required_evidence",
    *,
    outcome_type: str = "clinician-composite",
    outcome_properties: dict[str, bool] | None = None,
) -> dict:
    return {
        "outcome_type": outcome_type,
        "normalized_definition": "",
        "aliases": [],
        "outcome_properties": outcome_properties or dict(DEFAULT_OUTCOME_PROPERTIES),
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
    resolution["outcome_properties"] = _normalize_outcome_properties(
        resolution.get("outcome_properties", {})
    )
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
            constraints[0]["reason"],
            constraints[0]["constraint_type"],
            outcome_type=resolution.get("outcome_type", "clinician-composite"),
            outcome_properties=resolution["outcome_properties"],
        )
        fallback["normalized_definition"] = resolution.get("normalized_definition", "")
        fallback["aliases"] = resolution.get("aliases", [])
        fallback["outcome_classification_support"]["constraints"] = constraints
        return fallback
    resolution["outcome_classification_support"] = resolution.pop("support")
    return resolution


def _normalize_outcome_properties(raw_properties: object) -> dict[str, bool]:
    if hasattr(raw_properties, "model_dump"):
        raw = raw_properties.model_dump()
    elif isinstance(raw_properties, dict):
        raw = dict(raw_properties)
    else:
        raw = {}
    properties = {
        field: bool(raw.get(field, DEFAULT_OUTCOME_PROPERTIES[field]))
        for field in PROPERTY_FIELDS
    }
    if raw.get("death_only_objective_event"):
        properties["objective_event"] = True
        properties["clinician_judged"] = False
        properties["composite"] = False
    return properties


def outcome_resolver_node(state: RoB2State) -> RoB2State:
    sections = _evidence_sections(state)
    result = call_json_contract_llm(
        state,
        _build_prompt(state),
        "outcome_resolver",
        schema_model=OutcomeNormalizationArtifact,
        schema_version="outcome-normalization-v1",
        prompt_version="outcome-normalization-prompt-v1",
        fallback_factory=lambda reason: {
            "schema_version": "outcome-normalization-v1",
            "outcome_type": "clinician-composite",
            "normalized_definition": "",
            "aliases": [],
            "outcome_properties": dict(DEFAULT_OUTCOME_PROPERTIES),
            "support": {
                "support_level": "unsupported",
                "support_rationale": f"Invalid outcome resolver output: {reason}",
                "quotes": [],
                "constraints": [
                    {
                        "constraint_type": "missing_required_evidence",
                        "reason": f"Invalid outcome resolver output: {reason}",
                    }
                ],
            },
            "uncertainty": True,
        },
    )
    try:
        resolution = _validate_resolution(result.artifact, sections)
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
    normalization_notes = list(state.get("outcome_normalization_notes", []))
    previous_type = state.get("outcome_type", "")
    if previous_type and previous_type != resolution["outcome_type"]:
        normalization_notes.append(
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
        "outcome_normalization_notes": normalization_notes,
        "llm_call_log": result.log,
    }
