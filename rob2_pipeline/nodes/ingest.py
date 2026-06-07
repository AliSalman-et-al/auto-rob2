from typing import Literal

from pydantic import BaseModel, Field

from rob2_pipeline.ingestion.assessment import ingest_assessment_documents
from rob2_pipeline.llm_contracts import call_json_contract_llm
from rob2_pipeline.models import format_evidence
from rob2_pipeline.prompts import PROMPT_RCT_SCREEN
from rob2_pipeline.state import RoB2State


class RctScreeningArtifact(BaseModel):
    schema_version: Literal["rct-screening-v1"]
    is_rct: bool
    evidence: str = Field(min_length=1)
    study_design: str = Field(default="Not reported", min_length=1)
    note: str = ""


def pdf_ingest_node(state: RoB2State) -> RoB2State:
    precomputed = state.get("precomputed_ingestion")
    if precomputed is not None:
        return precomputed.to_state_update(include_llm_call_log=False)
    result = ingest_assessment_documents(
        state["pdf_path"], list(state.get("supplementary_paths") or [])
    )
    return result.to_state_update()


def rct_screener_node(state: RoB2State) -> RoB2State:
    evidence = state["evidence"]
    methods_text = "\n\n".join(
        part
        for part in [
            format_evidence(evidence["abstract"]),
            format_evidence(evidence["methods"]),
            format_evidence(evidence["d1_randomization"]),
            format_evidence(evidence["consort_flow"]),
        ]
        if part
    )
    prompt = PROMPT_RCT_SCREEN.format(methods_text=methods_text)
    prompt = (
        "Return JSON matching RctScreeningArtifact. Use is_rct=true only when "
        "participants were randomly assigned to intervention/comparator groups. "
        "Include concise quote-grounded evidence.\n\n"
        f"{prompt}"
    )
    result = call_json_contract_llm(
        state,
        prompt,
        "rct_screener",
        schema_model=RctScreeningArtifact,
        schema_version="rct-screening-v1",
        prompt_version="rct-screening-prompt-v1",
        fallback_factory=lambda reason: {
            "schema_version": "rct-screening-v1",
            "is_rct": False,
            "evidence": f"No validated RCT screening evidence: {reason}",
            "study_design": "Not reported",
            "note": "JSON contract fallback stopped the assessment.",
        },
    )
    artifact = result.artifact
    is_rct = bool(artifact["is_rct"])
    evidence = artifact.get("evidence") or "No relevant text found"
    errors = list(state.get("errors", []))
    if not is_rct:
        errors.append("Study screened as non-RCT; RoB 2 assessment stopped.")
    return {
        "is_rct": is_rct,
        "rct_screen_evidence": evidence,
        "errors": errors,
        "llm_call_log": result.log,
    }
