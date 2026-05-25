from rob2_pipeline.ingestion.assessment import ingest_assessment_documents
from rob2_pipeline.nodes.common import call_node_llm
from rob2_pipeline.models import format_evidence
from rob2_pipeline.prompts import PROMPT_RCT_SCREEN
from rob2_pipeline.state import RoB2State
from rob2_pipeline.xml_parser import extract_tag


def pdf_ingest_node(state: RoB2State) -> RoB2State:
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
    response, log, _ = call_node_llm(state, prompt, "rct_screener")
    is_rct = (extract_tag(response, "is_rct") or "NO").strip().upper() == "YES"
    evidence = extract_tag(response, "evidence") or "No relevant text found"
    errors = list(state.get("errors", []))
    if not is_rct:
        errors.append("Study screened as non-RCT; RoB 2 assessment stopped.")
    return {
        "is_rct": is_rct,
        "rct_screen_evidence": evidence,
        "errors": errors,
        "llm_call_log": log,
    }
