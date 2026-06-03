"""Mine schema-validated evidence-family facts from selected packet zones."""

from __future__ import annotations

from rob2_pipeline.evidence_store import mine_evidence_families
from rob2_pipeline.nodes.common import call_node_llm
from rob2_pipeline.state import RoB2State


def evidence_family_mining_node(state: RoB2State) -> RoB2State:
    return mine_evidence_families(state, call_fn=_call_for_family_mining)


def _call_for_family_mining(state: dict, prompt: str, node_name: str):
    response, log, parsed = call_node_llm(state, prompt, node_name)
    return response, log, parsed
