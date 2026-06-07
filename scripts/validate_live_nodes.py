from __future__ import annotations

import os
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("ROB2_REMOTE_EVIDENCE_EXTRACTION", "0")
os.environ.setdefault("ROB2_REQUEST_TIMEOUT", "20")
os.environ.setdefault("ROB2_MAX_RETRIES", "1")

from rob2_pipeline.nodes.ingest import pdf_ingest_node, rct_screener_node
from rob2_pipeline.nodes.preliminary import preliminary_info_node
from rob2_pipeline.nodes.outcome_resolver import outcome_resolver_node
from rob2_pipeline.nodes.trial_facts import trial_facts_node
from rob2_pipeline.nodes.rag_retrieval import rag_retrieval_node
from rob2_pipeline.nodes.evidence_packets import evidence_packet_builder_node
from rob2_pipeline.nodes.retrieval_repair import retrieval_repair_node
from rob2_pipeline.nodes.evidence_family_mining import evidence_family_mining_node
from rob2_pipeline.nodes.domain1 import domain1_judge_node, domain1_sq_node
from rob2_pipeline.nodes.domain2 import (
    domain2_analysis_node,
    domain2_conditional_node,
    domain2_judge_node,
    domain2_sq12_node,
)
from rob2_pipeline.nodes.domain3 import domain3_judge_node, domain3_sq_node
from rob2_pipeline.nodes.domain4 import domain4_judge_node, domain4_sq_node
from rob2_pipeline.nodes.domain5 import domain5_judge_node, domain5_sq_node
from rob2_pipeline.nodes.overall import overall_judge_node
from rob2_pipeline.nodes.reporter import report_formatter_node
from rob2_pipeline.nodes.verification import quote_verifier_node
from rob2_pipeline.state_factory import create_initial_state
from rob2_pipeline.trace import end_trace, record_node_span, start_trace


NODES = [
    ("pdf_ingest", pdf_ingest_node),
    ("rct_screener", rct_screener_node),
    ("preliminary_info", preliminary_info_node),
    ("outcome_resolver", outcome_resolver_node),
    ("trial_facts", trial_facts_node),
    ("rag_retrieval", rag_retrieval_node),
    ("evidence_packet_builder", evidence_packet_builder_node),
    ("retrieval_repair", retrieval_repair_node),
    ("evidence_family_mining", evidence_family_mining_node),
    ("domain1_sq", domain1_sq_node),
    ("domain1_judge", domain1_judge_node),
    ("domain2_sq12", domain2_sq12_node),
    ("domain2_conditional", domain2_conditional_node),
    ("domain2_analysis", domain2_analysis_node),
    ("domain2_judge", domain2_judge_node),
    ("domain3_sq", domain3_sq_node),
    ("domain3_judge", domain3_judge_node),
    ("domain4_sq", domain4_sq_node),
    ("domain4_judge", domain4_judge_node),
    ("domain5_sq", domain5_sq_node),
    ("domain5_judge", domain5_judge_node),
    ("quote_verifier", quote_verifier_node),
    ("overall_judge", overall_judge_node),
    ("report_formatter", report_formatter_node),
]


def main() -> int:
    tap_dir = os.environ.get("ROB2_LLM_TAP_DIR")
    if tap_dir:
        tap_path = Path(tap_dir) / "llm_calls.jsonl"
        if tap_path.exists():
            tap_path.unlink()
    state = create_initial_state(
        "inputs/benchmark/CHAARTED.pdf",
        "Overall Survival",
        "ITT",
        supplementary_paths=[],
    )
    start_trace(trial="CHAARTED", outcome="Overall Survival")
    try:
        for name, node in NODES:
            started = time.perf_counter()
            print(f"START {name}", flush=True)
            with record_node_span(name):
                update = node(state)
            if update:
                state.update(update)
            elapsed = time.perf_counter() - started
            print(f"END {name} {elapsed:.2f}s", flush=True)
            if name == "rct_screener" and not state.get("is_rct"):
                break
            if name == "domain2_sq12":
                answers = state.get("sq_answers", {})
                if (
                    answers.get("2.1", {}).get("answer") in {"N", "PN", "NI"}
                    and answers.get("2.2", {}).get("answer") in {"N", "PN", "NI"}
                ):
                    continue
    finally:
        trace = end_trace()
        output_dir = Path("outputs/validation_live_nodes")
        output_dir.mkdir(parents=True, exist_ok=True)
        if trace is not None:
            trace.write(str(output_dir))
        (output_dir / "CHAARTED_state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    print(f"OVERALL {state.get('overall_judgment')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
