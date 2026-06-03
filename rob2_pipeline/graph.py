from functools import wraps

from langgraph.graph import END, StateGraph

from rob2_pipeline.nodes.domain1 import domain1_judge_node, domain1_sq_node
from rob2_pipeline.nodes.domain2 import (
    d2_needs_conditional,
    domain2_analysis_node,
    domain2_conditional_node,
    domain2_judge_node,
    domain2_sq12_node,
)
from rob2_pipeline.nodes.domain3 import domain3_judge_node, domain3_sq_node
from rob2_pipeline.nodes.domain4 import domain4_judge_node, domain4_sq_node
from rob2_pipeline.nodes.domain5 import domain5_judge_node, domain5_sq_node
from rob2_pipeline.nodes.evidence_packets import evidence_packet_builder_node
from rob2_pipeline.nodes.evidence_family_mining import evidence_family_mining_node
from rob2_pipeline.nodes.ingest import pdf_ingest_node, rct_screener_node
from rob2_pipeline.nodes.overall import overall_judge_node
from rob2_pipeline.nodes.outcome_resolver import outcome_resolver_node
from rob2_pipeline.nodes.preliminary import preliminary_info_node
from rob2_pipeline.nodes.rag_retrieval import rag_retrieval_node
from rob2_pipeline.nodes.reporter import report_formatter_node
from rob2_pipeline.nodes.trial_facts import trial_facts_node
from rob2_pipeline.nodes.verification import quote_verifier_node
from rob2_pipeline.state import RoB2State
from rob2_pipeline.trace import record_node_span


DOMAIN_START_NODES = [
    "domain1_sq",
    "domain2_sq12",
    "domain3_sq",
    "domain4_sq",
    "domain5_sq",
]


def timed_node(node_name, fn):
    @wraps(fn)
    def _wrapped(*args, **kwargs):
        with record_node_span(node_name):
            return fn(*args, **kwargs)

    return _wrapped


def build_rob2_graph():
    """Build and compile the LangGraph for RoB 2 assessment."""
    g = StateGraph(RoB2State)

    g.add_node("pdf_ingest", timed_node("pdf_ingest", pdf_ingest_node))
    g.add_node("rct_screener", timed_node("rct_screener", rct_screener_node))
    g.add_node(
        "preliminary_info", timed_node("preliminary_info", preliminary_info_node)
    )
    g.add_node(
        "outcome_resolver", timed_node("outcome_resolver", outcome_resolver_node)
    )
    g.add_node("trial_facts", timed_node("trial_facts", trial_facts_node))
    g.add_node("rag_retrieval", timed_node("rag_retrieval", rag_retrieval_node))
    g.add_node(
        "evidence_packet_builder",
        timed_node("evidence_packet_builder", evidence_packet_builder_node),
    )
    g.add_node(
        "evidence_family_mining",
        timed_node("evidence_family_mining", evidence_family_mining_node),
    )

    g.add_node("domain1_sq", timed_node("domain1_sq", domain1_sq_node))
    g.add_node("domain1_judge", timed_node("domain1_judge", domain1_judge_node))

    g.add_node("domain2_sq12", timed_node("domain2_sq12", domain2_sq12_node))
    g.add_node(
        "domain2_conditional",
        timed_node("domain2_conditional", domain2_conditional_node),
    )
    g.add_node(
        "domain2_analysis", timed_node("domain2_analysis", domain2_analysis_node)
    )
    g.add_node("domain2_judge", timed_node("domain2_judge", domain2_judge_node))

    g.add_node("domain3_sq", timed_node("domain3_sq", domain3_sq_node))
    g.add_node("domain3_judge", timed_node("domain3_judge", domain3_judge_node))

    g.add_node("domain4_sq", timed_node("domain4_sq", domain4_sq_node))
    g.add_node("domain4_judge", timed_node("domain4_judge", domain4_judge_node))

    g.add_node("domain5_sq", timed_node("domain5_sq", domain5_sq_node))
    g.add_node("domain5_judge", timed_node("domain5_judge", domain5_judge_node))

    g.add_node("quote_verifier", timed_node("quote_verifier", quote_verifier_node))
    g.add_node("overall_judge", timed_node("overall_judge", overall_judge_node))
    g.add_node(
        "report_formatter", timed_node("report_formatter", report_formatter_node)
    )

    g.set_entry_point("pdf_ingest")
    g.add_edge("pdf_ingest", "rct_screener")
    g.add_conditional_edges(
        "rct_screener",
        lambda s: "continue" if s["is_rct"] else "stop",
        {"continue": "preliminary_info", "stop": END},
    )
    g.add_edge("preliminary_info", "outcome_resolver")
    g.add_edge("outcome_resolver", "trial_facts")
    g.add_edge("trial_facts", "rag_retrieval")
    g.add_edge("rag_retrieval", "evidence_packet_builder")
    g.add_edge("evidence_packet_builder", "evidence_family_mining")
    for domain_start in DOMAIN_START_NODES:
        g.add_edge("evidence_family_mining", domain_start)

    g.add_edge("domain1_sq", "domain1_judge")
    g.add_edge("domain1_judge", "quote_verifier")

    g.add_conditional_edges(
        "domain2_sq12",
        d2_needs_conditional,
        {"conditional": "domain2_conditional", "analysis": "domain2_analysis"},
    )
    g.add_edge("domain2_conditional", "domain2_analysis")
    g.add_edge("domain2_analysis", "domain2_judge")
    g.add_edge("domain2_judge", "quote_verifier")

    g.add_edge("domain3_sq", "domain3_judge")
    g.add_edge("domain3_judge", "quote_verifier")

    g.add_edge("domain4_sq", "domain4_judge")
    g.add_edge("domain4_judge", "quote_verifier")

    g.add_edge("domain5_sq", "domain5_judge")
    g.add_edge("domain5_judge", "quote_verifier")
    g.add_edge("quote_verifier", "overall_judge")
    g.add_edge("overall_judge", "report_formatter")
    g.add_edge("report_formatter", END)

    return g.compile()
