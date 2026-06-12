from dataclasses import dataclass

from rob2_pipeline.models import format_evidence
from rob2_pipeline.nodes.evidence_packets import packet_block_for_domain
from rob2_pipeline.state import RoB2State


@dataclass(frozen=True)
class Domain1Context:
    randomization_text: str
    baseline_text: str
    consort_text: str
    rag_text: str
    ctgov_design: str


@dataclass(frozen=True)
class Domain2Sq12Context:
    blinding_text: str
    methods_text: str
    rag_text: str
    ctgov_design: str


@dataclass(frozen=True)
class Domain2ConditionalContext:
    sq_2_1: str
    sq_2_2: str
    deviations_text: str
    concomitant_text: str
    rag_text: str


@dataclass(frozen=True)
class Domain2AnalysisContext:
    effect_of_interest: str
    analysis_text: str
    results_text: str
    rag_text: str


@dataclass(frozen=True)
class Domain3Context:
    n_randomized: str
    consort_text: str
    missing_data_text: str
    sensitivity_text: str
    rag_text: str
    ctgov_flow: str


@dataclass(frozen=True)
class Domain4Context:
    outcome_type: str
    sq_2_1: str
    outcome_measurement_text: str
    blinding_text: str
    rag_text: str


@dataclass(frozen=True)
class Domain5Context:
    outcome_type: str
    numerical_result: str
    registration_number: str
    registered_endpoint: str
    registered_secondary_endpoints: str
    reported_endpoint: str
    ctgov_outcomes: str
    ctgov_description: str
    registration_text: str
    sap_text: str
    results_text: str
    rag_text: str


def _join_nonempty(parts: list[str], separator: str = "\n\n") -> str:
    return separator.join(part for part in parts if part)


def _packet_text(state: RoB2State, domain: str) -> str:
    return packet_block_for_domain(state.get("evidence_packets", {}), domain)


def build_domain1_context(state: RoB2State) -> Domain1Context:
    evidence = state["evidence"]
    trial_facts = state.get("trial_facts", {})
    trial_level_text = _join_nonempty(
        [
            trial_facts.get("randomization", ""),
            trial_facts.get("allocation_concealment", ""),
        ],
    )
    packet_text = _packet_text(state, "d1")
    return Domain1Context(
        randomization_text=_join_nonempty(
            [
                format_evidence(evidence["d1_randomization"])
                or format_evidence(evidence["methods"]),
                trial_level_text,
            ]
        ),
        baseline_text=format_evidence(evidence["baseline_table"]),
        consort_text=format_evidence(evidence["consort_flow"]),
        rag_text=packet_text,
        ctgov_design=state.get(
            "ctgov_design", "(No ClinicalTrials.gov design metadata available)"
        ),
    )


def build_domain2_sq12_context(state: RoB2State) -> Domain2Sq12Context:
    evidence = state["evidence"]
    trial_facts = state.get("trial_facts", {})
    packet_text = _packet_text(state, "d2")
    return Domain2Sq12Context(
        blinding_text=_join_nonempty(
            [
                format_evidence(evidence["d2_blinding"]),
                trial_facts.get("masking", ""),
            ]
        ),
        methods_text=format_evidence(evidence["methods"]),
        rag_text=packet_text,
        ctgov_design=state.get(
            "ctgov_design", "(No ClinicalTrials.gov design metadata available)"
        ),
    )


def build_domain2_conditional_context(
    state: RoB2State,
) -> Domain2ConditionalContext:
    evidence = state["evidence"]
    trial_facts = state.get("trial_facts", {})
    packet_text = _packet_text(state, "d2")
    sq = state["sq_answers"]
    return Domain2ConditionalContext(
        sq_2_1=sq.get("2.1", {}).get("answer", "NI"),
        sq_2_2=sq.get("2.2", {}).get("answer", "NI"),
        deviations_text=_join_nonempty(
            [
                format_evidence(evidence["d2_blinding"]),
                format_evidence(evidence["results"]),
                trial_facts.get("protocol_deviations", ""),
                trial_facts.get("protocol_amendments", ""),
            ],
            separator="\n",
        ),
        concomitant_text=format_evidence(evidence["methods"]),
        rag_text=packet_text,
    )


def build_domain2_analysis_context(state: RoB2State) -> Domain2AnalysisContext:
    evidence = state["evidence"]
    trial_facts = state.get("trial_facts", {})
    packet_text = _packet_text(state, "d2")
    return Domain2AnalysisContext(
        effect_of_interest=state.get("effect_of_interest", "ITT"),
        analysis_text=format_evidence(evidence["d4_outcome_meas"]),
        results_text=_join_nonempty(
            [
                format_evidence(evidence["results"]),
                trial_facts.get("analysis_populations", ""),
            ]
        ),
        rag_text=packet_text,
    )


def build_domain3_context(state: RoB2State) -> Domain3Context:
    evidence = state["evidence"]
    packet_text = _packet_text(state, "d3")
    missing_data_text = format_evidence(evidence["d3_missing_data"]) or format_evidence(
        evidence["results"]
    )
    return Domain3Context(
        n_randomized=state.get("n_randomized", "Not reported"),
        consort_text=format_evidence(evidence["consort_flow"]),
        missing_data_text=missing_data_text,
        sensitivity_text=format_evidence(evidence["d4_outcome_meas"]),
        rag_text=packet_text,
        ctgov_flow=state.get(
            "ctgov_flow", "(No ClinicalTrials.gov participant flow available)"
        ),
    )


def build_domain4_context(state: RoB2State) -> Domain4Context:
    evidence = state["evidence"]
    packet_text = _packet_text(state, "d4")
    return Domain4Context(
        outcome_type=state.get("outcome_type", "clinician-composite"),
        sq_2_1=state.get("sq_answers", {}).get("2.1", {}).get("answer", "NI"),
        outcome_measurement_text=format_evidence(evidence["d4_outcome_meas"])
        or format_evidence(evidence["methods"]),
        blinding_text=format_evidence(evidence["d2_blinding"]),
        rag_text=packet_text,
    )


def build_domain5_context(state: RoB2State) -> Domain5Context:
    evidence = state["evidence"]
    packet_text = _packet_text(state, "d5")
    return Domain5Context(
        outcome_type=state.get("outcome_type", "clinician-composite"),
        numerical_result=state.get("numerical_result", "Not reported"),
        registration_number=state.get("registration_number", "Not reported"),
        registered_endpoint=state.get("registered_endpoint", "Not reported"),
        registered_secondary_endpoints=state.get(
            "registered_secondary_endpoints", "Not reported"
        ),
        reported_endpoint=state.get("outcome", "Not reported"),
        ctgov_outcomes=state.get("ctgov_outcomes", ""),
        ctgov_description=state.get(
            "ctgov_description", "(No ClinicalTrials.gov description available)"
        ),
        registration_text=format_evidence(evidence["d5_registration"]),
        sap_text=format_evidence(evidence["d4_outcome_meas"]),
        results_text=format_evidence(evidence["results"]),
        rag_text=packet_text,
    )
