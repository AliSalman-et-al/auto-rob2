"""Evidence packet contracts and shared matching constants."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceContract:
    sq_id: str
    domain: str
    required_evidence: tuple[str, ...]
    terms: tuple[str, ...]
    fallback_sections: tuple[str, ...] = ()
    needs_denominator: bool = False
    outcome_bound: bool = False
    needs_prespecification: bool = False
    # Each group is a set of terms the packet must cover with at least one
    # selected source when such a source exists, regardless of matched-term
    # rank. This guarantees an SQ that needs two distinct kinds of evidence
    # gets both, rather than letting the higher-ranking kind take every slot.
    coverage_groups: tuple[tuple[str, ...], ...] = ()


# SQ 5.3 ("was the result selected from multiple eligible analyses?") can only
# be answered N/PN when the model sees BOTH the pre-specified analysis plan
# ("what was planned") and the reported analysis methods ("what was done") and
# can check they match. These two coverage groups reserve one packet slot for
# each half so reported-methods sources cannot crowd the plan out (or vice
# versa). REPORTED_METHODS_TERMS is also reused as 5.3 matching terms so the
# best methods source ranks high within its group.
# High-precision only: coverage reserves a packet slot for a matching source,
# so a false match seats junk as the "plan" witness. Short ambiguous substrings
# are deliberately excluded ("nct" matches "function"/"junction"; "sap" matches
# "sapphire"); "registr" already covers registered/registration/registry.
PRESPECIFICATION_COVERAGE_TERMS = (
    "registr",
    "protocol",
    "statistical analysis plan",
    "pre-spec",
    "prespec",
    "clinicaltrials",
)
REPORTED_METHODS_TERMS = (
    "cox",
    "kaplan",
    "proportional",
    "log-rank",
    "log rank",
    "restricted mean",
)


CONTRACTS: dict[str, EvidenceContract] = {
    "1.1": EvidenceContract(
        "1.1",
        "d1",
        ("sequence_generation",),
        ("random", "sequence", "computer", "block", "stratif", "minimi", "assign"),
        ("d1_randomization", "methods"),
    ),
    "1.2": EvidenceContract(
        "1.2",
        "d1",
        ("allocation_concealment", "enrolment_timing"),
        (
            "conceal",
            "central",
            "web",
            "telephone",
            "pharmacy",
            "enrol",
            "assigned",
            "allocation",
        ),
        ("d1_randomization", "methods"),
    ),
    "1.3": EvidenceContract(
        "1.3",
        "d1",
        ("baseline_balance",),
        ("baseline", "characteristic", "imbalance", "prognostic", "table"),
        ("baseline_table", "results"),
    ),
    "2.1": EvidenceContract(
        "2.1",
        "d2",
        ("participant_awareness",),
        ("blind", "mask", "open-label", "open label", "aware"),
        ("d2_blinding", "methods"),
    ),
    "2.2": EvidenceContract(
        "2.2",
        "d2",
        ("personnel_awareness",),
        ("blind", "mask", "open-label", "investigator", "personnel", "carer"),
        ("d2_blinding", "methods"),
    ),
    "2.3": EvidenceContract(
        "2.3",
        "d2",
        ("trial_context_deviations",),
        (
            "deviation",
            "protocol",
            "amend",
            "cross-over",
            "adherence",
            "standard of care",
        ),
        ("d2_blinding", "results", "methods"),
    ),
    "2.4": EvidenceContract(
        "2.4",
        "d2",
        ("deviation_outcome_impact",),
        ("deviation", "affected", "outcome", "rescue", "benefit", "lack of effect"),
        ("results", "methods"),
    ),
    "2.5": EvidenceContract(
        "2.5",
        "d2",
        ("deviation_balance",),
        ("balanced", "between groups", "arm", "deviation", "adherence"),
        ("results", "methods"),
    ),
    "2.6": EvidenceContract(
        "2.6",
        "d2",
        ("analysis_population",),
        ("intention", "itt", "modified", "per-protocol", "as treated", "randomized"),
        ("results", "d4_outcome_meas", "methods"),
    ),
    "2.7": EvidenceContract(
        "2.7",
        "d2",
        ("analysis_failure_impact",),
        ("excluded", "randomized", "substantial", "impact", "analysis population"),
        ("results", "methods"),
    ),
    "3.1": EvidenceContract(
        "3.1",
        "d3",
        ("randomized_n", "outcome_data_n"),
        (
            "randomized",
            "randomised",
            "outcome data",
            "missing",
            "follow-up",
            "analysed",
            "analyzed",
        ),
        ("d3_missing_data", "consort_flow", "results"),
        needs_denominator=True,
    ),
    "3.2": EvidenceContract(
        "3.2",
        "d3",
        ("missing_bias_evidence",),
        ("missing", "sensitivity", "imputation", "lost", "withdraw", "reason"),
        ("d3_missing_data", "results"),
    ),
    "3.3": EvidenceContract(
        "3.3",
        "d3",
        ("missingness_true_value",),
        ("missing", "reason", "withdraw", "progression", "toxicity", "lost"),
        ("d3_missing_data", "results"),
    ),
    "3.4": EvidenceContract(
        "3.4",
        "d3",
        ("likely_informative_missingness",),
        ("censor", "sensitivity", "switch", "salvage", "second-line", "dropout"),
        ("d3_missing_data", "results"),
    ),
    "4.1": EvidenceContract(
        "4.1",
        "d4",
        ("measurement_method",),
        ("outcome", "measure", "definition", "endpoint", "instrument", "criteria"),
        ("d4_outcome_meas", "methods"),
        outcome_bound=True,
    ),
    "4.2": EvidenceContract(
        "4.2",
        "d4",
        ("between_group_measurement_difference",),
        ("differ", "same", "schedule", "assess", "visit", "criteria", "method"),
        ("d4_outcome_meas", "methods"),
        outcome_bound=True,
    ),
    "4.3": EvidenceContract(
        "4.3",
        "d4",
        ("assessor_awareness",),
        ("assessor", "blind", "mask", "open-label", "adjudicat", "central review"),
        ("d2_blinding", "d4_outcome_meas"),
        outcome_bound=True,
    ),
    "4.4": EvidenceContract(
        "4.4",
        "d4",
        ("assessment_could_be_influenced",),
        ("judg", "subjective", "symptom", "radiographic", "clinical", "mortality"),
        ("d4_outcome_meas", "methods"),
        outcome_bound=True,
    ),
    "4.5": EvidenceContract(
        "4.5",
        "d4",
        ("assessment_likely_influenced",),
        ("belief", "influence", "independent", "blinded", "adjudication", "standard"),
        ("d4_outcome_meas", "d2_blinding"),
        outcome_bound=True,
    ),
    "5.1": EvidenceContract(
        "5.1",
        "d5",
        ("prespecified_analysis_plan",),
        (
            "registr",
            "protocol",
            "sap",
            "statistical analysis plan",
            "pre-spec",
            "prespec",
            "clinicaltrials",
            "nct",
        ),
        ("d5_registration", "d4_outcome_meas", "methods"),
        outcome_bound=True,
        needs_prespecification=True,
    ),
    "5.2": EvidenceContract(
        "5.2",
        "d5",
        ("eligible_outcome_measurements", "assessed_outcome_binding"),
        (
            "primary",
            "secondary",
            "endpoint",
            "outcome",
            "definition",
            "time point",
            "protocol",
            "registr",
        ),
        ("d5_registration", "d4_outcome_meas", "results"),
        outcome_bound=True,
    ),
    "5.3": EvidenceContract(
        "5.3",
        "d5",
        ("eligible_analyses", "assessed_result_binding"),
        (
            "analysis",
            "subgroup",
            "adjusted",
            "unadjusted",
            "sap",
            "statistical analysis plan",
            "itt",
            # Reported analysis-methods lexicon ranks the survival-methods
            # sentence (Cox / Kaplan-Meier / log-rank) high within its coverage
            # group. High-specificity tokens only; results-collision tokens
            # ("hazard ratio", "stratified") and double-edged ones
            # ("sensitivity", "covariate") are deliberately excluded.
            *REPORTED_METHODS_TERMS,
        ),
        ("d5_registration", "d4_outcome_meas", "results"),
        outcome_bound=True,
        coverage_groups=(PRESPECIFICATION_COVERAGE_TERMS, REPORTED_METHODS_TERMS),
    ),
}


RESULT_STAT_RE = re.compile(
    r"\b(?:hr|or|rr|hazard ratio|p\s*=|confidence interval|ci\b|\d+(?:\.\d+)?\s*%)",
    re.I,
)
DENOMINATOR_RE = re.compile(
    r"\b\d[\d,]*\s*/\s*\d[\d,]*\b"
    r"|\b\d+(?:\.\d+)?\s*%"
    r"|\b\d[\d,]*\s+(?:participants|patients)\b.{0,80}\b"
    r"(?:all\s+had\s+outcome\s+data|no\s+missing\s+outcome\s+data)\b",
    re.I,
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
PRESPEC_TERMS = (
    "registr",
    "protocol",
    "sap",
    "statistical analysis plan",
    "pre-spec",
    "prespec",
    "nct",
)
OUTCOME_ALIASES = {
    "overall survival": ("overall survival", "os", "death", "mortality"),
    "progression-free survival": (
        "progression-free survival",
        "progression free survival",
        "pfs",
        "progression",
    ),
    "adverse events": (
        "adverse event",
        "adverse events",
        "toxicity",
        "safety",
        "grade",
    ),
}
