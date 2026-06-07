from typing import Literal, TypedDict, cast


class SectionEvidence(TypedDict):
    text: str
    tables: list[str]
    source: str


class PaperEvidence(TypedDict):
    abstract: SectionEvidence
    methods: SectionEvidence
    results: SectionEvidence
    d1_randomization: SectionEvidence
    d2_blinding: SectionEvidence
    d3_missing_data: SectionEvidence
    d4_outcome_meas: SectionEvidence
    d5_registration: SectionEvidence
    consort_flow: SectionEvidence
    baseline_table: SectionEvidence
    extraction_method: Literal[
        "json_contract",
        "structural_keywords",
        "parse_artifact",
        "fallback",
    ] | str
    warnings: list[str]


EVIDENCE_SECTION_FIELDS = (
    "abstract",
    "methods",
    "results",
    "d1_randomization",
    "d2_blinding",
    "d3_missing_data",
    "d4_outcome_meas",
    "d5_registration",
    "consort_flow",
    "baseline_table",
)


def empty_section_evidence(source: str = "") -> SectionEvidence:
    return {"text": "", "tables": [], "source": source}


def empty_paper_evidence(extraction_method: str = "") -> PaperEvidence:
    evidence: dict[str, object] = {
        field: empty_section_evidence() for field in EVIDENCE_SECTION_FIELDS
    }
    evidence["extraction_method"] = extraction_method
    evidence["warnings"] = []
    return cast(PaperEvidence, evidence)


def format_evidence(ev: SectionEvidence) -> str:
    parts = []
    text = ev.get("text", "").strip()
    if text:
        parts.append(text)
    parts.extend(table.strip() for table in ev.get("tables", []) if table.strip())
    return "\n\n".join(parts)
