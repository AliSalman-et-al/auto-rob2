import re
from typing import Protocol, cast

from pydantic import BaseModel, Field

from rob2_pipeline.llm_contracts import call_json_contract_llm
from rob2_pipeline.ingestion.settings import (
    CENSORING_PATTERNS,
    MAX_SECTION_CHARS,
    SECTION_ORDER,
    SECTION_PATTERNS,
)
from rob2_pipeline.models import (
    EVIDENCE_SECTION_FIELDS,
    PaperEvidence,
    SectionEvidence,
    empty_paper_evidence,
)
from rob2_pipeline.types import LLMCallLogEntry


class DocumentRepr(Protocol):
    full_text: str
    blocks: list

    def to_prompt_repr(self) -> str:
        ...


PROMPT_PAPER_EXTRACTION = """
You are a clinical trial analyst. Extract the following content from the paper below.
For each section, return all relevant narrative text AND any tables that belong to it.
If content is not present, return empty strings - do not invent content.

<paper>
{paper}
</paper>

Return only JSON matching PaperEvidenceExtractionArtifact. Each section has
text and tables. Use empty strings/lists for absent content.
""".strip()

PAPER_EXTRACTION_SYSTEM_MESSAGE = (
    "You are an expert systematic reviewer extracting clinical trial evidence. "
    "Respond only with JSON matching the requested schema."
)


class PaperSectionArtifact(BaseModel):
    text: str = ""
    tables: list[str] = Field(default_factory=list)


class PaperEvidenceExtractionArtifact(BaseModel):
    schema_version: str = Field(pattern=r"^paper-evidence-extraction-v1$")
    abstract: PaperSectionArtifact
    methods: PaperSectionArtifact
    results: PaperSectionArtifact
    d1_randomization: PaperSectionArtifact
    d2_blinding: PaperSectionArtifact
    d3_missing_data: PaperSectionArtifact
    d4_outcome_meas: PaperSectionArtifact
    d5_registration: PaperSectionArtifact
    consort_flow: PaperSectionArtifact
    baseline_table: PaperSectionArtifact


def _paper_evidence_from_artifact(artifact: dict, extraction_method: str) -> PaperEvidence:
    evidence = empty_paper_evidence(extraction_method)
    for field in EVIDENCE_SECTION_FIELDS:
        section = artifact.get(field, {})
        cast(dict[str, object], evidence)[field] = cast(
            SectionEvidence,
            {
                "text": str(section.get("text", "")).strip(),
                "tables": [
                    str(table).strip()
                    for table in section.get("tables", [])
                    if str(table).strip()
                ],
                "source": "llm_extract",
            },
        )
    return evidence


def extract_paper_evidence(
    doc_repr: DocumentRepr,
) -> tuple[PaperEvidence, list[LLMCallLogEntry]]:
    prompt = PROMPT_PAPER_EXTRACTION.format(paper=doc_repr.to_prompt_repr())
    result = call_json_contract_llm(
        {},
        prompt,
        "paper_evidence_extraction",
        schema_model=PaperEvidenceExtractionArtifact,
        schema_version="paper-evidence-extraction-v1",
        prompt_version="paper-evidence-extraction-prompt-v1",
        fallback_factory=lambda reason: {
            "schema_version": "paper-evidence-extraction-v1",
            **{
                field: {"text": "", "tables": []}
                for field in EVIDENCE_SECTION_FIELDS
            },
            "fallback_reason": reason,
        },
    )
    evidence = _paper_evidence_from_artifact(result.artifact, "json_contract")
    if result.status == "fallback":
        evidence = extract_structural_paper_evidence(doc_repr)
        evidence["warnings"] = [
            *evidence.get("warnings", []),
            f"LLM JSON evidence extraction failed validation: {result.failure_reason}",
        ]
    return evidence, result.log


def paper_evidence_from_sections(
    sections: dict[str, str],
    extraction_method: str = "fallback",
    source: str = "keyword_fallback",
    warnings: list[str] | None = None,
) -> PaperEvidence:
    evidence = empty_paper_evidence(extraction_method)
    mapping = {
        "abstract": ["abstract"],
        "methods": ["methods"],
        "results": ["results"],
        "d1_randomization": ["randomization", "methods"],
        "d2_blinding": ["blinding", "methods"],
        "d3_missing_data": ["missing_data", "results"],
        "d4_outcome_meas": ["outcomes", "analysis", "results"],
        "d5_registration": ["registration"],
        "consort_flow": ["consort"],
        "baseline_table": ["baseline"],
    }
    for field, section_names in mapping.items():
        text = "\n\n".join(
            sections.get(name, "") for name in section_names if sections.get(name, "")
        ).strip()
        cast(dict[str, object], evidence)[field] = {
            "text": cap_section(text) if text else "",
            "tables": [],
            "source": source,
        }
    evidence["warnings"] = warnings or []
    return evidence


def extract_structural_paper_evidence(doc_repr: DocumentRepr) -> PaperEvidence:
    evidence = paper_evidence_from_sections(
        parse_sections(doc_repr.to_prompt_repr() or doc_repr.full_text),
        extraction_method="structural_keywords",
        source="parser_neutral_sections",
        warnings=[
            "LLM evidence extraction failed; used structural keyword mapping."
        ],
    )
    table_mapping = {
        "baseline_table": SECTION_PATTERNS["baseline"],
        "consort_flow": SECTION_PATTERNS["consort"],
        "results": SECTION_PATTERNS["results"],
        "d4_outcome_meas": SECTION_PATTERNS["outcomes"] + SECTION_PATTERNS["analysis"],
        "d5_registration": SECTION_PATTERNS["registration"],
    }
    for block in getattr(doc_repr, "blocks", []):
        searchable = "\n".join([block.heading or "", block.text, *block.tables]).lower()
        for field, keywords in table_mapping.items():
            if block.tables and any(keyword in searchable for keyword in keywords):
                field_evidence = cast(dict[str, object], evidence)[field]
                tables = cast(
                    list[str], cast(dict[str, object], field_evidence)["tables"]
                )
                tables.extend(table for table in block.tables if table not in tables)
    return evidence


def cap_section(
    text: str,
    max_chars: int = MAX_SECTION_CHARS,
    keywords: list[str] | None = None,
) -> str:
    if len(text) <= max_chars:
        return text
    if keywords is None:
        keywords = [
            "random",
            "allocation",
            "conceal",
            "blind",
            "mask",
            "itt",
            "per-protocol",
            "missing",
            "imputation",
            "outcome",
            "endpoint",
            "register",
        ]
    chunk_size = 2000
    step = 1000
    chunks = []
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size]
        if not chunk:
            break
        score = sum(chunk.lower().count(keyword) for keyword in keywords)
        chunks.append((score, start, chunk))
        if start + chunk_size >= len(text):
            break
    chunks.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    top_chunks = [chunk for score, _, chunk in chunks if chunk and score > 0][:3]
    if not top_chunks:
        top_chunks = [chunk for _, _, chunk in chunks[:3] if chunk]
    marker = "\n[... truncated ...]\n"
    combined = marker.join(top_chunks)
    truncated = combined[:max_chars]
    return (
        truncated
        + f"\n\n[NOTE: Section truncated at {MAX_SECTION_CHARS} characters. Critical content may be absent.]"
    )


def _normalize_heading(line: str) -> str:
    line = line.strip().lower()
    line = re.sub(r"^#{1,6}\s*", "", line)
    line = line.strip("*_` ")
    line = re.sub(r"^\d+(?:\.\d+)*\s*", "", line)
    line = re.sub(r"[:.\s]+$", "", line)
    return line


def _detect_heading(line: str) -> str | None:
    stripped = line.strip()
    if stripped.startswith("|"):
        return None
    normalized = _normalize_heading(line)
    if not normalized or len(normalized) > 120:
        return None
    if stripped.endswith(".") and len(normalized.split()) > 3:
        return None

    for section, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            compact_normalized = normalized.replace(" ", "")
            compact_pattern = pattern.replace(" ", "")
            if pattern in normalized or compact_pattern in compact_normalized:
                return section
    return None


def _extract_keyword_context(full_text: str, section_name: str) -> str:
    keywords = SECTION_PATTERNS[section_name]
    lines = full_text.splitlines()
    windows = []
    seen_ranges = set()

    for index, line in enumerate(lines):
        lowered = line.lower()
        if not any(keyword in lowered for keyword in keywords):
            continue
        start = max(0, index - 3)
        end = min(len(lines), index + 8)
        window_key = (start, end)
        if window_key in seen_ranges:
            continue
        seen_ranges.add(window_key)
        window = "\n".join(lines[start:end]).strip()
        if window:
            windows.append(window)
        if len(windows) >= 8:
            break

    return (
        cap_section("\n\n[... nearby text ...]\n\n".join(windows), keywords=keywords)
        if windows
        else ""
    )


def _augment_consort_from_results(sections: dict) -> dict:
    """When CONSORT section is reference-only, search results/supplementary for flow numbers."""
    import re

    consort = sections.get("consort", "")
    if len(consort) < 300 and re.search(r"fig(?:ure)?|supplement", consort, re.I):
        pattern = re.compile(
            r"(\d[\d,]*)\s+(?:patients?|participants?|subjects?)\s+"
            r"(?:were\s+)?(?:enrolled|randomized|randomised|allocated|included|"
            r"excluded|withdrew|lost|assigned|eligible|screened)",
            re.I,
        )
        extra = []
        for section_name in ("results", "supplementary", "methods"):
            text = sections.get(section_name, "")
            matches = pattern.findall(text[:5000])
            if matches:
                extra.append(
                    f"\n[Patient flow numbers from {section_name} section: "
                    + "; ".join(f"{m} patients" for m in matches[:10])
                    + "]"
                )
        if extra:
            sections["consort"] = consort + "".join(extra)
    return sections


def extract_censoring_context(full_text: str, outcome: str) -> str:
    del outcome  # reserved for future outcome-specific filtering
    lines = full_text.splitlines()
    windows = []
    seen_ranges = set()

    for index, line in enumerate(lines):
        if not line.strip():
            continue
        if not any(pattern.search(line) for pattern in CENSORING_PATTERNS):
            continue
        start = max(0, index - 3)
        end = min(len(lines), index + 4)
        window_key = (start, end)
        if window_key in seen_ranges:
            continue
        seen_ranges.add(window_key)
        window = "\n".join(lines[start:end]).strip()
        if window:
            windows.append(window)
        if len(windows) >= 10:
            break

    if not windows:
        return ""

    return "\n\n[...]\n\n".join(windows)[:2000]


def parse_sections(full_text: str) -> dict[str, str]:
    sections = {name: "" for name in SECTION_ORDER}
    current_section: str | None = None
    buffers: dict[str, list[str]] = {name: [] for name in SECTION_ORDER}

    for raw_line in full_text.splitlines():
        line = raw_line.strip()
        if not line:
            if current_section is not None:
                buffers[current_section].append("")
            continue

        detected = _detect_heading(line)
        if detected is not None:
            current_section = detected
            buffers[current_section].append(line)
            continue

        if current_section is not None:
            buffers[current_section].append(raw_line)

    for name, lines in buffers.items():
        sections[name] = cap_section("\n".join(lines).strip()) if lines else ""

    if sections["methods"]:
        if not sections["randomization"]:
            sections["randomization"] = sections["methods"]
        if not sections["blinding"]:
            sections["blinding"] = sections["methods"]

    for name in (
        "randomization",
        "blinding",
        "outcomes",
        "analysis",
        "missing_data",
        "registration",
        "baseline",
        "consort",
        "supplementary",
    ):
        if not sections[name]:
            sections[name] = _extract_keyword_context(full_text, name)

    sections = _augment_consort_from_results(sections)

    return sections
