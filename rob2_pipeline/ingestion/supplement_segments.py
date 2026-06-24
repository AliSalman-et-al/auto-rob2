from __future__ import annotations

import re
from dataclasses import dataclass, replace

from rob2_pipeline.ingestion.parse_artifacts import SourceParseArtifact
from rob2_pipeline.providers.base import ContentBlock
from rob2_pipeline.supplement_retrieval import SupplementIndex, SupplementSegment
from rob2_pipeline.types import SourceDocument, SupplementSegmentArtifact


ALL_ROB2_DOMAINS = ["D1", "D2", "D3", "D4", "D5"]
MIN_STRUCTURAL_SEGMENTS = 3
ANNOTATION_CAP_PER_SUPPLEMENT = 24
FALLBACK_ANNOTATION = "No risk-of-bias relevant content."

DOCUMENT_ROLE_LEXICONS = {
    "sap": (
        "statistical analysis plan",
        "sap",
        "analysis plan",
    ),
    "protocol": (
        "clinical trial protocol",
        "study protocol",
        "trial protocol",
        "protocol",
    ),
    "appendix": (
        "appendix",
        "supplementary appendix",
        "supplemental appendix",
        "supplementary material",
        "supplementary materials",
    ),
}

DOMAIN_LEXICONS = {
    "D1": (
        "randomisation",
        "randomization",
        "allocation",
        "concealment",
        "sequence",
    ),
    "D2": (
        "blinding",
        "masking",
        "deviation",
        "adherence",
        "intention-to-treat",
        "per-protocol",
        "analysis population",
    ),
    "D3": (
        "missing",
        "withdrawal",
        "dropout",
        "lost to follow",
        "incomplete",
        "denominator",
    ),
    "D4": (
        "outcome",
        "assessment",
        "assessor",
        "adjudication",
        "measurement",
        "blinding",
        "masking",
    ),
    "D5": (
        "statistical analysis",
        "analysis plan",
        "prespecified",
        "pre-specified",
        "endpoint",
        "primary outcome",
        "selective reporting",
    ),
}


@dataclass(frozen=True)
class SupplementIngestionArtifacts:
    source_document: SourceDocument
    segments: list[SupplementSegment]
    index: SupplementIndex | None
    warnings: list[str]


def build_supplement_ingestion_artifacts(
    artifact: SourceParseArtifact,
) -> SupplementIngestionArtifacts:
    source = _source_with_content_detected_role(artifact)
    segments, warnings = _segments_from_artifact(artifact, source)
    index = SupplementIndex.from_segments(segments) if segments else None
    return SupplementIngestionArtifacts(
        source_document=source,
        segments=segments,
        index=index,
        warnings=warnings,
    )


def supplement_segment_artifacts(
    segments: list[SupplementSegment],
) -> list[SupplementSegmentArtifact]:
    return [SupplementSegmentArtifact(**segment.to_dict()) for segment in segments]


def supplement_annotation_user_blocks(
    segment: SupplementSegment,
    *,
    document_preamble: str,
) -> list[ContentBlock]:
    segment_text = (
        "Annotate this supplement segment for risk-of-bias retrieval.\n\n"
        f"Document: {segment.document_name}\n"
        f"Role: {segment.document_role}\n"
        f"Heading: {segment.heading}\n"
        f"Pages: {', '.join(str(page) for page in segment.page_numbers) or 'unknown'}\n"
        f"Domain tags: {', '.join(segment.domain_tags) or 'none'}\n\n"
        f"{segment.text}"
    )
    return [
        {
            "type": "text",
            "text": document_preamble,
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": segment_text},
    ]


def _source_with_content_detected_role(
    artifact: SourceParseArtifact,
) -> SourceDocument:
    source = SourceDocument(**artifact.source_identity)
    detected_role = _detect_document_role_from_headers(artifact)
    if detected_role in {"sap", "protocol", "appendix"}:
        source["document_role"] = detected_role
    return source


def _detect_document_role_from_headers(artifact: SourceParseArtifact) -> str:
    headers = [
        _normalize_text(box.get("text", ""))
        for page in artifact.pages[:10]
        for box in page.get("section_header_boxes", [])
        if box.get("text")
    ]
    for role, terms in DOCUMENT_ROLE_LEXICONS.items():
        if any(term in header for header in headers for term in terms):
            return role
    return "unknown_supplement"


def _segments_from_artifact(
    artifact: SourceParseArtifact,
    source: SourceDocument,
) -> tuple[list[SupplementSegment], list[str]]:
    structural_segments = _structural_segments(artifact, source)
    if len(structural_segments) < MIN_STRUCTURAL_SEGMENTS:
        page_segments = _page_segments(artifact.pages, source)
        structural_segments = page_segments or [_full_document_segment(artifact, source)]

    annotation_allowed, warnings = _annotation_allowed_by_segment_id(
        structural_segments, source
    )
    return [
        _annotated_segment(
            segment,
            force_fallback=segment.segment_id not in annotation_allowed,
        )
        for segment in structural_segments
    ], warnings


def _structural_segments(
    artifact: SourceParseArtifact,
    source: SourceDocument,
) -> list[SupplementSegment]:
    all_segments: list[SupplementSegment] = []
    for page in artifact.pages:
        page_number = int(page.get("page_number", 0))
        headers = [
            str(box.get("text", "")).strip()
            for box in page.get("section_header_boxes", [])
        ]
        if not headers:
            continue
        page_segments = _split_page_text_at_headings(
            page_text=page.get("text", ""),
            headings=headers,
        )
        for heading, text in page_segments:
            if not text.strip():
                continue
            segment_number = len(all_segments) + 1
            domain_tags = _domain_tags(heading, text)
            all_segments.append(
                SupplementSegment(
                    segment_id=(
                        f"{source.get('document_id', 'supplement')}:segment:"
                        f"{segment_number:04d}"
                    ),
                    document_id=source.get("document_id", ""),
                    document_name=source.get("document_name", ""),
                    document_role=source.get("document_role", "unknown_supplement"),
                    source_path=source.get("path", ""),
                    heading=heading,
                    page_numbers=[page_number] if page_number else [],
                    domain_tags=domain_tags,
                    annotation="",
                    text=text,
                )
            )
    return all_segments


def _split_page_text_at_headings(
    *,
    page_text: str,
    headings: list[str],
) -> list[tuple[str, str]]:
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    heading_lookup = {_normalize_text(heading): heading for heading in headings}
    segments: list[tuple[str, list[str]]] = []
    current_heading = ""
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        if current_heading:
            segments.append((current_heading, current_lines))
        current_lines = []

    for line in lines:
        heading = heading_lookup.get(_normalize_text(line))
        if heading is not None:
            flush()
            current_heading = heading
            continue
        current_lines.append(line)
    flush()
    return [(heading, "\n".join(body).strip()) for heading, body in segments]


def _full_document_segment(
    artifact: SourceParseArtifact,
    source: SourceDocument,
) -> SupplementSegment:
    text = "\n\n".join(
        page.get("text", "").strip()
        for page in artifact.pages
        if page.get("text", "").strip()
    ).strip()
    page_numbers = [
        int(page.get("page_number", 0))
        for page in artifact.pages
        if int(page.get("page_number", 0))
    ]
    return SupplementSegment(
        segment_id=f"{source.get('document_id', 'supplement')}:segment:0001",
        document_id=source.get("document_id", ""),
        document_name=source.get("document_name", ""),
        document_role=source.get("document_role", "unknown_supplement"),
        source_path=source.get("path", ""),
        heading="Full document",
        page_numbers=sorted(set(page_numbers)),
        domain_tags=list(ALL_ROB2_DOMAINS),
        annotation="",
        text=text,
    )


def _page_segments(
    pages: list[dict],
    source: SourceDocument,
) -> list[SupplementSegment]:
    segments: list[SupplementSegment] = []
    for page in pages:
        text = str(page.get("text", "")).strip()
        if not text:
            continue
        page_number = int(page.get("page_number", 0) or 0)
        segment_number = len(segments) + 1
        segments.append(
            SupplementSegment(
                segment_id=(
                    f"{source.get('document_id', 'supplement')}:segment:"
                    f"{segment_number:04d}"
                ),
                document_id=source.get("document_id", ""),
                document_name=source.get("document_name", ""),
                document_role=source.get("document_role", "unknown_supplement"),
                source_path=source.get("path", ""),
                heading=f"Page {page_number}" if page_number else "Page",
                page_numbers=[page_number] if page_number else [],
                domain_tags=list(ALL_ROB2_DOMAINS),
                annotation="",
                text=text,
            )
        )
    return segments


def _annotation_allowed_by_segment_id(
    segments: list[SupplementSegment],
    source: SourceDocument,
) -> tuple[set[str], list[str]]:
    if len(segments) <= ANNOTATION_CAP_PER_SUPPLEMENT:
        return {segment.segment_id for segment in segments}, []
    tagged = [segment for segment in segments if segment.domain_tags]
    untagged = [segment for segment in segments if not segment.domain_tags]
    allowed = {
        segment.segment_id
        for segment in (tagged + untagged)[:ANNOTATION_CAP_PER_SUPPLEMENT]
    }
    warning = (
        f"Supplement {source.get('document_name', '')} exceeded annotation cap; "
        "remaining segments received fallback annotations."
    )
    return allowed, [warning]


def _annotated_segment(
    segment: SupplementSegment,
    *,
    force_fallback: bool = False,
) -> SupplementSegment:
    if force_fallback:
        return replace(segment, annotation=FALLBACK_ANNOTATION)
    annotation = _annotation_for_segment(segment)
    if annotation:
        return replace(segment, annotation=annotation)
    return replace(segment, annotation=FALLBACK_ANNOTATION)


def _annotation_for_segment(segment: SupplementSegment) -> str:
    tags = ", ".join(segment.domain_tags)
    if tags:
        return f"Potential RoB 2 evidence for {tags}: {_first_sentence(segment.text)}"
    return FALLBACK_ANNOTATION


def _domain_tags(heading: str, text: str) -> list[str]:
    haystack = _normalize_text(f"{heading} {text[:300]}")
    return [
        domain
        for domain, terms in DOMAIN_LEXICONS.items()
        if any(term in haystack for term in terms)
    ]


def _first_sentence(text: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return FALLBACK_ANNOTATION
    sentence = re.split(r"(?<=[.!?])\s+", compact, maxsplit=1)[0]
    return sentence[:240]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()
