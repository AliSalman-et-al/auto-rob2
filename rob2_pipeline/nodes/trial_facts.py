import re

from rob2_pipeline.models import format_evidence
from rob2_pipeline.state import RoB2State
from rob2_pipeline.types import MaskingFact, MaskingFacts


_FACT_PATTERNS = {
    "randomization": re.compile(
        r"\b(randomi[sz]\w*|minimi[sz]\w*|allocation sequence|assigned)\b", re.I
    ),
    "allocation_concealment": re.compile(
        r"\b(conceal|central|sealed|envelope|telephone|web|interactive|accessed only|not disclosed)\b",
        re.I,
    ),
    "masking": re.compile(r"\b(mask|blind|open[- ]label|aware|unaware)\b", re.I),
    "protocol_deviations": re.compile(
        r"\b(deviation|non[- ]adherence|cross[- ]over|contamination|withdraw|discontinu)\b",
        re.I,
    ),
    "protocol_amendments": re.compile(
        r"\b(protocol amendment|amended|modification|standard of care|standard-of-care|regulatory approval)\b",
        re.I,
    ),
    "analysis_populations": re.compile(
        r"\b(intention[- ]to[- ]treat|ITT|modified intention|per[- ]protocol|as treated|safety population|analysis population)\b",
        re.I,
    ),
}


def _sentences(text: str) -> list[str]:
    compact = re.sub(r"\s+", " ", text).strip()
    return [
        part.strip() for part in re.split(r"(?<=[.!?])\s+", compact) if part.strip()
    ]


def _matching_snippets(text: str, pattern: re.Pattern, limit: int = 3) -> str:
    snippets = []
    for sentence in _sentences(text):
        if pattern.search(sentence):
            snippets.append(sentence)
        if len(snippets) >= limit:
            break
    return " ".join(snippets)


_OPEN_LABEL_RE = re.compile(
    r"\b(open[- ]label|non[- ]blinded|unblinded|not masked|not blinded|masking:\s*none)\b",
    re.I,
)
_AWARE_RE = re.compile(
    r"\b(aware|knew|knowledge of).{0,60}\b(assign|treatment|group|intervention)\b", re.I
)
_BLINDED_RE = re.compile(r"\b(blinded|masked|double[- ]blind|single[- ]blind)\b", re.I)
_PARTICIPANT_RE = re.compile(r"\b(participant|patient|subject)s?\b", re.I)
_PERSONNEL_RE = re.compile(
    r"\b(investigator|personnel|clinician|physician|staff|carer|provider)s?\b", re.I
)
_ASSESSOR_RE = re.compile(
    r"\b(outcome assessor|assessor|evaluator|reviewer|radiologist|investigator)s?\b",
    re.I,
)
_ADJUDICATION_RE = re.compile(
    r"\b(blinded|masked|independent|central).{0,80}\b(adjudication|committee|review|assessor)\b",
    re.I,
)


def _should_set_awareness(existing: MaskingFact, has_open_label: bool) -> bool:
    return existing.get("status") == "unclear" or (
        has_open_label
        and "open"
        not in (existing.get("quotes") or [{}])[0].get("quote", "").casefold()
    )


def _source_texts(state: RoB2State) -> list[tuple[str, str, str]]:
    evidence = state.get("evidence", {})
    parts = [
        (
            "primary",
            "paper_evidence:d2_blinding",
            format_evidence(evidence.get("d2_blinding", {})),
        ),
        (
            "primary",
            "paper_evidence:methods",
            format_evidence(evidence.get("methods", {})),
        ),
        (
            "primary",
            "paper_evidence:d4_outcome_meas",
            format_evidence(evidence.get("d4_outcome_meas", {})),
        ),
        ("registry", "ctgov_design", state.get("ctgov_design", "")),
        (
            "source_text",
            "rag_contexts:d2_blinding",
            state.get("rag_contexts", {}).get("d2_blinding", ""),
        ),
        (
            "source_text",
            "rag_contexts:d4_assessor",
            state.get("rag_contexts", {}).get("d4_assessor", ""),
        ),
        (
            "source_text",
            "rag_contexts:d4_measurement",
            state.get("rag_contexts", {}).get("d4_measurement", ""),
        ),
    ]
    return [(kind, label, text) for kind, label, text in parts if text]


def _quote(sentence: str, source_kind: str, source_label: str) -> list[dict[str, str]]:
    return [
        {
            "quote": sentence,
            "source_kind": source_kind,
            "source_label": source_label,
        }
    ]


def _fact(
    status: str,
    source_strength: str = "none",
    sentence: str = "",
    source_kind: str = "",
    source_label: str = "",
) -> MaskingFact:
    fact: MaskingFact = {
        "status": status,
        "source_strength": source_strength,
        "quotes": [],
    }
    if sentence:
        fact["quotes"] = _quote(sentence, source_kind, source_label)
    return fact


def extract_masking_facts(state: RoB2State) -> MaskingFacts:
    facts: MaskingFacts = {
        "participant_awareness": _fact("unclear"),
        "personnel_awareness": _fact("unclear"),
        "outcome_assessor_awareness": _fact("unclear"),
        "blinded_adjudication": _fact("unclear"),
    }

    for source_kind, source_label, text in _source_texts(state):
        for sentence in _sentences(text):
            has_open_label = bool(_OPEN_LABEL_RE.search(sentence))
            has_awareness = bool(_AWARE_RE.search(sentence))
            has_blinding = bool(_BLINDED_RE.search(sentence))

            if _ADJUDICATION_RE.search(sentence):
                facts["blinded_adjudication"] = _fact(
                    "present", source_kind, sentence, source_kind, source_label
                )

            if has_open_label or has_awareness:
                if (
                    has_open_label or _PARTICIPANT_RE.search(sentence)
                ) and _should_set_awareness(
                    facts["participant_awareness"], has_open_label
                ):
                    facts["participant_awareness"] = _fact(
                        "aware", source_kind, sentence, source_kind, source_label
                    )
                if (
                    has_open_label or _PERSONNEL_RE.search(sentence)
                ) and _should_set_awareness(
                    facts["personnel_awareness"], has_open_label
                ):
                    facts["personnel_awareness"] = _fact(
                        "aware", source_kind, sentence, source_kind, source_label
                    )
                if (
                    has_open_label or _ASSESSOR_RE.search(sentence)
                ) and _should_set_awareness(
                    facts["outcome_assessor_awareness"], has_open_label
                ):
                    facts["outcome_assessor_awareness"] = _fact(
                        "aware", source_kind, sentence, source_kind, source_label
                    )

            if has_blinding and not has_open_label:
                if _PARTICIPANT_RE.search(sentence):
                    facts["participant_awareness"] = _fact(
                        "unaware", source_kind, sentence, source_kind, source_label
                    )
                if _PERSONNEL_RE.search(sentence):
                    facts["personnel_awareness"] = _fact(
                        "unaware", source_kind, sentence, source_kind, source_label
                    )
                if _ASSESSOR_RE.search(sentence):
                    facts["outcome_assessor_awareness"] = _fact(
                        "unaware", source_kind, sentence, source_kind, source_label
                    )

    if facts["blinded_adjudication"]["status"] == "unclear":
        text = " ".join(text for _, _, text in _source_texts(state))
        if _OPEN_LABEL_RE.search(text) and not _ADJUDICATION_RE.search(text):
            facts["blinded_adjudication"] = _fact("absent", "inferred")

    return facts


def extract_trial_facts(state: RoB2State) -> dict[str, str]:
    evidence = state.get("evidence", {})
    text = "\n\n".join(
        part
        for part in [
            format_evidence(evidence.get("d1_randomization", {})),
            format_evidence(evidence.get("d2_blinding", {})),
            format_evidence(evidence.get("d5_registration", {})),
            format_evidence(evidence.get("methods", {})),
            format_evidence(evidence.get("results", {})),
        ]
        if part
    )
    facts = {
        name: _matching_snippets(text, pattern)
        for name, pattern in _FACT_PATTERNS.items()
    }
    facts["source"] = "paper_evidence"
    facts["masking_facts"] = extract_masking_facts(state)
    return facts


def trial_facts_node(state: RoB2State) -> RoB2State:
    facts = extract_trial_facts(state)
    return {"trial_facts": facts, "masking_facts": facts["masking_facts"]}
