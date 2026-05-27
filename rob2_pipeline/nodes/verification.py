import re

from rob2_pipeline.models import EVIDENCE_SECTION_FIELDS, format_evidence
from rob2_pipeline.state import RoB2State


_BYPASS_QUOTES = {"", "not applicable", "no relevant text found", "not reported"}
_NA_QUOTES = {"not applicable"}


def _normalize_text(text: str) -> str:
    text = re.sub(
        r"\([^)]*(?:section|primary evidence|additional retrieved context|results|methods)[^)]*\)",
        "",
        text,
        flags=re.I,
    )
    text = text.strip().strip('"').strip("'")
    text = re.sub(r"\s+", " ", text)
    return text.casefold()


def _source_text(state: RoB2State) -> str:
    evidence = state.get("evidence", {})
    parts = [state.get("full_text", "")]
    for field in EVIDENCE_SECTION_FIELDS:
        section = evidence.get(field) if evidence else None
        if section:
            parts.append(format_evidence(section))
    parts.extend((state.get("rag_contexts") or {}).values())
    return _normalize_text("\n\n".join(part for part in parts if part))


def quote_is_supported(quote: str, source_text: str) -> bool:
    return classify_evidence_support(quote, source_text)["status"] in {
        "supported",
        "paraphrase-supported",
        "not-applicable-by-control",
    }


def classify_evidence_support(
    claim: str, source_text: str, provenance_text: str | None = None
) -> dict:
    normalized_claim = _normalize_text(claim)
    if normalized_claim in _NA_QUOTES:
        return {
            "status": "not-applicable-by-control",
            "provenance": {"source_scope": "control_flow"},
        }
    if normalized_claim in _BYPASS_QUOTES:
        return {
            "status": "supported",
            "provenance": {"source_scope": "assessment_context"},
        }

    normalized_source = _normalize_text(source_text)
    provenance_source = (
        _normalize_text(provenance_text) if provenance_text is not None else None
    )
    global_match = _support_match(normalized_claim, normalized_source)
    provenance_match = (
        _support_match(normalized_claim, provenance_source)
        if provenance_source is not None
        else None
    )
    if provenance_source is not None and not provenance_match and global_match:
        return {
            "status": "source-mismatched",
            "provenance": {
                "source_scope": "declared_provenance",
                "matched_elsewhere": True,
            },
        }
    if provenance_match or global_match:
        match = provenance_match or global_match or {}
        return {
            "status": match["status"],
            "provenance": {
                "source_scope": "declared_provenance"
                if provenance_match
                else "assessment_context"
            },
        }
    return {
        "status": "unsupported",
        "provenance": {
            "source_scope": "declared_provenance"
            if provenance_text is not None
            else "assessment_context"
        },
    }


def _support_match(normalized_claim: str, normalized_source: str | None) -> dict | None:
    if not normalized_source:
        return None
    if normalized_claim in normalized_source:
        return {"status": "supported"}
    words = [_word_key(word) for word in re.findall(r"[a-z0-9]+", normalized_claim)]
    words = [word for word in words if len(word) > 3]
    if len(words) < 4:
        return None
    source_words = {
        _word_key(word) for word in re.findall(r"[a-z0-9]+", normalized_source)
    }
    hits = sum(1 for word in words if word in source_words)
    if hits >= 3 and hits / len(words) >= 0.5:
        return {"status": "paraphrase-supported"}
    return None


def _word_key(word: str) -> str:
    for suffix in ("ization", "isation", "ized", "ised", "ally", "ly", "ed", "ing"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _fragile_sq_issue(sq_id: str, answer: dict) -> str | None:
    value = answer.get("answer", "")
    justification = (answer.get("justification") or "").casefold()
    if (
        sq_id == "3.1"
        and value in ("Y", "PY")
        and not re.search(r"\d+\s*/\s*\d+|\d+(?:\.\d+)?\s*%", justification)
    ):
        return "D3 completeness answer lacks a denominator or percentage calculation."
    if (
        sq_id in ("5.2", "5.3")
        and value in ("Y", "PY")
        and "multiple" not in justification
    ):
        return "D5 selective-reporting answer does not identify multiple eligible measurements or analyses."
    return None


def verify_sq_evidence(state: RoB2State) -> list[dict]:
    source = _source_text(state)
    flags = []
    for sq_id, answer in sorted((state.get("sq_answers") or {}).items()):
        quote = answer.get("quote", "")
        support = classify_evidence_support(quote, source)
        if support["status"] in {"unsupported", "source-mismatched"}:
            flags.append(
                {
                    "sq_id": sq_id,
                    "issue": "quote_not_found_in_source_context",
                    "quote": quote,
                    "support_status": support["status"],
                    "provenance": support["provenance"],
                }
            )
        fragile_issue = _fragile_sq_issue(sq_id, answer)
        if fragile_issue:
            flags.append({"sq_id": sq_id, "issue": fragile_issue, "quote": quote})
    flags.extend(verify_packet_evidence(state))
    return flags


def collect_evidence_support_statuses(state: RoB2State) -> list[dict]:
    source = _source_text(state)
    statuses = []
    for sq_id, answer in sorted((state.get("sq_answers") or {}).items()):
        quote = answer.get("quote", "")
        support = classify_evidence_support(quote, source)
        statuses.append(
            {
                "sq_id": sq_id,
                "claim_type": "sq_quote",
                "quote": quote,
                "support_status": support["status"],
                "provenance": support["provenance"],
            }
        )
    for sq_id, facts in sorted((state.get("evidence_facts") or {}).items()):
        for fact in facts:
            statuses.append(
                {
                    "sq_id": sq_id,
                    "claim_type": "packet_fact",
                    "quote": fact.get("quote", ""),
                    "support_status": fact.get("support_status", "unsupported"),
                    "provenance": {
                        "source_scope": "packet_source",
                        "document_id": fact.get("document_id", ""),
                        "document_name": fact.get("document_name", ""),
                        "document_role": fact.get("document_role", ""),
                        "source_kind": fact.get("source_kind", ""),
                        "source_path": fact.get("source_path", ""),
                    },
                }
            )
    return statuses


def verify_packet_evidence(state: RoB2State) -> list[dict]:
    flags = []
    for sq_id, packet in sorted((state.get("evidence_packets") or {}).items()):
        grade = packet.get("packet_grade") or {}
        missing = grade.get("missing_evidence") or packet.get("missing_evidence") or []
        negative_flags = packet.get("negative_flags") or []
        if grade.get("retry_recommended") or missing or negative_flags:
            details = []
            if missing:
                details.append("missing: " + ", ".join(missing))
            if negative_flags:
                details.append("negative_flags: " + ", ".join(negative_flags))
            flags.append(
                {
                    "sq_id": sq_id,
                    "issue": "packet_verification_failed"
                    + (f" ({'; '.join(details)})" if details else ""),
                    "quote": "",
                }
            )
    return flags


def _verification_actions_from_flags(flags: list[dict]) -> list[dict]:
    actions = []
    for flag in flags:
        issue = flag.get("issue", "")
        if "packet_verification_failed" in issue:
            actions.append(
                {
                    "sq_id": flag.get("sq_id", ""),
                    "action": "retry_packet_or_escalate",
                    "reason": issue,
                }
            )
        elif flag.get("issue") == "quote_not_found_in_source_context":
            actions.append(
                {
                    "sq_id": flag.get("sq_id", ""),
                    "action": "retry_sq_with_verified_packet",
                    "reason": issue,
                }
            )
    return actions


def quote_verifier_node(state: RoB2State) -> RoB2State:
    flags = verify_sq_evidence(state)
    support_statuses = collect_evidence_support_statuses(state)
    trace = list(state.get("verifier_trace", []))
    actions = _verification_actions_from_flags(flags)
    if flags:
        trace.append(
            {
                "node": "quote_verifier",
                "action": "flag",
                "reason": f"{len(flags)} evidence validation issue(s) found",
            }
        )
    return {
        "evidence_validation_flags": flags,
        "evidence_support_statuses": support_statuses,
        "verifier_trace": trace,
        "verification_actions": actions,
    }
