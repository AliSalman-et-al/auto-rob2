import re
from typing import Optional

from lxml import etree  # type: ignore[import-untyped]


VALID_ANSWERS = {"Y", "PY", "PN", "N", "NI", "NA"}
VALID_SUPPORT_LEVELS = {"strong", "moderate", "weak", "unsupported"}
ANSWER_MAPPING = {
    "YES": "Y",
    "NO": "N",
    "PROBABLY YES": "PY",
    "PROBABLY NO": "PN",
    "Y/PY": "PY",
    "N/PN": "PN",
    "NO INFORMATION": "NI",
    "NOT APPLICABLE": "NA",
}


def extract_tag(xml_string: str, tag: str) -> Optional[str]:
    """Extract a single tag value from model XML fragments."""
    sanitized = sanitize_stray_lt(xml_string)
    wrapped = f"<root>{sanitized}</root>"
    parser = etree.XMLParser(recover=True)
    root = etree.fromstring(wrapped.encode(), parser=parser)
    el = root.find(f".//{tag}")
    return el.text.strip() if el is not None and el.text else None


def _normalize_answer(answer: str) -> str:
    answer = (answer or "NI").strip().upper()
    if answer in VALID_ANSWERS:
        return answer
    if answer in ANSWER_MAPPING:
        return ANSWER_MAPPING[answer]
    raise ValueError(f"Invalid signaling-question answer: {answer}")


def _normalize_support_level(level: Optional[str], answer: str) -> str:
    normalized = (level or "").strip().lower()
    if not normalized:
        return "unsupported" if answer == "NA" else "weak"
    if normalized in VALID_SUPPORT_LEVELS:
        return normalized
    raise ValueError(f"Invalid evidence support level: {level}")


def _safe_text(value: Optional[str], default: str = "") -> str:
    return value.strip() if value and value.strip() else default


def sanitize_stray_lt(xml_string: str) -> str:
    """Escape `<` that isn't part of a tag/comment/declaration.

    LLM outputs often include text like `"<70 years"` or `"P<0.05"` inside
    `<quote>...</quote>`. A raw `<` followed by a digit, space, or other
    non-tag-start character makes lxml raise `StartTag: invalid element
    name`. Escaping those to `&lt;` lets the parser treat them as text.
    """
    return re.sub(r"<(?![a-zA-Z_/!?])", "&lt;", xml_string)


def parse_sq_response(xml_string: str, sq_ids: list[str]) -> dict[str, dict]:
    """
    Parse a multi-SQ XML response into a dict keyed by SQ ID.

    Returns: {"1.1": {"answer": "Y", "quote": "...", "justification": "...",
                      "uncertainty_flag": "NORMAL"}, ...}
    """
    result = {}
    cleaned_xml = re.sub(r"```xml\s*|\s*```", "", xml_string).strip()
    cleaned_xml = sanitize_stray_lt(cleaned_xml)
    root = etree.fromstring(f"<root>{cleaned_xml}</root>".encode())

    for sq_id in sq_ids:
        tag_name = f"sq_{sq_id.replace('.', '_')}"
        sq_el = root.find(f".//{tag_name}")
        if sq_el is None:
            raise ValueError(f"Missing signaling question: {sq_id}")
        parsed = {
            "answer": sq_el.findtext("answer") or "NI",
            "quote": sq_el.findtext("quote") or "No relevant text found",
            "justification": sq_el.findtext("justification")
            or "No relevant text found",
            "uncertainty_flag": sq_el.findtext("uncertainty_flag") or "NORMAL",
            "support_level": sq_el.findtext("support_level"),
            "support_rationale": sq_el.findtext("support_rationale"),
        }

        answer = _normalize_answer(parsed.get("answer", "NI"))
        quote_default = "Not applicable" if answer == "NA" else "No relevant text found"
        justification_default = (
            "Not applicable" if answer == "NA" else "No relevant text found"
        )

        quote = _safe_text(parsed.get("quote") if parsed else None, quote_default)
        justification = _safe_text(
            parsed.get("justification") if parsed else None, justification_default
        )
        if answer == "NA":
            if quote == "No relevant text found":
                quote = "Not applicable"
            if justification == "No relevant text found":
                justification = "Not applicable"
        support_level = _normalize_support_level(parsed.get("support_level"), answer)
        support_rationale_default = "Not applicable" if answer == "NA" else ""
        support_rationale = _safe_text(
            parsed.get("support_rationale"), support_rationale_default
        )
        if answer != "NA" and not support_rationale:
            if parsed.get("support_level") is not None:
                raise ValueError(
                    f"Non-NA signaling question {sq_id} requires a support rationale"
                )
            support_rationale = (
                "Support rationale was not provided by the legacy response."
            )

        result[sq_id] = {
            "answer": answer,
            "quote": quote,
            "justification": justification,
            "uncertainty_flag": _safe_text(
                parsed.get("uncertainty_flag"), "NORMAL"
            ).upper(),
            "support_level": support_level,
            "support_rationale": support_rationale,
        }

    return result


def validate_sq_answers(parsed: dict[str, dict], expected_ids: list[str]) -> list[str]:
    suspected = []
    for sq_id in expected_ids:
        answer = parsed.get(sq_id, {})
        if (
            answer.get("answer") == "NI"
            and answer.get("justification") == "No relevant text found"
        ):
            suspected.append(sq_id)
    return suspected
