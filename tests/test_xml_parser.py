import pytest

from rob2_pipeline.xml_parser import (
    sanitize_stray_lt,
    extract_tag,
    parse_sq_response,
    validate_sq_answers,
)


def test_extract_tag_well_formed_fragment():
    assert extract_tag("<screening><is_rct>YES</is_rct></screening>", "is_rct") == "YES"


def test_parse_sq_response_valid_xml():
    xml = """
    <domain1>
      <sq_1_1>
        <answer>Y</answer>
        <quote>"Computer-generated sequence" (Methods)</quote>
        <justification>Random method was stated.</justification>
        <support_level>strong</support_level>
        <support_rationale>Direct sequence-generation quote.</support_rationale>
      </sq_1_1>
      <sq_1_2>
        <answer>PY</answer>
        <quote>"Central allocation" (Methods)</quote>
        <justification>Central allocation implies concealment.</justification>
        <uncertainty_flag>normal</uncertainty_flag>
        <support_level>Moderate</support_level>
        <support_rationale>Indirect but relevant allocation evidence.</support_rationale>
      </sq_1_2>
    </domain1>
    """

    parsed = parse_sq_response(xml, ["1.1", "1.2"])

    assert parsed["1.1"]["answer"] == "Y"
    assert parsed["1.1"]["quote"] == '"Computer-generated sequence" (Methods)'
    assert parsed["1.1"]["support_level"] == "strong"
    assert parsed["1.2"]["answer"] == "PY"
    assert parsed["1.2"]["uncertainty_flag"] == "NORMAL"
    assert parsed["1.2"]["support_level"] == "moderate"


def test_parse_sq_response_malformed_xml_raises():
    xml = """
    <domain1>
      <sq_1_1>
        <answer>Probably Yes</answer>
        <quote>"Randomized centrally" (Methods)</quote>
        <justification>Supports a probably yes answer.</justification>
    """

    with pytest.raises(Exception):
        parse_sq_response(xml, ["1.1"])


def test_parse_sq_response_missing_sq_raises():
    xml = """
    <domain1>
      <sq_1_1>
        <answer>Y</answer>
        <quote>"Randomized" (Abstract)</quote>
        <justification>States randomized.</justification>
      </sq_1_1>
    </domain1>
    """

    with pytest.raises(ValueError, match="Missing signaling question"):
        parse_sq_response(xml, ["1.1", "1.2"])


def test_parse_sq_response_unexpected_answer_values_raise():
    xml = """
    <domain5>
      <sq_5_1>
        <answer>Definitely</answer>
        <quote></quote>
        <justification></justification>
      </sq_5_1>
      <sq_5_2>
        <answer>Not applicable</answer>
      </sq_5_2>
    </domain5>
    """

    with pytest.raises(ValueError, match="Invalid signaling-question answer"):
        parse_sq_response(xml, ["5.1", "5.2"])


def test_parse_sq_response_rejects_invalid_support_level():
    xml = """
    <domain1>
      <sq_1_1>
        <answer>Y</answer>
        <quote>"Randomized" (Abstract)</quote>
        <justification>States randomized.</justification>
        <support_level>definitive</support_level>
        <support_rationale>Unsupported label.</support_rationale>
      </sq_1_1>
    </domain1>
    """

    with pytest.raises(ValueError, match="Invalid evidence support level"):
        parse_sq_response(xml, ["1.1"])


def test_parse_sq_response_requires_support_rationale_for_non_na_answers():
    xml = """
    <domain1>
      <sq_1_1>
        <answer>Y</answer>
        <quote>"Randomized" (Abstract)</quote>
        <justification>States randomized.</justification>
        <support_level>strong</support_level>
        <support_rationale></support_rationale>
      </sq_1_1>
    </domain1>
    """

    with pytest.raises(ValueError, match="support rationale"):
        parse_sq_response(xml, ["1.1"])


def test_parse_sq_response_defaults_support_metadata_for_legacy_xml():
    xml = """
    <domain1>
      <sq_1_1>
        <answer>NA</answer>
        <quote>Not applicable</quote>
        <justification>Not applicable</justification>
      </sq_1_1>
    </domain1>
    """

    parsed = parse_sq_response(xml, ["1.1"])

    assert parsed["1.1"]["support_level"] == "unsupported"
    assert parsed["1.1"]["support_rationale"] == "Not applicable"


def test_parse_sq_response_strips_xml_code_fences():
    xml = """
    ```xml
    <domain1>
      <sq_1_1>
        <answer>Y</answer>
        <quote>\"Randomized\" (Methods)</quote>
        <justification>Directly reported.</justification>
      </sq_1_1>
    </domain1>
    ```
    """
    parsed = parse_sq_response(xml, ["1.1"])
    assert parsed["1.1"]["answer"] == "Y"


def test_validate_sq_answers_flags_suspected_parse_failures():
    parsed = {
        "1.1": {"answer": "NI", "justification": "No relevant text found"},
        "1.2": {"answer": "Y", "justification": "Found explicit method"},
    }
    suspected = validate_sq_answers(parsed, ["1.1", "1.2"])
    assert suspected == ["1.1"]


def testsanitize_stray_lt_escapes_age_threshold():
    assert sanitize_stray_lt("age <70 years") == "age &lt;70 years"


def testsanitize_stray_lt_escapes_p_value():
    assert sanitize_stray_lt("P<0.05") == "P&lt;0.05"


def testsanitize_stray_lt_preserves_valid_tags():
    xml = "<quote>text</quote><justification>more</justification>"
    assert sanitize_stray_lt(xml) == xml


def testsanitize_stray_lt_preserves_closing_tags_and_comments():
    xml = "<a>text</a><!-- comment --><?pi data?>"
    assert sanitize_stray_lt(xml) == xml


def test_parse_sq_response_handles_stray_lt_in_quote():
    xml = """
    <domain1>
      <sq_1_1>
        <answer>Y</answer>
        <quote>"Eligible if age <70 years" (Methods)</quote>
        <justification>Age threshold reported.</justification>
      </sq_1_1>
    </domain1>
    """
    parsed = parse_sq_response(xml, ["1.1"])
    assert parsed["1.1"]["answer"] == "Y"
    assert "<70 years" in parsed["1.1"]["quote"]


def test_parse_sq_response_handles_p_value_in_justification():
    xml = """
    <domain1>
      <sq_1_3>
        <answer>N</answer>
        <quote>"Baseline characteristics balanced" (Table 1)</quote>
        <justification>No imbalance; all P<0.05 thresholds met for stratification.</justification>
      </sq_1_3>
    </domain1>
    """
    parsed = parse_sq_response(xml, ["1.3"])
    assert parsed["1.3"]["answer"] == "N"
    assert "P<0.05" in parsed["1.3"]["justification"]
