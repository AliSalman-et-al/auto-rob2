"""Regression: Domain 5 SQ 5.2/5.3 answer polarity must reach the model on BOTH paths.

Background: the weak benchmark model (gpt-oss-120b) emitted ``Y`` on D5 SQ 5.2/5.3 for trials whose own justification
described pre-specified, fully-reported analyses (which are ``N``). The Domain 5 judge
maps ``5.2 or 5.3 = Y/PY -> High``, so the inverted code forced High over-calls.

The polarity disambiguators lived only in ``RuleCard.notes``. The XML prompt path
(``render_methodology``) renders notes, but the JSON evidence-packet classifier path
(``build_decision_table``) renders only the per-code ``response_rules`` guidance, so on
that path the model never saw them. These tests lock that the Y=selective-reporting-
present / N=selective-reporting-absent polarity is explicit in EACH render channel, so
the fix cannot silently regress by living somewhere only one path surfaces.
"""

from rob2_pipeline.methodology import DOMAIN5_METHODOLOGY
from rob2_pipeline.methodology.render import render_methodology
from rob2_pipeline.nodes.evidence_contracts import CONTRACTS
from rob2_pipeline.nodes.evidence_packets import build_decision_table


def _json_path_rules(sq_id: str) -> dict[str, str]:
    """Answer-code -> rule text as the JSON evidence-packet classifier path renders it."""
    table = build_decision_table(
        contract=CONTRACTS[sq_id], facts=[], gaps=[], missing=[]
    )
    return {row["answer"]: row["rule"] for row in table["rows"]}


def _xml_path_text() -> str:
    """Full D5 methodology block as the XML prompt path renders it (lowercased)."""
    return render_methodology(DOMAIN5_METHODOLOGY, ["5.1", "5.2", "5.3"]).lower()


def test_d5_sq52_polarity_is_explicit_in_both_render_paths():
    json_rules = _json_path_rules("5.2")
    assert "selective reporting present" in json_rules["Y"].lower()
    n_rule = json_rules["N"].lower()
    assert "selective reporting absent" in n_rule
    # The specific disambiguation that fixes the inversion: a pre-specified result
    # reported as planned is N, not Y.
    assert "pre-specified" in n_rule
    assert "report" in n_rule

    xml = _xml_path_text()
    assert "selective reporting present" in xml
    assert "selective reporting absent" in xml


def test_d5_sq53_polarity_is_explicit_in_both_render_paths():
    json_rules = _json_path_rules("5.3")
    assert "selective reporting present" in json_rules["Y"].lower()
    n_rule = json_rules["N"].lower()
    assert "selective reporting absent" in n_rule
    assert "pre-specified" in n_rule
    assert "report" in n_rule
