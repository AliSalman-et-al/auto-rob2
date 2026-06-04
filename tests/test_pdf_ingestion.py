from types import SimpleNamespace

import rob2_pipeline.pdf_ingestion as pdf_ingestion
from langchain_core.documents import Document
from rob2_pipeline.models import empty_paper_evidence, format_evidence
from rob2_pipeline.ingestion.supplements import (
    apply_source_metadata,
    primary_source_document,
)
from rob2_pipeline.pdf_ingestion import (
    _build_docling_chunks,
    build_document_repr,
    cap_section,
    extract_censoring_context,
    extract_full_text,
    extract_paper_evidence,
    extract_structural_paper_evidence,
    parse_sections,
)


def test_pdf_ingestion_facade_reexports_core_ingestion_api():
    assert callable(pdf_ingestion.extract_full_text)
    assert callable(pdf_ingestion.extract_paper_evidence)
    assert callable(pdf_ingestion.extract_structural_paper_evidence)
    assert callable(pdf_ingestion.parse_sections)
    assert callable(pdf_ingestion.extract_censoring_context)
    assert callable(pdf_ingestion._configure_docling_runtime)
    assert callable(pdf_ingestion._get_docling_converter)
    assert callable(pdf_ingestion._build_docling_chunks)
    assert callable(pdf_ingestion.build_document_repr)
    assert callable(pdf_ingestion.paper_evidence_from_sections)
    assert callable(pdf_ingestion.appears_rct_candidate)
    assert callable(pdf_ingestion.allow_remote_evidence_extraction)


def test_docling_chunker_can_still_be_monkeypatched_via_facade(monkeypatch):
    mock_conv = type("ConversionResult", (), {"document": object()})()
    mock_chunks = [_make_mock_chunk("Facade chunk.", ["Methods"], [5])]

    class MockChunker:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def chunk(self, document):
            return mock_chunks

    monkeypatch.setattr(pdf_ingestion, "HybridChunker", MockChunker)

    result = pdf_ingestion._build_docling_chunks(mock_conv)

    assert result[0].page_content == "Facade chunk."
    assert result[0].metadata["page_numbers"] == [5]


def test_extract_full_text_uses_docling(monkeypatch):
    calls = []

    def fake_docling(pdf_path):
        calls.append(("docling", pdf_path))
        return "Docling text\xa0with hyphen-\nbreaks"

    monkeypatch.setattr(pdf_ingestion, "_extract_with_docling", fake_docling)

    text = extract_full_text("trial.pdf")

    assert calls == [("docling", "trial.pdf")]
    assert text == "Docling text with hyphenbreaks"


def test_extract_full_text_raises_when_docling_fails(monkeypatch):
    """Single-path docling: if docling fails, the exception propagates up. No
    silent fallback to a different parser."""

    def fake_docling(pdf_path):
        raise RuntimeError("docling exploded")

    monkeypatch.setattr(pdf_ingestion, "_extract_with_docling", fake_docling)

    try:
        extract_full_text("trial.pdf")
    except RuntimeError as error:
        assert "docling exploded" in str(error)
    else:
        raise AssertionError("extract_full_text should raise when docling fails")


def _make_mock_chunk(text: str, headings: list[str], pages: list[int]):
    class MockMeta:
        def __init__(self):
            self.headings = headings
            self.page_numbers = pages

        def export_json_dict(self):
            return {"headings": headings, "page_numbers": pages}

    class MockChunk:
        def __init__(self):
            self.text = text
            self.meta = MockMeta()

    return MockChunk()


def test_build_docling_chunks_returns_langchain_documents(monkeypatch):
    mock_conv = type("ConversionResult", (), {"document": object()})()
    mock_chunks = [
        _make_mock_chunk("Patients were randomly allocated.", ["Methods"], [2]),
        _make_mock_chunk("Allocation was concealed.", ["Methods"], [2]),
        _make_mock_chunk("Baseline characteristics.", ["Baseline"], [3]),
    ]

    class MockChunker:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def chunk(self, document):
            return mock_chunks

    monkeypatch.setattr(pdf_ingestion, "HybridChunker", MockChunker)

    result = _build_docling_chunks(mock_conv)

    assert len(result) == 3
    assert all(doc.page_content for doc in result)


def test_build_docling_chunks_preserves_metadata(monkeypatch):
    mock_conv = type("ConversionResult", (), {"document": object()})()
    mock_chunks = [_make_mock_chunk("Text about randomization.", ["Methods"], [2])]

    class MockChunker:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def chunk(self, document):
            return mock_chunks

    monkeypatch.setattr(pdf_ingestion, "HybridChunker", MockChunker)

    result = _build_docling_chunks(mock_conv)

    assert result[0].metadata["section"] == "Methods"
    assert result[0].metadata["page_numbers"] == [2]
    assert result[0].metadata["dl_meta"] == {
        "headings": ["Methods"],
        "page_numbers": [2],
    }


def test_build_docling_chunks_handles_no_headings(monkeypatch):
    mock_conv = type("ConversionResult", (), {"document": object()})()
    mock_chunks = [_make_mock_chunk("Plain text.", [], [1])]

    class MockChunker:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def chunk(self, document):
            return mock_chunks

    monkeypatch.setattr(pdf_ingestion, "HybridChunker", MockChunker)

    result = _build_docling_chunks(mock_conv)

    assert result[0].metadata["section"] == ""


def test_build_docling_chunks_configures_tokenizer_for_long_docling_counts(monkeypatch):
    mock_conv = type("ConversionResult", (), {"document": object()})()
    mock_chunks = [_make_mock_chunk("Text about randomization.", ["Methods"], [2])]
    tokenizer_calls = []

    class MockTokenizer:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            tokenizer_calls.append((model_name, kwargs))
            return "configured-tokenizer"

    class MockChunker:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def chunk(self, document):
            assert self.tokenizer == "configured-tokenizer"
            return mock_chunks

    monkeypatch.setattr(
        pdf_ingestion, "HuggingFaceTokenizer", MockTokenizer, raising=False
    )
    monkeypatch.setattr(pdf_ingestion, "HybridChunker", MockChunker)

    result = _build_docling_chunks(mock_conv)

    assert tokenizer_calls == [
        (
            "BAAI/bge-small-en-v1.5",
            {"max_tokens": 256, "model_max_length": 10**9},
        )
    ]
    assert result[0].page_content == "Text about randomization."
    assert result[0].metadata["section"] == "Methods"
    assert result[0].metadata["page_numbers"] == [2]


def test_primary_source_metadata_can_be_applied_to_docling_chunks(tmp_path):
    pdf_path = tmp_path / "ARCHES.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    chunks = [
        Document(
            page_content="Primary paper methods.",
            metadata={"section": "Methods", "page_numbers": [2]},
        )
    ]

    enriched = apply_source_metadata(chunks, primary_source_document(pdf_path))

    assert enriched[0].metadata["document_id"] == "primary"
    assert enriched[0].metadata["document_name"] == "ARCHES.pdf"
    assert enriched[0].metadata["document_role"] == "primary"
    assert enriched[0].metadata["source_kind"] == "rag_chunk"


def test_parse_sections_detects_expected_sections():
    text = """
    ABSTR ACT
    This randomized trial compared drug A with placebo.
    Methods
    Participants were randomly assigned in a 1:1 ratio.
    Randomization
    A computer-generated randomization schedule was used.
    Blinding
    Participants and investigators were double-blind.
    Results
    The primary outcome improved.
    Trial registration
    ClinicalTrials.gov NCT00000000.
    """

    sections = parse_sections(text)

    assert "randomized trial" in sections["abstract"]
    assert "Participants were randomly assigned" in sections["methods"]
    assert "computer-generated" in sections["randomization"]
    assert "double-blind" in sections["blinding"]
    assert "primary outcome" in sections["results"]
    assert "ClinicalTrials.gov" in sections["registration"]


def test_parse_sections_detects_markdown_headings():
    text = """
    ## Methods
    Participants were randomly assigned.
    **Outcomes**
    Overall survival was the primary endpoint.
    """

    sections = parse_sections(text)

    assert "randomly assigned" in sections["methods"]
    assert "Overall survival" in sections["outcomes"]


def test_parse_sections_recovers_keyword_context_without_headings():
    text = """
    The paper describes a phase 3 trial.
    Patients were assigned to treatment groups centrally.
    Overall survival was the primary endpoint and progression-free survival was secondary.
    Analyses used the intention-to-treat population.
    """

    sections = parse_sections(text)

    assert "primary endpoint" in sections["outcomes"]
    assert "intention-to-treat" in sections["analysis"]


def test_parse_sections_falls_back_to_methods_for_randomization_and_blinding():
    text = """
    Methods
    This was a randomized double-blind controlled trial using identical placebo.
    Results
    Participants completed follow-up.
    """

    sections = parse_sections(text)

    assert sections["methods"]
    assert sections["randomization"] == sections["methods"]
    assert sections["blinding"] == sections["methods"]


def test_parse_sections_returns_empty_strings_for_missing_sections():
    sections = parse_sections("Abstract\nBrief abstract only.")

    assert set(sections) == {
        "abstract",
        "methods",
        "randomization",
        "blinding",
        "outcomes",
        "analysis",
        "results",
        "missing_data",
        "registration",
        "baseline",
        "consort",
        "supplementary",
    }
    assert sections["methods"] == ""


def test_cap_section_returns_unchanged_when_under_limit():
    text = "A" * 7999
    capped = cap_section(text)

    assert capped == text


def test_cap_section_prefers_keyword_dense_chunks():
    text = (
        "A" * 3000
        + " random allocation conceal blind " * 40
        + "B" * 3000
        + " outcome endpoint register " * 40
        + "C" * 3000
    )
    capped = cap_section(text)

    assert "[... truncated ...]" in capped
    assert "allocation" in capped.lower()
    assert (
        "[NOTE: Section truncated at 10000 characters. Critical content may be absent.]"
        in capped
    )
    assert len(capped) <= 10000 + len(
        "\n\n[NOTE: Section truncated at 10000 characters. Critical content may be absent.]"
    )


def test_parse_sections_from_docling_document_routes_correctly():
    class MockItem:
        def __init__(self, label, text="", table_md=""):
            self.label = label
            self.text = text
            self._table_md = table_md

        def export_to_markdown(self, doc=None):
            return self._table_md

    class MockDoc:
        def __init__(self, items):
            self._items = items

        def iterate_items(self):
            for item in self._items:
                yield item, 1

        def export_to_text(self):
            return "\n".join(getattr(item, "text", "") for item in self._items)

    items = [
        MockItem("section_header", text="Methods"),
        MockItem("text", text="Patients were randomly assigned in a 1:1 ratio."),
        MockItem("section_header", text="Randomization"),
        MockItem("text", text="Computer-generated sequence was used."),
        MockItem("table", table_md="| baseline characteristics | age |\n|---|---|"),
    ]
    sections = pdf_ingestion._parse_sections_from_docling_document(MockDoc(items))

    assert sections is not None
    assert "randomly assigned" in sections["methods"]
    assert "Computer-generated" in sections["randomization"]
    assert "baseline characteristics" in sections["baseline"]


def test_build_document_repr_groups_text_and_tables_by_heading():
    class MockItem:
        def __init__(self, label, text="", table_md=""):
            self.label = label
            self.text = text
            self._table_md = table_md

        def export_to_markdown(self, doc=None):
            return self._table_md

    class MockDoc:
        def __init__(self, items):
            self._items = items

        def iterate_items(self):
            for item in self._items:
                yield item, 1

        def export_to_markdown(self):
            return "# Methods\nPatients were randomized.\n| baseline | age |"

    doc_repr = build_document_repr(
        MockDoc(
            [
                MockItem("section_header", text="Methods"),
                MockItem("text", text="Patients were randomized centrally."),
                MockItem(
                    "table", table_md="| baseline characteristics | age |\n|---|---|"
                ),
                MockItem("section_header", text="Results"),
                MockItem(
                    "paragraph", text="All randomized participants were analysed."
                ),
            ]
        )
    )

    assert doc_repr.full_text.startswith("# Methods")
    assert doc_repr.blocks[0].heading == "Methods"
    assert "randomized centrally" in doc_repr.blocks[0].text
    assert doc_repr.blocks[0].tables == [
        "| baseline characteristics | age |\n|---|---|"
    ]
    assert doc_repr.blocks[1].heading == "Results"
    assert "All randomized" in doc_repr.to_prompt_repr()
    assert "[TABLE]" in doc_repr.to_prompt_repr()


def test_extract_paper_evidence_accepts_validated_json_contract(monkeypatch):
    def fake_contract(state, prompt, node_name, **kwargs):
        assert "<paper>" in prompt
        return SimpleNamespace(
            artifact={
                "schema_version": "paper-evidence-extraction-v1",
                "abstract": {"text": "Trial abstract.", "tables": []},
                "methods": {"text": "Randomized methods.", "tables": []},
                "results": {"text": "Result text.", "tables": ["| result |"]},
                "d1_randomization": {"text": "Central sequence.", "tables": []},
                "d2_blinding": {"text": "Double blind.", "tables": []},
                "d3_missing_data": {"text": "Complete follow-up.", "tables": []},
                "d4_outcome_meas": {"text": "Mortality outcome.", "tables": []},
                "d5_registration": {"text": "NCT00000000.", "tables": []},
                "consort_flow": {"text": "100 randomized.", "tables": []},
                "baseline_table": {"text": "", "tables": ["| baseline | age |"]},
            },
            log=[{"node": node_name, "validation_status": "validated"}],
            status="validated",
            failure_reason=None,
        )

    monkeypatch.setattr(pdf_ingestion._evidence, "call_json_contract_llm", fake_contract)
    doc_repr = pdf_ingestion.DocumentRepr(
        blocks=[],
        full_text="Methods\nPatients were randomized.",
    )

    evidence, log = extract_paper_evidence(doc_repr)

    assert evidence["extraction_method"] == "docling_llm"
    assert evidence["d1_randomization"]["text"] == "Central sequence."
    assert evidence["baseline_table"]["tables"] == ["| baseline | age |"]
    assert format_evidence(evidence["results"]) == "Result text.\n\n| result |"
    assert log[0]["node"] == "paper_evidence_extraction"


def test_structural_paper_evidence_preserves_tables_by_heading():
    doc_repr = pdf_ingestion.DocumentRepr(
        blocks=[
            pdf_ingestion.DocBlock(
                heading="Baseline characteristics",
                level=2,
                text="Baseline table caption.",
                tables=["| baseline characteristics | age |\n|---|---|"],
                page_start=1,
            ),
            pdf_ingestion.DocBlock(
                heading="Participant flow",
                level=2,
                text="100 participants were randomized.",
                tables=["| randomized | analysed |\n|---|---|"],
                page_start=2,
            ),
        ],
        full_text="",
    )

    evidence = extract_structural_paper_evidence(doc_repr)

    assert evidence["extraction_method"] == "docling_struct"
    assert evidence["baseline_table"]["tables"] == [
        "| baseline characteristics | age |\n|---|---|"
    ]
    assert evidence["consort_flow"]["tables"] == [
        "| randomized | analysed |\n|---|---|"
    ]


def test_extract_censoring_context_finds_event_sentences():
    full_text = "\n".join(
        [
            "Introduction line.",
            "Irrelevant details.",
            "At final analysis, 415 events were observed in 917 participants.",
            "More methods text.",
            "The study reports data maturity of 74% at the data cutoff.",
            "Conclusion line.",
        ]
    )
    result = extract_censoring_context(full_text, "Overall Survival")

    assert "415 events" in result
    assert "74%" in result
    assert result
    assert len(result) <= 2000


def test_extract_censoring_context_returns_empty_for_no_matches():
    full_text = (
        "This study compared two interventions. Outcomes improved with treatment."
    )
    assert extract_censoring_context(full_text, "Overall Survival") == ""


def test_assessment_ingestion_falls_back_to_text_parse_when_docling_structure_fails(
    monkeypatch,
):
    import rob2_pipeline.ingestion.assessment as assessment

    """If the docling converter fails, fall back to a text keyword parse of the
    already-extracted full text. extract_full_text itself still raises on
    failure (no pymupdf fallback)."""
    known_text = (
        "Methods\nParticipants were randomly assigned in a 1:1 ratio.\nResults\nDone."
    )
    monkeypatch.setattr(
        "rob2_pipeline.ingestion.assessment.extract_full_text", lambda _: known_text
    )

    class BrokenConverter:
        def convert(self, _):
            raise RuntimeError("docling structured parse failed")

    monkeypatch.setattr(
        "rob2_pipeline.ingestion.assessment._get_docling_converter",
        lambda use_ocr: BrokenConverter(),
    )

    state = {"pdf_path": "trial.pdf"}
    result = assessment.ingest_assessment_documents(
        state["pdf_path"], state.get("supplementary_paths")
    ).to_state_update()

    assert "evidence" in result
    assert result["evidence"]["extraction_method"] == "fallback"
    assert result["docling_doc"] is None
    assert result["docling_chunks"] == []
    assert "randomly assigned" in result["evidence"]["methods"]["text"]
    assert "randomly assigned" in result["evidence"]["d1_randomization"]["text"]


def test_ingest_node_records_skipped_supplements_when_primary_docling_falls_back(
    monkeypatch,
):
    import rob2_pipeline.ingestion.assessment as assessment

    known_text = (
        "Methods\nParticipants were randomly assigned in a 1:1 ratio.\nResults\nDone."
    )
    monkeypatch.setattr(
        "rob2_pipeline.ingestion.assessment.extract_full_text", lambda _: known_text
    )

    class BrokenConverter:
        def convert(self, _):
            raise RuntimeError("docling structured parse failed")

    monkeypatch.setattr(
        "rob2_pipeline.ingestion.assessment._get_docling_converter",
        lambda use_ocr: BrokenConverter(),
    )

    state = {
        "pdf_path": "trial.pdf",
        "supplementary_paths": ["inputs/benchmark/supplement/TITAN/protocol.pdf"],
    }
    result = assessment.ingest_assessment_documents(
        state["pdf_path"], state.get("supplementary_paths")
    ).to_state_update()

    assert result["docling_chunks"] == []
    assert result["source_documents"][1]["document_name"] == "protocol.pdf"
    assert result["source_documents"][1]["status"] == "failed"
    assert "Supplement not ingested" in result["supplement_warnings"][0]


def test_pdf_ingest_node_adapts_assessment_ingestion_result(monkeypatch):
    import rob2_pipeline.nodes.ingest as ingest_node
    from rob2_pipeline.ingestion.assessment import AssessmentIngestionResult

    evidence = empty_paper_evidence("docling_struct")
    captured = {}
    llm_log = {
        "node": "paper_evidence_extraction",
        "prompt_length_chars": 120,
        "response_length_chars": 80,
        "latency_ms": 5,
        "cache_hit": False,
    }

    def fake_ingest_assessment_documents(pdf_path, supplementary_paths):
        captured["pdf_path"] = pdf_path
        captured["supplementary_paths"] = supplementary_paths
        return AssessmentIngestionResult(
            full_text="Primary text",
            evidence=evidence,
            docling_doc=None,
            docling_chunks=[],
            source_documents=[],
            supplement_warnings=["warning"],
            llm_call_log=[llm_log],
        )

    monkeypatch.setattr(
        ingest_node,
        "ingest_assessment_documents",
        fake_ingest_assessment_documents,
    )

    result = ingest_node.pdf_ingest_node(
        {"pdf_path": "primary.pdf", "supplementary_paths": ["protocol.pdf"]}
    )

    assert captured == {
        "pdf_path": "primary.pdf",
        "supplementary_paths": ["protocol.pdf"],
    }
    assert result == {
        "full_text": "Primary text",
        "evidence": evidence,
        "docling_doc": None,
        "docling_chunks": [],
        "source_documents": [],
        "parse_artifacts": [],
        "supplement_warnings": ["warning"],
        "llm_call_log": [llm_log],
    }


def test_ingest_node_stores_docling_conversion_result(monkeypatch):
    import rob2_pipeline.ingestion.assessment as assessment

    known_text = "Methods\nParticipants were randomly assigned."
    monkeypatch.setattr(
        "rob2_pipeline.ingestion.assessment.extract_full_text", lambda _: known_text
    )

    class MockConverter:
        def __init__(self):
            self.conversion_result = type(
                "ConversionResult", (), {"document": object()}
            )()

        def convert(self, _):
            return self.conversion_result

    converter = MockConverter()
    monkeypatch.setattr(
        "rob2_pipeline.ingestion.assessment._get_docling_converter",
        lambda use_ocr: converter,
    )
    monkeypatch.setattr(
        "rob2_pipeline.ingestion.assessment.build_document_repr",
        lambda doc: pdf_ingestion.DocumentRepr(blocks=[], full_text=known_text),
    )
    monkeypatch.setattr(
        "rob2_pipeline.ingestion.assessment.extract_paper_evidence",
        lambda doc_repr: (empty_paper_evidence("docling_llm"), []),
    )
    monkeypatch.setattr(
        "rob2_pipeline.ingestion.assessment._build_docling_chunks",
        lambda conv_result: ["chunk"],
    )

    result = assessment.ingest_assessment_documents("trial.pdf").to_state_update()

    assert result["docling_doc"] is converter.conversion_result
    assert result["docling_chunks"] == ["chunk"]


def test_ingest_node_skips_remote_extraction_when_disabled(monkeypatch):
    import rob2_pipeline.ingestion.assessment as assessment

    known_text = "Methods\nParticipants were randomly assigned."
    monkeypatch.setattr(
        "rob2_pipeline.ingestion.assessment.extract_full_text", lambda _: known_text
    )

    class MockConverter:
        def __init__(self):
            self.conversion_result = type(
                "ConversionResult", (), {"document": object()}
            )()

        def convert(self, _):
            return self.conversion_result

    converter = MockConverter()
    monkeypatch.setattr(
        "rob2_pipeline.ingestion.assessment._get_docling_converter",
        lambda use_ocr: converter,
    )
    monkeypatch.setattr(
        "rob2_pipeline.ingestion.assessment.build_document_repr",
        lambda doc: pdf_ingestion.DocumentRepr(blocks=[], full_text=known_text),
    )
    monkeypatch.setattr(
        "rob2_pipeline.ingestion.assessment.allow_remote_evidence_extraction",
        lambda: False,
    )
    monkeypatch.setattr(
        "rob2_pipeline.ingestion.assessment._build_docling_chunks",
        lambda conv_result: ["chunk"],
    )

    def fail_if_called(_doc_repr):
        raise AssertionError("remote extraction should be skipped when disabled")

    monkeypatch.setattr(
        "rob2_pipeline.ingestion.assessment.extract_paper_evidence", fail_if_called
    )

    result = assessment.ingest_assessment_documents("trial.pdf").to_state_update()

    assert result["evidence"]["extraction_method"] == "docling_struct"
    assert result["docling_chunks"] == ["chunk"]


def test_ingest_node_skips_remote_extraction_for_apparent_non_rct(monkeypatch):
    import rob2_pipeline.ingestion.assessment as assessment

    known_text = (
        "Editorial commentary describing mechanism without any trial assignment."
    )
    monkeypatch.setattr(
        "rob2_pipeline.ingestion.assessment.extract_full_text", lambda _: known_text
    )

    class MockConverter:
        def __init__(self):
            self.conversion_result = type(
                "ConversionResult", (), {"document": object()}
            )()

        def convert(self, _):
            return self.conversion_result

    converter = MockConverter()
    monkeypatch.setattr(
        "rob2_pipeline.ingestion.assessment._get_docling_converter",
        lambda use_ocr: converter,
    )
    monkeypatch.setattr(
        "rob2_pipeline.ingestion.assessment.build_document_repr",
        lambda doc: pdf_ingestion.DocumentRepr(blocks=[], full_text=known_text),
    )
    monkeypatch.setattr(
        "rob2_pipeline.ingestion.assessment.allow_remote_evidence_extraction",
        lambda: True,
    )
    monkeypatch.setattr(
        "rob2_pipeline.ingestion.assessment._build_docling_chunks",
        lambda conv_result: ["chunk"],
    )

    def fail_if_called(_doc_repr):
        raise AssertionError(
            "remote extraction should be skipped for apparent non-RCT text"
        )

    monkeypatch.setattr(
        "rob2_pipeline.ingestion.assessment.extract_paper_evidence", fail_if_called
    )

    result = assessment.ingest_assessment_documents("trial.pdf").to_state_update()

    assert result["evidence"]["extraction_method"] == "docling_struct"
    assert result["docling_chunks"] == ["chunk"]
