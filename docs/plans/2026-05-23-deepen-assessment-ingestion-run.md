# Deepen Assessment Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move primary-paper plus supplement ingestion behind a typed Assessment ingestion module while preserving existing graph behavior and the `pdf_ingestion.py` compatibility facade.

**Architecture:** Add `rob2_pipeline/ingestion/assessment.py` as the deep module for Assessment ingestion behavior. `rob2_pipeline/nodes/ingest.py` becomes a graph adapter that calls `ingest_assessment_documents()` and converts `AssessmentIngestionResult` into the existing `RoB2State` update shape. Keep `rob2_pipeline/pdf_ingestion.py` exports intact for current tests and external compatibility.

**Tech Stack:** Python 3.13, dataclasses, Docling, LangChain `Document`, pytest, existing `RoB2State`, `PaperEvidence`, `SourceDocument`, and `LLMCallLogEntry` typed dicts.

---

## File Structure

- Create `rob2_pipeline/ingestion/assessment.py`
  - Owns `AssessmentIngestionResult`.
  - Owns `ingest_assessment_documents(pdf_path, supplementary_paths)`.
  - Owns the fallback order currently embedded in `pdf_ingest_node()`.
  - Owns conversion of primary and supplement chunks into one provenance-bearing chunk list.
  - Imports directly from focused ingestion modules (`docling_extract.py`, `document_repr.py`, `evidence.py`, and `supplements.py`) instead of routing through `pdf_ingestion.py`.
  - Does not own RCT screening, trial metadata, retrieval, evidence packets, or judging.

- Modify `rob2_pipeline/nodes/ingest.py`
  - Replace ingestion orchestration in `pdf_ingest_node()` with a call to `ingest_assessment_documents()`.
  - Keep `rct_screener_node()` unchanged.
  - Keep graph output keys unchanged.
  - Do not preserve ingestion-internal monkeypatch points on this module; tests for extraction, Docling conversion, supplement ingestion, and fallback behavior should patch `rob2_pipeline.ingestion.assessment`.

- Modify `rob2_pipeline/ingestion/__init__.py`
  - Export `AssessmentIngestionResult` and `ingest_assessment_documents`.

- Modify `CONTEXT.md`
  - Update the ingestion module map so future agents start in `ingestion/assessment.py` for primary plus supplement ingestion behavior.

- Test `tests/test_assessment_ingestion.py`
  - New focused tests for the typed ingestion interface and fallback behavior.

- Modify `tests/test_supplements.py`
  - Keep supplement-specific tests.
  - Move or adjust `pdf_ingest_node` orchestration tests so they target the new deep module where appropriate.

- Modify `tests/test_pdf_ingestion.py`
  - Keep facade compatibility assertions.
  - Do not remove monkeypatch-compatible facade coverage.
  - Keep only graph-adapter tests on `pdf_ingest_node()`; move ingestion-internal tests to `tests/test_assessment_ingestion.py`.

- Keep `rob2_pipeline/pdf_ingestion.py`
  - Preserve as an outward-facing compatibility adapter.
  - Do not use it as an internal dependency from `rob2_pipeline/ingestion/assessment.py`.

- Preserve current primary-failure behavior
  - If strict full-text extraction fails, the run still halts.
  - If primary Docling structural extraction fails after full text is available, build fallback primary `PaperEvidence`, emit no Docling chunks, and record requested supplements as skipped rather than parsing supplements independently.

---

### Task 1: Add The Typed Assessment Ingestion Result

**Files:**
- Create: `rob2_pipeline/ingestion/assessment.py`
- Modify: `rob2_pipeline/ingestion/__init__.py`
- Test: `tests/test_assessment_ingestion.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_assessment_ingestion.py` with this initial content:

```python
from rob2_pipeline.ingestion.assessment import AssessmentIngestionResult
from rob2_pipeline.models import empty_paper_evidence


def test_assessment_ingestion_result_to_state_update_omits_empty_llm_log():
    evidence = empty_paper_evidence("docling_struct")
    result = AssessmentIngestionResult(
        full_text="Primary text",
        evidence=evidence,
        docling_doc=None,
        docling_chunks=[],
        source_documents=[],
        supplement_warnings=[],
    )

    assert result.to_state_update() == {
        "full_text": "Primary text",
        "evidence": evidence,
        "docling_doc": None,
        "docling_chunks": [],
        "source_documents": [],
        "supplement_warnings": [],
    }


def test_assessment_ingestion_result_to_state_update_includes_llm_log_when_present():
    evidence = empty_paper_evidence("docling_llm")
    result = AssessmentIngestionResult(
        full_text="Primary text",
        evidence=evidence,
        docling_doc=None,
        docling_chunks=[],
        source_documents=[],
        supplement_warnings=[],
        llm_call_log=[
            {
                "node": "paper_evidence_extraction",
                "prompt_length_chars": 120,
                "response_length_chars": 80,
                "latency_ms": 5,
                "cache_hit": False,
            }
        ],
    )

    assert result.to_state_update()["llm_call_log"] == [
        {
            "node": "paper_evidence_extraction",
            "prompt_length_chars": 120,
            "response_length_chars": 80,
            "latency_ms": 5,
            "cache_hit": False,
        }
    ]
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
uv run pytest tests/test_assessment_ingestion.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'rob2_pipeline.ingestion.assessment'`.

- [ ] **Step 3: Add the dataclass and export**

Create `rob2_pipeline/ingestion/assessment.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document

from rob2_pipeline.models import PaperEvidence
from rob2_pipeline.types import LLMCallLogEntry, SourceDocument


@dataclass(frozen=True)
class AssessmentIngestionResult:
    full_text: str
    evidence: PaperEvidence
    docling_doc: Any | None
    docling_chunks: list[Document]
    source_documents: list[SourceDocument]
    supplement_warnings: list[str]
    llm_call_log: list[LLMCallLogEntry] = field(default_factory=list)

    def to_state_update(self) -> dict:
        update = {
            "full_text": self.full_text,
            "evidence": self.evidence,
            "docling_doc": self.docling_doc,
            "docling_chunks": self.docling_chunks,
            "source_documents": self.source_documents,
            "supplement_warnings": self.supplement_warnings,
        }
        if self.llm_call_log:
            update["llm_call_log"] = self.llm_call_log
        return update
```

Modify `rob2_pipeline/ingestion/__init__.py` by adding these imports:

```python
from rob2_pipeline.ingestion.assessment import (
    AssessmentIngestionResult,
    ingest_assessment_documents,
)
```

Add both names to `__all__`:

```python
    "AssessmentIngestionResult",
    "ingest_assessment_documents",
```

In this task, `ingest_assessment_documents` does not exist yet, so add a temporary stub below the dataclass:

```python
def ingest_assessment_documents(
    pdf_path: str, supplementary_paths: list[str] | None = None
) -> AssessmentIngestionResult:
    raise NotImplementedError("ingest_assessment_documents is added in Task 2")
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
uv run pytest tests/test_assessment_ingestion.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rob2_pipeline/ingestion/assessment.py rob2_pipeline/ingestion/__init__.py tests/test_assessment_ingestion.py
git commit -m "refactor: add assessment ingestion result"
```

---

### Task 2: Move Primary Ingestion Success Path Behind The New Interface

**Files:**
- Modify: `rob2_pipeline/ingestion/assessment.py`
- Test: `tests/test_assessment_ingestion.py`

- [ ] **Step 1: Write the failing test**

Append this test to `tests/test_assessment_ingestion.py`:

```python
from langchain_core.documents import Document


def test_ingest_assessment_documents_returns_primary_structural_result_when_remote_disabled(
    monkeypatch,
):
    import rob2_pipeline.ingestion.assessment as assessment

    evidence = empty_paper_evidence("docling_struct")
    primary_chunk = Document(
        page_content="Primary chunk",
        metadata={"section": "Methods", "page_numbers": [1]},
    )

    monkeypatch.setattr(assessment, "extract_full_text", lambda path: "Primary text")
    monkeypatch.setattr(assessment, "_configure_docling_runtime", lambda: None)
    monkeypatch.setattr(assessment, "_build_docling_chunks", lambda conv_result: [primary_chunk])
    monkeypatch.setattr(
        assessment,
        "build_document_repr",
        lambda doc: type(
            "DocRepr",
            (),
            {
                "full_text": "Primary text",
                "to_prompt_repr": lambda self: "Primary text",
                "blocks": [],
            },
        )(),
    )
    monkeypatch.setattr(assessment, "extract_structural_paper_evidence", lambda doc_repr: evidence)
    monkeypatch.setattr(assessment, "allow_remote_evidence_extraction", lambda: False)
    monkeypatch.setattr(assessment, "ingest_supplements", lambda paths: ([], [], []))

    class Result:
        document = object()

    class Converter:
        def convert(self, path):
            return Result()

    monkeypatch.setattr(assessment, "_get_docling_converter", lambda use_ocr=False: Converter())

    result = assessment.ingest_assessment_documents("primary.pdf", [])

    assert result.full_text == "Primary text"
    assert result.evidence is evidence
    assert len(result.docling_chunks) == 1
    assert result.docling_chunks[0].metadata["document_id"] == "primary"
    assert result.source_documents == [
        {
            "document_id": "primary",
            "document_name": "primary.pdf",
            "document_role": "primary",
            "source_kind": "rag_chunk",
            "path": "primary.pdf",
            "is_primary": True,
            "status": "parsed",
        }
    ]
    assert result.supplement_warnings == []
    assert result.llm_call_log == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
uv run pytest tests/test_assessment_ingestion.py::test_ingest_assessment_documents_returns_primary_structural_result_when_remote_disabled -q
```

Expected: FAIL with the `NotImplementedError` from Task 1.

- [ ] **Step 3: Implement the structural success path**

Replace the stub in `rob2_pipeline/ingestion/assessment.py` with imports and implementation:

```python
from pathlib import Path

from rob2_pipeline.ingestion.docling_extract import (
    _build_docling_chunks,
    _configure_docling_runtime,
    _get_docling_converter,
    extract_full_text,
)
from rob2_pipeline.ingestion.document_repr import build_document_repr
from rob2_pipeline.ingestion.evidence import extract_structural_paper_evidence
from rob2_pipeline.ingestion.settings import allow_remote_evidence_extraction
from rob2_pipeline.ingestion.supplements import (
    apply_source_metadata,
    ingest_supplements,
    primary_source_document,
)
```

Use this function body:

```python
def ingest_assessment_documents(
    pdf_path: str, supplementary_paths: list[str] | None = None
) -> AssessmentIngestionResult:
    supplementary_paths = list(supplementary_paths or [])
    full_text = extract_full_text(pdf_path)
    primary_source = primary_source_document(Path(pdf_path))

    _configure_docling_runtime()
    converter = _get_docling_converter(use_ocr=False)
    conv_result = converter.convert(pdf_path)
    docling_chunks = apply_source_metadata(
        _build_docling_chunks(conv_result), primary_source
    )
    supplement_chunks, supplement_documents, supplement_warnings = ingest_supplements(
        supplementary_paths
    )
    docling_chunks = [*docling_chunks, *supplement_chunks]
    source_documents = [primary_source, *supplement_documents]

    doc_repr = build_document_repr(conv_result.document)
    if not doc_repr.full_text:
        doc_repr.full_text = full_text
    evidence = extract_structural_paper_evidence(doc_repr)

    if not allow_remote_evidence_extraction():
        evidence["warnings"].append(
            "Remote evidence extraction disabled by ROB2_REMOTE_EVIDENCE_EXTRACTION."
        )
        return AssessmentIngestionResult(
            full_text=full_text,
            evidence=evidence,
            docling_doc=conv_result,
            docling_chunks=docling_chunks,
            source_documents=source_documents,
            supplement_warnings=supplement_warnings,
        )

    return AssessmentIngestionResult(
        full_text=full_text,
        evidence=evidence,
        docling_doc=conv_result,
        docling_chunks=docling_chunks,
        source_documents=source_documents,
        supplement_warnings=supplement_warnings,
    )
```

- [ ] **Step 4: Run the focused test**

Run:

```bash
uv run pytest tests/test_assessment_ingestion.py::test_ingest_assessment_documents_returns_primary_structural_result_when_remote_disabled -q
```

Expected: PASS.

- [ ] **Step 5: Run all assessment ingestion tests**

Run:

```bash
uv run pytest tests/test_assessment_ingestion.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add rob2_pipeline/ingestion/assessment.py tests/test_assessment_ingestion.py
git commit -m "refactor: move primary ingestion success path"
```

---

### Task 3: Move Supplement Tolerance Into The Assessment Ingestion Module

**Files:**
- Modify: `rob2_pipeline/ingestion/assessment.py`
- Modify: `tests/test_assessment_ingestion.py`
- Modify: `tests/test_supplements.py`

- [ ] **Step 1: Write the failing test**

Append this test to `tests/test_assessment_ingestion.py`:

```python
def test_ingest_assessment_documents_preserves_primary_when_supplement_ingestion_escapes(
    monkeypatch,
):
    import rob2_pipeline.ingestion.assessment as assessment

    evidence = empty_paper_evidence("docling_struct")
    primary_chunk = Document(
        page_content="Primary chunk",
        metadata={"section": "Methods", "page_numbers": [1]},
    )

    monkeypatch.setattr(assessment, "extract_full_text", lambda path: "Primary text")
    monkeypatch.setattr(assessment, "_configure_docling_runtime", lambda: None)
    monkeypatch.setattr(assessment, "_build_docling_chunks", lambda conv_result: [primary_chunk])
    monkeypatch.setattr(
        assessment,
        "build_document_repr",
        lambda doc: type(
            "DocRepr",
            (),
            {
                "full_text": "Primary text",
                "to_prompt_repr": lambda self: "Primary text",
                "blocks": [],
            },
        )(),
    )
    monkeypatch.setattr(assessment, "extract_structural_paper_evidence", lambda doc_repr: evidence)
    monkeypatch.setattr(assessment, "allow_remote_evidence_extraction", lambda: False)
    monkeypatch.setattr(
        assessment,
        "ingest_supplements",
        lambda paths: (_ for _ in ()).throw(RuntimeError("unexpected supplement error")),
    )

    class Result:
        document = object()

    class Converter:
        def convert(self, path):
            return Result()

    monkeypatch.setattr(assessment, "_get_docling_converter", lambda use_ocr=False: Converter())

    result = assessment.ingest_assessment_documents("primary.pdf", ["protocol.pdf"])

    assert len(result.docling_chunks) == 1
    assert result.docling_chunks[0].metadata["document_id"] == "primary"
    assert result.source_documents == [
        {
            "document_id": "primary",
            "document_name": "primary.pdf",
            "document_role": "primary",
            "source_kind": "rag_chunk",
            "path": "primary.pdf",
            "is_primary": True,
            "status": "parsed",
        }
    ]
    assert result.supplement_warnings == [
        "Supplement ingestion failed: unexpected supplement error"
    ]
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
uv run pytest tests/test_assessment_ingestion.py::test_ingest_assessment_documents_preserves_primary_when_supplement_ingestion_escapes -q
```

Expected: FAIL because `RuntimeError("unexpected supplement error")` escapes.

- [ ] **Step 3: Implement supplement exception tolerance**

In `ingest_assessment_documents()`, replace:

```python
    supplement_chunks, supplement_documents, supplement_warnings = ingest_supplements(
        supplementary_paths
    )
```

with:

```python
    try:
        supplement_chunks, supplement_documents, supplement_warnings = ingest_supplements(
            supplementary_paths
        )
    except Exception as error:  # noqa: BLE001
        supplement_chunks = []
        supplement_documents = []
        supplement_warnings = [f"Supplement ingestion failed: {error}"]
```

- [ ] **Step 4: Move graph-node supplement tests to the new module**

In `tests/test_supplements.py`, rename:

```python
def test_pdf_ingest_node_appends_supplement_chunks(monkeypatch):
```

to:

```python
def test_assessment_ingestion_appends_supplement_chunks(monkeypatch):
```

Change:

```python
    import rob2_pipeline.nodes.ingest as node
```

to:

```python
    import rob2_pipeline.ingestion.assessment as node
```

Change:

```python
    result = node.pdf_ingest_node(
        {"pdf_path": "primary.pdf", "supplementary_paths": ["protocol.pdf"]}
    )
```

to:

```python
    result = node.ingest_assessment_documents("primary.pdf", ["protocol.pdf"]).to_state_update()
```

In the same file, rename:

```python
def test_pdf_ingest_node_preserves_primary_chunks_when_supplement_ingestion_escapes(
```

to:

```python
def test_assessment_ingestion_preserves_primary_chunks_when_supplement_ingestion_escapes(
```

Change its `import rob2_pipeline.nodes.ingest as node` line to:

```python
    import rob2_pipeline.ingestion.assessment as node
```

Change its call from:

```python
    result = node.pdf_ingest_node(
        {"pdf_path": "primary.pdf", "supplementary_paths": ["protocol.pdf"]}
    )
```

to:

```python
    result = node.ingest_assessment_documents("primary.pdf", ["protocol.pdf"]).to_state_update()
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
uv run pytest tests/test_assessment_ingestion.py tests/test_supplements.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add rob2_pipeline/ingestion/assessment.py tests/test_assessment_ingestion.py tests/test_supplements.py
git commit -m "refactor: keep supplement tolerance inside assessment ingestion"
```

---

### Task 4: Move Remote Evidence Extraction And RCT Skip Behavior

**Files:**
- Modify: `rob2_pipeline/ingestion/assessment.py`
- Modify: `tests/test_assessment_ingestion.py`
- Modify: `tests/test_pdf_ingestion.py`

- [ ] **Step 1: Write the failing tests**

Append these tests to `tests/test_assessment_ingestion.py`:

```python
def test_ingest_assessment_documents_skips_remote_extraction_for_apparent_non_rct(
    monkeypatch,
):
    import rob2_pipeline.ingestion.assessment as assessment

    structural = empty_paper_evidence("docling_struct")
    monkeypatch.setattr(assessment, "extract_full_text", lambda path: "Primary text")
    monkeypatch.setattr(assessment, "_configure_docling_runtime", lambda: None)
    monkeypatch.setattr(assessment, "_build_docling_chunks", lambda conv_result: [])
    monkeypatch.setattr(
        assessment,
        "build_document_repr",
        lambda doc: type(
            "DocRepr",
            (),
            {
                "full_text": "Editorial commentary.",
                "to_prompt_repr": lambda self: "Editorial commentary.",
                "blocks": [],
            },
        )(),
    )
    monkeypatch.setattr(assessment, "extract_structural_paper_evidence", lambda doc_repr: structural)
    monkeypatch.setattr(assessment, "allow_remote_evidence_extraction", lambda: True)
    monkeypatch.setattr(assessment, "appears_rct_candidate", lambda text: False)
    monkeypatch.setattr(assessment, "ingest_supplements", lambda paths: ([], [], []))

    def fail_if_called(doc_repr):
        raise AssertionError("remote extraction should be skipped")

    monkeypatch.setattr(assessment, "extract_paper_evidence", fail_if_called)

    class Result:
        document = object()

    class Converter:
        def convert(self, path):
            return Result()

    monkeypatch.setattr(assessment, "_get_docling_converter", lambda use_ocr=False: Converter())

    result = assessment.ingest_assessment_documents("primary.pdf", [])

    assert result.evidence["extraction_method"] == "docling_struct"
    assert (
        "Remote evidence extraction skipped for apparent non-RCT document."
        in result.evidence["warnings"]
    )


def test_ingest_assessment_documents_returns_llm_evidence_and_log(monkeypatch):
    import rob2_pipeline.ingestion.assessment as assessment

    structural = empty_paper_evidence("docling_struct")
    remote = empty_paper_evidence("docling_llm")
    monkeypatch.setattr(assessment, "extract_full_text", lambda path: "Primary text")
    monkeypatch.setattr(assessment, "_configure_docling_runtime", lambda: None)
    monkeypatch.setattr(assessment, "_build_docling_chunks", lambda conv_result: [])
    monkeypatch.setattr(
        assessment,
        "build_document_repr",
        lambda doc: type(
            "DocRepr",
            (),
            {
                "full_text": "Randomized trial.",
                "to_prompt_repr": lambda self: "Randomized trial.",
                "blocks": [],
            },
        )(),
    )
    monkeypatch.setattr(assessment, "extract_structural_paper_evidence", lambda doc_repr: structural)
    monkeypatch.setattr(assessment, "allow_remote_evidence_extraction", lambda: True)
    monkeypatch.setattr(assessment, "appears_rct_candidate", lambda text: True)
    monkeypatch.setattr(
        assessment,
        "extract_paper_evidence",
        lambda doc_repr: (
            remote,
            [
                {
                    "node": "paper_evidence_extraction",
                    "prompt_length_chars": 120,
                    "response_length_chars": 80,
                    "latency_ms": 5,
                    "cache_hit": False,
                }
            ],
        ),
    )
    monkeypatch.setattr(assessment, "ingest_supplements", lambda paths: ([], [], []))

    class Result:
        document = object()

    class Converter:
        def convert(self, path):
            return Result()

    monkeypatch.setattr(assessment, "_get_docling_converter", lambda use_ocr=False: Converter())

    result = assessment.ingest_assessment_documents("primary.pdf", [])

    assert result.evidence is remote
    assert result.llm_call_log == [
        {
            "node": "paper_evidence_extraction",
            "prompt_length_chars": 120,
            "response_length_chars": 80,
            "latency_ms": 5,
            "cache_hit": False,
        }
    ]


def test_ingest_assessment_documents_falls_back_when_remote_extraction_fails(
    monkeypatch,
):
    import rob2_pipeline.ingestion.assessment as assessment

    structural = empty_paper_evidence("docling_struct")
    monkeypatch.setattr(assessment, "extract_full_text", lambda path: "Primary text")
    monkeypatch.setattr(assessment, "_configure_docling_runtime", lambda: None)
    monkeypatch.setattr(assessment, "_build_docling_chunks", lambda conv_result: [])
    monkeypatch.setattr(
        assessment,
        "build_document_repr",
        lambda doc: type(
            "DocRepr",
            (),
            {
                "full_text": "Randomized trial.",
                "to_prompt_repr": lambda self: "Randomized trial.",
                "blocks": [],
            },
        )(),
    )
    monkeypatch.setattr(assessment, "extract_structural_paper_evidence", lambda doc_repr: structural)
    monkeypatch.setattr(assessment, "allow_remote_evidence_extraction", lambda: True)
    monkeypatch.setattr(assessment, "appears_rct_candidate", lambda text: True)
    monkeypatch.setattr(
        assessment,
        "extract_paper_evidence",
        lambda doc_repr: (_ for _ in ()).throw(RuntimeError("bad xml")),
    )
    monkeypatch.setattr(assessment, "ingest_supplements", lambda paths: ([], [], []))

    class Result:
        document = object()

    class Converter:
        def convert(self, path):
            return Result()

    monkeypatch.setattr(assessment, "_get_docling_converter", lambda use_ocr=False: Converter())

    result = assessment.ingest_assessment_documents("primary.pdf", [])

    assert result.evidence["extraction_method"] == "docling_struct"
    assert "LLM evidence extraction failed: bad xml" in result.evidence["warnings"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run pytest tests/test_assessment_ingestion.py::test_ingest_assessment_documents_skips_remote_extraction_for_apparent_non_rct tests/test_assessment_ingestion.py::test_ingest_assessment_documents_returns_llm_evidence_and_log tests/test_assessment_ingestion.py::test_ingest_assessment_documents_falls_back_when_remote_extraction_fails -q
```

Expected: at least the remote-success and remote-failure tests FAIL because Task 2 returns structural evidence whenever remote extraction is enabled.

- [ ] **Step 3: Implement remote extraction behavior**

Add imports to `rob2_pipeline/ingestion/assessment.py`:

```python
from rob2_pipeline.ingestion.evidence import (
    extract_paper_evidence,
    extract_structural_paper_evidence,
)
from rob2_pipeline.ingestion.settings import (
    allow_remote_evidence_extraction,
    appears_rct_candidate,
)
```

Replace the final return in `ingest_assessment_documents()` with:

```python
    if not appears_rct_candidate(doc_repr.to_prompt_repr() or doc_repr.full_text):
        evidence["warnings"].append(
            "Remote evidence extraction skipped for apparent non-RCT document."
        )
        return AssessmentIngestionResult(
            full_text=full_text,
            evidence=evidence,
            docling_doc=conv_result,
            docling_chunks=docling_chunks,
            source_documents=source_documents,
            supplement_warnings=supplement_warnings,
        )

    try:
        evidence, log = extract_paper_evidence(doc_repr)
        return AssessmentIngestionResult(
            full_text=full_text,
            evidence=evidence,
            docling_doc=conv_result,
            docling_chunks=docling_chunks,
            source_documents=source_documents,
            supplement_warnings=supplement_warnings,
            llm_call_log=log,
        )
    except Exception as error:  # noqa: BLE001
        evidence = extract_structural_paper_evidence(doc_repr)
        evidence["warnings"].append(f"LLM evidence extraction failed: {error}")
        return AssessmentIngestionResult(
            full_text=full_text,
            evidence=evidence,
            docling_doc=conv_result,
            docling_chunks=docling_chunks,
            source_documents=source_documents,
            supplement_warnings=supplement_warnings,
        )
```

- [ ] **Step 4: Move existing node remote-extraction tests to the deep module**

In `tests/test_pdf_ingestion.py`, for these tests:

- `test_ingest_node_stores_docling_conversion_result`
- `test_ingest_node_skips_remote_extraction_when_disabled`
- `test_ingest_node_skips_remote_extraction_for_apparent_non_rct`

Change imports and calls so they target `rob2_pipeline.ingestion.assessment.ingest_assessment_documents()` instead of `pdf_ingest_node()`.

For each test, replace:

```python
from rob2_pipeline.nodes.ingest import pdf_ingest_node
```

with local imports inside the affected test:

```python
import rob2_pipeline.ingestion.assessment as assessment
```

Replace monkeypatch targets like:

```python
"rob2_pipeline.nodes.ingest.extract_full_text"
```

with:

```python
"rob2_pipeline.ingestion.assessment.extract_full_text"
```

Replace:

```python
result = pdf_ingest_node({"pdf_path": "trial.pdf"})
```

with:

```python
result = assessment.ingest_assessment_documents("trial.pdf").to_state_update()
```

Keep `from rob2_pipeline.nodes.ingest import pdf_ingest_node` only for tests that still specifically verify the graph adapter.

- [ ] **Step 5: Run focused tests**

Run:

```bash
uv run pytest tests/test_assessment_ingestion.py tests/test_pdf_ingestion.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add rob2_pipeline/ingestion/assessment.py tests/test_assessment_ingestion.py tests/test_pdf_ingestion.py
git commit -m "refactor: move remote paper evidence extraction"
```

---

### Task 5: Move Structural Fallback When Primary Docling Conversion Fails

**Files:**
- Modify: `rob2_pipeline/ingestion/assessment.py`
- Modify: `tests/test_assessment_ingestion.py`
- Modify: `tests/test_pdf_ingestion.py`

- [ ] **Step 1: Write the failing test**

Append this test to `tests/test_assessment_ingestion.py`:

```python
def test_ingest_assessment_documents_falls_back_to_keyword_parse_when_docling_structure_fails(
    monkeypatch,
):
    import rob2_pipeline.ingestion.assessment as assessment

    known_text = (
        "Methods\nParticipants were randomly assigned in a 1:1 ratio.\nResults\nDone."
    )
    monkeypatch.setattr(assessment, "extract_full_text", lambda path: known_text)

    class BrokenConverter:
        def convert(self, path):
            raise RuntimeError("docling structured parse failed")

    monkeypatch.setattr(assessment, "_get_docling_converter", lambda use_ocr=False: BrokenConverter())

    result = assessment.ingest_assessment_documents(
        "trial.pdf",
        ["inputs/benchmark/supplement/TITAN/protocol.pdf"],
    )

    assert result.evidence["extraction_method"] == "fallback"
    assert result.docling_doc is None
    assert result.docling_chunks == []
    assert "randomly assigned" in result.evidence["methods"]["text"]
    assert result.source_documents[0]["document_role"] == "primary"
    assert result.source_documents[1]["document_name"] == "protocol.pdf"
    assert result.source_documents[1]["status"] == "failed"
    assert "Supplement not ingested" in result.supplement_warnings[0]
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
uv run pytest tests/test_assessment_ingestion.py::test_ingest_assessment_documents_falls_back_to_keyword_parse_when_docling_structure_fails -q
```

Expected: FAIL because the converter exception escapes.

- [ ] **Step 3: Implement structural fallback**

Add imports in `rob2_pipeline/ingestion/assessment.py`:

```python
from rob2_pipeline.ingestion.evidence import (
    extract_paper_evidence,
    extract_structural_paper_evidence,
    paper_evidence_from_sections,
    parse_sections,
)
from rob2_pipeline.ingestion.supplements import (
    apply_source_metadata,
    ingest_supplements,
    primary_source_document,
    skipped_source_documents,
)
```

Wrap the Docling conversion and downstream structural/remote work in:

```python
    try:
        _configure_docling_runtime()
        converter = _get_docling_converter(use_ocr=False)
        conv_result = converter.convert(pdf_path)
        ...
    except Exception as error:
        sections = parse_sections(full_text)
        evidence = paper_evidence_from_sections(
            sections,
            extraction_method="fallback",
            source="keyword_fallback",
            warnings=[
                "Docling structural extraction failed; used text keyword fallback."
            ],
        )
        supplement_documents, supplement_warnings = skipped_source_documents(
            supplementary_paths,
            f"primary Docling structural extraction failed: {error}",
        )
        return AssessmentIngestionResult(
            full_text=full_text,
            evidence=evidence,
            docling_doc=None,
            docling_chunks=[],
            source_documents=[primary_source, *supplement_documents],
            supplement_warnings=supplement_warnings,
        )
```

The `extract_full_text(pdf_path)` call must stay before this `try`. If full-text extraction fails, it must still halt the run.

- [ ] **Step 4: Move existing fallback tests to the new module**

In `tests/test_pdf_ingestion.py`, update these tests to target `assessment.ingest_assessment_documents()`:

- `test_ingest_node_falls_back_to_text_parse_when_docling_structure_fails`
- `test_ingest_node_records_skipped_supplements_when_primary_docling_falls_back`

Replace node monkeypatch targets with `rob2_pipeline.ingestion.assessment.*` targets.

Replace:

```python
result = pdf_ingest_node(state)
```

with:

```python
result = assessment.ingest_assessment_documents(
    state["pdf_path"], state.get("supplementary_paths")
).to_state_update()
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
uv run pytest tests/test_assessment_ingestion.py tests/test_pdf_ingestion.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add rob2_pipeline/ingestion/assessment.py tests/test_assessment_ingestion.py tests/test_pdf_ingestion.py
git commit -m "refactor: move primary docling fallback"
```

---

### Task 6: Turn pdf_ingest_node Into A Thin Graph Adapter

**Files:**
- Modify: `rob2_pipeline/nodes/ingest.py`
- Test: `tests/test_pdf_ingestion.py`
- Test: `tests/test_pipeline.py`
- Test: `tests/test_graph.py`

- [ ] **Step 1: Write the failing graph-adapter test**

Append this test to `tests/test_pdf_ingestion.py`:

```python
def test_pdf_ingest_node_adapts_assessment_ingestion_result(monkeypatch):
    import rob2_pipeline.nodes.ingest as ingest_node
    from rob2_pipeline.ingestion.assessment import AssessmentIngestionResult

    evidence = empty_paper_evidence("docling_struct")
    captured = {}

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
            llm_call_log=[
                {
                    "node": "paper_evidence_extraction",
                    "prompt_length_chars": 120,
                    "response_length_chars": 80,
                    "latency_ms": 5,
                    "cache_hit": False,
                }
            ],
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
        "supplement_warnings": ["warning"],
        "llm_call_log": [
            {
                "node": "paper_evidence_extraction",
                "prompt_length_chars": 120,
                "response_length_chars": 80,
                "latency_ms": 5,
                "cache_hit": False,
            }
        ],
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
uv run pytest tests/test_pdf_ingestion.py::test_pdf_ingest_node_adapts_assessment_ingestion_result -q
```

Expected: FAIL because `rob2_pipeline.nodes.ingest` does not yet import `ingest_assessment_documents`.

- [ ] **Step 3: Replace ingestion orchestration in the node**

In `rob2_pipeline/nodes/ingest.py`, remove imports that are only used by `pdf_ingest_node()`:

```python
from pathlib import Path
from rob2_pipeline.ingestion.supplements import ...
from rob2_pipeline.pdf_ingestion import ...
```

Keep imports used by `rct_screener_node()`:

```python
from rob2_pipeline.nodes.common import call_node_llm
from rob2_pipeline.models import format_evidence
from rob2_pipeline.prompts import PROMPT_RCT_SCREEN
from rob2_pipeline.state import RoB2State
from rob2_pipeline.xml_parser import extract_tag
```

Add:

```python
from rob2_pipeline.ingestion.assessment import ingest_assessment_documents
```

Replace the whole `pdf_ingest_node()` body with:

```python
def pdf_ingest_node(state: RoB2State) -> RoB2State:
    result = ingest_assessment_documents(
        state["pdf_path"], list(state.get("supplementary_paths") or [])
    )
    return result.to_state_update()
```

- [ ] **Step 4: Run adapter tests**

Run:

```bash
uv run pytest tests/test_pdf_ingestion.py::test_pdf_ingest_node_adapts_assessment_ingestion_result tests/test_graph.py tests/test_pipeline.py -q
```

Expected: PASS.

- [ ] **Step 5: Run ingestion-related tests**

Run:

```bash
uv run pytest tests/test_assessment_ingestion.py tests/test_pdf_ingestion.py tests/test_supplements.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add rob2_pipeline/nodes/ingest.py tests/test_pdf_ingestion.py
git commit -m "refactor: make pdf ingest node a graph adapter"
```

---

### Task 7: Preserve pdf_ingestion.py Compatibility Explicitly

**Files:**
- Modify: `tests/test_pdf_ingestion.py`
- Optionally modify: `rob2_pipeline/pdf_ingestion.py`

- [ ] **Step 1: Strengthen facade compatibility tests**

In `tests/test_pdf_ingestion.py`, extend `test_pdf_ingestion_facade_reexports_core_ingestion_api()` to assert these compatibility exports:

```python
    assert callable(pdf_ingestion._configure_docling_runtime)
    assert callable(pdf_ingestion._get_docling_converter)
    assert callable(pdf_ingestion._build_docling_chunks)
    assert callable(pdf_ingestion.build_document_repr)
    assert callable(pdf_ingestion.extract_structural_paper_evidence)
    assert callable(pdf_ingestion.paper_evidence_from_sections)
    assert callable(pdf_ingestion.appears_rct_candidate)
    assert callable(pdf_ingestion.allow_remote_evidence_extraction)
```

- [ ] **Step 2: Run the compatibility test**

Run:

```bash
uv run pytest tests/test_pdf_ingestion.py::test_pdf_ingestion_facade_reexports_core_ingestion_api -q
```

Expected: PASS. If it fails, restore the missing export in `rob2_pipeline/pdf_ingestion.py` by importing it from the focused ingestion module and adding it to `__all__`.

- [ ] **Step 3: Run all facade tests**

Run:

```bash
uv run pytest tests/test_pdf_ingestion.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_pdf_ingestion.py rob2_pipeline/pdf_ingestion.py
git commit -m "test: lock pdf ingestion facade compatibility"
```

---

### Task 8: Update Domain Documentation

**Files:**
- Modify: `CONTEXT.md`
- Optionally modify: `ARCHITECTURE.md`

- [ ] **Step 1: Update the module map**

In `CONTEXT.md`, replace the ingestion bullet:

```markdown
- `rob2_pipeline/nodes/ingest.py` is the graph-facing ingestion node. It owns
  the fallback order and combines primary plus supplement chunks.
```

with:

```markdown
- `rob2_pipeline/nodes/ingest.py` is the graph-facing ingestion adapter. It
  calls the Assessment ingestion module and returns the existing `RoB2State`
  update shape.
- `rob2_pipeline/ingestion/assessment.py` owns Assessment ingestion behavior:
  strict primary full-text extraction, primary Docling structural parsing,
  primary plus supplement chunk assembly, source-document inventory, remote
  remote paper-evidence extraction orchestration, and fallback order.
```

- [ ] **Step 2: Update the architecture document**

In `ARCHITECTURE.md`, in the ingestion subsystem table, add this row:

```markdown
| `rob2_pipeline/ingestion/assessment.py` | Primary plus supplement Assessment ingestion behavior and fallback order |
```

Change the `rob2_pipeline/nodes/ingest.py` row to:

```markdown
| `rob2_pipeline/nodes/ingest.py` | Graph adapter for ingestion plus RCT screening node |
```

- [ ] **Step 3: Run documentation grep checks**

Run:

```bash
rg "fallback order and combines primary plus supplement chunks|graph-facing ingestion node" CONTEXT.md ARCHITECTURE.md
```

Expected: no output.

Run:

```bash
rg "ingestion/assessment.py|Assessment ingestion behavior" CONTEXT.md ARCHITECTURE.md
```

Expected: output from both files.

- [ ] **Step 4: Commit**

```bash
git add CONTEXT.md ARCHITECTURE.md
git commit -m "docs: document assessment ingestion module"
```

---

### Task 9: Final Verification

**Files:**
- No code changes expected.

- [ ] **Step 1: Run the full test suite**

Run:

```bash
uv run pytest -q
```

Expected: PASS.

- [ ] **Step 2: Run ruff if available through the project**

Run:

```bash
uv run ruff check .
```

Expected: PASS. If this reports pre-existing unrelated issues, record them in the implementation summary and do not refactor unrelated files.

- [ ] **Step 3: Inspect changed files**

Run:

```bash
git diff --stat
```

Expected: changes limited to:

```text
CONTEXT.md
ARCHITECTURE.md
rob2_pipeline/ingestion/__init__.py
rob2_pipeline/ingestion/assessment.py
rob2_pipeline/nodes/ingest.py
tests/test_assessment_ingestion.py
tests/test_pdf_ingestion.py
tests/test_supplements.py
```

- [ ] **Step 4: Commit any verification fixes**

If final verification required small fixes, commit them:

```bash
git add CONTEXT.md ARCHITECTURE.md rob2_pipeline/ingestion/__init__.py rob2_pipeline/ingestion/assessment.py rob2_pipeline/nodes/ingest.py tests/test_assessment_ingestion.py tests/test_pdf_ingestion.py tests/test_supplements.py
git commit -m "test: verify assessment ingestion refactor"
```

If no fixes were required, do not create an empty commit.

---

## Self-Review

**Spec coverage:** The plan creates a new deep Assessment ingestion module, preserves the `pdf_ingestion.py` facade, returns a typed dataclass, keeps `pdf_ingest_node()` as the graph adapter, and updates domain documentation.

**Placeholder scan:** No task contains TBD, TODO, future-only error handling, or unspecified test instructions. Every code-changing step includes concrete code or exact replacement instructions.

**Type consistency:** `AssessmentIngestionResult` fields match the current `pdf_ingest_node()` output keys. `to_state_update()` preserves the existing omission of empty `llm_call_log` and includes the log when remote extraction succeeds.
