from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read_doc(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_domain_and_architecture_docs_describe_json_contract_sq_outputs() -> None:
    docs = "\n\n".join(
        [
            _read_doc("CONTEXT.md"),
            _read_doc("ARCHITECTURE.md"),
            _read_doc("README.md"),
        ]
    )

    assert "Signaling Question Answer\n\nThe parsed JSON-contract answer" in docs
    assert "JSON contract validation and repair" in docs
    assert "XML parsing" not in docs
    assert "XML answer" not in docs


def test_architecture_docs_describe_parser_neutral_ingestion_state() -> None:
    architecture = _read_doc("ARCHITECTURE.md")

    assert "page-aware parser artifacts" in architecture
    assert "parse_artifacts" in architecture
    assert "docling_chunks" not in architecture


def test_architecture_docs_name_evidence_store_and_workspace_boundaries() -> None:
    docs = "\n\n".join([_read_doc("CONTEXT.md"), _read_doc("ARCHITECTURE.md")])

    assert "EvidenceStore" in docs
    assert "Trial Workspace artifacts" in docs
    assert "Outcome Workspace artifacts" in docs
    assert "JSON-contract SQ answers" in docs


def test_trial_reuse_adr_marks_docling_design_as_superseded() -> None:
    adr = _read_doc("docs/adr/0002-reuse-trial-level-ingestion-and-retrieval-indexes.md")

    assert "Superseded implementation note" in adr
    assert "LiteParse" in adr
    assert "Docling conversion" not in adr
