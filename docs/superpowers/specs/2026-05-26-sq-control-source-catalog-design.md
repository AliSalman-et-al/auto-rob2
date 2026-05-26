# SQ Control Flow And SourceDocument Catalog Design

Date: 2026-05-26

## Scope

This design covers two sequenced, behavior-preserving refactors:

1. Deepen SQ control flow for D2, D3, and D4.
2. Create a SourceDocument catalog module for provenance and inventory behavior.

The refactors are sequenced. SQ control flow comes first because it is lower
I/O risk and establishes the deep-module pattern before changing ingestion
provenance code.

## Goals

- Concentrate RoB 2 SQ applicability, NA propagation, D2 routing, and D4
  outcome-type corrections in one module.
- Concentrate SourceDocument role, status, error, and chunk metadata behavior in
  one module.
- Preserve existing `RoB2State`, JSON output, benchmark behavior, and dict-shaped
  `SourceDocument` values.
- Add focused tests around the new interfaces while keeping runtime behavior
  unchanged.

## Non-Goals

- Do not move prompt construction, LLM calls, evidence packet selection, or
  deterministic domain judgments into the SQ control module.
- Do not move Docling setup, converter caching, supplement page-window parsing,
  or supplement parsing orchestration into the SourceDocument catalog.
- Do not migrate `SourceDocument` to dataclasses in this refactor.
- Do not alter `DomainEvidenceContext`; ADR-0001 remains accepted and unchanged.

## Design 1: SQ Control Flow

Create `rob2_pipeline/nodes/sq_control.py` as the module for SQ control behavior.
Its interface is domain/stage-level rather than individual-rule-level:

```python
apply_domain2_sq12_control(state, sq_answers) -> dict[str, dict]
next_domain2_stage(state) -> Literal["conditional", "analysis"]
apply_domain2_conditional_control(state, sq_answers) -> dict[str, dict]
apply_domain2_analysis_control(state, sq_answers) -> dict[str, dict]
apply_domain3_control(state, sq_answers) -> dict[str, dict]
apply_domain4_control(state, sq_answers) -> dict[str, dict]
```

`domain2.py`, `domain3.py`, and `domain4.py` keep their existing graph-facing
responsibilities: prompt construction, `DomainSqStage` declarations, node
functions, and deterministic judge calls. They import stage-level control
functions from `sq_control.py`.

`graph.py` should route D2 using `next_domain2_stage`. The import can come
directly from `sq_control.py`, or `domain2.py` can re-export it if that keeps
graph imports more coherent. The implementation should choose the cleaner local
shape during planning.

The SQ control module may keep smaller private helper functions internally, but
callers should not depend on individual rule functions. The deletion test should
pass: deleting `sq_control.py` would force D2/D3/D4 branching rules back into
multiple domain modules.

## Design 2: SourceDocument Catalog

Create `rob2_pipeline/ingestion/source_catalog.py` as the module for
SourceDocument provenance and inventory behavior. It keeps returning the current
dict-shaped `SourceDocument` values.

Target interface:

```python
classify_document_role(path: Path) -> str
primary_source_document(path: Path) -> SourceDocument
supplement_source_document(path: Path, index: int) -> SourceDocument
mark_missing(source, path) -> SourceDocument
mark_failed(source, message) -> SourceDocument
mark_parsed(source) -> SourceDocument
mark_partial(source, warnings) -> SourceDocument
skipped_source_documents(paths, reason) -> tuple[list[SourceDocument], list[str]]
apply_source_metadata(chunks, source) -> list
```

`ingestion/supplements.py` keeps supplement parsing orchestration, Docling
converter setup, page-window parsing, and environment-derived page limits. It
uses the catalog to classify roles, create source records, update statuses, and
enrich chunk metadata.

`ingestion/assessment.py` uses the catalog for the primary source record and
for skipped supplement records after primary structural extraction failure.
`pipeline.py` and `benchmark.py` should not need behavior changes beyond any
import cleanup caused by the move.

The catalog owns these invariants:

- Primary paper records always use `document_id="primary"`,
  `document_role="primary"`, `source_kind="rag_chunk"`, `is_primary=True`, and
  `status="parsed"`.
- Supplement records use stable ids in `supplement:NNN` format.
- Missing and failed supplement records always include both `status` and
  `error`.
- Chunk metadata receives document id, name, role, source kind, and source path
  without discarding existing chunk metadata.

## Data Flow

SQ flow after the refactor:

1. A domain SQ node builds a prompt from `DomainEvidenceContext`.
2. `DomainSqStage` executes the LLM call and merges parsed SQ answers.
3. The stage calls a public `sq_control.py` function for stage-local control.
4. Domain judge nodes remain unchanged and consume the controlled SQ answers.

SourceDocument flow after the refactor:

1. Assessment ingestion creates the primary paper SourceDocument through the
   catalog.
2. Supplement ingestion creates and updates supplement SourceDocuments through
   the catalog while retaining Docling parsing ownership.
3. The catalog applies SourceDocument metadata to chunks.
4. Existing state and JSON output receive the same dict-shaped records.

## Error Handling

SQ control functions should preserve the current default behavior for missing
answers, including `NI` and `NA` defaults already present in D2/D3/D4 logic.
They should use the existing `set_na` behavior or equivalent output shape.

SourceDocument catalog functions should keep current warning strings stable
where tests and benchmark behavior depend on them. Status-marking helpers should
return updated dict-shaped records and avoid mutating unrelated fields.

## Testing

Testing is behavior-preserving.

For SQ control flow:

- Add direct tests for D2 SQ12 control, D2 conditional control, D2 analysis
  control, D2 routing, D3 control, and D4 outcome-type correction plus NA
  propagation.
- Keep existing domain-stage tests passing.
- Avoid changing judge expectations or prompt text.

For SourceDocument catalog:

- Move or update tests for role classification, primary records, skipped
  supplement records, missing and failed records, and metadata enrichment.
- Add focused invariant tests for primary id, supplement id format, missing and
  failed status/error shape, and metadata preservation.
- Keep assessment ingestion, supplement ingestion, pipeline, and benchmark tests
  passing.

## Rollout Plan

1. Implement SQ control flow module and update D2/D3/D4 imports.
2. Run focused SQ/domain tests.
3. Implement SourceDocument catalog and update ingestion imports.
4. Run focused ingestion/catalog tests.
5. Run the full test suite.

## Implementation Note

No open design decisions remain. The implementation plan may choose whether
`graph.py` imports `next_domain2_stage` directly from `sq_control.py` or through
a re-export from `domain2.py`, based on the smaller call-site change.
