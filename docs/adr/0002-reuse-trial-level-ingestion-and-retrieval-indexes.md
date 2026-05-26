# ADR-0002: Reuse Trial-Level Ingestion And Retrieval Indexes

Date: 2026-05-26

## Status

Accepted

## Context

Benchmark runs often assess multiple outcomes for the same trial. Each
outcome-specific `Assessment` currently repeats primary PDF Docling conversion,
supplement Docling conversion, primary-paper evidence extraction, chunk
creation, embedding, and FAISS index construction.

The timing traces show `pdf_ingest` dominating benchmark wall time. Repeating
the same trial-level parsing work for each outcome is wasteful because the
primary paper, selected supplements, source-document inventory, chunks, and
primary-paper evidence do not depend on the assessed outcome.

The risk is boundary confusion: an `Assessment` is outcome-specific, and RoB 2
signaling-question evidence selection must remain tied to the assessed outcome.

## Decision

Reuse trial-level ingestion artifacts and trial-level retrieval indexes across
multiple outcome-specific Assessments for the same primary paper and selected
supplements.

Primary PDF ingestion should avoid duplicate Docling conversion. One primary
Docling conversion should supply strict full text, primary-paper structural
representation, and primary chunks. OCR fallback can remain available when the
non-OCR conversion produces too little usable text.

Primary Docling table structure remains enabled by default. A fast-mode switch
may disable table structure for measurement or throughput-sensitive runs, but
that mode must be explicit because baseline, CONSORT, and results tables can
support RoB 2 evidence.

Supplement role classification is provenance, not a parsing gate. Supplement
ingestion should stay inclusive because useful RoB 2 evidence can appear in
documents with uncertain or low-value-looking roles. Speed improvements should
first come from bounded parsing: page-count-aware windows, exhaustion detection,
reasonable per-document caps, and evidence-seeking windows rather than skipping
documents solely by role. The default supplement scan cap should be 300 pages,
still bounded by real page count and overridable for benchmark experiments.

Reusable trial-level artifacts may include:

- primary-paper `full_text`
- primary-paper `PaperEvidence`
- primary Docling conversion output needed by downstream code
- primary and supplement Docling chunks
- source-document inventory
- supplement warnings
- the vector index built from those chunks

Outcome-specific artifacts must not be reused across outcomes:

- retrieved domain contexts
- RAG source selections emitted into output JSON
- evidence packets and evidence facts
- signaling-question prompts and answers
- domain judgments and overall judgment
- verification flags and report output

Initial implementation should use benchmark-only in-memory reuse. Persistent
disk caching can be added later after cache keys, invalidation, and option
compatibility are proven.

If primary-paper evidence extraction becomes outcome-specific, `PaperEvidence`
must move out of the trial-level artifact.

## Consequences

Benchmark runs avoid repeating the slowest Docling and embedding work for every
outcome of the same trial.

The public single-assessment behavior can remain unchanged while benchmark
orchestration passes precomputed trial-level artifacts into the internal
pipeline path.

The implementation needs explicit cache keys based on the primary PDF,
supplement paths, file identity, and ingestion/index options, even for
in-memory reuse.

Tests should protect the boundary: trial-level artifacts can be shared, but
retrieval output, evidence packets, prompts, judgments, and reports remain
outcome-specific.
