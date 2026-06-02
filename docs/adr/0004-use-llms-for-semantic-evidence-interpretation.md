# ADR-0004: Use LLMs For Semantic Evidence Interpretation

Date: 2026-06-02

## Status

Accepted

## Context

Benchmark failures showed that broad regex heuristics can misclassify assessed
outcomes and over-trust weak signaling-question answers. The pipeline needs to
work across randomized trials from many medical fields, where patient-reported,
clinician-assessed, objective, composite, safety, and adjudicated outcomes are
semantic distinctions grounded in local trial text.

## Decision

LLMs own semantic interpretation of trial evidence when clinical or
methodological judgment is required. This includes assessed-outcome-bound
classification, SQ evidence support, and whether quoted evidence actually
supports a RoB 2 claim. Deterministic code owns fixed RoB 2 decision tables,
graph routing, schema validation, provenance and quote-traceability checks,
support gating, and benchmark comparison.

Regex and other deterministic heuristics may be used when they are the simplest
fit for retrieval aids, cheap provenance checks, contradiction flags, or fallback
diagnostics. Redundant regex classifiers should be removed rather than retained
behind new abstractions, and regex must not be the primary authority for
semantic classification of trial evidence.

## Consequences

`outcome_resolver` should become an evidence-bounded semantic resolver rather
than a broad paper-wide regex classifier. Weak or unsupported semantic
classifications should remain visible as support levels and flow through the
same audit/adjudication model used for SQ answers when they are pivotal.

LLM-based semantic resolvers should have lean fallbacks. If structured semantic
resolution fails validation, the pipeline should emit unsupported classification
with audit/review flags rather than rebuilding the same semantic decision with a
parallel regex fallback.
