# ADR-0005: Use Packet-Bound SQ Classification For Domain 1

Date: 2026-06-07

## Status

Accepted

## Context

Domain 1 previously kept bespoke deterministic controls that could rewrite SQ
answers after classification and again after support adjudication. Those
controls encoded benchmark-driven heuristics for randomized design evidence,
allocation concealment, and baseline-balance polarity.

## Decision

Domain 1 should use the same packet-bound SQ classification concept and generic
domain SQ artifact shape as D2-D5. The selected Evidence Packet and its decision
table are the evidence authority for D1 SQ answers; support audit and SQ Support
Adjudication handle weak or unsupported pivotal answers. D1-specific
deterministic answer-rewriting controls and D1-only SQ answer artifact shapes
should be removed rather than preserved behind the generic classifier seam.
Classifier artifacts should be stored in a nested
`domain_sq_classifier_artifacts[domain][stage]` state shape instead of
domain-specific top-level keys. Workspace SQ answer artifacts should remain
per-domain files written through the generic domain writer. D1-only engineering
diagnostics should be removed rather than generalized in this consolidation.

## Consequences

The packet-bound classifier owns shared schema validation, required-SQ checks,
fallbacks, and outside-packet quote rejection for D1-D5. Workspace persistence,
reporting, and tests should treat D1 as a generic domain SQ answer set instead
of using D1-only writer modules, artifact builders, state keys, filenames, or
schema versions. The migration should remove and rename D1-only interfaces
rather than keeping compatibility aliases. Domain 1 tests should cover
packet-bound classification, quote binding, support audit, and deterministic
judging rather than local guard mutation.
