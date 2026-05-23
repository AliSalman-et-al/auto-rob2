# ADR-0001: Keep DomainEvidenceContext Limited To Prompt Assembly

Date: 2026-05-23

## Status

Accepted

## Context

`DomainEvidenceContext` was introduced to deepen prompt evidence assembly for
D1-D5 and Domain 2 prompt stages. Before this module, each domain SQ node knew
too much about primary-paper evidence fields, evidence-packet rendering, RAG
compatibility keys, trial facts, and registry defaults.

The tempting next step is to move adjacent behavior into the same module:
retrieval, evidence-packet source selection, SQ branching, NA control logic, or
deterministic judging. That would make the name broader but would also blur
separate RoB 2 responsibilities.

## Decision

`DomainEvidenceContext` is prompt assembly only.

It may combine existing state into prompt-ready dataclass values:

- primary-paper `PaperEvidence`
- evidence-packet blocks
- RAG compatibility text
- trial facts
- registry fields
- domain-specific prompt defaults

It must not own:

- RAG retrieval
- evidence-packet source selection or grading
- SQ branching
- NA control logic
- D2 ITT versus per-protocol routing
- D4 outcome-type corrections
- deterministic domain or overall judging

Those responsibilities stay in the retrieval, evidence-packet, domain-node, and
judge modules.

## Consequences

Prompt evidence assembly has one place to change and test.

The domain SQ nodes still own RoB 2 control flow, so D2 and D4 remain more
explicit than D1, D3, and D5.

Future changes that alter which evidence appears in a prompt should usually
start in `rob2_pipeline/nodes/domain_context.py`.

Future changes that alter which sources are selected, ranked, or graded should
stay in the evidence-packet modules.

Future changes that alter SQ applicability, NA propagation, algorithm routing,
or final judgment labels should stay in the domain nodes or judges.
