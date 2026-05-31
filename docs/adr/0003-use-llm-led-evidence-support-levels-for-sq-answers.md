# ADR-0003: Use LLM-Led Evidence Support Levels For SQ Answers

Date: 2026-05-31

## Status

Accepted

## Context

Benchmark failures showed that a signaling-question answer can cite relevant-looking evidence while still being weakly supported, semantically unsupported, or about the wrong RoB 2 concept. Brittle deterministic rules such as requiring exact denominators would miss common valid reporting language, while a full second verifier pass for every answer would add substantial cost.

## Decision

Each signaling-question answer should carry an LLM-assessed Evidence Support Level: `strong`, `moderate`, `weak`, or `unsupported`. The support level is part of the SQ reasoning, not a mechanical keyword or denominator check.

Domain judges continue to apply the RoB 2 decision tables, but weak or unsupported SQ answers receive a cheap Pivotality Test. The test is an explicit audit artifact, not a hidden mutation of the emitted SQ answer. If a weak or unsupported answer is pivotal to the Domain Judgment, the pipeline should run a focused SQ Support Adjudication retry rather than silently allowing that answer to drive the final judgment. Deterministic code should remain limited to schema, provenance, and quote-traceability checks; it should not try to make semantic RoB 2 support decisions with brittle rules.

## Consequences

SQ prompts, parsed answers, JSON output, reports, verification, and retry logic need to preserve support levels and support rationales. Additional LLM cost is targeted to weak or unsupported pivotal answers instead of every SQ answer. Reviewer-facing Markdown reports should display support levels compactly so the reasoning remains explainable. Full Pivotality Test details belong in JSON audit artifacts, while Markdown reports should summarize only adjudication-relevant cases.

Human review escalation should be conservative. Moderate support and isolated non-pivotal weak support should be visible but should not automatically raise review priority. Weak or unsupported pivotal answers should trigger focused SQ Support Adjudication first. Human review priority should rise only when adjudication remains weak or unsupported, conflicts with the first answer, or repeated weak support suggests systemic fragility.

Benchmark agreement should be scored against post-adjudication final judgments because adjudication is part of the pipeline. Benchmark reports should also preserve initial-versus-final deltas, counts of weak or unsupported answers, Pivotality Tests, adjudications triggered, SQ/support changes, and extra LLM cost.
