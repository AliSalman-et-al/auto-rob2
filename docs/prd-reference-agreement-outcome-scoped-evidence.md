# PRD: Improve Reference Agreement With Outcome-Scoped Evidence Slices

## Problem Statement

Benchmark Reference Agreement is poor despite ingesting the expected primary
PDFs, supplements, and registry data. The failure is downstream of ingestion:
outcome-property resolution is contaminated by paper-wide evidence, D4
overreacts or underreacts to open-label PFS evidence, D3 treats ITT/censoring
language as enough missingness evidence for time-to-event outcomes, and D2 can
silently switch adverse-event outcomes to an adhering/per-protocol estimand.

The result is that correct sources are present, but the pipeline gives the LLM
and deterministic control logic poorly scoped evidence and unstable endpoint
properties.

## Solution

Split accuracy optimization into independently shippable vertical slices that
improve domain-level Reference Agreement while preserving official RoB 2
judgment behavior as a separate policy.

Slice 1 builds outcome-scoped, source-prioritized LLM outcome resolution. Slice
2 calibrates D4 for objective OS and Open-Label PFS Concern. Slice 3 calibrates
D3 and D2 around Time-To-Event Missingness Evidence and the Benchmark AE
Estimand.

## User Stories

1. As a benchmark reviewer, I want OS, PFS, and AE Assessments to keep their Outcome Scope, so that unrelated paper-wide text does not change endpoint properties.
2. As a benchmark reviewer, I want the resolver to use the requested outcome and benchmark code, so that user intent constrains interpretation.
3. As a benchmark reviewer, I want outcome-property resolution to happen after registry enrichment, so that ClinicalTrials.gov evidence can support endpoint matching.
4. As a benchmark reviewer, I want primary paper, protocol, and SAP endpoint definitions to outrank registry text, so that stale or broad registry fields do not override clearer trial documents.
5. As a benchmark reviewer, I want LLM-derived Outcome Properties to include cited evidence snippets, so that endpoint classification is auditable.
6. As a benchmark reviewer, I want OS/death-from-any-cause endpoints treated as vital-status/objective outcomes, so that D4 does not inherit unrelated patient-reported or safety text.
7. As a benchmark reviewer, I want open-label PFS with clinician or investigator-assessed progression and no blinded adjudication to map to Some concerns, so that D4 matches the benchmark convention.
8. As a benchmark reviewer, I want PFS packets to isolate the assessed progression endpoint definition, so that D4 does not mix OS, toxicity, and unrelated endpoints.
9. As a benchmark reviewer, I want D3 to require direct Time-To-Event Missingness Evidence, so that ITT/censoring language alone does not imply nearly complete data.
10. As a benchmark reviewer, I want AE outcomes to keep the effect of assignment by default in benchmark runs, so that safety endpoint identity does not silently change the Domain 2 estimand.
11. As a benchmark reviewer, I want PEACE-1 D2 concern to come from evidence, not a trial-name prior, so that benchmark tuning generalizes to new studies.
12. As a developer, I want each slice to produce its own benchmark delta, so that I can tell which change improved or harmed Reference Agreement.
13. As a developer, I want deterministic guardrails around LLM outcome resolution, so that malformed or overbroad LLM output cannot destabilize downstream control logic.
14. As a developer, I want official overall judgment and benchmark-reference overall policy to remain distinct, so that Reference Agreement does not redefine clinical correctness.
15. As a human reviewer, I want generated reports to show the resolved Outcome Properties and evidence basis, so that I can audit why a domain followed a particular path.

## Implementation Decisions

- Slice 1: Replace regex-primary outcome classification with an LLM outcome resolver that emits constrained Outcome Properties and evidence citations.
- Slice 1: The resolver input must be outcome-local: requested outcome, benchmark outcome code when available, endpoint definitions, matched registry endpoint, and D4-relevant measurement/adjudication snippets.
- Slice 1: The resolver must use Source-Prioritized Outcome Resolution: requested outcome and benchmark code, primary paper endpoint definition, protocol or SAP endpoint definition, then registry endpoint match.
- Slice 1: Regexes may remain only as guardrails, validators, or fallback behavior when LLM resolution is unavailable; they should not be the primary classifier.
- Slice 1: Deterministic guardrails should prevent OS/death-from-any-cause from resolving to patient-reported or safety-harm merely because the paper mentions symptoms, toxicity, or adverse events elsewhere.
- Slice 1: The best deep module is an outcome resolution service with a small interface: input scoped evidence plus outcome code, output structured Outcome Properties, Outcome Type, cited support, and warnings.
- Slice 2: D4 control should distinguish objective OS from Open-Label PFS Concern before NA propagation and final judging.
- Slice 2: Open-label PFS with clinician/investigator progression assessment and no demonstrated blinded adjudication should produce a Some concerns path, not Low or High by default.
- Slice 2: D4 Evidence Packets should be outcome-bound so packet sources for PFS do not mix in OS, toxicity, or unrelated endpoint language unless explicitly relevant.
- Slice 2: Keep D4 outcome-type corrections in domain-node control flow, consistent with the existing architecture boundary.
- Slice 3: D3 prompts/control should require direct Time-To-Event Missingness Evidence for 3.1=Y/PY in survival outcomes.
- Slice 3: ITT population language, Cox models, Kaplan-Meier methods, and censoring rules are not by themselves enough to establish negligible missing outcome data.
- Slice 3: Benchmark AE Estimand should default to effect of assignment; do not auto-switch to adhering/per-protocol solely because an outcome is safety-related.
- Slice 3: D2 deviation concern must be Evidence-Derived Deviation Concern; do not hard-code trial-name priors.
- Preserve ADR-0001: DomainEvidenceContext remains prompt assembly only and must not own packet source selection, SQ branching, NA control, D4 corrections, or judging.
- Preserve ADR-0002: trial-level ingestion artifacts can remain reusable, but all outcome resolver outputs, retrieved contexts, Evidence Packets, SQ answers, judgments, and reports remain outcome-specific.

## Testing Decisions

- Tests should assert external behavior: resolved Outcome Properties, Domain Judgments, SQ control outcomes, benchmark comparison deltas, and audit fields, not internal prompt wording unless prompt contract text is the behavior under test.
- Slice 1 tests should cover OS, PFS, and AE examples where paper-wide evidence contains misleading symptoms, toxicity, imaging, or adverse-event text.
- Slice 1 tests should verify source priority: registry evidence can support but not override clearer primary paper or protocol endpoint definitions.
- Slice 1 tests should include LLM-output parser/validator cases for missing citations, invalid booleans, contradictory properties, and fallback warnings.
- Slice 2 tests should extend existing Domain 4 and SQ-control coverage with open-label PFS, objective OS, blinded adjudication present, and blinded adjudication absent cases.
- Slice 2 tests should assert D4 for open-label PFS maps to Some concerns when progression includes clinician/investigator assessment and no blinded adjudication is shown.
- Slice 3 tests should extend D3 coverage so time-to-event ITT/censoring alone does not force 3.1=Y.
- Slice 3 tests should verify AE benchmark runs retain effect of assignment unless an explicit per-protocol/adhering effect is requested.
- Benchmark regression tests should run the current CHAARTED, PEACE-1, and STAMPEDE supplement outputs or fixtures and report per-domain Reference Agreement before and after each slice.
- Prior test locations include outcome resolver tests, SQ control tests, judge tests, prompt tests, benchmark tests, and domain context tests.

## Out of Scope

- Do not change primary PDF or supplement ingestion strategy for this PRD.
- Do not add trial-name priors for PEACE-1, STAMPEDE, CHAARTED, or any other named study.
- Do not merge official RoB 2 overall judgment policy with benchmark-reference overall policy.
- Do not move source selection or packet grading into DomainEvidenceContext.
- Do not implement persistent disk caching for trial artifacts.
- Do not optimize runtime except where needed to keep new LLM calls bounded and auditable.

## Further Notes

The current benchmark pattern suggests D1 and D5 are mostly stable, while D2,
D3, and especially D4 need scoped evidence and calibration. The highest-leverage
first slice is outcome-scoped LLM outcome resolution because it removes the
paper-wide contamination that cascades into D4 and AE estimand behavior.
