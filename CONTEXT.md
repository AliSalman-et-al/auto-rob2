# Context

`auto-rob2` produces automated Cochrane Risk of Bias 2 assessments for
randomized controlled trial reports. The system creates reviewer-facing drafts
and audit artifacts; human review remains required.

## Core Flow

An assessment starts at `run_assessment()` and executes a LangGraph over a
shared `RoB2State`.

1. `pdf_ingest` parses the primary paper and optional supplements.
2. `rct_screener` stops early when the study is not an RCT.
3. `preliminary_info` extracts trial metadata and enriches it with
   ClinicalTrials.gov when possible.
4. `outcome_resolver` normalizes outcome properties and outcome type.
5. `trial_facts` extracts reusable trial-level snippets.
6. `rag_retrieval` builds per-study retrieval context from primary and
   supplement chunks.
7. `evidence_packet_builder` builds SQ-specific evidence packets.
8. D1-D5 SQ nodes ask LLMs for signaling-question answers.
9. D1-D5 judges convert SQ answers into deterministic domain judgments.
10. `quote_verifier`, `overall_judge`, and `report_formatter` produce quality
    flags, overall judgment, review priority, and the Markdown draft.

## Domain Terms

### Assessment

A single RoB 2 run for one primary PDF and one assessed outcome. It may include
supplementary PDFs, registry enrichment, RAG retrieval, LLM signaling-question
answers, deterministic judgments, verification flags, and output artifacts.

### Primary Paper

The main randomized trial report. It stays central. Primary-paper text becomes
`full_text`, structured `PaperEvidence`, primary retrieval chunks, and the primary
`SourceDocument`.

### Supplement

An optional supporting PDF such as a protocol, SAP, appendix, disclosure, or
data-sharing document. Supplements are parsed into chunks with provenance and
join retrieval, but they do not replace `full_text` or primary-paper evidence.

### SourceDocument

The inventory record for a primary paper or supplement. It records
`document_id`, `document_name`, `document_role`, `source_kind`, path, primary
status, parse status, and any parse error. Supplement statuses are `parsed`,
`partial`, `failed`, or `missing`.

### Document Role

The role assigned to a source document or packet source. Current roles include
`primary`, `protocol`, `sap`, `appendix`, `disclosure`, `data_sharing`,
`unknown_supplement`, and `registry`.

### Source Kind

The provenance class for evidence text. Current values include `rag_chunk`,
`section_text`, and `ctgov`. Only real `rag_chunk` sources are expected to have
page numbers.

### PaperEvidence

The structured primary-paper evidence object used by downstream nodes. It has
sections for abstract, methods, results, D1 randomization, D2 blinding, D3
missing data, D4 outcome measurement, D5 registration, CONSORT flow, and
baseline table, plus extraction method and warnings. It is extracted evidence,
not authoritative evidence; when it conflicts with quote-grounded source text,
the quote-grounded source text is authoritative.

### Quote-Grounded Evidence

Evidence whose claim can be traced to matching text in `full_text`, a parser artifact
chunk, or another provenance-bearing `SourceDocument`. It is the authority used
to resolve conflicts with unverified extracted evidence.

### Raw Character Stream

The minimally processed text extracted from the primary PDF for quote
traceability checks. It is not prompt context and must not replace the
layout-aware primary-paper `full_text`.

### Trial Ingestion Artifact

The reusable trial-level ingestion result for a primary paper and its selected
supplements. It can be shared by multiple outcome-specific Assessments for the
same trial because primary-paper text, primary-paper evidence, retrieval chunks,
source-document inventory, and supplement warnings do not depend on the assessed
outcome. If primary-paper evidence extraction becomes outcome-specific, that
evidence must move out of the trial-level artifact.

### Parser-Neutral Artifact

The ingestion representation produced from page-aware parser output. It carries
source identity, page text, diagnostics, provenance, and retrieval chunks without
exposing parser-native document objects.

### Trial Workspace

The trial-level artifact workspace for source identities, parser-neutral
`ParseArtifact` records, page-aware artifacts, parser diagnostics, and
EvidenceStore outputs. Trial Workspace artifacts are reusable only when source,
parser, configuration, and upstream artifact hashes still match.

### Outcome Workspace

The outcome-specific artifact workspace for outcome normalization,
JSON-contract SQ answers, deterministic domain judgments, support-escalation
diagnostics, and other artifacts that depend on the assessed outcome or RoB 2
settings. Outcome Workspace artifacts are invalidated by
changed outcome definitions, settings, trial-workspace inputs, or contract
versions.

### Trial Metadata

The trial-level labels extracted before domain assessment: intervention,
comparator, assessed outcome, numerical result, registration number, registered
endpoint, registered analysis, number randomized, and ClinicalTrials.gov
fields.

### Registry Evidence

ClinicalTrials.gov enrichment for NCT-registered studies. It can provide
registered outcomes, design/masking metadata, registry description, and
participant flow. It is supporting evidence and may disagree with the primary
paper or protocol.

### Outcome Properties

Boolean features inferred from the assessed outcome and evidence text:
patient-reported, safety harm, time-to-event, death-only objective event,
composite, lab/imaging threshold, blinded adjudication, objective event, and
clinician judged. These resolve `outcome_type`.

### Outcome Type

The prompt and D4-control category derived from outcome properties. Current
values are `patient-reported`, `clinician-graded`, `biomarker`, `vital-status`,
and `clinician-composite`. Outcome type should carry evidence support when it
is semantically resolved, because weak or unsupported classification can be
pivotal to Domain 4.

### Assessed-Outcome-Bound Classification

Outcome classification should be driven by the assessed outcome and evidence
specifically tied to how that outcome is measured. Trial-wide mentions of other
endpoint families should not determine the assessed outcome's properties.

### Semantic Evidence Interpretation

The meaning of trial text when it requires clinical, methodological, or
cross-specialty judgment. LLMs should handle semantic evidence interpretation;
deterministic code should be reserved for RoB 2 algorithm tables, schema checks,
provenance checks, contradiction flags, support gating, and explicit control
flow. Broad deterministic heuristics should not override semantic
classification of trial evidence.

### Effect Of Interest

The Domain 2 estimand mode. `ITT` means effect of assignment to intervention.
`per-protocol` means effect of adhering to intervention. Safety endpoints may
auto-switch to per-protocol when the user has not explicitly overridden the
environment default.

### TrialFacts

Reusable trial-level snippets extracted from primary-paper evidence:
randomization, allocation concealment, masking, protocol deviations, protocol
amendments, and analysis populations. Domain prompt context can use these
snippets alongside section evidence and evidence packets.

### RAG Context

Domain-oriented retrieved text stored in `rag_contexts`. It is compatibility
text for prompts, currently keyed by `d1`, `d2_blinding`, `d2_deviations`,
`d2_analysis`, `d3`, `d4_measurement`, `d4_assessor`, and `d5`.

### RAG Sources

The JSON-facing retrieval provenance emitted from `rag_chunk_metadata`. Sources
include retrieved text, section, page numbers, score, document id/name/role,
source kind, and source path.

### Trial Retrieval Index

The reusable trial-level vector index built from parser-neutral retrieval chunks for a primary
paper and its selected supplements. The index can be shared across
outcome-specific Assessments for the same trial, but retrieved contexts,
evidence packets, signaling-question prompts, and judgments remain
outcome-specific.

### EvidenceContract

The SQ-level contract that describes what an evidence packet needs: domain,
required evidence labels, matching terms, fallback primary-paper sections, and
flags such as denominator, outcome binding, or prespecification requirements. It
can also declare coverage groups: sets of terms where the packet must keep at
least one selected source per group when one exists, so an SQ that needs two
distinct kinds of evidence (e.g. D5 SQ 5.3's pre-specified-plan source and
reported-analysis-methods source) does not let the higher-ranking kind take
every slot.

### Evidence Packet

An SQ-specific packet built from ranked candidate sources. It combines RAG
chunks, ClinicalTrials.gov-derived sources, and section-text fallbacks, then
records selected sources, candidate facts, missing evidence, negative flags,
retrieval confidence, and packet grade.

### Evidence Fact

A compact fact derived from a selected packet source. It records fact type,
domain, SQ ids, claim, quote, source section, page numbers, confidence, support
status, document identity, role, source kind, and source path. Its Evidence
Support Level describes how well the source supports the extracted claim.

### Evidence Support Level

The reviewer-facing strength of support for a claim or signaling-question
answer. It captures the human judgment that evidence may be strong, moderate,
weak, or unsupported even when a relevant quote or number is present. It is a
reasoned assessment of support, not a mechanical keyword or denominator check.
Canonical levels are `strong`, `moderate`, `weak`, and `unsupported`. Support
level primarily shapes the SQ answer itself; domain judging uses it as an audit
brake when a weak or unsupported SQ answer is pivotal to the final judgment.

### EvidenceStore

The typed store of quote-grounded Evidence Facts, failed claims, and evidence
gaps. It validates support status, provenance, family-specific fields, and
outcome binding before facts are selected for SQ packets or workspace
artifacts.

### D3 Completeness Evidence

Evidence supporting Domain 3 SQ 3.1 that outcome data were available for nearly
all participants. Missing denominator, percentage, or count evidence is a
provenance trigger for weak support, not a standalone domain-judgment rule.

### Support Constraint

A typed evidence-quality finding attached to an evidence claim, such as a
signaling-question answer or outcome classification. Support constraints cover
untraceable quotes, missing required evidence, wrong outcome context, and
conflicts between semantic support and provenance checks. They do not directly
assign RoB 2 answers or labels, and they do not overwrite the original Evidence
Support Level. The SQ Support Audit combines constraints with Evidence Support
Levels to decide acceptance, adjudication, or audit limitation. Initial
constraint types are `quote_untraceable`, `missing_required_evidence`,
`wrong_outcome_context`, and `semantic_support_conflict`.

### Pivotal SQ Answer

A signaling-question answer whose more conservative interpretation would change
the Domain Judgment. Weak or unsupported pivotal answers should be retried or
escalated rather than silently driving the final judgment.

### Pivotality Test

An explicit audit artifact that records whether a signaling-question answer is
pivotal. It compares the original Domain Judgment with the judgment produced by
a documented conservative test answer, without silently mutating the emitted SQ
answer. The conservative test answer is the nearest less-confident answer
(`Y` to `PY` to `NI`, or `N` to `PN` to `NI`) unless the evidence semantically
contradicts the original answer.

### SQ Support Adjudication

A focused retry of one pivotal signaling-question answer. It re-evaluates
whether the original answer is semantically supported by the selected evidence
without re-running the whole domain judgment.

### SQ Support Audit

The pipeline step that identifies weak or unsupported pivotal signaling-question
answers and routes them to SQ Support Adjudication before final domain
judgments are accepted. It gates acceptance of final judgments rather than
changing the deterministic RoB 2 decision tables directly. The acceptance gate
applies across D1-D5, not only to domains with known benchmark failures.

### Acceptance Status

The support-audit state for an evidence claim. Canonical statuses are
`accepted`, `needs_adjudication`, and `audit_limited`.

### DomainEvidenceContext

Prompt-ready evidence context for one RoB 2 domain or one Domain 2 prompt
stage. It combines primary-paper evidence, evidence packets, RAG compatibility
text, trial facts, and registry fields without changing source selection or
signaling-question control logic.

### DomainSqStage

The shared interface for one LLM signaling-question stage. It records the graph
node name, SQ ids, source-domain key, prompt builder, parser, and optional
stage-local post-processing for answer corrections or NA control flow.

### Packet-Bound SQ Classification

The signaling-question classification mode where the selected Evidence Packet
and its decision table are the only evidence authority for an SQ answer. D1-D5
should use the same packet-bound classification concept even when domains have
different branching or post-processing rules.

### Signaling Question Answer

The parsed JSON-contract answer for one RoB 2 signaling question. It contains
an answer code (`Y`, `PY`, `PN`, `N`, `NI`, or `NA`), quote, justification,
uncertainty flag, Evidence Support Level, and support rationale so reviewers
can see how strongly the full evidence set supports that SQ answer, not just
whether one cited fact exists.

### Domain Judgment

The deterministic D1-D5 judgment produced by a Python judge from SQ answers.
LLMs do not directly assign final domain labels.

### Final Domain Judgment

The accepted D1-D5 judgment after pivotal weak or unsupported SQ answers have
been audited and, when needed, adjudicated. Benchmark agreement should compare
against final domain judgments, not pre-audit initial judgments. RoB 2 labels
remain `Low`, `Some concerns`, or `High`; unresolved pivotal weak support is
reported as an audit limitation and human-review signal, not as a fourth domain
label.

### Overall Judgment

The deterministic overall RoB 2 label derived from D1-D5 domain judgments.
The default policy follows official RoB 2-style escalation. The benchmark
reference policy can be more permissive for benchmark comparison.

### Methodology Authority

The hierarchy used when assessment outputs disagree. RoB 2 methodology is the
governing authority, benchmark references are calibration fixtures, and current
pipeline output is evidence to inspect rather than a source of truth.

### Benchmark Agreement

The comparison between benchmark reference labels and final RoB 2 labels.
Agreement scoring compares labels directly, while benchmark reports may also
measure whether label mismatches were caught as audit-limited or high-priority
human-review cases.

### Verification Flag

An audit warning produced after SQ answering. Current flags cover unsupported
quotes, fragile SQ answers, and packet verification failures. Verification
flags should be represented as Support Constraints when they affect SQ
acceptance or adjudication.

### Human Review Priority

The review triage level emitted with the result. It is driven by overall
judgment, the count of `NI` answers, high-uncertainty SQs, and multiple
Some-concerns domains.

## Module Map

### Entry Points

- `main.py` runs one PDF or a directory of PDFs. It handles explicit
  supplements, per-trial supplement discovery, cache bypass, and debug output.
- `benchmark.py` runs reference comparisons for configured trial/outcome pairs.
- `rob2_pipeline/pipeline.py` owns `run_assessment()`, output writing, and the
  JSON output schema.

### Graph And State

- `rob2_pipeline/graph.py` wires the LangGraph node order and wraps every node
  with timing spans.
- `rob2_pipeline/state.py` defines `RoB2State` and LangGraph reducers.
- `rob2_pipeline/state_factory.py` defines the initial state and default
  outcome properties.

### Ingestion

- `rob2_pipeline/nodes/ingest.py` is the graph-facing ingestion adapter. It
  calls the Assessment ingestion module and returns the existing `RoB2State`
  update shape.
- `rob2_pipeline/ingestion/assessment.py` owns Assessment ingestion behavior:
  strict primary full-text extraction from parser artifacts, primary plus
  supplement chunk assembly, source-document inventory, and remote
  paper-evidence extraction orchestration.
- `rob2_pipeline/ingestion/parse_artifacts.py` owns LiteParse adaptation,
  parser-neutral page artifacts, page-aware section artifacts, and retrieval
  chunk conversion.
- `rob2_pipeline/ingestion/evidence.py` owns paper-evidence extraction,
  structural keyword mapping, section parsing, section capping, CONSORT
  augmentation, and censoring-context extraction.
- `rob2_pipeline/ingestion/supplements.py` owns supplement classification,
  source-document records, source metadata application, windowed supplement
  parsing, and supplement warnings.

### Trial Metadata

- `rob2_pipeline/nodes/preliminary.py` extracts trial metadata, fetches
  ClinicalTrials.gov data, reconciles registered endpoints, and may auto-set
  safety outcomes to per-protocol.
- `rob2_pipeline/registration_api.py` fetches and formats ClinicalTrials.gov
  outcomes, design, description, and participant flow.
- `rob2_pipeline/nodes/outcome_resolver.py` semantically resolves assessed-
  outcome-bound properties and normalizes `outcome_type`.
- `rob2_pipeline/nodes/trial_facts.py` extracts reusable snippets for prompt
  context.

### Retrieval And Evidence Packets

- `rob2_pipeline/rag.py` owns per-study FAISS index building, section-filtered
  retrieval, adaptive token budgeting, and retrieval grades.
- `rob2_pipeline/rag_queries.py` owns SQ and domain query text.
- `rob2_pipeline/nodes/rag_retrieval.py` is the graph node that emits
  compatibility `rag_contexts`, structured `rag_chunk_metadata`, and retrieval
  grades. It falls back to primary-paper sections when vector retrieval fails
  or no chunks exist.
- `rob2_pipeline/nodes/evidence_contracts.py` defines SQ evidence contracts,
  matching regexes, prespecification terms, and outcome aliases.
- `rob2_pipeline/nodes/evidence_source_selection.py` builds candidate sources
  from RAG metadata, registry fields, and section-text fallbacks, then annotates
  matched terms and provenance.
- `rob2_pipeline/nodes/evidence_packet_grading.py` detects missing evidence,
  negative flags, confidence, packet grade, and evidence facts.
- `rob2_pipeline/nodes/evidence_packets.py` orchestrates packet construction
  and renders packet blocks for prompts.

### Domain SQ Nodes And Judges

- `rob2_pipeline/nodes/domain_context.py` builds prompt-ready context objects
  for D1-D5 and D2 prompt stages.
- `rob2_pipeline/nodes/domain_helpers.py` owns `DomainSqStage`, the shared
  interface for SQ-stage prompt building, LLM invocation, parsed-answer
  merging, chunk-source logging, and optional stage-local SQ control flow.
- `rob2_pipeline/nodes/domain1.py` handles D1 randomization SQs and D1 judge.
- `rob2_pipeline/nodes/domain2.py` handles D2 SQ12, conditional, analysis, ITT
  versus per-protocol routing, D2 NA control logic, and D2 judge.
- `rob2_pipeline/nodes/domain3.py` handles D3 missing-outcome-data SQs, D3 NA
  control logic, and D3 judge.
- `rob2_pipeline/nodes/domain4.py` handles D4 measurement SQs, outcome-type
  corrections, D4 NA control logic, and D4 judge.
- `rob2_pipeline/nodes/domain5.py` handles D5 selective-reporting SQs,
  missing-intervention review priority, and D5 judge.
- `rob2_pipeline/judges/` owns deterministic D1-D5 and overall decision tables.

### LLM Calls, JSON Contracts, Providers, Cache, And Trace

- `rob2_pipeline/nodes/common.py` is the shared LLM call module for graph nodes.
  It handles prompt cache lookup/write, provider invocation, JSON-contract
  parsing and repair, trace logging, SQ answer merging, `NA` setting, source
  labels, and domain judgment insertion.
- `rob2_pipeline/llm_contracts.py` validates JSON LLM artifacts against local
  Pydantic schemas and records contract trace metadata.
- `rob2_pipeline/prompts.py` owns prompt templates and rendered methodology
  text.
- `rob2_pipeline/methodology/` owns canonical RoB 2 rule-card data and prompt
  rendering.
- `rob2_pipeline/config.py` reads provider/model/rate-limit configuration at
  import time and builds provider adapters.
- `rob2_pipeline/providers/` contains the LLM provider interface and adapters
  for OpenRouter, Anthropic, and OpenAI.
- `rob2_pipeline/cache.py` owns optional prompt cache behavior.
- `rob2_pipeline/trace.py` records LLM calls and graph-node spans for audit and
  timing analysis.
- `rob2_pipeline/evidence_store.py` defines EvidenceStore schemas and
  family-typed fact selection for quote-grounded evidence.
- `rob2_pipeline/trial_workspace.py` writes Trial Workspace and Outcome
  Workspace manifests and artifacts with source, config, contract, and upstream
  hash identity.

### Verification, Output, And Benchmarking

- `rob2_pipeline/nodes/verification.py` checks SQ quote support, fragile SQ
  patterns, packet grades, and verification actions.
- `rob2_pipeline/nodes/overall.py` applies overall judgment policy and human
  review priority rules.
- `rob2_pipeline/nodes/reporter.py` formats the Markdown draft report.
- `rob2_pipeline/benchmark.py` resolves benchmark inputs, applies supplement
  policy, runs assessments, compares against reference CSVs, summarizes
  confusion/timing data, and writes benchmark outputs.
- `rob2_pipeline/io.py` discovers input PDFs and per-trial supplement folders.

## Important Invariants

- The primary paper stays central; supplements and registries enrich but do not
  replace it.
- `full_text` extraction is strict. If full-text extraction fails, the run
  halts. Later structural or remote evidence extraction can fall back.
- Supplements are best-effort by default; benchmark `required` mode treats
  missing, failed, or not-ingested requested supplements as benchmark failures.
- LLMs answer SQs; deterministic judges assign D1-D5 and overall labels.
- `rag_contexts` are prompt compatibility text. `rag_chunk_metadata` is the
  provenance-bearing source record emitted as `rag_sources` in JSON.
- Evidence packets are the main prompt context protection against overload.
- `DomainEvidenceContext` is prompt assembly only. It does not own packet
  selection, retrieval, SQ branching, NA control logic, or judges.
- `DomainSqStage` owns SQ-stage execution shape: prompt build, LLM call,
  parsed-answer merge, chunk-source logging, and optional stage-local
  post-processing. It must not own source selection, evidence-packet grading,
  or deterministic judging.
- D2 has three prompt stages because the official algorithm branches by
  awareness, deviations, analysis, and effect of interest.
- D4 includes outcome-type-specific corrections before final D4 NA control
  logic.
- ClinicalTrials.gov evidence is supporting evidence and is marked as registry
  provenance.
- Node timing is centralized in graph wrappers; individual nodes should not add
  ad hoc timing spans.
- Generated Markdown is a draft for human review; JSON and trace outputs are
  the audit artifacts.

## Common Change Paths

### Add Or Change A Signaling Question

Update methodology cards, prompt templates, RAG queries, evidence contracts,
packet grading or source selection if evidence requirements changed, the
domain node's `DomainSqStage`, the deterministic judge, and focused tests.

### Add A New Evidence Source

Prefer adding it as a packet candidate with explicit `source_kind`,
`document_role`, and provenance. Avoid appending non-primary evidence directly
to `full_text` or primary `PaperEvidence`.

### Change Prompt Evidence Assembly

Start in `domain_context.py`. Keep packet source selection in evidence-packet
modules and keep SQ control logic in `DomainSqStage` post-processing inside
domain SQ modules unless the behavior is being deliberately redesigned.

### Investigate A Wrong Judgment

Inspect `domain_judgments` and `domain_rationales`, then relevant `sq_answers`,
`evidence_packets`, domain `rag_sources`, `retrieval_grades`, `packet_grades`,
`evidence_validation_flags`, and `verification_actions`.

### Investigate Missing Or Weak Evidence

Inspect `evidence.warnings`, `source_documents`, `supplement_warnings`,
retrieved source metadata, packet missing-evidence labels, and the trace for
LLM parse or repair failures.
