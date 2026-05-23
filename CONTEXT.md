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
`full_text`, structured `PaperEvidence`, primary Docling chunks, and the primary
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
baseline table, plus extraction method and warnings.

### DocumentRepr

The prompt-facing representation of a Docling document. It groups text and
tables by heading into `DocBlock`s and can render a compact Markdown-like
document for LLM evidence extraction.

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
and `clinician-composite`.

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

### EvidenceContract

The SQ-level contract that describes what an evidence packet needs: domain,
required evidence labels, matching terms, fallback primary-paper sections, and
flags such as denominator, outcome binding, or prespecification requirements.

### Evidence Packet

An SQ-specific packet built from ranked candidate sources. It combines RAG
chunks, ClinicalTrials.gov-derived sources, and section-text fallbacks, then
records selected sources, candidate facts, missing evidence, negative flags,
retrieval confidence, and packet grade.

### Evidence Fact

A compact fact derived from a selected packet source. It records fact type,
domain, SQ ids, claim, quote, source section, page numbers, confidence, support
status, document identity, role, source kind, and source path.

### DomainEvidenceContext

Prompt-ready evidence context for one RoB 2 domain or one Domain 2 prompt
stage. It combines primary-paper evidence, evidence packets, RAG compatibility
text, trial facts, and registry fields without changing source selection or
signaling-question control logic.

### Signaling Question Answer

The parsed XML answer for one RoB 2 signaling question. It contains an answer
code (`Y`, `PY`, `PN`, `N`, `NI`, or `NA`), quote, justification, and
uncertainty flag.

### Domain Judgment

The deterministic D1-D5 judgment produced by a Python judge from SQ answers.
LLMs do not directly assign final domain labels.

### Overall Judgment

The deterministic overall RoB 2 label derived from D1-D5 domain judgments.
The default policy follows official RoB 2-style escalation. The benchmark
reference policy can be more permissive for benchmark comparison.

### Verification Flag

An audit warning produced after SQ answering. Current flags cover unsupported
quotes, fragile SQ answers, and packet verification failures. Verification
flags can produce retry or escalation actions.

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

- `rob2_pipeline/nodes/ingest.py` is the graph-facing ingestion node. It owns
  the fallback order and combines primary plus supplement chunks.
- `rob2_pipeline/ingestion/docling_extract.py` owns Docling full-text
  extraction, OCR retry, converter caching, chunk building, and chunk page
  metadata.
- `rob2_pipeline/ingestion/document_repr.py` owns `DocumentRepr` and `DocBlock`.
- `rob2_pipeline/ingestion/evidence.py` owns paper-evidence extraction,
  structural keyword mapping, section parsing, section capping, CONSORT
  augmentation, and censoring-context extraction.
- `rob2_pipeline/ingestion/supplements.py` owns supplement classification,
  source-document records, source metadata application, windowed supplement
  parsing, and supplement warnings.
- `rob2_pipeline/pdf_ingestion.py` is a compatibility facade for ingestion
  helpers and test-used monkeypatch points.

### Trial Metadata

- `rob2_pipeline/nodes/preliminary.py` extracts trial metadata, fetches
  ClinicalTrials.gov data, reconciles registered endpoints, and may auto-set
  safety outcomes to per-protocol.
- `rob2_pipeline/registration_api.py` fetches and formats ClinicalTrials.gov
  outcomes, design, description, and participant flow.
- `rob2_pipeline/nodes/outcome_resolver.py` infers outcome properties and
  normalizes `outcome_type`.
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
- `rob2_pipeline/nodes/domain_helpers.py` centralizes the simple domain SQ LLM
  call pattern.
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

### LLM Calls, XML, Providers, Cache, And Trace

- `rob2_pipeline/nodes/common.py` is the shared LLM call module for graph nodes.
  It handles prompt cache lookup/write, provider invocation, XML parse/repair,
  trace logging, SQ answer merging, `NA` setting, source labels, and domain
  judgment insertion.
- `rob2_pipeline/xml_parser.py` extracts XML fragments, sanitizes stray `<`,
  normalizes answer codes, parses SQ responses, and validates expected SQ ids.
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
domain node, the deterministic judge, and focused tests.

### Add A New Evidence Source

Prefer adding it as a packet candidate with explicit `source_kind`,
`document_role`, and provenance. Avoid appending non-primary evidence directly
to `full_text` or primary `PaperEvidence`.

### Change Prompt Evidence Assembly

Start in `domain_context.py`. Keep packet source selection in evidence-packet
modules and keep SQ control logic in domain SQ modules unless the behavior is
being deliberately redesigned.

### Investigate A Wrong Judgment

Inspect `domain_judgments` and `domain_rationales`, then relevant `sq_answers`,
`evidence_packets`, domain `rag_sources`, `retrieval_grades`, `packet_grades`,
`evidence_validation_flags`, and `verification_actions`.

### Investigate Missing Or Weak Evidence

Inspect `evidence.warnings`, `source_documents`, `supplement_warnings`,
retrieved source metadata, packet missing-evidence labels, and the trace for
LLM parse or repair failures.
