# Architecture

This document explains how `auto-rob2` works internally and where to look when
changing or debugging it. For installation and command-line usage, start with
`README.md`.

The repo-level domain glossary and change-path map lives in `CONTEXT.md`.
Read it first when making architectural changes or when choosing names for new
modules.

## Design Goals

`auto-rob2` is built around four constraints:

1. **The primary paper stays central.** Supplements and registries enrich the
   evidence base, but they do not replace the study report.
2. **Context is selected before prompting.** Primary-paper evidence is
   structured, supplement sections are retrieved, and evidence packets package
   targeted sources so domain prompts do not receive whole documents.
3. **LLMs do not make final labels directly.** LLMs answer structured RoB 2
   signaling questions and rate evidence support. Domain judge nodes audit
   pivotal weak or constrained answers before deterministic judges produce
   final D1-D5 and overall judgments.
4. **Outputs must be auditable.** Reports are accompanied by JSON diagnostics,
   source provenance, retrieval grades, evidence packets, LLM traces, and
   timing traces.

## End-To-End Flow

The compiled LangGraph workflow lives in `rob2_pipeline/graph.py`.

```text
pdf_ingest
  -> rct_screener
  -> preliminary_info
  -> outcome_resolver
  -> trial_facts
  -> evidence_packet_builder
  -> D1-D5 signaling-question nodes
  -> support audit and deterministic domain judges
  -> quote_verifier
  -> overall_judge
  -> report_formatter
```

The RCT screener can stop the graph early. Otherwise, the run proceeds through
ingestion, evidence selection, signaling-question answering, support audit,
deterministic judgment, verification, and report formatting.

`rob2_pipeline/pipeline.py` owns the public `run_assessment()` API and writes
the Markdown report, data JSON, and trace JSON after graph execution. Graph
nodes are wrapped centrally in `rob2_pipeline/graph.py` so every node execution
records a timing span in the active trace.

## Major Subsystems

### Entry Points

| File | Responsibility |
| --- | --- |
| `main.py` | CLI for one PDF or a directory of PDFs |
| `benchmark.py` | CLI for benchmark runs |
| `rob2_pipeline/pipeline.py` | `run_assessment()` API and output writing |
| `rob2_pipeline/benchmark.py` | Benchmark orchestration, comparisons, and summaries |

### Ingestion

Primary paper ingestion is strict because the assessment cannot proceed without
the main article. Supplement ingestion is best-effort unless benchmark
`supplement_policy="required"` is used.

| File | Responsibility |
| --- | --- |
| `rob2_pipeline/nodes/ingest.py` | Graph adapter for ingestion plus RCT screening node |
| `rob2_pipeline/ingestion/assessment.py` | Primary plus supplement Assessment ingestion from parser-neutral artifacts |
| `rob2_pipeline/ingestion/parse_artifacts.py` | PyMuPDF/PyMuPDF4LLM adaptation, page-aware parser artifacts, and raw character streams |
| `rob2_pipeline/ingestion/evidence.py` | Primary-paper structured evidence extraction |
| `rob2_pipeline/ingestion/settings.py` | Ingestion constants and environment controls |

`pdf_ingest` produces primary-paper text, structured primary-paper evidence,
parser artifacts, source documents, optional `SupplementSegment` artifacts, and
supplement warnings. Supplement text is not appended to `full_text`.

Benchmark runs can pass `precomputed_ingestion` back into the ingestion node
when the primary PDF and selected supplements match a previous outcome run.
That reuse is trial-level only; outcome-specific resolution and judgments still
run independently.

### Trial Metadata And Registry Enrichment

`preliminary_info` extracts trial metadata such as intervention, comparator,
outcome, and registration number. When an NCT number is available,
`registration_api.py` fetches ClinicalTrials.gov API v2 data.

Registry fields populate state keys such as:

- `registered_endpoint`
- `registered_secondary_endpoints`
- `registered_analysis`
- `ctgov_outcomes`
- `ctgov_design`
- `ctgov_description`
- `ctgov_flow`

ClinicalTrials.gov evidence enters later evidence packets as a structured
source with `source_kind="ctgov"` and `document_role="registry"`.

### Retrieval

Evidence packet construction is the retrieval boundary. The primary paper stays
available as structured `PaperEvidence` section text; optional supplements are
retrieved through BM25S-backed `SupplementIndex` artifacts.

The primary-paper evidence is not BM25S-indexed. It enters evidence packets and
prompt context through structured evidence, parser artifacts, trial facts,
registry evidence, and primary-paper `section_text` fallbacks. These
section_text fallbacks stay outside BM25S; BM25S is reserved for supplement
retrieval.

| File | Responsibility |
| --- | --- |
| `rob2_pipeline/supplement_retrieval.py` | BM25S supplement indexes over SupplementSegments |
| `rob2_pipeline/nodes/evidence_contracts.py` | SQ-specific evidence contracts and retrieval terms |
| `rob2_pipeline/nodes/evidence_source_selection.py` | Supplement, registry, and section-text packet candidates |
| `rob2_pipeline/nodes/evidence_packets.py` | Evidence packet orchestration and prompt rendering |

Evidence packets are the only prompt-facing retrieval product. JSON output
exposes `supplement_segments`, `supplement_retrieval_grades`, `evidence_packets`,
and packet diagnostics; it does not emit legacy `rag_sources`, `rag_contexts`,
or generic `retrieval_grades`.

`SupplementIndex` instances are rebuilt in memory for each run. Trial-level
ingestion artifacts such as source documents, parse artifacts, and supplement
segments can still be reused across outcomes for the same primary PDF and
supplements.

ADR-0002's vector-index reuse detail is superseded by the BM25S supplement
retrieval design, while trial-level ingestion artifact reuse remains valid.

If no supplements are supplied, downstream nodes still receive Evidence Packets
built from registry fields and deterministic primary-paper fallback sections.

### Evidence Packets

Evidence packets are the main protection against context overload. They combine
supplement segments, ClinicalTrials.gov fields, and primary-paper fallback
sections into signaling-question-specific inputs.

| File | Responsibility |
| --- | --- |
| `rob2_pipeline/nodes/evidence_contracts.py` | Required evidence for each signaling question |
| `rob2_pipeline/nodes/evidence_source_selection.py` | Candidate source creation and ranking |
| `rob2_pipeline/nodes/evidence_packet_grading.py` | Missing-evidence and quality flags |
| `rob2_pipeline/nodes/evidence_packets.py` | Packet construction and prompt rendering |
| `rob2_pipeline/evidence_store.py` | Typed quote-grounded evidence facts, failed claims, and gaps |
| `rob2_pipeline/trial_workspace.py` | Trial Workspace and Outcome Workspace artifact manifests |

Contracts define what each signaling question needs: required labels, matching
terms, fallback sections, denominator requirements, outcome-binding
requirements, and prespecification requirements.

Source ranking is domain-aware. For example, D5 prefers protocol, SAP, and
registry sources; D3 gives weight to appendix and SAP missing-data evidence;
D4 values outcome-definition and adjudication sources.

### EvidenceStore And Workspaces

`rob2_pipeline/evidence_store.py` defines the typed EvidenceStore used for
quote-grounded evidence facts, failed claims, and evidence gaps. It validates
support status, provenance, family-specific fields, and outcome binding before
facts are selected for signaling-question packets or persisted as audit
artifacts.

`rob2_pipeline/trial_workspace.py` separates reusable Trial Workspace artifacts
from Outcome Workspace artifacts. The Trial Workspace records source identities,
PyMuPDF-derived parser-neutral `ParseArtifact` records, page-aware artifacts,
parser diagnostics, and EvidenceStore outputs with content/config/upstream
hashes. Outcome Workspaces record outcome normalization, JSON-contract SQ
answers, deterministic domain judgments, and support-escalation diagnostics
that depend on the assessed outcome, RoB 2 settings, and contract versions.

### Domain Prompt Context

`rob2_pipeline/nodes/domain_context.py` owns prompt-ready evidence context for
D1-D5 and for each Domain 2 prompt stage. It combines primary-paper evidence,
evidence-packet blocks, trial facts, and registry fields into dataclass values
consumed by the domain SQ nodes.

Domain prompt context is intentionally not responsible for source selection,
retrieval, signaling-question branching, NA control logic, support audit, or
deterministic judging. Those remain in the evidence-packet, retrieval,
domain-node, verification, and judge modules.

| File | Responsibility |
| --- | --- |
| `rob2_pipeline/nodes/domain_context.py` | Prompt-ready evidence context objects |
| `rob2_pipeline/nodes/domain_helpers.py` | `DomainSqStage` interface and shared SQ-stage runner |
| `rob2_pipeline/nodes/domain1.py` - `domain5.py` | Prompt builders, SQ-stage declarations, stage-local SQ control logic, and judge node glue |

Every domain SQ node runs through `DomainSqStage`. The stage interface keeps
LLM invocation, chunk-source logging, parsed-answer merging, and optional
post-processing in one place while leaving prompt assembly in
`domain_context.py` and deterministic judging in `judges/`.

When ready evidence packets are available, domain SQ nodes use the generic
packet-bound JSON classifier and persist classifier artifacts under
`domain_sq_classifier_artifacts[domain][stage]`. Domain 1 follows this generic
path specifically because ADR-0005 removed its former D1-only answer-rewriting
controls, artifact state keys, and engineering diagnostics.

### LLM Calls

All graph LLM calls go through `call_node_llm()` in
`rob2_pipeline/nodes/common.py`. That layer handles provider selection, prompt
caching, JSON contract validation and repair, trace logging, and error
normalization. Signaling-question responses are parsed into answer code, quote,
justification, uncertainty flag, support level, and support rationale.

| File | Responsibility |
| --- | --- |
| `rob2_pipeline/prompts.py` | Prompt templates |
| `rob2_pipeline/methodology/` | RoB 2 rule cards rendered into prompts |
| `rob2_pipeline/providers/` | OpenRouter, Anthropic, and OpenAI adapters |
| `rob2_pipeline/cache.py` | Optional prompt cache |
| `rob2_pipeline/trace.py` | LLM input/output records and graph-node timing spans |
| `rob2_pipeline/llm_contracts.py` | JSON schema validation and repair/fallback helpers |

Avoid direct provider SDK calls inside graph nodes. Keeping calls behind the
provider abstraction makes traces, caching, retries, and tests consistent.

### Timing Instrumentation

The active trace records two timing layers:

| Layer | Trace field | Meaning |
| --- | --- | --- |
| Graph node spans | `node_spans` | Wall-clock duration, status, timestamps, and error text for each LangGraph node |
| LLM calls | `llm_calls` | Provider-facing latency, cache hits, token counts, repairs, parse errors, and model metadata |

Node spans are produced by the central graph wrapper, not by individual node
implementations. This keeps instrumentation additive as the graph evolves and
ensures exceptions still close spans before the original error is re-raised.

LLM latency and node duration are intentionally separate. A node span includes
all work inside the node, including local parsing, retrieval, ingestion work, and
any nested LLM calls. Benchmark summaries use these fields to estimate non-LLM
time as `max(total_wall_ms - llm_total_ms, 0)`.

### Judging, Verification, And Reporting

| File | Responsibility |
| --- | --- |
| `rob2_pipeline/judges/` | Deterministic D1-D5 and overall judgment logic |
| `rob2_pipeline/nodes/verification.py` | Quote support and packet quality verification |
| `rob2_pipeline/nodes/reporter.py` | Markdown report payload |

The domain judges consume parsed signaling-question answers, not raw free-form
model text. Each domain judge records the initial deterministic label, runs
pivotality tests for weak, unsupported, or constrained SQ answers, and may call
a targeted SQ support adjudication LLM node before writing the final domain
label.

Pivotality tests do not silently rewrite labels. They record the conservative
answer that would change the domain judgment and an acceptance status:
`accepted`, `needs_adjudication`, or `audit_limited`. Targeted adjudication may
change an SQ answer or only its support level. Both the initial and final
domain judgments are emitted to JSON when available.

The quote verifier adds post-judgment audit flags, typed support constraints,
and suggested actions when quotes are not traceable, packet quality is low, or
required evidence is missing.

## State Model

State is defined in `rob2_pipeline/state.py` and initialized in
`rob2_pipeline/state_factory.py`.

Important state groups:

| Group | Representative keys |
| --- | --- |
| Inputs | `pdf_path`, `supplementary_paths`, `precomputed_ingestion` |
| Primary ingestion | `full_text`, `evidence`, page-aware parser artifacts, `parse_artifacts` |
| Source inventory | `source_documents`, `supplement_warnings` |
| Trial metadata | `intervention`, `comparator`, `outcome`, `registration_number` |
| Outcome resolution | `outcome_type`, `outcome_properties`, `outcome_classification_support` |
| Registry enrichment | `registered_endpoint`, `ctgov_outcomes`, `ctgov_design`, `ctgov_flow` |
| Supplement retrieval | `supplement_segments`, `supplement_indexes`, `supplement_retrieval_grades` |
| Packets | `evidence_packets`, `evidence_facts`, `packet_grades`, `packet_readiness` |
| Prompt context | Derived in `domain_context.py`; not persisted in state |
| Judgments | `sq_answers`, `domain_sq_classifier_artifacts`, `initial_domain_judgments`, `domain_judgments`, `overall_judgment` |
| Support audit | `pivotality_tests`, `sq_support_adjudications`, `support_constraints` |
| Quality | `evidence_validation_flags`, `verification_actions`, `human_review_priority` |
| Diagnostics | `errors`, `llm_call_log`, `verifier_trace` |

Several graph nodes run in parallel, so reducers in `state.py` merge node
outputs safely. Dict-like fields merge by key; logs concatenate.

## Source Provenance

Every retrieved or packetized source should be traceable. Source metadata
commonly includes:

- `document_id`
- `document_name`
- `document_role`
- `source_kind`
- `source_path`
- `section`
- `page_numbers`
- `score`

Common `document_role` values are `primary`, `protocol`, `sap`, `appendix`,
`disclosure`, `data_sharing`, `unknown_supplement`, and `registry`.

Common `source_kind` values are `supplement_segment`, `section_text`, and
`ctgov`.

`supplement_segment` sources should carry page numbers when the parser provides
them. Missing page numbers on supplement segments are provenance warnings rather
than fatal packet defects. Structured fallbacks such as `ctgov` and
`section_text` are exempt from missing-page validation.

## Supplement Ingestion

Supplement parsing lives in `rob2_pipeline/ingestion/supplements.py`.

Key behavior:

- Supplements are optional by default.
- Supplements never replace `full_text` or primary-paper `evidence`.
- Content-based document type detection is authoritative for `protocol`, `sap`,
  and `appendix` when it can identify the role; filename heuristics remain a
  fallback for roles such as `protocol`, `sap`, `appendix`, `disclosure`, and
  `data_sharing`.
- Long supplements are parsed in page windows.
- Failed windows are recorded and skipped; later windows continue.
- Empty windows do not stop scanning.
- Parsed content is segmented into `SupplementSegment` artifacts with headings,
  page ranges, document roles, domain tags, annotations, and raw text.
- Annotated `SupplementSegment` artifacts feed in-memory BM25S
  `SupplementIndex` objects stored in state under `supplement_indexes`.
- Selected supplement packet sources use `source_kind="supplement_segment"`.
- Output JSON exposes serializable `supplement_segments` and
  `supplement_retrieval_grades`; BM25S index internals are not serialized.

Runtime controls:

| Setting | Default | Meaning |
| --- | --- | --- |
| `ROB2_SUPPLEMENT_PAGE_WINDOW` | `20` | Number of pages parsed per supplement window |
| `ROB2_SUPPLEMENT_MAX_SCAN_PAGES` | `1000` | Defensive scan limit for very long supplements |
| `ROB2_SUPPLEMENT_MAX_PAGES` | unset | Legacy alias for max scan pages |

`source_documents` records one status per requested source:

| Status | Meaning |
| --- | --- |
| `parsed` | Clean parse |
| `partial` | One or more windows failed; inspect warnings and retrieved sources for usable chunks |
| `failed` | No usable content could be extracted |
| `missing` | Requested file did not exist |

Benchmark `required` mode accepts `parsed` and `partial` as present source
documents, but `partial` still requires review of warnings and retrieved
sources. It fails missing, failed, or not-ingested requested supplements.

## Benchmark Architecture

Benchmark execution is implemented in `rob2_pipeline/benchmark.py`; the
top-level `benchmark.py` file handles CLI parsing.

Reference CSVs live in:

```text
data/references/overall_survival.csv
data/references/progression_free_survival.csv
data/references/adverse_events.csv
```

Benchmark inputs are selected with:

```text
--outcome-map TRIAL:OUTCOME[:COHORT]
```

Primary PDFs resolve from `inputs/benchmark/<TRIAL>.pdf`. When
`--use-supplements` is enabled, supplements resolve from
`inputs/benchmark/supplement/<TRIAL>/*.pdf` unless another supplement directory
is supplied.

Benchmark orchestration caches trial-level ingestion artifacts by primary PDF
identity plus selected supplement identities. If the same trial is assessed for
multiple outcomes in one run, later outcomes reuse the cached primary text,
evidence, parse artifacts, source inventory, supplement segments, and supplement
warnings. Outcome resolution, Evidence Packets, SQ answers, support audit, and
judgments remain outcome-specific.

Benchmark results include the reference row, pipeline judgments, agreement
comparisons, supplement counts, errors, aggregate confusion matrices, timing
summaries, audit-caught mismatch metrics, and adjudication metrics.

Each benchmark result gets a `timing` object when an assessment is attempted or
fails before execution in a non-skipped path. It includes:

- total wall-clock runtime for the assessment attempt
- whether the assessment trace was available
- total graph-node duration
- total LLM latency, LLM call count, cache hits, repairs, and parse errors
- adjudication LLM calls, latency, and token counts
- estimated non-LLM time
- slowest nodes
- LLM latency grouped by node

`benchmark_report.md` renders a `Timing Summary` section with aggregate
wall-clock timing, slowest runs, and node timing totals. It also renders an
`Adjudication Summary` when support-audit artifacts are present. Raw per-node
span payloads remain in the per-assessment trace JSON rather than the public
benchmark summary JSON.

## Configuration Reference

| Setting | Default | Notes |
| --- | --- | --- |
| `ROB2_PROVIDER` | `openrouter` | Provider adapter |
| `ROB2_MODEL` | provider default | Model used for graph LLM calls |
| `ROB2_TEMPERATURE` | provider setting | Generation temperature |
| `ROB2_MAX_TOKENS` | provider setting | Output token limit |
| `ROB2_EFFECT_OF_INTEREST` | `ITT` | Default effect of interest |
| `ROB2_USE_CACHE` | off | Prompt cache in `.rob2_cache/` |
| `ROB2_CTGOV_CACHE` | unset | ClinicalTrials.gov cache path |
| `ROB2_REMOTE_EVIDENCE_EXTRACTION` | enabled | Set `0` to skip ingestion-time LLM refinement |
| `ROB2_SUPPLEMENT_PAGE_WINDOW` | `20` | Supplement parsing window size |
| `ROB2_SUPPLEMENT_MAX_SCAN_PAGES` | `1000` | Supplement scan safety limit |
| `ROB2_RPM_LIMIT`, `ROB2_RPD_LIMIT` | provider setting | OpenRouter rate-limit controls |
| `ANTHROPIC_RPM_LIMIT`, `ANTHROPIC_TPM_LIMIT` | provider setting | Anthropic rate-limit controls |

## Debugging Guide

Useful commands:

```bash
uv run python main.py inputs/example.pdf --debug
uv run python benchmark.py --outcome-map CHAARTED:OS --dry-run
uv run python -m pytest -q
uv run python -m pytest tests/test_supplements.py -q
```

For a wrong judgment, inspect in this order:

1. `domain_judgments` and `domain_rationales`
2. `initial_domain_judgments`, `pivotality_tests`, and `sq_support_adjudications`
3. relevant `sq_answers`
4. relevant `evidence_packets`
5. relevant `DomainEvidenceContext` builder in `nodes/domain_context.py`
6. `packet_grades`, `packet_readiness`, and supplement retrieval diagnostics
7. `support_constraints`, `evidence_validation_flags`, and `verification_actions`

For ingestion problems, inspect:

1. `evidence.warnings`
2. `source_documents`
3. `supplement_warnings`
4. the LLM trace for extraction failures
5. `node_spans` for slow or failed ingestion nodes

Common failure modes:

| Problem | First check |
| --- | --- |
| JSON contract failure | Trace JSON and `llm_contracts.py` |
| RCT stops early | `is_rct`, `rct_screen_evidence`, `errors` |
| Missing randomization or masking evidence | D1/D2 packets and selected packet sources |
| Missing-data uncertainty | D3 packets, denominator flags, appendix/SAP sources |
| Selective-reporting uncertainty | D5 packets, CT.gov fields, protocol/SAP sources |
| Supplement parse issue | `source_documents`, `supplement_warnings` |
| Parser failure | Source document diagnostics and parser provenance |
| Empty packet evidence | `evidence_packets`, `packet_grades`, `packet_readiness`, and supplement warnings |
| Unexpected final-vs-initial label | `pivotality_tests`, `sq_support_adjudications`, and adjudication LLM trace |
| Slow benchmark run | `benchmark_report.md` Timing Summary, per-result `timing`, and trace `node_spans` |

## Extension Guide

### Add Or Change A Signaling Question

1. Update the relevant RoB 2 methodology card under `rob2_pipeline/methodology/`.
2. Update prompt templates in `rob2_pipeline/prompts.py`.
3. Update the evidence contract in `nodes/evidence_contracts.py`.
4. Update packet selection or grading if the evidence requirements changed.
5. Update `nodes/domain_context.py` if prompt evidence fields change.
6. Update the relevant `DomainSqStage` prompt builder, SQ IDs, or
   stage-local control flow in the domain node.
7. Update the deterministic judge if final-label behavior changes.
8. Update support-audit expectations if the SQ can be pivotal when weak or
   unsupported.
10. Add tests for stage behavior, context, parsing, packets, pivotality, and
    judge behavior.

### Add A New Evidence Source

Prefer adding the source as an evidence-packet candidate with explicit
`source_kind`, `document_role`, and provenance metadata. Avoid blending external
source text into primary-paper evidence unless it was extracted from the primary
publication itself.

### Change Prompt Evidence Assembly

Start in `nodes/domain_context.py`. Keep supplement retrieval and packet
selection in the SupplementIndex/evidence-packet modules, and keep SQ branching
and NA control logic in the domain node's `DomainSqStage` post-processing unless
the behavior is being deliberately redesigned.

### Add A New LLM Node

Use `call_node_llm()` from `nodes/common.py`. This keeps provider calls,
caching, JSON contract validation, trace logging, and error handling
consistent across the graph.

## Production Notes

- Human review remains required.
- Prompt cache is opt-in with `ROB2_USE_CACHE=1`.
- Rate limiting is lock-protected for concurrent graph fan-out.
- Timing instrumentation is always on and additive; it does not alter pipeline
  decisions or benchmark accuracy calculations.
- Support audit can add targeted adjudication calls and initial-vs-final
  judgment deltas; final labels still use standard RoB 2 labels.
- ClinicalTrials.gov evidence is supporting evidence; it may disagree with
  protocols or publications.
- Supplement ingestion is intentionally tolerant in normal runs and stricter in
  benchmark `required` mode.
- The JSON artifacts are as important as the Markdown report for auditing.
