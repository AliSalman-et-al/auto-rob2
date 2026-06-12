# auto-rob2

`auto-rob2` produces automated Cochrane Risk of Bias 2 (RoB 2)
assessments for randomized controlled trial reports.

The pipeline ingests a primary study PDF, optionally adds supplementary PDFs
such as protocols or appendices, enriches the record with ClinicalTrials.gov
data when available, retrieves targeted evidence, asks LLMs to answer RoB 2
signaling questions with evidence-support ratings, audits pivotal weak or
constrained answers, and then applies deterministic Python judges for the final
D1-D5 and overall judgments.

The output is a reviewer-facing draft plus detailed JSON diagnostics. It is
intended to support human review, not replace it.

## Quick Start

Requires Python `>=3.13`. `uv sync` will create or use a compatible environment
when one is available.

Install dependencies:

```bash
uv sync
```

Create a `.env` file with at least one provider key:

```text
OPENROUTER_API_KEY=your_key_here

# Optional alternatives:
# ANTHROPIC_API_KEY=your_key_here
# OPENAI_API_KEY=your_key_here
```

Run one assessment:

```bash
uv run python main.py inputs/example.pdf --output-dir outputs
```

Run one assessment for a specific outcome:

```bash
uv run python main.py inputs/example.pdf \
  --outcome "Overall Survival" \
  --effect ITT \
  --output-dir outputs
```

Run with supplements discovered from a per-study supplement folder:

```bash
uv run python main.py inputs/benchmark/CHAARTED.pdf \
  --outcome "Overall Survival" \
  --supplement-dir inputs/benchmark/supplement \
  --output-dir outputs
```

Run a benchmark dry run to validate inputs without LLM calls:

```bash
uv run python benchmark.py \
  --outcome-map CHAARTED:OS ARCHES:PFS \
  --dry-run
```

## What The Pipeline Uses

- PyMuPDF4LLM and PyMuPDF for parser-neutral PDF page artifacts, raw character
  streams, structured primary-paper evidence, and parser artifacts.
- LangGraph for the workflow.
- BM25S `SupplementIndex` retrieval over annotated `SupplementSegments` from
  supplementary documents.
- ClinicalTrials.gov API v2 for registry/design/outcome enrichment.
- LLMs for RCT screening, metadata extraction, and signaling-question answers.
- Deterministic Python judges for final domain and overall RoB 2 labels.

Primary-paper evidence is not BM25S-indexed. It enters prompts through
structured evidence sections, parser artifacts, trial facts, registry evidence,
and `section_text` fallbacks inside evidence packets.

## Inputs

Primary PDFs can be passed directly to `main.py`, or placed under
`inputs/benchmark/` for benchmark runs. Local inputs and outputs are ignored by
git.

Recommended benchmark layout:

```text
inputs/benchmark/
  CHAARTED.pdf
  ARCHES.pdf
  supplement/
    CHAARTED/
      protocol.pdf
      appendix.pdf
    ARCHES/
      supplementary_appendix.pdf
```

Supplement discovery maps the primary PDF stem to a folder under the supplement
directory. For example, `inputs/benchmark/CHAARTED.pdf` uses files from
`inputs/benchmark/supplement/CHAARTED/`.

For single-PDF runs, you can also pass explicit supplement files:

```bash
uv run python main.py inputs/benchmark/CHAARTED.pdf \
  --outcome "Overall Survival" \
  --supplement inputs/benchmark/supplement/CHAARTED/protocol.pdf \
  --supplement inputs/benchmark/supplement/CHAARTED/appendix.pdf
```

`--supplement` is only valid for single-PDF input. For directories, use
`--supplement-dir` so supplements cannot be accidentally applied to the wrong
study.

## Outputs

Each completed assessment writes:

```text
outputs/<pdf_basename>_rob2_report.md
outputs/<pdf_basename>_rob2_data.json
outputs/<pdf_basename>_trace.json
```

- The Markdown report is the human-readable draft RoB 2 assessment.
- The data JSON is the main audit artifact.
- The trace JSON captures LLM inputs/outputs and graph-node timing spans for
  debugging and performance analysis.

Useful JSON fields:

| Field                                  | What it tells you                                    |
| -------------------------------------- | ---------------------------------------------------- |
| `domain_judgments`, `overall_judgment` | Final deterministic RoB 2 labels                     |
| `sq_answers`                           | Parsed LLM signaling-question answers                |
| `initial_domain_judgments`             | Pre-adjudication deterministic domain labels          |
| `pivotality_tests`                     | Conservative tests for weak or constrained SQ answers |
| `sq_support_adjudications`             | Targeted LLM re-checks for pivotal weak or constrained SQ answers |
| `evidence`                             | Structured evidence extracted from the primary paper |
| `source_documents`                     | Primary and supplement parse inventory               |
| `supplement_segments`                  | Parsed and annotated supplement evidence sections    |
| `supplement_retrieval_grades`          | BM25S supplement retrieval diagnostics by SQ/domain  |
| `supplement_warnings`                  | Non-fatal supplement ingestion issues                |
| `evidence_packets`                     | Evidence selected for each signaling question        |
| `packet_grades`                        | Evidence packet quality diagnostics                  |
| `evidence_validation_flags`            | Quote-support and quality flags                      |
| `support_constraints`                  | Typed support issues such as untraceable quotes or missing required evidence |
| `verification_actions`                 | Suggested retry or review actions                    |

Useful trace JSON fields:

| Field        | What it tells you                                               |
| ------------ | --------------------------------------------------------------- |
| `llm_calls`  | Per-call prompt/response metadata, token counts, cache hits, repairs, parse errors, and latency |
| `node_spans` | Per-graph-node wall-clock timing, status, timestamps, and errors |

If the paper is screened as non-RCT, the graph stops early. JSON is still
written, but report and judgment fields may be absent.

## Benchmarking

Benchmarks compare pipeline judgments against reference RoB 2 CSVs in
`data/references/`.

Run selected trial/outcome pairs:

```bash
uv run python benchmark.py \
  --outcome-map CHAARTED:OS ARCHES:PFS PEACE-1:AE
```

Outcome codes:

| Code  | Outcome                   |
| ----- | ------------------------- |
| `OS`  | Overall Survival          |
| `PFS` | Progression-Free Survival |
| `AE`  | Adverse Events            |

Outcome-map entries may include a cohort label:

```bash
uv run python benchmark.py \
  --outcome-map CHAARTED:OS:calibration ARCHES:PFS:validation
```

Run with benchmark supplements:

```bash
uv run python benchmark.py \
  --outcome-map CHAARTED:OS CHAARTED:PFS \
  --use-supplements \
  --supplement-dir inputs/benchmark/supplement \
  --output-dir outputs/benchmark/chaarted_supplement
```

Supplement policies:

| Policy     | Behavior                                                          |
| ---------- | ----------------------------------------------------------------- |
| `auto`     | Use supplements when found; continue on supplement warnings       |
| `required` | Treat missing or failed requested supplements as benchmark errors |
| `none`     | Ignore supplements                                                |

Benchmark outputs:

```text
<output-dir>/benchmark_report.md
<output-dir>/benchmark_results.json
<output-dir>/<TRIAL>_<OUTCOME_CODE>/...
```

Benchmark results include timing data for each attempted assessment. Each
per-result `timing` object reports total wall time, trace availability, total
LLM latency, estimated non-LLM time, LLM call/cache/repair counts, slowest
nodes, adjudication LLM calls, and LLM latency grouped by node.
`benchmark_report.md` also includes a `Timing Summary` section with aggregate
wall-clock timing, slowest runs, and node timing totals.

When the same trial is benchmarked for multiple outcomes, benchmark execution
reuses trial-level ingestion artifacts for later outcomes with the same primary
PDF and supplements. BM25S `SupplementIndex` internals are rebuilt in memory
from reusable `SupplementSegments`; outcome resolution, evidence packets, SQ
answers, support adjudication, and judgments remain outcome-specific.

Benchmark reports also include an `Adjudication Summary` when support audit
artifacts are present. It counts weak and unsupported SQ answers, pivotality
tests, targeted SQ support adjudications, and initial-vs-final judgment deltas.

Timing data is instrumentation-only. It does not change prompts, provider
selection, graph behavior, cache policy, or benchmark accuracy calculations.

## Supplement Handling

Supplements are supporting evidence sources. They are not concatenated into the
primary paper text and do not replace the primary publication.

At ingestion time, the pipeline:

1. Parses the primary PDF normally.
2. Runs supplement document type detection, using content-based detection for
   protocol, SAP, and appendix roles when available and filename classification
   as a fallback.
3. Parses supplements in bounded page windows.
4. Segments supplements into `SupplementSegments` with headings, page ranges,
   document roles, RoB 2 domain tags, and short methodological annotations.
5. Records warnings and fallback annotations when parsing or annotation is
   partial, while keeping usable segments retrievable.
6. Builds in-memory BM25S `SupplementIndex` instances from the segments.
7. Selects supplement hits into SQ-specific evidence packets alongside
   ClinicalTrials.gov evidence and primary-paper `section_text` fallbacks.
8. Surfaces supplement source name, role, page, path, selected segments, and
   retrieval diagnostics in JSON.

These supplement parsing, annotation, and retrieval diagnostics are the first
place to inspect when supplement evidence is missing or unexpectedly weak.

Supplement statuses in `source_documents`:

| Status    | Meaning                                                                            |
| --------- | ---------------------------------------------------------------------------------- |
| `parsed`  | All attempted supplement windows parsed cleanly                                    |
| `partial` | One or more windows failed; check warnings and retrieved sources for usable chunks |
| `failed`  | No usable content could be extracted                                               |
| `missing` | The requested supplement file did not exist                                        |

Windowed parsing avoids losing an entire long protocol or appendix because one
page triggers a parser error.

## Configuration

Common environment variables:

| Setting                                      | Purpose                                          |
| -------------------------------------------- | ------------------------------------------------ |
| `ROB2_PROVIDER`                              | `openrouter` (default), `anthropic`, or `openai` |
| `ROB2_MODEL`                                 | Model name for LLM calls                         |
| `ROB2_TEMPERATURE`                           | LLM generation temperature                       |
| `ROB2_MAX_TOKENS`                            | LLM output token limit                           |
| `ROB2_EFFECT_OF_INTEREST`                    | Default effect of interest, usually `ITT`        |
| `ROB2_USE_CACHE=1`                           | Enable prompt cache in `.rob2_cache/`            |
| `ROB2_CTGOV_CACHE`                           | ClinicalTrials.gov response cache path           |
| `ROB2_REMOTE_EVIDENCE_EXTRACTION=0`          | Disable ingestion-time LLM evidence refinement   |
| `ROB2_SUPPLEMENT_PAGE_WINDOW`                | Supplement page-window size, default `20`        |
| `ROB2_SUPPLEMENT_MAX_SCAN_PAGES`             | Defensive supplement scan limit, default `1000`  |
| `ROB2_RPM_LIMIT`, `ROB2_RPD_LIMIT`           | OpenRouter rate-limit controls                   |
| `ANTHROPIC_RPM_LIMIT`, `ANTHROPIC_TPM_LIMIT` | Anthropic rate-limit controls                    |

Provider and model settings are read when modules are imported, so export them
before invoking the CLI.

## Project Map

```text
data/references/             benchmark reference CSVs
CONTEXT.md                   domain glossary and subsystem map for agents
inputs/                      local PDFs, ignored by git
inputs/benchmark/            benchmark primary PDFs
inputs/benchmark/supplement/ benchmark supplements by trial name
outputs/                     generated reports and diagnostics, ignored by git
rob2_pipeline/               pipeline package
tests/                       unit and integration-style tests
```

Key files:

| Path                                      | Responsibility                                   |
| ----------------------------------------- | ------------------------------------------------ |
| `main.py`                                 | CLI for one or more PDFs                         |
| `benchmark.py`                            | CLI for benchmark runs                           |
| `rob2_pipeline/pipeline.py`               | Public `run_assessment()` API and output writing |
| `rob2_pipeline/graph.py`                  | LangGraph workflow wiring                        |
| `rob2_pipeline/ingestion/`                | Primary and supplement PDF ingestion             |
| `rob2_pipeline/supplement_retrieval.py`   | BM25S SupplementIndex over SupplementSegments    |
| `rob2_pipeline/nodes/domain_context.py`   | Prompt-ready D1-D5 evidence context              |
| `rob2_pipeline/nodes/domain_helpers.py`   | Shared `DomainSqStage` SQ-stage runner           |
| `rob2_pipeline/nodes/evidence_packets.py` | SQ-specific evidence packets                     |
| `rob2_pipeline/evidence_store.py`         | Typed quote-grounded evidence facts and gaps     |
| `rob2_pipeline/trial_workspace.py`        | Trial and outcome artifact workspace manifests   |
| `rob2_pipeline/nodes/verification.py`     | Quote, packet, and support-constraint checks     |
| `rob2_pipeline/judges/`                   | Deterministic RoB 2 judgment logic               |
| `rob2_pipeline/providers/`                | LLM provider adapters                            |
| `CONTEXT.md`                              | Shared domain vocabulary and change-path map     |

## Python API

```python
from rob2_pipeline.pipeline import run_assessment

state = run_assessment(
    "inputs/example.pdf",
    outcome="Overall Survival",
    effect_of_interest="ITT",
    output_dir="outputs",
    supplementary_paths=["inputs/example-protocol.pdf"],
)

print(state["overall_judgment"])
```

`run_assessment()` also accepts `precomputed_ingestion` and
`supplement_indexes` for internal benchmark-style reuse across multiple
outcomes from the same trial. Normal callers should usually leave those unset.

## Development

Run all tests:

```bash
uv run python -m pytest -q
```

Run focused tests:

```bash
uv run python -m pytest tests/test_domain_stages.py -q
uv run python -m pytest tests/test_domain_context.py -q
uv run python -m pytest tests/test_supplements.py -q
uv run python -m pytest tests/test_benchmark.py -q
```

Syntax-check selected files:

```bash
uv run python -m py_compile rob2_pipeline/benchmark.py benchmark.py
```

## Troubleshooting

| Symptom                     | First places to inspect                                          |
| --------------------------- | ---------------------------------------------------------------- |
| LLM JSON contract failure   | Trace JSON and `rob2_pipeline/llm_contracts.py`                  |
| Early non-RCT stop          | `is_rct`, `rct_screen_evidence`, `errors`                        |
| Missing evidence            | `evidence`, `evidence_packets`, `packet_grades`                  |
| Prompt evidence mismatch    | `rob2_pipeline/nodes/domain_context.py` and relevant `DomainSqStage` |
| Weak D3/D5 support          | `packet_grades`, `verification_actions`, supplement sources      |
| Unexpected final-vs-initial label | `pivotality_tests`, `sq_support_adjudications`, `initial_domain_judgments` |
| Supplement parse errors     | `source_documents`, `supplement_warnings`                        |
| Supplement annotation gaps  | `supplement_warnings`, fallback annotations in `supplement_segments` |
| Sparse supplement retrieval | `supplement_retrieval_grades`, `supplement_segments`, evidence packet sources |
| ClinicalTrials.gov mismatch | Registered endpoint fields and CT.gov-derived `evidence_packets` |

For large supplements with repeated skipped windows, reduce
`ROB2_SUPPLEMENT_PAGE_WINDOW`. To scan deeper into very long supplements,
increase `ROB2_SUPPLEMENT_MAX_SCAN_PAGES`.
