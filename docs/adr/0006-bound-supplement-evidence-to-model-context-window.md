# ADR-0006: Bound Supplement Evidence To The Model Context Window

Date: 2026-06-23

## Status

Proposed

## Context

Benchmarking the current master (`82d8b0e`) on the paid `openai/gpt-oss-120b`
model (131,072-token context window) showed that 3 of 8 RoB 2 trials - ARCHES,
ENZAMET, and STAMPEDE, the three with the largest supplement PDFs - fail
deterministically across all runs with `HTTP 400: maximum context length is
131072 tokens`. Their signaling-question and evidence-mining prompts assemble
220,000-396,000 tokens, far over the model limit. The other 5 trials score, but
their prompts are already large (ARASENS's D4 SQ prompt is ~90,000 tokens).

Reading the actual recorded prompts (`*_trace.json` `user_prompt`) pinpoints two
compounding causes:

1. A supplement that does not split into at least `MIN_STRUCTURAL_SEGMENTS` (3)
   structural sections collapses into ONE uncapped full-document segment
   (`rob2_pipeline/ingestion/supplement_segments.py:163-165` ->
   `_full_document_segment`, which concatenates every page's text). BM25S then
   retrieves that whole-document segment into the prompt. The `"Full document"`
   heading emitted only by this fallback is present in the failing prompts,
   confirming the path fires.
2. There is no total-prompt token budget between evidence assembly and
   `provider.complete()`, so an oversized payload is sent and rejected.

This contradicts the intent of PRD #159 ("Replace legacy RAG with BM25S
supplement retrieval"), which states that "injecting large supplements wholesale
is too expensive and degrades accuracy" and that supplements should be "searched
selectively ... instead of entire documents." The overflow is an implementation
gap in the `<3-segment` fallback, not the design intent; the fix is therefore
with-the-grain of the existing architecture.

## Decision

Two changes, plus a verification gate.

1. **Page-level fallback chunking.** When structural segmentation yields fewer
   than `MIN_STRUCTURAL_SEGMENTS`, fall back to one segment per page (all-domain
   tagged) instead of a single whole-document segment. BM25S `top_k` retrieval
   then selects only the most relevant pages. This also restores page-level
   provenance (PRD #159 user story 3). For a supplement whose page count is at
   or below `top_k`, retrieval returns every page, so behavior is unchanged for
   small documents.

2. **Total-prompt token budget.** Before an SQ / evidence-mining prompt is sent,
   if the assembled prompt exceeds a configurable ceiling
   (`ROB2_PROMPT_TOKEN_BUDGET`, default 115,000 tokens - headroom under the
   131,072 limit for the JSON schema, instructions, and the ~2,000-token
   output), drop the lowest-ranked evidence pieces until the prompt fits, and
   log each dropped piece. Trials already under the ceiling are unaffected.
   Token size is estimated conservatively; an exact tokenizer is not required
   for a safety margin.

3. **No-regression verification.** Re-run the k=3 benchmark on all 8 trials and
   diff against the pre-fix run. Acceptance: the 5 previously-scoreable trials
   are unchanged or improved, and ARCHES / ENZAMET / STAMPEDE now complete.
   Because some scoreable trials also hit the fallback (ARASENS does),
   byte-identical evidence is not promised; verified no-regression is.

The deterministic judges, skip/NA logic, and the published RoB 2 algorithm are
not touched. This change is confined to evidence assembly / retrieval.

## Consequences

- The 3 overflow trials become scoreable on gpt-oss; a real 8/8 baseline becomes
  possible.
- Retrieval becomes selective for large supplements, matching PRD #159 intent;
  likely a small accuracy gain on the big trials (less irrelevant text drowning
  the relevant evidence) and lower cost.
- Risk: the budget could drop evidence a question needed. Mitigated by dropping
  lowest-ranked pieces first, logging every drop (no silent truncation), and the
  no-regression gate.
- The already-large prompts on the 5 working trials are NOT globally reduced
  here; that broader "tighten retrieval everywhere" work is deliberately left as
  a follow-up, to be judged against this baseline.

## Implementation notes

- Change 1: `rob2_pipeline/ingestion/supplement_segments.py` - replace the
  single `_full_document_segment` fallback with a per-page segment builder;
  preserve segment fields (document role, per-page page numbers, all-domain
  tags, empty annotation). Keep `_full_document_segment` only if still needed for
  a genuinely single-page document.
- Change 2: enforce the budget at the evidence-assembly / prompt-assembly
  boundary. Per ADR-0001, source selection lives in the evidence-packet modules
  and prompt assembly in `rob2_pipeline/nodes/domain_context.py`; the exact
  insertion point is pinned in the implementation plan.
- New config: `ROB2_PROMPT_TOKEN_BUDGET` (default 115000), read in
  `rob2_pipeline/config.py` alongside the other env-driven limits.
- TDD: (a) a failing test that a synthetic oversized supplement yields an
  over-limit payload on the old path and a bounded payload on the new path;
  (b) a test that a small supplement (<= `top_k` pages) still retrieves in full
  (no behavior change); (c) a test that the budget drops lowest-ranked evidence
  and logs each drop.
