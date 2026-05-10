# Spec: D2 and D4 Prompt Accuracy Fixes

**Date:** 2026-05-10  
**Status:** Proposed

---

## Context

After the D4/D5 fixes landed (`spec/pfs-d4-d5-reasoning-fix`), the CHAARTED benchmark now shows
D5 = perfect agreement on both outcomes. Two failures remain on the PFS outcome:

| Domain | Reference | Pipeline | Direction |
|--------|-----------|----------|-----------|
| D2 | Low | Some concerns | False positive |
| D4 | Some concerns | Low | False negative |
| Overall | Low (PFS), Low (OS) | Some concerns (PFS), Low (OS) | PFS wrong |

Evidence files:
- `outputs/benchmark/chaarted/CHAARTED_pfs/CHAARTED_rob2_data.json`
- `outputs/benchmark/chaarted/benchmark_results.json`

Both failures are caused by incorrect LLM signaling question (SQ) answers. The deterministic
judge tables in `domain2.py` and `domain4.py` correctly implement the Cochrane RoB 2 algorithm;
only the upstream LLM extraction needs to change.

---

## Failure 1 — D2 False Positive (pipeline=Some concerns, reference=Low)

### Decision path

| SQ | Answer | Notes |
|----|--------|-------|
| 2.1 | Y | Correct — open-label trial |
| 2.2 | Y | Correct |
| 2.3 | **NI** | **Wrong** — should be N/PN |
| 2.6 | Y | Correct — ITT used |

`_part1()` in `rob2_pipeline/judges/domain2.py`:

```python
# Line 46 — fires when Q2.3=NI
if (s21 in ("Y","PY","NI") or s22 in ("Y","PY","NI")) and s23 == "NI":
    return "Some concerns", "Part1 Some concerns: no information on trial-context deviations"
```

Part2=Low (Q2.6=Y). Combined result: **Some concerns** (Part1 drives it up, line 87).

For D2=Low we need Q2.3=N/PN, which fires line 44:

```python
if (s21 in ("Y","PY","NI") or s22 in ("Y","PY","NI")) and s23 in ("N","PN"):
    return "Low", "Part1 Low: awareness present/unclear but no trial-context deviations"
```

### Root cause

**The LLM answers Q2.3=NI because the report does not explicitly state whether 6 patients who
did not start therapy were prevented by trial-context factors.**

Q2.3 asks: *"Were there deviations from the intended intervention that arose because of the
trial context?"* The current prompt correctly defines trial-context deviations as "changes
inconsistent with protocol that occurred because of the trial context, such as
recruitment/engagement effects or trial personnel undermining protocol implementation," and
explicitly excludes "protocol-consistent changes such as dose cessation for toxicity."

However, the LLM treats absence of an explicit explanation as genuine uncertainty (NI) rather
than applying the correct default. Under RoB 2 methodology, routine pre-treatment non-starts
(patients declining to begin chemotherapy before the first dose for clinical reasons — performance
status decline, patient preference, early comorbidities) are normal clinical management events
that do not arise from trial context. The correct answer is N/PN: the report's silence about
why those 6 patients did not start is not evidence that trial context caused the non-starts; it
is the expected absence of note for an unremarkable clinical event.

The prompt's NI definition ("the report does not state whether deviations arose because of trial
context") is being applied too broadly — the LLM uses it whenever an explicit attribution is
absent, even when the only described events are unmistakably routine.

### Fix A — Strengthen Q2.3 NI vs N/PN guidance

**File:** `rob2_pipeline/prompts.py`, `PROMPT_DOMAIN2_CONDITIONAL` (Q2.3 guidance block).

Add two clarifications:

1. **Strengthen the `N` definition** to explicitly include routine pre-treatment non-starts:

   > "Routine pre-treatment non-starts (e.g., a small number of participants in the experimental
   > arm who do not begin therapy before the first dose for clinical reasons such as performance
   > status decline, patient preference, or comorbidity) are normal clinical management and should
   > be scored N or PN — not NI — unless the report specifically attributes them to trial-context
   > influence."

2. **Narrow the `NI` definition** so it requires both (a) clear evidence that deviations occurred
   AND (b) genuine uncertainty about whether trial context caused them:

   > "Answer NI only when deviations are described that could plausibly have arisen from trial
   > context but the report does not clarify their origin. Do not answer NI merely because routine
   > non-adherence events lack an explicit statement that they were unrelated to trial context."

---

## Failure 2 — D4 False Negative (pipeline=Low, reference=Some concerns)

### Decision path

| SQ | Answer | Notes |
|----|--------|-------|
| 4.1 | N | Correct |
| 4.2 | N | Correct |
| 4.3 | **NI** | Wrong — should be PY for open-label |
| 4.4 | **N** | **Wrong** — reasoning borrowed from OS, not PFS |
| 4.5 | NA | Correct (skipped because Q4.4=N) |

`judge_domain4()` fires line 18–19:

```python
# s41=N, s42=N, s43=NI, s44=N → Low
if s41 in ("N","PN","NI") and s42 in ("N","PN") and s43 in ("Y","PY","NI") and s44 in ("N","PN"):
    return "Low", "..."
```

For D4=Some concerns we need Q4.4=PY (not N), which fires line 25–26:

```python
# s43=Y/PY/NI, s44=Y/PY/NI, s45=N/PN → Some concerns
if ... s44 in ("Y","PY","NI") and s45 in ("N","PN"):
    return "Some concerns", "..."
```

### Root cause — three layered issues

**Layer A: `outcome_type` misclassification** (upstream, in `preliminary_info`)

The PFS assessment JSON shows `"outcome_type": "vital-status"`. CHAARTED's PFS endpoint is:
> *"time to biochemical, symptomatic, or radiographic progression with testosterone ≤50 ng/dL"*

This is a **clinician-composite** endpoint — it requires clinical and radiological judgment. It
is not a vital-status (all-cause mortality) endpoint.

The preliminary prompt defines:
- `vital-status`: all-cause mortality or an event defined purely as death
- `clinician-composite`: composite or time-to-event outcome requiring clinical or radiological judgment

The misclassification occurs because the `d4_outcome_meas` evidence section leads with the OS
definition ("Overall survival was defined as time from randomization to death from any cause"),
and the LLM anchors to this sentence even when the assessed outcome is PFS. The correct five
outcome-type categories must exclude composite/progression endpoints from `vital-status`
regardless of whether death is one component.

**Layer B: Q4.3 defaults to NI in open-label trials**

The D4 prompt receives `Q2.1 participants aware of assignment: Y` as context but does not
instruct the LLM to use this to infer assessor awareness. In an open-label trial with no
mention of central blinded adjudication or an independent blinded review committee, outcome
assessors are necessarily aware of treatment assignment. Q4.3 should default to PY (assessors
likely aware) rather than NI.

**Layer C: Q4.4=N reasoning borrowed from OS**

The LLM justifies Q4.4=N with: *"The outcome is inherently objective, so knowledge of
intervention assignment is unlikely to influence assessment."* This is correct for death (OS)
but wrong for PFS. Symptomatic and radiographic progression assessment involves clinical and
radiologist judgment and CAN be influenced by knowledge of treatment assignment. Q4.4 should
be PY for PFS in an open-label trial.

### Fix B — `PROMPT_PRELIMINARY_INFO`: harden `vital-status` vs `clinician-composite`

**File:** `rob2_pipeline/prompts.py`, the `<outcome_type>` block.

Amend the `vital-status` definition to exclude composites that include death as one component:

> "`vital-status`: all-cause mortality or disease-specific mortality **assessed as a single
> criterion** (i.e., death is the only event that counts). Do not use this category for
> composite endpoints that combine death with non-mortality criteria such as progression,
> relapse, or hospitalisation — even if death is one component."

Add examples at the end of the block:

> "Examples: OS (all-cause death) = vital-status; PFS (progression or death) =
> clinician-composite; CRPC (biochemical, symptomatic, or radiographic progression) =
> clinician-composite; RECIST response rate = clinician-graded."

### Fix C — `PROMPT_DOMAIN4`: infer assessor awareness from Q2.1 context

**File:** `rob2_pipeline/prompts.py`, `PROMPT_DOMAIN4` Q4.3 guidance block.

Add an inference rule after the bullet definitions:

> "If the trial is open-label (Q2.1=Y as shown above) and the report contains no mention of a
> central blinded outcome adjudication committee or independent blinded assessors, answer PY
> (assessors likely aware of assignment) rather than NI. Reserve NI only when the blinding
> status of assessors genuinely cannot be inferred from any available evidence — which is
> unusual when Q2.1 is already established."

### Fix D — `PROMPT_DOMAIN4`: Q4.4 cannot be N for composite/clinical endpoints

**File:** `rob2_pipeline/prompts.py`, `PROMPT_DOMAIN4` Q4.4 guidance block.

Strengthen the `N` definition and add an outcome-type rule:

> "N applies only when the outcome event is physiologically determined and cannot plausibly be
> influenced by assessor knowledge — specifically `vital-status` outcomes (all-cause or
> disease-specific mortality). For `clinician-composite`, `clinician-graded`, and
> `patient-reported` outcome types, N is almost never appropriate in an open-label trial because
> at least some components require clinical or radiological judgment. For composite progression
> endpoints (PFS, TTP, CRPC), where components include symptomatic or radiographic progression,
> answer PY unless there is explicit evidence that all assessment components are objective and
> mechanically determined."

---

## Why These Fixes Do Not Overfit to CHAARTED

All four changes are grounded in published RoB 2 methodology and apply across the full
10-trial benchmark:

| Fix | Generalisation |
|-----|---------------|
| A — Q2.3 NI narrowing | Any trial with routine clinical non-adherence (ARASENS, ENZAMET, STAMPEDE all have chemotherapy non-starts) |
| B — outcome_type vital-status exclusion | Any composite/TTP endpoint (ARCHES, LATITUDE, PEACE-1 all have PFS) |
| C — Q4.3 open-label inference | Every unblinded trial in the 10-trial benchmark |
| D — Q4.4 composite guidance | PFS/CRPC endpoints in all 10 trials |

OS assessments are unaffected: death = vital-status; Q4.4=N is correct for all-cause mortality.

---

## Files to Modify

| File | Change |
|------|--------|
| `rob2_pipeline/prompts.py` | 4 prompt-text edits (Fixes A–D) |

No changes to judge logic, state model, graph, or test infrastructure.

---

## Implementation Steps

- [ ] **Step 1: Update `PROMPT_DOMAIN2_CONDITIONAL`** — apply Fix A to Q2.3 guidance
- [ ] **Step 2: Update `PROMPT_PRELIMINARY_INFO`** — apply Fix B to `outcome_type` block
- [ ] **Step 3: Update `PROMPT_DOMAIN4`** — apply Fix C to Q4.3 guidance
- [ ] **Step 4: Update `PROMPT_DOMAIN4`** — apply Fix D to Q4.4 guidance
- [ ] **Step 5: Run test suite** — `uv run pytest tests/` → no regressions expected (judge
      logic is unchanged; only prompt text changes)
- [ ] **Step 6: Re-run CHAARTED benchmark** — verify PFS D2=Low, D4=Some concerns, Overall=Low;
      verify OS all-domain agreement unchanged

---

## Verification Checklist

- [ ] `uv run pytest tests/` passes with no regressions
- [ ] CHAARTED:PFS D2 matches reference (Low)
- [ ] CHAARTED:PFS D4 matches reference (Some concerns)
- [ ] CHAARTED:PFS Overall matches reference (Low)
- [ ] CHAARTED:OS 6/6 agreement unchanged
- [ ] `outcome_type` block excludes composite endpoints from `vital-status`
- [ ] D4 prompt contains open-label inference rule for Q4.3
- [ ] D4 prompt contains outcome-type rule for Q4.4
- [ ] D2 conditional prompt narrows NI and widens N for routine non-adherence
