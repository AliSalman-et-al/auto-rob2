# Spec: D2 and D4 Prompt Accuracy Fixes

**Date:** 2026-05-10  
**Status:** Proposed

---

## Context

After the D4/D5 fixes landed (`spec/pfs-d4-d5-reasoning-fix`), the CHAARTED benchmark shows D5
in perfect agreement on both outcomes. Two failures remain on the PFS outcome:

| Domain | Reference | Pipeline | Direction |
|--------|-----------|----------|-----------|
| D2 | Low | Some concerns | False positive |
| D4 | Some concerns | Low | False negative |
| Overall | Low (PFS), Low (OS) | Some concerns (PFS), Low (OS) | PFS wrong |

Evidence files:
- `outputs/benchmark/chaarted/CHAARTED_pfs/CHAARTED_rob2_data.json`
- `outputs/benchmark/chaarted/benchmark_results.json`

### Reference judgment rationale

The benchmark reference judgments were assigned using the following published criteria:

> "Risk of bias was assessed using Cochrane risk of bias for randomized controlled trials
> guidelines (v2) for each trial across patient important outcomes (overall survival, progression
> free survival, and grade 3 or higher adverse events). Overall bias for each trial was deemed to
> be low if there were low risk of bias in all domains or some concerns in one domain. PEACE-1
> trial raised some concerns over the deviation from intended intervention considering the trial
> protocol was modified to include docetaxel for some patients owing to change in standard of
> care. For STAMPEDE, LATITUDE, and ARCHES some concerns were raised for potential missing outcome
> data in at least 10% of the total population. Some concerns were raised for trials assessing
> progression free survival and adverse events which followed an open-label design and did not mask
> the outcome assessment. Only four trials followed a double-blind design. The outcome assessment
> for overall survival was deemed to be void of any potential biases due to unblinded assessment."

This establishes three systematic rules relevant to the current failures:

| Rule | Domain | Condition | Reference judgment |
|------|--------|-----------|-------------------|
| **R1** | D2 | Protocol formally modified due to external change in standard of care | Some concerns |
| **R2** | D4 | Open-label design + PFS or AE outcome (no masked outcome assessment) | Some concerns |
| **R3** | D4 | OS outcome in any design | Low (objective, unblinded assessment void of bias) |

Both failures are caused by the pipeline not applying these rules correctly. The deterministic
judge tables in `domain2.py` and `domain4.py` are correct Cochrane RoB 2 implementations;
only the upstream LLM signaling question (SQ) answers need to change.

---

## Failure 1 — D2 False Positive (pipeline=Some concerns, reference=Low)

### Decision path

| SQ | Answer | Notes |
|----|--------|-------|
| 2.1 | Y | Correct — open-label trial |
| 2.2 | Y | Correct |
| 2.3 | **NI** | **Wrong** — should be N/PN |
| 2.6 | Y | Correct — ITT used |

`_part1()` in `rob2_pipeline/judges/domain2.py` line 46 fires on Q2.3=NI:

```python
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

**Q2.3=NI from over-broad application of the NI default to routine clinical non-adherence.**

The LLM sees "6 patients in the combination group did not start the assigned therapy" and answers
NI because the report does not explicitly state whether this was due to trial-context factors.

The correct answer is N/PN. Rule R1 above clarifies exactly what constitutes a D2-elevating
trial-context deviation: a **formal protocol modification driven by an external change in standard
of care** (as in PEACE-1, where the protocol was amended to allow docetaxel for control-arm
patients because docetaxel became standard outside the trial). Routine pre-treatment non-starts —
a small number of patients who do not begin assigned chemotherapy for clinical reasons (performance
status decline, patient preference, early comorbidities) — are normal clinical management events.
They are protocol-consistent, do not reflect trial-context influence, and should be scored N/PN.

The contrast:

| Event | Q2.3 | Reason |
|-------|------|--------|
| Protocol formally amended to add docetaxel (PEACE-1) | Y/PY | External standard-of-care change drove protocol change — a trial-context event |
| 6 patients did not start assigned chemo (CHAARTED) | N/PN | Routine pre-treatment clinical decision — normal management, not trial-context |

The current `NI` definition ("the report does not state whether deviations arose because of trial
context") is applied too broadly: the LLM uses it whenever an explicit attribution is absent,
even when the only described events are unmistakably routine.

### Fix A — Strengthen Q2.3 NI vs N/PN guidance

**File:** `rob2_pipeline/prompts.py` → `PROMPT_DOMAIN2_CONDITIONAL`, Q2.3 guidance block.

1. **Strengthen the `N` definition** to include routine pre-treatment non-starts:

   > "Routine pre-treatment non-starts — a small number of participants in the experimental arm
   > who do not begin therapy before the first dose for clinical reasons such as performance status
   > decline, patient preference, or comorbidity — are normal clinical management. Score N or PN,
   > not NI, unless the report specifically attributes them to trial-context influence."

2. **Narrow the `NI` definition** so it requires genuine uncertainty about deviations that
   actually occurred, not just absence of an explicit statement:

   > "Answer NI only when deviations are described that could plausibly have arisen from trial
   > context but the report does not clarify their origin — for example, when it is unclear
   > whether a formal protocol amendment or external standard-of-care change drove the deviation.
   > Do not answer NI merely because routine non-adherence events lack an explicit statement that
   > they were unrelated to trial context."

---

## Failure 2 — D4 False Negative (pipeline=Low, reference=Some concerns)

### Applicable rule

Rule R2: **open-label design + PFS outcome → D4 = Some concerns** because outcome assessors
were not masked and progression assessment (biochemical, symptomatic, or radiographic) involves
clinical and radiological judgment that can be influenced by knowledge of treatment assignment.

Rule R3: **OS outcome → D4 = Low** because death is objective and cannot be influenced by
assessor knowledge of treatment assignment.

### Decision path

| SQ | Answer | Notes |
|----|--------|-------|
| 4.1 | N | Correct |
| 4.2 | N | Correct |
| 4.3 | **NI** | Wrong — open-label trial, assessors are aware (should be PY) |
| 4.4 | **N** | **Wrong** — PFS requires judgment; only OS/death warrants N |
| 4.5 | NA | Skipped because Q4.4=N |

`judge_domain4()` fires line 18–19 (Q4.4=N → Low):

```python
if s41 in ("N","PN","NI") and s42 in ("N","PN") and s43 in ("Y","PY","NI") and s44 in ("N","PN"):
    return "Low", "..."
```

For D4=Some concerns we need Q4.4=PY, which fires line 25–26:

```python
if ... s44 in ("Y","PY","NI") and s45 in ("N","PN"):
    return "Some concerns", "..."
```

### Root cause — two issues

**Issue 1: Q4.3 = NI in open-label trial (should be PY)**

The D4 prompt passes `Q2.1 participants aware of assignment: Y` but does not instruct the LLM
to use this fact to infer assessor awareness. In an open-label trial with no mention of central
blinded adjudication, outcome assessors are necessarily aware of treatment assignment. Q4.3
should default to PY, not NI.

**Issue 2: Q4.4 = N with reasoning borrowed from OS**

The LLM justifies Q4.4=N with: *"The outcome is inherently objective, so knowledge of
intervention assignment is unlikely to influence assessment."* This is correct for death (Rule R3)
but incorrect for PFS (Rule R2). CHAARTED's PFS endpoint is:
> *"time to biochemical, symptomatic, or radiographic progression with testosterone ≤50 ng/dL"*

Symptomatic and radiographic progression requires clinical and radiological judgment. In an
open-label trial, knowledge of treatment assignment CAN influence whether a clinician or
radiologist calls a scan as progression. The correct answer is Q4.4=PY.

**Contributing factor: `outcome_type` misclassification**

The PFS JSON shows `"outcome_type": "vital-status"` — incorrect. This endpoint is
`clinician-composite` (composite progression definition requiring clinical/radiological
judgment). The misclassification strips outcome-type context from the D4 prompt, making it
easier for the LLM to apply OS-style reasoning to a PFS assessment. Fixing the
`outcome_type` classification makes the correct Q4.4 reasoning more accessible.

### Fix B — `PROMPT_PRELIMINARY_INFO`: exclude composite endpoints from `vital-status`

**File:** `rob2_pipeline/prompts.py`, the `<outcome_type>` block.

Amend the `vital-status` definition:

> "`vital-status`: all-cause mortality or disease-specific mortality assessed as a **single
> criterion** — death is the only event that counts. Do not use this category for composite
> endpoints that combine death with non-mortality criteria such as progression, relapse, or
> hospitalisation, even if death is one component."

Add examples at the end of the block:

> "Examples: OS (all-cause death) = vital-status; PFS (progression or death) =
> clinician-composite; CRPC (biochemical, symptomatic, or radiographic progression) =
> clinician-composite; RECIST response rate = clinician-graded."

### Fix C — `PROMPT_DOMAIN4`: infer assessor awareness from Q2.1 context

**File:** `rob2_pipeline/prompts.py`, `PROMPT_DOMAIN4` Q4.3 guidance block.

Add an inference rule after the bullet definitions:

> "If the trial is open-label (Q2.1=Y as shown above) and the report contains no mention of a
> central blinded outcome adjudication committee or independent blinded assessors, answer PY
> (assessors likely aware of assignment) rather than NI. Reserve NI only when assessor blinding
> status genuinely cannot be inferred — which is unusual once Q2.1=Y is established."

### Fix D — `PROMPT_DOMAIN4`: Q4.4 must reflect endpoint type, not just trial design

**File:** `rob2_pipeline/prompts.py`, `PROMPT_DOMAIN4` Q4.4 guidance block.

Strengthen the `N` definition with an explicit endpoint-type rule:

> "N applies only when the outcome is physiologically determined and cannot be influenced by
> assessor knowledge — specifically `vital-status` outcomes (all-cause or disease-specific
> mortality, per Rule R3: OS outcome assessment is void of bias from unblinded design). For
> `clinician-composite`, `clinician-graded`, and `patient-reported` outcomes in open-label
> trials, N is incorrect because clinical or radiological judgment is involved. For composite
> progression endpoints (PFS, TTP, CRPC) that include symptomatic or radiographic components,
> answer PY in an open-label trial unless explicit evidence shows all progression criteria are
> mechanical and judgment-free."

---

## Scope: Other Benchmark Failures These Rules Predict

The reference rationale identifies additional systematic failures not yet benchmarked here:

| Trial | Domain | Reason | Expected reference |
|-------|--------|--------|--------------------|
| PEACE-1 | D2 | Protocol modified to include docetaxel (change in standard of care) | Some concerns |
| STAMPEDE | D3 | ≥10% missing outcome data | Some concerns |
| LATITUDE | D3 | ≥10% missing outcome data | Some concerns |
| ARCHES | D3 | ≥10% missing outcome data | Some concerns |
| All open-label trials (PFS/AE outcomes) | D4 | No masked outcome assessment | Some concerns |

Fix A (Q2.3 guidance) will help the PEACE-1 case by ensuring that evidence of a formal protocol
amendment IS recognised as a trial-context deviation (the PEACE-1 amendment is not a routine
non-start — it is a structural change to the protocol). Fixes B–D will apply to all open-label
PFS/AE assessments across the benchmark. D3 missing-data failures are out of scope for this spec.

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
      logic unchanged; only prompt text changes)
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
- [ ] D4 prompt contains endpoint-type rule for Q4.4
- [ ] D2 conditional prompt narrows NI and widens N for routine non-adherence
