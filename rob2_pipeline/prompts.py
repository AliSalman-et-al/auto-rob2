from rob2_pipeline.methodology import (
    DOMAIN1_METHODOLOGY,
    DOMAIN2_ADHERING_METHODOLOGY,
    DOMAIN2_ASSIGNMENT_METHODOLOGY,
    DOMAIN3_METHODOLOGY,
    DOMAIN4_METHODOLOGY,
    DOMAIN5_METHODOLOGY,
)
from rob2_pipeline.methodology.render import render_methodology


DOMAIN1_METHODOLOGY_TEXT = render_methodology(
    DOMAIN1_METHODOLOGY, ["1.1", "1.2", "1.3"]
)
DOMAIN2_SQ12_METHODOLOGY_TEXT = render_methodology(
    DOMAIN2_ASSIGNMENT_METHODOLOGY, ["2.1", "2.2"]
)
DOMAIN2_CONDITIONAL_METHODOLOGY_TEXT = render_methodology(
    DOMAIN2_ASSIGNMENT_METHODOLOGY, ["2.3", "2.4", "2.5"]
)
DOMAIN2_ANALYSIS_METHODOLOGY_TEXT = render_methodology(
    DOMAIN2_ASSIGNMENT_METHODOLOGY, ["2.6", "2.7"]
)
DOMAIN2_ADHERING_CONDITIONAL_METHODOLOGY_TEXT = render_methodology(
    DOMAIN2_ADHERING_METHODOLOGY, ["2.3a", "2.4a", "2.5a"]
)
DOMAIN2_ADHERING_ANALYSIS_METHODOLOGY_TEXT = render_methodology(
    DOMAIN2_ADHERING_METHODOLOGY, ["2.6a"]
)
DOMAIN3_METHODOLOGY_TEXT = render_methodology(
    DOMAIN3_METHODOLOGY, ["3.1", "3.2", "3.3", "3.4"]
)
DOMAIN4_METHODOLOGY_TEXT = render_methodology(
    DOMAIN4_METHODOLOGY, ["4.1", "4.2", "4.3", "4.4", "4.5"]
)
DOMAIN5_METHODOLOGY_TEXT = render_methodology(
    DOMAIN5_METHODOLOGY, ["5.1", "5.2", "5.3"]
)


PROMPT_RCT_SCREEN = """You are a systematic review methodologist. Verify whether the input study is a randomized controlled trial before any RoB 2 assessment.

Read the following text carefully:

<methods_section>
{methods_text}
</methods_section>

RoB 2 applies only to randomized controlled trials: participants must be randomly assigned to intervention vs control/comparator groups. Random methods include a computer-generated sequence or computer-generated random numbers, random number tables, coin tossing, shuffling cards or envelopes, throwing dice, minimization with a random element, or drawing lots.

Studies that are not RCTs include observational studies, cohort studies, case-control studies, non-randomized experimental designs, and quasi-randomized trials where allocation is predictable, such as alternation, date of birth, admission date, record number, clinician judgment, or intervention availability.

If the study is not randomized, stop the assessment and state that ROBINS-I, not RoB 2, should be used.

Respond in this exact XML format:

<screening>
  <is_rct>YES or NO</is_rct>
  <evidence>"[exact quote from text supporting your decision]"</evidence>
  <study_design>[brief description of the study design as reported]</study_design>
  <note>[if NO: state that ROBINS-I should be used instead; if YES: leave blank]</note>
</screening>"""

PROMPT_PRELIMINARY_INFO = """You are an expert systematic reviewer applying the Cochrane Risk of Bias 2 (RoB 2) tool. Before answering signaling questions, collect the preliminary information required by RoB 2.

Read these sections of the trial report:

<abstract>
{abstract_text}
</abstract>

<methods>
{methods_text}
</methods>

<results>
{results_text}
</results>

<registration>
{registration_text}
</registration>

<consort_flow>
{consort_text}
</consort_flow>

Extract the experimental intervention, comparator intervention, outcome being assessed, numerical result, effect-relevant analysis details, trial registration, and source information. RoB 2 is result-specific, not trial-wide; if no user-specified outcome is available, use the primary outcome. If co-primary endpoints are apparent and the target outcome is unclear, identify the selected outcome and mark uncertainty in the quote/justification fields.

For every extracted value, provide the exact quoted text from the available source. Do not paraphrase quotes. If genuinely absent, write "Not reported" for the value and "No relevant text found" for the quote.

<preliminary_info>
  <experimental_intervention>
    <value>[name and description of the treatment being tested]</value>
    <quote>"[exact text]" ([Section])</quote>
  </experimental_intervention>

  <comparator_intervention>
    <value>[name and description of the control arm]</value>
    <quote>"[exact text]" ([Section])</quote>
  </comparator_intervention>

  <outcome_assessed>
    <value>[specific outcome; use the primary outcome if user did not specify another]</value>
    <quote>"[exact text]" ([Section])</quote>
    <is_primary>YES or NO or UNCLEAR</is_primary>
  </outcome_assessed>

  <outcome_type>Classify the outcome type using exactly one of these five values:
  - vital-status: all-cause mortality or disease-specific mortality assessed as a single criterion: death is the only event that counts. Do not use this category for composite endpoints that combine death with non-mortality criteria such as relapse, hospitalisation, or treatment failure, even if death is one component.
  - biomarker: laboratory or imaging measurement with a pre-defined numerical threshold
  - clinician-composite: composite or time-to-event outcome requiring clinical or radiological judgment
  - clinician-graded: outcome assessed using a standardized clinical grading scale that still requires judgment
  - patient-reported: outcome assessed by the participant using a questionnaire or self-report instrument
  Examples: mortality-only endpoint = `vital-status`; composite endpoint combining death with another clinical event = `clinician-composite`; standardized clinician-rated response scale = `clinician-graded`; participant questionnaire = `patient-reported`.
</outcome_type>

  <numerical_result>
    <value>[effect estimate with CI, e.g., HR 0.64 (95% CI 0.54-0.77)]</value>
    <quote>"[exact text]" ([Section])</quote>
  </numerical_result>

  <n_randomized>
    <value>[total number randomized, as integer string]</value>
    <quote>"[exact text]" ([Section])</quote>
  </n_randomized>

  <trial_registration>
    <number>[registration number or "Not reported"]</number>
    <registry>[ClinicalTrials.gov or ISRCTN or other registry or "Not reported"]</registry>
    <quote>"[exact text]" or "Not reported"</quote>
  </trial_registration>

  <registered_primary_endpoint>
    <value>[primary endpoint listed in trial registry or protocol]</value>
    <quote>"[exact text]" ([Section])</quote>
  </registered_primary_endpoint>

  <registered_secondary_endpoints>List any secondary or co-primary endpoints from the registration or protocol exactly as stated. Separate multiple endpoints with semicolons. If none found, write "Not reported".</registered_secondary_endpoints>

  <registered_analysis>
    <value>[pre-specified analysis approach from registry/protocol, e.g., ITT or per-protocol]</value>
    <quote>"[exact text]" ([Section])</quote>
  </registered_analysis>
</preliminary_info>

RULES:
- Every quote must be the exact source text, not a paraphrase.
- Do not guess numerical results.
- Use "No relevant text found" only when the available sources genuinely do not contain the information."""

PROMPT_DOMAIN1 = (
    """You are an expert systematic reviewer applying the Cochrane Risk of Bias 2 (RoB 2) tool.

TRIAL: {intervention} vs {comparator} | Outcome: {outcome}

Read the following evidence. The Primary Evidence section was extracted specifically for this domain. Use it as your primary source. The Additional Retrieved Context supplements it; it may contain supporting detail not present in the primary section.

=== PRIMARY EVIDENCE (domain-extracted - treat as authoritative) ===

<randomization_section>
{randomization_text}
</randomization_section>

<baseline_characteristics>
{baseline_text}
</baseline_characteristics>

<consort_flow>
{consort_text}
</consort_flow>

<registry_design_metadata>
{ctgov_design}
</registry_design_metadata>

=== ADDITIONAL RETRIEVED CONTEXT (full-document search) ===
{rag_text}

Answer Domain 1 signaling questions: Bias arising from the randomization process.
"""
    + DOMAIN1_METHODOLOGY_TEXT
    + """

If ClinicalTrials.gov design metadata is provided above, treat it as authoritative evidence about the trial's registered design:
- An allocation type of RANDOMIZED is evidence that the registry classifies the trial as randomized; without sequence-generation details, this supports PY rather than Y for Q1.1.
- Masking = NONE confirms an open-label design, which is context for assessors but is not directly scored in D1.
- Presence of a DMC or a research network lead sponsor is contextual registry information only; do not treat it as direct evidence of allocation concealment unless the paper or registry also describes a concealment or central allocation process.
- Use NI only when both the paper text and registry metadata provide no meaningful basis for a judgment.

For each question, choose exactly one answer: Y, PY, PN, N, or NI.
Y/N means firm evidence is stated. PY/PN means a reasonable inference from indirect evidence. NI is reserved for genuine absence of enough information; it is not a default.

<domain1>
  <sq_1_1>
    <answer>[Y/PY/PN/N/NI]</answer>
    <quote>"[exact text from paper]" ([Section reference])</quote>
    <justification>[one sentence explaining why this quote supports this answer]</justification>
  </sq_1_1>
  <sq_1_2>
    <answer>[Y/PY/PN/N/NI]</answer>
    <quote>"[exact text from paper]" ([Section reference])</quote>
    <justification>[one sentence]</justification>
  </sq_1_2>
  <sq_1_3>
    <answer>[Y/PY/PN/N/NI]</answer>
    <quote>"[exact text from paper]" ([Section reference])</quote>
    <justification>[one sentence]</justification>
  </sq_1_3>
</domain1>"""
)

PROMPT_DOMAIN2_SQ12 = (
    """You are an expert systematic reviewer applying the Cochrane Risk of Bias 2 (RoB 2) tool.

TRIAL: {intervention} vs {comparator} | Outcome: {outcome} | Effect of interest: effect of assignment to intervention (intention-to-treat)

Read the following evidence. The Primary Evidence section was extracted specifically for this domain. Use it as your primary source. The Additional Retrieved Context supplements it; it may contain supporting detail not present in the primary section.

=== PRIMARY EVIDENCE (domain-extracted - treat as authoritative) ===

<blinding_section>
{blinding_text}
</blinding_section>

<methods_interventions>
{methods_text}
</methods_interventions>

<registry_design_metadata>
{ctgov_design}
</registry_design_metadata>

=== ADDITIONAL RETRIEVED CONTEXT (full-document search) ===
{rag_text}

Answer the first two Domain 2 signaling questions: Bias due to deviations from intended interventions.
"""
    + DOMAIN2_SQ12_METHODOLOGY_TEXT
    + """

Important RoB 2 principle: an open-label trial is not automatically high risk. Risk depends on whether awareness led to deviations from intended interventions that arose because of the trial context, whether those deviations affected the outcome, whether they were balanced, and whether the analysis was appropriate.

If ClinicalTrials.gov design metadata is provided above, use the masking field as authoritative confirmation when it maps to the person being assessed: masking = NONE confirms participants and carers were aware of their assignment (supports Y for Q2.1 and Q2.2). For blinded designs, check any listed masked parties before judging participants separately from carers or intervention deliverers.

If both 2.1 and 2.2 are N/PN, 2.3-2.5 are not applicable and the pipeline will skip them.

<domain2_part1>
  <sq_2_1>
    <answer>[Y/PY/PN/N/NI]</answer>
    <quote>"[exact text]" ([Section])</quote>
    <justification>[one sentence]</justification>
  </sq_2_1>
  <sq_2_2>
    <answer>[Y/PY/PN/N/NI]</answer>
    <quote>"[exact text]" ([Section])</quote>
    <justification>[one sentence]</justification>
  </sq_2_2>
</domain2_part1>"""
)

PROMPT_DOMAIN2_CONDITIONAL = (
    """You are an expert systematic reviewer applying the Cochrane Risk of Bias 2 (RoB 2) tool.

TRIAL: {intervention} vs {comparator} | Outcome: {outcome}
Prior answers: Q2.1 = {sq_2_1} | Q2.2 = {sq_2_2}

Read the following evidence. The Primary Evidence section was extracted specifically for this domain. Use it as your primary source. The Additional Retrieved Context supplements it; it may contain supporting detail not present in the primary section.

=== PRIMARY EVIDENCE (domain-extracted - treat as authoritative) ===

<protocol_adherence>
{deviations_text}
</protocol_adherence>

<concomitant_medications>
{concomitant_text}
</concomitant_medications>

=== ADDITIONAL RETRIEVED CONTEXT (full-document search) ===
{rag_text}

Because 2.1 or 2.2 was Y/PY/NI, answer the conditional Domain 2 questions for the effect of assignment to intervention.
"""
    + DOMAIN2_CONDITIONAL_METHODOLOGY_TEXT
    + """

If 2.3 is N/PN, answer 2.4 and 2.5 as NA. If 2.3 is Y/PY/NI, answer 2.4.

For SQ 2.3 in aware/open-label trials, N/PN requires affirmative evidence that no relevant deviations occurred or that described deviations were not trial-context deviations. Do not infer N/PN from generic methods/results text, ITT language, "No relevant text found", or simple absence of reported deviations; use NI when that is all the sources provide.

If 2.4 is N/PN/NI/NA, answer 2.5 as NA. If 2.4 is Y/PY, answer 2.5.

<domain2_conditional>
  <sq_2_3>
    <answer>[Y/PY/PN/N/NI]</answer>
    <quote>"[exact text]" ([Section]) or "No relevant text found"</quote>
    <justification>[one sentence]</justification>
  </sq_2_3>
  <sq_2_4>
    <answer>[Y/PY/PN/N/NI or NA]</answer>
    <quote>"[exact text]" or "Not applicable"</quote>
    <justification>[one sentence or "Not applicable"]</justification>
    <uncertainty_flag>[HIGH if subjective judgment with limited evidence, otherwise NORMAL]</uncertainty_flag>
  </sq_2_4>
  <sq_2_5>
    <answer>[Y/PY/PN/N/NI or NA]</answer>
    <quote>"[exact text]" or "Not applicable"</quote>
    <justification>[one sentence or "Not applicable"]</justification>
  </sq_2_5>
</domain2_conditional>"""
)

PROMPT_DOMAIN2_ADHERING_CONDITIONAL = (
    """You are an expert systematic reviewer applying the Cochrane Risk of Bias 2 (RoB 2) tool.

TRIAL: {intervention} vs {comparator} | Outcome: {outcome}
Prior answers: Q2.1 = {sq_2_1} | Q2.2 = {sq_2_2}
Effect of interest: effect of adhering to intervention (per-protocol)

Read the following evidence. The Primary Evidence section was extracted specifically for this domain. Use it as your primary source. The Additional Retrieved Context supplements it; it may contain supporting detail not present in the primary section.

=== PRIMARY EVIDENCE (domain-extracted - treat as authoritative) ===

<protocol_adherence>
{deviations_text}
</protocol_adherence>

<concomitant_medications>
{concomitant_text}
</concomitant_medications>

=== ADDITIONAL RETRIEVED CONTEXT (full-document search) ===
{rag_text}

Answer the Domain 2 Version B conditional questions for bias due to deviations from intended interventions when estimating the effect of adhering to intervention.
"""
    + DOMAIN2_ADHERING_CONDITIONAL_METHODOLOGY_TEXT
    + """

<domain2_conditional>
  <sq_2_3>
    <answer>[Y/PY/PN/N/NI or NA]</answer>
    <quote>"[exact text]" ([Section]) or "Not applicable"</quote>
    <justification>[one sentence or "Not applicable"]</justification>
  </sq_2_3>
  <sq_2_4>
    <answer>[Y/PY/PN/N/NI or NA]</answer>
    <quote>"[exact text]" or "Not applicable"</quote>
    <justification>[one sentence or "Not applicable"]</justification>
    <uncertainty_flag>[HIGH if subjective judgment with limited evidence, otherwise NORMAL]</uncertainty_flag>
  </sq_2_4>
  <sq_2_5>
    <answer>[Y/PY/PN/N/NI or NA]</answer>
    <quote>"[exact text]" or "Not applicable"</quote>
    <justification>[one sentence or "Not applicable"]</justification>
  </sq_2_5>
</domain2_conditional>"""
)

PROMPT_DOMAIN2_ANALYSIS = (
    """You are an expert systematic reviewer applying the Cochrane Risk of Bias 2 (RoB 2) tool.

TRIAL: {intervention} vs {comparator} | Outcome: {outcome}
Effect of interest in pipeline state: {effect_of_interest}

Read the following evidence. The Primary Evidence section was extracted specifically for this domain. Use it as your primary source. The Additional Retrieved Context supplements it; it may contain supporting detail not present in the primary section.

=== PRIMARY EVIDENCE (domain-extracted - treat as authoritative) ===

<statistical_analysis>
{analysis_text}
</statistical_analysis>

<results_participants>
{results_text}
</results_participants>

=== ADDITIONAL RETRIEVED CONTEXT (full-document search) ===
{rag_text}

Answer Domain 2 analysis questions for the effect of assignment to intervention unless the user explicitly configured a per-protocol/adhering effect.
"""
    + DOMAIN2_ANALYSIS_METHODOLOGY_TEXT
    + """

If 2.6 is Y/PY, answer 2.7 as NA.

<domain2_analysis>
  <sq_2_6>
    <answer>[Y/PY/PN/N/NI]</answer>
    <quote>"[exact text]" ([Section])</quote>
    <justification>[one sentence]</justification>
  </sq_2_6>
  <sq_2_7>
    <answer>[Y/PY/PN/N/NI or NA]</answer>
    <quote>"[exact text]" or "Not applicable"</quote>
    <justification>[one sentence or "Not applicable"]</justification>
  </sq_2_7>
</domain2_analysis>"""
)

PROMPT_DOMAIN2_ADHERING_ANALYSIS = (
    """You are an expert systematic reviewer applying the Cochrane Risk of Bias 2 (RoB 2) tool.

TRIAL: {intervention} vs {comparator} | Outcome: {outcome}
Effect of interest in pipeline state: {effect_of_interest}

Read the following evidence. The Primary Evidence section was extracted specifically for this domain. Use it as your primary source. The Additional Retrieved Context supplements it; it may contain supporting detail not present in the primary section.

=== PRIMARY EVIDENCE (domain-extracted - treat as authoritative) ===

<statistical_analysis>
{analysis_text}
</statistical_analysis>

<results_participants>
{results_text}
</results_participants>

=== ADDITIONAL RETRIEVED CONTEXT (full-document search) ===
{rag_text}

Answer Domain 2 Version B analysis question for the effect of adhering to intervention.
"""
    + DOMAIN2_ADHERING_ANALYSIS_METHODOLOGY_TEXT
    + """

2.7 is not applicable for the effect of adhering to intervention; answer NA.

<domain2_analysis>
  <sq_2_6>
    <answer>[Y/PY/PN/N/NI]</answer>
    <quote>"[exact text]" ([Section])</quote>
    <justification>[one sentence]</justification>
  </sq_2_6>
  <sq_2_7>
    <answer>NA</answer>
    <quote>Not applicable</quote>
    <justification>Not applicable for effect of adhering to intervention.</justification>
  </sq_2_7>
</domain2_analysis>"""
)

PROMPT_DOMAIN3 = (
    """You are an expert systematic reviewer applying the Cochrane Risk of Bias 2 (RoB 2) tool.

TRIAL: {intervention} vs {comparator} | Outcome: {outcome} | N randomized: {n_randomized}

Read the following evidence. The Primary Evidence section was extracted specifically for this domain. Use it as your primary source. The Additional Retrieved Context supplements it; it may contain supporting detail not present in the primary section.

=== PRIMARY EVIDENCE (domain-extracted - treat as authoritative) ===

<consort_flow>
{consort_text}
</consort_flow>

<missing_data_handling>
{missing_data_text}
</missing_data_handling>

<sensitivity_analyses>
{sensitivity_text}
</sensitivity_analyses>

<registry_participant_flow>
{ctgov_flow}
</registry_participant_flow>

=== ADDITIONAL RETRIEVED CONTEXT (full-document search) ===
{rag_text}

Answer Domain 3 signaling questions: Bias due to missing outcome data.
"""
    + DOMAIN3_METHODOLOGY_TEXT
    + """

Before answering, calculate the percentage of randomised participants whose outcome data are included in the analysis: (analysis N / randomised N) x 100. If >= 10% of randomised participants are excluded from or absent in the analysis, answer N or PN rather than Y or PY. Report this calculation in the <completeness_calculation> field.

If ClinicalTrials.gov participant flow data is provided above, use it as supporting participant disposition evidence for Q3.1. Do not assume treatment completion equals outcome-data availability; compare it with paper text about the assessed outcome and missing outcome data.

For time-to-event outcomes, answer 3.1 as Y/PY only when the evidence directly addresses loss to follow-up, missing outcome data, or completeness of event/status ascertainment. ITT population language, Cox models, Kaplan-Meier methods, and censoring rules do not by themselves establish that missing outcome data were negligible.

If 3.1 is Y/PY, answer 3.2-3.4 as NA.

If 3.2 is Y/PY, answer 3.3-3.4 as NA. If 3.2 is N/PN, answer 3.3.

If 3.3 is N/PN, answer 3.4 as NA. If 3.3 is Y/PY/NI, answer 3.4.

<domain3>
  <sq_3_1>
    <answer>[Y/PY/PN/N/NI]</answer>
    <quote>"[exact text]" ([Section])</quote>
    <completeness_calculation>[e.g. "234/249 = 94.0%" or "Not calculable from available text"]</completeness_calculation>
    <justification>[one sentence]</justification>
  </sq_3_1>
  <sq_3_2>
    <answer>[Y/PY/PN/N/NI or NA]</answer>
    <quote>"[exact text]" or "Not applicable"</quote>
    <justification>[one sentence or "Not applicable"]</justification>
  </sq_3_2>
  <sq_3_3>
    <answer>[Y/PY/PN/N/NI or NA]</answer>
    <quote>"[exact text]" or "Not applicable"</quote>
    <justification>[one sentence or "Not applicable"]</justification>
    <uncertainty_flag>[HIGH or NORMAL]</uncertainty_flag>
  </sq_3_3>
  <sq_3_4>
    <answer>[Y/PY/PN/N/NI or NA]</answer>
    <quote>"[exact text]" or "Not applicable"</quote>
    <justification>[one sentence or "Not applicable"]</justification>
    <uncertainty_flag>[HIGH or NORMAL]</uncertainty_flag>
  </sq_3_4>
</domain3>"""
)

PROMPT_DOMAIN4 = (
    """You are an expert systematic reviewer applying the Cochrane Risk of Bias 2 (RoB 2) tool.

TRIAL: {intervention} vs {comparator} | Outcome: {outcome} | Outcome type: {outcome_type}
Q2.1 participants aware of assignment: {sq_2_1}

Read the following evidence. The Primary Evidence section was extracted specifically for this domain. Use it as your primary source. The Additional Retrieved Context supplements it; it may contain supporting detail not present in the primary section.

=== PRIMARY EVIDENCE (domain-extracted - treat as authoritative) ===

<outcome_measurement>
{outcome_measurement_text}
</outcome_measurement>

<blinding_section>
{blinding_text}
</blinding_section>

=== ADDITIONAL RETRIEVED CONTEXT (full-document search) ===
{rag_text}

Answer Domain 4 signaling questions: Bias in measurement of the outcome.
"""
    + DOMAIN4_METHODOLOGY_TEXT
    + """

Outcome-specific instruction: first identify the outcome currently being assessed: {outcome}. When the outcome_measurement evidence contains definitions for multiple outcomes, answer based only on the definition for {outcome}. Do not anchor Domain 4 reasoning to a different endpoint, even if that endpoint is described first or in more detail.

If 4.1 or 4.2 is Y/PY, answer 4.3-4.5 as NA.

If 4.3 is N/PN, answer 4.4-4.5 as NA.

If 4.4 is N/PN, answer 4.5 as NA.

<domain4>
  <sq_4_1>
    <answer>[Y/PY/PN/N/NI]</answer>
    <quote>"[exact text]" ([Section])</quote>
    <justification>[one sentence]</justification>
  </sq_4_1>
  <sq_4_2>
    <answer>[Y/PY/PN/N/NI]</answer>
    <quote>"[exact text]" ([Section])</quote>
    <justification>[one sentence]</justification>
  </sq_4_2>
  <sq_4_3>
    <answer>[Y/PY/PN/N/NI or NA]</answer>
    <auto_set_reason>[leave blank unless pipeline auto-set this answer]</auto_set_reason>
    <quote>"[exact text]" or "Not applicable"</quote>
    <justification>[one sentence or "Not applicable"]</justification>
  </sq_4_3>
  <sq_4_4>
    <answer>[Y/PY/PN/N/NI or NA]</answer>
    <quote>"[exact text]" or "Not applicable"</quote>
    <justification>[one sentence or "Not applicable"]</justification>
  </sq_4_4>
  <sq_4_5>
    <answer>[Y/PY/PN/N/NI or NA]</answer>
    <quote>"[exact text]" or "Not applicable"</quote>
    <justification>[one sentence or "Not applicable"]</justification>
    <uncertainty_flag>[HIGH or NORMAL]</uncertainty_flag>
  </sq_4_5>
</domain4>"""
)

PROMPT_DOMAIN5 = (
    """You are an expert systematic reviewer applying the Cochrane Risk of Bias 2 (RoB 2) tool.

TRIAL: {intervention} vs {comparator} | Outcome: {outcome} | Outcome type: {outcome_type}
Numerical result being assessed: {numerical_result}
Trial registration: {registration_number}
Registered primary endpoint: {registered_endpoint}
Reported outcome being assessed: {reported_endpoint}

Read the following evidence. The Primary Evidence section was extracted specifically for this domain. Use it as your primary source. The Additional Retrieved Context supplements it; it may contain supporting detail not present in the primary section.

=== PRIMARY EVIDENCE (domain-extracted - treat as authoritative) ===

Registered primary endpoint: {registered_endpoint}
Registered secondary endpoints: {registered_secondary_endpoints}

<registration_or_protocol>
{registration_text}
</registration_or_protocol>

<statistical_analysis_plan>
{sap_text}
</statistical_analysis_plan>

<results_section>
{results_text}
</results_section>

<authoritative_registration_outcomes>
{ctgov_outcomes}
</authoritative_registration_outcomes>

<authoritative_registration_description>
{ctgov_description}
</authoritative_registration_description>

=== ADDITIONAL RETRIEVED CONTEXT (full-document search) ===
{rag_text}

Answer Domain 5 signaling questions: Bias in selection of the reported result.
"""
    + DOMAIN5_METHODOLOGY_TEXT
    + """

IMPORTANT: You are assessing Domain 5 for the specific outcome: {outcome}. All three questions concern whether the {outcome} result was selectively reported. Do NOT reason about whether other outcomes were selectively reported or chosen. Each outcome is assessed independently.

If a trial registration number is available, compare the registry/protocol outcomes and analysis intentions against the result being assessed. Focus on whether the numerical result was selected on the basis of the results, not merely whether the assessed outcome was primary or secondary.

<domain5>
  <sq_5_1>
    <answer>[Y/PY/PN/N/NI]</answer>
    <quote>"[exact text]" ([Section])</quote>
    <justification>[one sentence]</justification>
    <registration_comparison>[note any discrepancy between registered and reported endpoint/analysis, or "No registration information available"]</registration_comparison>
  </sq_5_1>
  <sq_5_2>
    <answer>[Y/PY/PN/N/NI]</answer>
    <quote>"[exact text]" or "No relevant text found"</quote>
    <justification>[one sentence]</justification>
  </sq_5_2>
  <sq_5_3>
    <answer>[Y/PY/PN/N/NI]</answer>
    <quote>"[exact text]" or "No relevant text found"</quote>
    <justification>[one sentence]</justification>
  </sq_5_3>
</domain5>"""
)
