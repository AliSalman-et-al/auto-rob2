from rob2_pipeline.methodology.types import (
    Citation,
    DomainMethodology,
    ResponseRule,
    RuleCard,
)

SUPP = "Sterne 2019 supplement"
BMJ = "Sterne 2019 BMJ"

_AWARENESS_21 = RuleCard(
    "2.1",
    "Were participants aware of their assigned intervention during the trial?",
    {
        "Y": ResponseRule(
            "Participants were explicitly aware; the intervention made blinding impossible; or intervention-specific side effects/toxicities revealed assignment."
        ),
        "PY": ResponseRule(
            "Design or side effects likely allowed participants to guess assignment, or blinding was attempted but likely ineffective."
        ),
        "PN": ResponseRule(
            "Blinding was used but not verified, interventions were similar, or no evidence of compromised participant blinding is reported."
        ),
        "N": ResponseRule(
            "Participants were explicitly and successfully blinded, or robust indistinguishable placebo/sham procedures were used and verified."
        ),
        "NI": ResponseRule(
            "Participant awareness or blinding is not reported and cannot be inferred."
        ),
    },
    [Citation(SUPP, "p.6"), Citation(BMJ, "p.4")],
)

_AWARENESS_22 = RuleCard(
    "2.2",
    "Were carers and people delivering interventions aware of participants' assigned intervention?",
    {
        "Y": ResponseRule(
            "Carers/deliverers were explicitly aware; blinding was impossible; visible side effects revealed assignment; or allocation was not concealed from care staff."
        ),
        "PY": ResponseRule(
            "Design or side effects likely revealed assignment to carers/deliverers."
        ),
        "PN": ResponseRule(
            "Blinding methods for carers/deliverers were described but not verified, or interventions were similar enough that awareness was unlikely."
        ),
        "N": ResponseRule(
            "Carers/deliverers were explicitly and successfully blinded, or robust blinding was implemented and verified."
        ),
        "NI": ResponseRule(
            "Awareness of carers/deliverers is not reported and cannot be inferred."
        ),
    },
    [Citation(SUPP, "p.6"), Citation(BMJ, "p.4")],
)

DOMAIN2_ASSIGNMENT_METHODOLOGY = DomainMethodology(
    domain_id="D2_ASSIGNMENT",
    title="Bias due to deviations from intended interventions: effect of assignment",
    principles=[
        "Open-label trials are not automatically high risk; bias depends on trial-context deviations and analysis appropriateness.",
        "In aware or open-label trials, 2.3=N/PN requires affirmative no-deviation evidence. Generic methods/results text, ITT language, or absence of reported deviations is insufficient evidence rather than proof of no trial-context deviations.",
    ],
    rule_cards={
        "2.1": _AWARENESS_21,
        "2.2": _AWARENESS_22,
        "2.3": RuleCard(
            "2.3",
            "Were there deviations from intended intervention that arose because of the trial context?",
            {
                "Y": ResponseRule(
                    "Clear evidence that trial context caused protocol-inconsistent deviations or non-protocol interventions."
                ),
                "PY": ResponseRule(
                    "Strong indications that recruitment, engagement, unblinding, or trial personnel led to protocol-inconsistent intervention changes or influenced adherence in ways that would not happen outside the trial."
                ),
                "PN": ResponseRule(
                    "Deviations appear consistent with what could occur outside the trial context, or no indication suggests trial-context influence."
                ),
                "N": ResponseRule(
                    "Affirmative evidence shows no relevant deviations, or deviations are explicitly unrelated to trial context, protocol-consistent, or normal clinical management that could occur outside the trial context."
                ),
                "NI": ResponseRule(
                    "Use when evidence about trial-context deviations is insufficient or when described deviations cannot be classified as trial-context or non-trial-context."
                ),
            },
            [Citation(SUPP, "p.7"), Citation(BMJ, "p.3")],
            notes=[
                "This question concerns changes from assigned intervention that are inconsistent with the protocol and occurred because of the trial context, such as recruitment, engagement, unblinding, or trial personnel undermining protocol implementation in ways that would not happen outside the trial. Do not count protocol-consistent changes such as dose cessation for toxicity, treatment changes after outcome events, or additional interventions used to treat consequences of the assigned intervention."
            ],
        ),
        "2.4": RuleCard(
            "2.4",
            "Were these deviations likely to have affected the outcome?",
            {
                "Y": ResponseRule(
                    "Clear evidence deviations affected the outcome, or large protocol-inconsistent changes likely altered outcomes."
                ),
                "PY": ResponseRule(
                    "Deviations were substantial and plausibly affect the assessed outcome."
                ),
                "PN": ResponseRule(
                    "Deviations were minimal or unlikely to influence the assessed outcome."
                ),
                "N": ResponseRule(
                    "Deviations were minor, unrelated to the outcome, or shown not to affect the outcome."
                ),
                "NI": ResponseRule(
                    "Deviations are described but their outcome impact cannot be determined."
                ),
            },
            [Citation(SUPP, "p.7")],
            applicability="Only if 2.3 is Y/PY/NI.",
        ),
        "2.5": RuleCard(
            "2.5",
            "Were these deviations balanced between groups?",
            {
                "Y": ResponseRule(
                    "Clear evidence deviations were equally distributed or balanced by design."
                ),
                "PY": ResponseRule(
                    "Data suggest similar patterns of deviations across groups."
                ),
                "PN": ResponseRule(
                    "Data suggest different patterns, but not conclusively."
                ),
                "N": ResponseRule("Clear evidence of unequal distribution."),
                "NI": ResponseRule(
                    "Deviations are known or likely to affect the outcome, but their distribution between groups is not reported."
                ),
            },
            [Citation(SUPP, "p.7")],
            applicability="Only if 2.4 is Y/PY.",
        ),
        "2.6": RuleCard(
            "2.6",
            "Was an appropriate analysis used to estimate the effect of assignment?",
            {
                "Y": ResponseRule(
                    "Clear ITT analysis, appropriate mITT, or exclusions limited to independently determined ineligible participants."
                ),
                "PY": ResponseRule(
                    "Analysis appears to follow ITT/mITT principles but lacks full detail, with only minimal likely irrelevant exclusions."
                ),
                "PN": ResponseRule(
                    "Analysis suggests deviation from ITT principles or some inappropriate exclusions/regrouping."
                ),
                "N": ResponseRule(
                    "Naive per-protocol, as-treated, analysis by treatment received, or substantial post-randomization exclusions of eligible participants."
                ),
                "NI": ResponseRule(
                    "Analysis method is not specified or cannot be assessed."
                ),
            },
            [Citation(SUPP, "p.8"), Citation(BMJ, "p.3")],
            notes=[
                "Both intention-to-treat (ITT) analyses and modified intention-to-treat (mITT) analyses excluding participants with missing outcome data should be considered appropriate.",
                "Both naive per-protocol analyses and as-treated analyses should be considered inappropriate.",
                "Analyses excluding eligible trial participants post-randomization should be considered inappropriate, but post-randomization exclusions of ineligible participants, when eligibility was not confirmed until after randomization and could not have been influenced by intervention group assignment, can be considered appropriate.",
            ],
        ),
        "2.7": RuleCard(
            "2.7",
            "Was there potential for substantial impact from failure to analyse participants as randomized?",
            {
                "Y": ResponseRule(
                    "Many participants were excluded or analysed in the wrong group, outcome is rare, or exclusions/misclassification are related to prognostic factors."
                ),
                "PY": ResponseRule(
                    "A moderate number of exclusions/misanalyses could affect the result."
                ),
                "PN": ResponseRule(
                    "Only a small number of participants were affected and a material effect is unlikely."
                ),
                "N": ResponseRule(
                    "Very few/no participants were affected, or effects are clearly unrelated to prognosis/outcome."
                ),
                "NI": ResponseRule(
                    "Insufficient information about exclusions or wrong-group analyses."
                ),
            },
            [Citation(SUPP, "p.8")],
            applicability="Only if 2.6 is N/PN/NI.",
            notes=[
                "There may be potential for substantial impact even if fewer than 5% of participants were analysed in the wrong group or excluded, if the outcome is rare or if exclusions are strongly related to prognostic factors."
            ],
        ),
    },
)

DOMAIN2_ADHERING_METHODOLOGY = DomainMethodology(
    domain_id="D2_ADHERING",
    title="Bias due to deviations from intended interventions: effect of adhering",
    principles=[
        "Use only when the assessment targets the effect of adhering to intervention."
    ],
    rule_cards={
        "2.1": _AWARENESS_21,
        "2.2": _AWARENESS_22,
        "2.3a": RuleCard(
            "2.3a",
            "Were important non-protocol interventions balanced?",
            {
                "Y": ResponseRule(
                    "Important non-protocol interventions were clearly balanced."
                ),
                "PY": ResponseRule("Available data suggest balance."),
                "PN": ResponseRule(
                    "Available data suggest imbalance, but not conclusively."
                ),
                "N": ResponseRule(
                    "Important non-protocol interventions were clearly not balanced."
                ),
                "NI": ResponseRule("Insufficient information to judge balance."),
                "NA": ResponseRule(
                    "Not applicable because the assessment is not addressing non-protocol interventions or participants/carers/deliverers were unaware."
                ),
            },
            [Citation(SUPP, "p.12")],
            notes=[
                "Important non-protocol interventions are additional interventions or exposures that are inconsistent with the trial protocol, may be received with or after starting assigned intervention, and are prognostic for the outcome."
            ],
        ),
        "2.4a": RuleCard(
            "2.4a",
            "Were there failures in implementing the intervention that could have affected the outcome?",
            {
                "Y": ResponseRule(
                    "Implementation failures clearly could have affected the outcome."
                ),
                "PY": ResponseRule(
                    "Implementation failures probably could have affected the outcome."
                ),
                "PN": ResponseRule(
                    "Implementation was mostly successful or failures were unlikely to affect the outcome."
                ),
                "N": ResponseRule(
                    "No relevant implementation failures occurred or they could not affect the outcome."
                ),
                "NI": ResponseRule(
                    "Insufficient information about implementation failures."
                ),
                "NA": ResponseRule(
                    "Not applicable because this deviation type is not being assessed."
                ),
            },
            [Citation(SUPP, "p.12")],
        ),
        "2.5a": RuleCard(
            "2.5a",
            "Was non-adherence to assigned intervention likely to affect outcomes?",
            {
                "Y": ResponseRule(
                    "Non-adherence clearly could have affected outcomes."
                ),
                "PY": ResponseRule(
                    "Non-adherence probably could have affected outcomes."
                ),
                "PN": ResponseRule(
                    "Non-adherence was limited or unlikely to affect outcomes."
                ),
                "N": ResponseRule(
                    "Participants adhered to the assigned regimen, or adherence issues could not affect outcomes."
                ),
                "NI": ResponseRule("Insufficient information about adherence."),
                "NA": ResponseRule(
                    "Not applicable because this deviation type is not being assessed."
                ),
            },
            [Citation(SUPP, "p.13")],
            notes=[
                "Non-adherence includes imperfect compliance, intervention cessation, crossovers to comparator intervention, and switches to another active intervention."
            ],
        ),
        "2.6a": RuleCard(
            "2.6a",
            "Was an appropriate analysis used to estimate the effect of adhering?",
            {
                "Y": ResponseRule(
                    "An appropriate causal method for the adherence effect was clearly used and justified, such as suitable instrumental variable analysis or inverse probability weighting."
                ),
                "PY": ResponseRule(
                    "An appropriate adherence-effect method appears to have been used, but some assumptions/details are unclear."
                ),
                "PN": ResponseRule(
                    "The analysis is probably inappropriate or insufficiently justified for the adherence effect."
                ),
                "N": ResponseRule(
                    "ITT, naive per-protocol, as-treated, analysis by treatment received, or another inappropriate method was used."
                ),
                "NI": ResponseRule(
                    "Insufficient information to judge whether the adherence-effect analysis was appropriate."
                ),
            },
            [Citation(SUPP, "p.14"), Citation(BMJ, "p.3")],
            notes=[
                "Naive per-protocol analyses, as-treated analyses, ITT analyses, and analysis by treatment received will usually be inappropriate for estimating the effect of adhering to intervention. Appropriate methods may include instrumental variable analyses for a single all-or-nothing baseline intervention, or inverse probability weighting to adjust for censoring of participants who cease adherence in sustained treatment strategies. Such methods depend on strong assumptions that should be appropriate and justified.",
                "If an important non-protocol intervention was administered to all participants in one intervention group, adjustments cannot be made to overcome this.",
            ],
        ),
    },
)
