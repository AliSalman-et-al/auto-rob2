from rob2_pipeline.methodology.types import (
    Citation,
    DomainMethodology,
    ResponseRule,
    RuleCard,
)

SUPP = "Sterne 2019 supplement"

DOMAIN5_METHODOLOGY = DomainMethodology(
    domain_id="D5",
    title="Bias in selection of the reported result",
    principles=[
        "Assess the specific numerical result for the outcome under review, not selective non-reporting of other outcomes at review level."
    ],
    rule_cards={
        "5.1": RuleCard(
            "5.1",
            "Were data analysed according to a pre-specified plan finalized before unblinded outcome data were available?",
            {
                "Y": ResponseRule(
                    "A sufficiently detailed pre-specified plan was finalized before unblinded data and followed."
                ),
                "PY": ResponseRule(
                    "Strong indication of pre-specification exists, with some detail missing or minor justified deviations."
                ),
                "PN": ResponseRule(
                    "A plan is mentioned but not detailed enough, or unexplained deviations raise concern."
                ),
                "N": ResponseRule(
                    "Clear post hoc decisions, endpoint switching, or result-based changes occurred."
                ),
                "NI": ResponseRule(
                    "No adequate information on pre-specified analysis intentions or timing."
                ),
            },
            [Citation(SUPP, "p.26")],
            notes=[
                "Answer Y or PY for 5.1 if a trial registration number is cited and the registration predates the primary analysis, or if the paper explicitly states that primary endpoints or the statistical analysis plan were prespecified or publicly available. Do not require every statistical detail, such as covariate lists, imputation methods, or sensitivity analyses, to be reprinted in the paper itself. A registration number combined with a prespecification claim is sufficient for Y. Answer PN only if there is specific evidence that the analysis plan changed after data unblinding, or if no registration exists and no prespecification is documented anywhere.",
                "If a ClinicalTrials.gov registry description is provided above and lists PRIMARY, SECONDARY, or TERTIARY objectives, treat these as evidence that objectives were described in the registry. Objectives alone are not the same as prespecified endpoint definitions or a finalized pre-unblinding statistical analysis plan; use them together with protocol, SAP, amendment, and results-reporting evidence when judging Q5.1.",
            ],
        ),
        "5.2": RuleCard(
            "5.2",
            "Was the result selected, on the basis of the results, from multiple eligible outcome measurements within the outcome domain?",
            {
                "Y": ResponseRule(
                    "Selective reporting present (higher risk): the reported result for this outcome was chosen, on the basis of its value, from several pre-specified ways of measuring the same outcome (different scales, definitions, or time points). Answer Y only with specific evidence of such result-based selection."
                ),
                "PY": ResponseRule(
                    "Selective reporting may be present: multiple eligible measurements of this outcome likely existed and the reporting looks potentially result-driven, but result-based selection is not clearly shown."
                ),
                "PN": ResponseRule(
                    "Selective reporting probably absent: most intended measurements of this outcome are reported, or any omission is explained and unrelated to the result."
                ),
                "N": ResponseRule(
                    "Selective reporting absent (lower risk): the pre-specified measurement of this outcome is reported as planned, only one measurement was possible, or any inconsistency is unrelated to the result. A pre-specified result reported as planned is N, not Y. Other distinct outcomes listed in the registry (for example a different endpoint) are not multiple eligible measurements of this outcome."
                ),
                "NI": ResponseRule(
                    "Analysis intentions are unavailable or insufficient and multiple eligible measurements of this outcome could have existed."
                ),
            },
            [Citation(SUPP, "p.26-27")],
            notes=[
                "Examples include different scales, definitions, or time points.",
                "A pre-specified composite endpoint is NOT multiple eligible outcome measurements merely because it combines several components into one measure. Answer Q5.2=N for composite endpoints unless there is evidence that specific components were selected post-hoc. Answer Q5.2=Y/PY only when the paper reports one specific scale, definition, component, or time point chosen from several separately pre-specified alternatives based on the observed results.",
                "Pre-specified co-primary endpoints are not multiple eligible outcome measurements relative to each other: each is pre-specified and expected to be reported. Reporting multiple pre-specified co-primary endpoints does not by itself constitute selective outcome selection for any one endpoint.",
            ],
        ),
        "5.3": RuleCard(
            "5.3",
            "Was the result selected, on the basis of the results, from multiple eligible analyses of the data?",
            {
                "Y": ResponseRule(
                    "Selective reporting present (higher risk): the reported result was chosen, on the basis of its value, from several pre-specified eligible analyses of the same data (for example adjusted vs unadjusted models, different covariate sets, or different handling of missing data). Answer Y only with specific evidence of such result-based selection."
                ),
                "PY": ResponseRule(
                    "Selective reporting may be present: multiple eligible analyses likely existed and the reporting looks potentially result-driven, but result-based selection is not clearly shown."
                ),
                "PN": ResponseRule(
                    "Selective reporting probably absent: the intended analyses are mostly reported, or any omission is explained and unrelated to the result."
                ),
                "N": ResponseRule(
                    "Selective reporting absent (lower risk): the pre-specified analyses are reported as planned, only one analysis was possible, or any inconsistency is unrelated to the result. A pre-specified analysis reported as planned is N, not Y."
                ),
                "NI": ResponseRule(
                    "Analysis intentions are unavailable or insufficient and multiple eligible analyses could have existed."
                ),
            },
            [Citation(SUPP, "p.27-28")],
            notes=[
                "A particular outcome measurement may be analysed in multiple ways. Examples include: unadjusted and adjusted models; final value vs change from baseline vs analysis of covariance; transformations of variables; different definitions of composite outcomes; conversion of continuously scaled outcome to categorical data with different cut-points; different sets of covariates for adjustment; and different strategies for dealing with missing data."
            ],
        ),
    },
)
