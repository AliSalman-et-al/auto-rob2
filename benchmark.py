import argparse
import os
from pathlib import Path

from rob2_pipeline.benchmark import (
    OUTCOME_LABELS,
    find_supplements_for_trial,
    load_reference,
    run_benchmark,
    summarize_benchmark,
    write_benchmark_report,
)


DEFAULT_REFERENCE_OS = "data/references/overall_survival.csv"
DEFAULT_REFERENCE_PFS = "data/references/progression_free_survival.csv"
DEFAULT_REFERENCE_AE = "data/references/adverse_events.csv"


def _parse_outcome_map(values: list[str]) -> list[dict[str, str]]:
    parsed: list[dict[str, str]] = []
    for item in values:
        if ":" not in item:
            raise ValueError(
                f"Invalid outcome map item '{item}'. Expected TRIAL:OUTCOME[:COHORT]"
            )
        parts = item.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(
                f"Invalid outcome map item '{item}'. Expected TRIAL:OUTCOME[:COHORT]"
            )
        trial, outcome = parts[0], parts[1]
        cohort = parts[2].strip() if len(parts) == 3 else "unspecified"
        trial = trial.strip()
        outcome = outcome.strip().upper()
        if not trial or not outcome:
            raise ValueError(
                f"Invalid outcome map item '{item}'. Expected TRIAL:OUTCOME[:COHORT]"
            )
        if outcome not in OUTCOME_LABELS:
            raise ValueError(
                f"Invalid outcome code '{outcome}'. Use one of: {', '.join(OUTCOME_LABELS)}"
            )
        parsed.append(
            {"trial": trial, "outcome_code": outcome, "cohort": cohort or "unspecified"}
        )
    return parsed


def _resolve_reference_row(reference: dict[str, dict], trial: str) -> dict | None:
    key = trial.strip().casefold()
    for trial_name, row in reference.items():
        if trial_name.strip().casefold() == key:
            return row
    return None


def _resolve_pdf(input_dir: Path, trial: str) -> Path | None:
    direct = input_dir / f"{trial}.pdf"
    if direct.exists() and direct.is_file():
        return direct
    target = f"{trial}.pdf".casefold()
    for pdf in input_dir.glob("*.pdf"):
        if pdf.is_file() and pdf.name.casefold() == target:
            return pdf
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Run RoB 2 benchmark against reference judgments."
    )
    parser.add_argument(
        "--input-dir",
        default="inputs/benchmark/",
        help="Directory containing trial PDFs.",
    )
    parser.add_argument(
        "--reference-os",
        default=DEFAULT_REFERENCE_OS,
        help="CSV path for Overall Survival references.",
    )
    parser.add_argument(
        "--reference-pfs",
        default=DEFAULT_REFERENCE_PFS,
        help="CSV path for Progression-Free Survival references.",
    )
    parser.add_argument(
        "--reference-ae",
        default=DEFAULT_REFERENCE_AE,
        help="CSV path for Adverse Events references.",
    )
    parser.add_argument(
        "--outcome-map",
        nargs="+",
        required=True,
        help="Trial:outcome mappings, e.g. CHAARTED:OS ARCHES:PFS or CHAARTED:OS:calibration",
    )
    parser.add_argument(
        "--output-dir", default="outputs/benchmark/", help="Benchmark output directory."
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Bypass prompt cache for this run."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print extra progress details."
    )
    parser.add_argument(
        "--use-supplements",
        action="store_true",
        help="Use discovered supplements for each benchmark trial.",
    )
    parser.add_argument(
        "--supplement-dir",
        default=None,
        help="Directory containing per-trial supplement folders.",
    )
    parser.add_argument(
        "--supplement-policy",
        choices=["auto", "required", "none"],
        default="auto",
        help="How benchmark should treat missing supplements.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print planned runs only.",
    )
    parser.add_argument(
        "--gold-evidence",
        default=None,
        help="Optional JSON fixture path for expected pages/snippets by trial/outcome/SQ.",
    )
    args = parser.parse_args()

    if args.no_cache:
        os.environ["ROB2_USE_CACHE"] = "0"

    outcome_map = _parse_outcome_map(args.outcome_map)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    reference_csvs = {
        "OS": Path(args.reference_os),
        "PFS": Path(args.reference_pfs),
        "AE": Path(args.reference_ae),
    }

    references = {code: load_reference(path) for code, path in reference_csvs.items()}

    if args.dry_run:
        print("Dry run: validating inputs")
        for code, path in reference_csvs.items():
            print(f"- {code}: {path} ({len(references[code])} rows)")
        print("Planned trial runs:")
        for item in outcome_map:
            trial = item["trial"]
            code = item["outcome_code"]
            row = _resolve_reference_row(references[code], trial)
            pdf = _resolve_pdf(input_dir, trial)
            status = []
            if row is None:
                status.append("Trial not in reference")
            if pdf is None:
                status.append("PDF not found")
            details = "; ".join(status) if status else "OK"
            print(
                f"- {trial} -> {OUTCOME_LABELS[code]} [{item.get('cohort', 'unspecified')}] ({details})"
            )
            if args.use_supplements and args.supplement_dir:
                supplements = find_supplements_for_trial(
                    Path(args.supplement_dir), trial
                )
                supplement_names = (
                    ", ".join(path.name for path in supplements) or "none"
                )
                print(f"  supplements: {supplement_names}")
        return

    results = run_benchmark(
        pdf_dir=input_dir,
        reference_csvs=reference_csvs,
        outcome_map=outcome_map,
        output_dir=output_dir,
        supplement_dir=Path(args.supplement_dir) if args.supplement_dir else None,
        use_supplements=args.use_supplements,
        supplement_policy=args.supplement_policy,
        gold_evidence_path=Path(args.gold_evidence) if args.gold_evidence else None,
    )
    summary = summarize_benchmark(results)
    write_benchmark_report(results, summary, output_dir / "benchmark_report.md")

    print(f"Benchmark complete: {summary['evaluated_trials']} trial(s) evaluated")
    if args.debug:
        for item in results:
            print(
                f"- {item.get('id') or item.get('trial')}: outcome={item.get('outcome')} "
                f"skipped={item.get('skipped')} error={bool(item.get('error'))}"
            )
        print(f"Results written to: {output_dir / 'benchmark_results.json'}")
        print(f"Report written to: {output_dir / 'benchmark_report.md'}")


if __name__ == "__main__":
    main()
