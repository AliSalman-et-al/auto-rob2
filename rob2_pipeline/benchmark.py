import csv
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from rob2_pipeline.ingestion.assessment import AssessmentIngestionResult
from rob2_pipeline.pipeline import run_assessment


LOGGER = logging.getLogger(__name__)
DOMAINS = ("D1", "D2", "D3", "D4", "D5")
AGREEMENT_FIELDS = (*DOMAINS, "Overall", "Benchmark Overall")
OUTCOME_LABELS = {
    "OS": "Overall Survival",
    "PFS": "Progression-Free Survival",
    "AE": "Adverse Events",
}
JUDGMENT_ORDER = ("Low", "Some concerns", "High")


def _strip(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_trial(value: str) -> str:
    return _strip(value).casefold()


def _normalize_judgment(value: Any) -> str:
    raw = _strip(value)
    compact = " ".join(raw.split()).casefold()
    mapping = {
        "l": "Low",
        "low": "Low",
        "s": "Some concerns",
        "some concerns": "Some concerns",
        "h": "High",
        "high": "High",
    }
    return mapping.get(compact, raw)


def benchmark_reference_overall_judgment(domain_judgments: dict) -> str:
    values = [_normalize_judgment(value) for value in domain_judgments.values()]
    if any(value == "High" for value in values):
        return "High"
    if values and sum(1 for value in values if value == "Some concerns") <= 1:
        return "Low"
    if any(value == "Some concerns" for value in values):
        return "Some concerns"
    return ""


def _file_cache_identity(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))


def _trial_artifact_cache_key(pdf_path: Path, supplement_paths: list[Path]) -> tuple:
    return (
        _file_cache_identity(pdf_path),
        tuple(_file_cache_identity(path) for path in supplement_paths),
    )


def _state_ingestion_artifact(state: dict) -> AssessmentIngestionResult:
    return AssessmentIngestionResult(
        full_text=state.get("full_text", ""),
        evidence=state.get("evidence"),
        docling_doc=state.get("docling_doc"),
        docling_chunks=list(state.get("docling_chunks") or []),
        source_documents=list(state.get("source_documents") or []),
        supplement_warnings=list(state.get("supplement_warnings") or []),
    )


def _find_pdf_for_trial(pdf_dir: Path, trial_name: str) -> Path | None:
    direct = pdf_dir / f"{trial_name}.pdf"
    if direct.exists() and direct.is_file():
        return direct
    target = f"{trial_name}.pdf".casefold()
    for candidate in pdf_dir.glob("*.pdf"):
        if candidate.is_file() and candidate.name.casefold() == target:
            return candidate
    return None


def find_supplements_for_trial(supplement_dir: Path, trial_name: str) -> list[Path]:
    if not supplement_dir.exists() or not supplement_dir.is_dir():
        return []
    target = trial_name.strip().casefold()
    for candidate in supplement_dir.iterdir():
        if candidate.is_dir() and candidate.name.strip().casefold() == target:
            return sorted(path for path in candidate.glob("*.pdf") if path.is_file())
    return []


def _required_supplement_failures(
    requested_paths: list[Path], source_documents: list[dict]
) -> list[str]:
    def key(value: object) -> str:
        text = _strip(value)
        if not text:
            return ""
        try:
            text = str(Path(text).resolve())
        except OSError:
            pass
        return text.replace("\\", "/").casefold()

    non_primary_documents = [
        document for document in source_documents if not document.get("is_primary")
    ]
    documents_by_path = {
        key(document.get("path")): document
        for document in non_primary_documents
        if _strip(document.get("path"))
    }
    failures: list[str] = []
    for requested in requested_paths:
        requested_key = key(requested)
        document = documents_by_path.get(requested_key)
        if document is None:
            failures.append(f"{requested.name} (not ingested)")
        elif document.get("status") not in {"parsed", "partial"}:
            failures.append(f"{requested.name} ({document.get('status', 'unknown')})")
    return failures


def _coerce_int_ms(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _format_seconds(value_ms: object) -> str:
    return f"{_coerce_int_ms(value_ms) / 1000:.1f}s"


def _summarize_trace_timing(trace_path: Path, total_wall_ms: int) -> dict[str, Any]:
    timing = {
        "total_wall_ms": total_wall_ms,
        "trace_available": False,
        "node_total_ms": 0,
        "llm_total_ms": 0,
        "non_llm_estimated_ms": max(total_wall_ms, 0),
        "llm_calls": 0,
        "llm_cache_hits": 0,
        "llm_repairs": 0,
        "llm_parse_errors": 0,
        "slowest_nodes": [],
        "llm_by_node": {},
        "_node_spans": [],
    }

    try:
        trace_data = json.loads(trace_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        timing["trace_error"] = "trace file not found"
        return timing
    except Exception as exc:  # noqa: BLE001
        timing["trace_error"] = str(exc)
        return timing

    raw_llm_calls = trace_data.get("llm_calls") or []
    llm_calls = [call for call in raw_llm_calls if isinstance(call, dict)]
    raw_node_spans = trace_data.get("node_spans") or []
    node_spans = [span for span in raw_node_spans if isinstance(span, dict)]

    llm_total_ms = 0
    llm_cache_hits = 0
    llm_repairs = 0
    llm_parse_errors = 0
    llm_by_node: dict[str, dict[str, int]] = {}
    for call in llm_calls:
        node = _strip(call.get("node")) or "unknown"
        node_summary = llm_by_node.setdefault(
            node,
            {
                "calls": 0,
                "latency_ms": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_hits": 0,
                "repairs": 0,
                "parse_errors": 0,
            },
        )
        latency_ms = _coerce_int_ms(call.get("latency_ms"))
        node_summary["calls"] += 1
        node_summary["latency_ms"] += latency_ms
        node_summary["input_tokens"] += _coerce_int_ms(call.get("input_tokens"))
        node_summary["output_tokens"] += _coerce_int_ms(call.get("output_tokens"))
        if call.get("cache_hit"):
            node_summary["cache_hits"] += 1
            llm_cache_hits += 1
        if call.get("is_repair"):
            node_summary["repairs"] += 1
            llm_repairs += 1
        if call.get("parse_error"):
            node_summary["parse_errors"] += 1
            llm_parse_errors += 1
        llm_total_ms += latency_ms

    sorted_spans = sorted(
        (
            {
                "node": _strip(span.get("node")) or "unknown",
                "duration_ms": _coerce_int_ms(span.get("duration_ms")),
                "status": _strip(span.get("status")) or "ok",
                "error": _strip(span.get("error")) or None,
                "timestamp_start": span.get("timestamp_start"),
                "timestamp_end": span.get("timestamp_end"),
            }
            for span in node_spans
        ),
        key=lambda span: (-span["duration_ms"], span["node"]),
    )

    timing.update(
        {
            "trace_available": True,
            "node_total_ms": sum(span["duration_ms"] for span in sorted_spans),
            "llm_total_ms": llm_total_ms,
            "non_llm_estimated_ms": max(total_wall_ms - llm_total_ms, 0),
            "llm_calls": len(llm_calls),
            "llm_cache_hits": llm_cache_hits,
            "llm_repairs": llm_repairs,
            "llm_parse_errors": llm_parse_errors,
            "slowest_nodes": [
                {
                    "node": span["node"],
                    "duration_ms": span["duration_ms"],
                    "status": span["status"],
                }
                for span in sorted_spans[:3]
            ],
            "llm_by_node": llm_by_node,
            "_node_spans": sorted_spans,
        }
    )
    return timing


def _timing_without_private_fields(timing: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in timing.items()
        if not key.startswith("_") and key != "node_spans"
    }


def _public_result(result: dict[str, Any]) -> dict[str, Any]:
    public = dict(result)
    timing = public.get("timing")
    if isinstance(timing, dict):
        public["timing"] = _timing_without_private_fields(timing)
    return public


def _iter_outcome_map(outcome_map) -> list[tuple[str, str, str]]:
    if isinstance(outcome_map, dict):
        return [(trial, code, "unspecified") for trial, code in outcome_map.items()]
    pairs = []
    for item in outcome_map:
        if isinstance(item, dict):
            pairs.append(
                (item["trial"], item["outcome_code"], item.get("cohort", "unspecified"))
            )
        else:
            if len(item) == 2:
                trial, outcome_code = item
                cohort = "unspecified"
            else:
                trial, outcome_code, cohort = item
            pairs.append((trial, outcome_code, cohort))
    return pairs


def load_reference(csv_path: Path) -> dict[str, dict]:
    references: dict[str, dict] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            trial = _strip(row.get("Trial"))
            if not trial:
                continue
            references[trial] = {
                "D1": _strip(row.get("D1")),
                "D2": _strip(row.get("D2")),
                "D3": _strip(row.get("D3")),
                "D4": _strip(row.get("D4")),
                "D5": _strip(row.get("D5")),
                "Overall Risk": _strip(row.get("Overall Risk")),
            }
    return references


def compare_judgments(pipeline: dict, reference: dict) -> dict[str, bool]:
    domain_judgments = pipeline.get("domain_judgments") or {}
    result: dict[str, bool] = {}
    for domain in DOMAINS:
        left = _normalize_judgment(domain_judgments.get(domain, ""))
        right = _normalize_judgment(reference.get(domain, ""))
        result[domain] = left.casefold() == right.casefold()

    overall_pipeline = _normalize_judgment(pipeline.get("overall_judgment", ""))
    overall_ref = _normalize_judgment(reference.get("Overall Risk", ""))
    result["Overall"] = overall_pipeline.casefold() == overall_ref.casefold()
    benchmark_overall = benchmark_reference_overall_judgment(domain_judgments)
    result["Benchmark Overall"] = benchmark_overall.casefold() == overall_ref.casefold()
    return result


def run_benchmark(
    pdf_dir,
    reference_csvs,
    outcome_map,
    output_dir,
    supplement_dir=None,
    use_supplements: bool = False,
    supplement_policy: str = "auto",
    **run_kwargs,
) -> list[dict]:
    pdf_dir_path = Path(pdf_dir)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    ingestion_cache: dict[tuple, dict[str, Any]] = {}

    normalized_refs: dict[str, dict[str, dict]] = {}
    for outcome_code, csv_path in reference_csvs.items():
        loaded = load_reference(Path(csv_path))
        normalized_refs[outcome_code.upper()] = {
            _normalize_trial(trial): {"trial": trial, "row": row}
            for trial, row in loaded.items()
        }

    results: list[dict] = []
    for trial_name, outcome_code, cohort in _iter_outcome_map(outcome_map):
        code = _strip(outcome_code).upper()
        outcome_label = OUTCOME_LABELS.get(code, "")
        trial_result: dict[str, Any] = {
            "id": f"{trial_name}:{code}",
            "trial": trial_name,
            "outcome_code": code,
            "outcome": outcome_label,
            "cohort": _strip(cohort) or "unspecified",
            "supplementary_paths": [],
            "supplements_found": 0,
            "supplement_policy": supplement_policy,
            "skipped": False,
            "error": None,
            "notes": "",
        }

        references_for_outcome = normalized_refs.get(code)
        if references_for_outcome is None:
            trial_result["skipped"] = True
            trial_result["notes"] = f"Unknown outcome code: {code}"
            LOGGER.warning(
                "Skipping trial %s: unknown outcome code '%s'", trial_name, code
            )
            results.append(trial_result)
            continue

        reference_row_entry = references_for_outcome.get(_normalize_trial(trial_name))
        if reference_row_entry is None:
            trial_result["skipped"] = True
            trial_result["notes"] = "Trial not in reference"
            LOGGER.warning(
                "Skipping trial %s: not present in %s reference", trial_name, code
            )
            results.append(trial_result)
            continue

        pdf_path = _find_pdf_for_trial(pdf_dir_path, trial_name)
        if pdf_path is None:
            trial_result["skipped"] = True
            trial_result["notes"] = f"PDF not found in {pdf_dir_path}"
            LOGGER.warning(
                "Skipping trial %s: PDF not found in %s", trial_name, pdf_dir_path
            )
            results.append(trial_result)
            continue

        trial_result["pdf_path"] = str(pdf_path)
        trial_result["reference"] = reference_row_entry["row"]
        supplement_paths: list[Path] = []
        if (
            use_supplements
            and supplement_policy != "none"
            and supplement_dir is not None
        ):
            supplement_paths = find_supplements_for_trial(
                Path(supplement_dir), trial_name
            )
        trial_result["supplementary_paths"] = [str(path) for path in supplement_paths]
        trial_result["supplements_found"] = len(supplement_paths)
        if use_supplements and supplement_policy == "required" and not supplement_paths:
            trial_result["error"] = (
                f"Required supplements not found in {supplement_dir}"
            )
            trial_result["notes"] = trial_result["error"]
            trial_result["comparison"] = {}
            trace_path = (
                output_dir_path
                / f"{pdf_path.stem}_{code.lower()}"
                / f"{pdf_path.stem}_trace.json"
            )
            trial_result["timing"] = _summarize_trace_timing(
                trace_path,
                0,
            )
            trial_result["timing"]["trace_error"] = "assessment not run"
            results.append(trial_result)
            continue

        assessment_output_dir = output_dir_path / f"{pdf_path.stem}_{code.lower()}"
        cache_key = _trial_artifact_cache_key(pdf_path, supplement_paths)
        cached_artifacts = ingestion_cache.get(cache_key, {})
        start_wall = time.perf_counter()
        run_error: Exception | None = None
        try:
            state = run_assessment(
                pdf_path=str(pdf_path),
                outcome=outcome_label,
                output_dir=str(assessment_output_dir),
                supplementary_paths=[str(path) for path in supplement_paths],
                precomputed_ingestion=cached_artifacts.get("ingestion"),
                trial_retrieval_indexes=cached_artifacts.get("retrieval_indexes"),
                **run_kwargs,
            )
            if state is not None and cache_key not in ingestion_cache:
                ingestion_cache[cache_key] = {
                    "ingestion": _state_ingestion_artifact(state),
                    "retrieval_indexes": state.get("trial_retrieval_indexes") or {},
                }
        except Exception as exc:  # noqa: BLE001
            run_error = exc
            trial_result["error"] = str(exc)
            trial_result["notes"] = str(exc)
            trial_result["comparison"] = {}
            LOGGER.exception("run_assessment failed for trial %s", trial_name)
        finally:
            total_wall_ms = int((time.perf_counter() - start_wall) * 1000)
            trace_path = assessment_output_dir / f"{pdf_path.stem}_trace.json"
            trial_result["timing"] = _summarize_trace_timing(trace_path, total_wall_ms)

        if run_error is not None:
            results.append(trial_result)
            continue

        try:
            output_json = assessment_output_dir / f"{pdf_path.stem}_rob2_data.json"
            pipeline_output = json.loads(output_json.read_text(encoding="utf-8"))
            if supplement_policy == "required":
                failures = _required_supplement_failures(
                    supplement_paths,
                    list(pipeline_output.get("source_documents") or []),
                )
                if failures:
                    raise RuntimeError(
                        "Required supplement ingestion failed: " + ", ".join(failures)
                    )
            trial_result["pipeline"] = {
                "domain_judgments": pipeline_output.get("domain_judgments") or {},
                "overall_judgment": pipeline_output.get("overall_judgment"),
                "benchmark_reference_overall_judgment": benchmark_reference_overall_judgment(
                    pipeline_output.get("domain_judgments") or {}
                ),
            }
            trial_result["comparison"] = compare_judgments(
                trial_result["pipeline"], reference_row_entry["row"]
            )
        except Exception as exc:  # noqa: BLE001
            trial_result["error"] = str(exc)
            trial_result["notes"] = str(exc)
            trial_result["comparison"] = {}
            LOGGER.exception("run_assessment failed for trial %s", trial_name)

        results.append(trial_result)

    return results


def _empty_confusion() -> dict[str, dict[str, int]]:
    return {row: {col: 0 for col in JUDGMENT_ORDER} for row in JUDGMENT_ORDER}


def _summarize_results_subset(results) -> dict:
    fields = list(AGREEMENT_FIELDS)
    counts = {field: {"matches": 0, "total": 0} for field in fields}
    confusion = {field: _empty_confusion() for field in fields}

    evaluated_trials = 0
    for result in results:
        if result.get("error") or result.get("skipped"):
            continue
        comparison = result.get("comparison") or {}
        reference = result.get("reference") or {}
        pipeline = result.get("pipeline") or {}
        domain_judgments = pipeline.get("domain_judgments") or {}
        evaluated_trials += 1

        for domain in DOMAINS:
            if domain in comparison:
                counts[domain]["total"] += 1
                counts[domain]["matches"] += 1 if comparison[domain] else 0
            ref_value = _normalize_judgment(reference.get(domain, ""))
            pred_value = _normalize_judgment(domain_judgments.get(domain, ""))
            if ref_value in JUDGMENT_ORDER and pred_value in JUDGMENT_ORDER:
                confusion[domain][ref_value][pred_value] += 1

        if "Overall" in comparison:
            counts["Overall"]["total"] += 1
            counts["Overall"]["matches"] += 1 if comparison["Overall"] else 0
        overall_ref = _normalize_judgment(reference.get("Overall Risk", ""))
        overall_pred = _normalize_judgment(pipeline.get("overall_judgment", ""))
        if overall_ref in JUDGMENT_ORDER and overall_pred in JUDGMENT_ORDER:
            confusion["Overall"][overall_ref][overall_pred] += 1

        benchmark_overall_pred = _normalize_judgment(
            pipeline.get("benchmark_reference_overall_judgment", "")
            or benchmark_reference_overall_judgment(domain_judgments)
        )
        benchmark_overall_matches = (
            benchmark_overall_pred.casefold() == overall_ref.casefold()
            if benchmark_overall_pred and overall_ref
            else comparison.get("Benchmark Overall")
        )
        if benchmark_overall_matches is not None:
            counts["Benchmark Overall"]["total"] += 1
            counts["Benchmark Overall"]["matches"] += (
                1 if benchmark_overall_matches else 0
            )
        if (
            overall_ref in JUDGMENT_ORDER
            and benchmark_overall_pred in JUDGMENT_ORDER
        ):
            confusion["Benchmark Overall"][overall_ref][benchmark_overall_pred] += 1

    rates = {}
    for field, field_counts in counts.items():
        total = field_counts["total"]
        rates[field] = (field_counts["matches"] / total) if total else 0.0

    return {
        "evaluated_trials": evaluated_trials,
        "agreement_counts": counts,
        "agreement_rates": rates,
        "confusion_matrices": confusion,
        "judgment_order": list(JUDGMENT_ORDER),
    }


def _summarize_timing_results(results) -> dict[str, Any]:
    timed_results = [
        result for result in results if isinstance(result.get("timing"), dict)
    ]
    if not timed_results:
        return {
            "evaluated_runs": 0,
            "total_wall_ms": 0,
            "mean_wall_ms": 0,
            "median_wall_ms": 0,
            "total_node_duration_ms": 0,
            "total_llm_latency_ms": 0,
            "total_llm_calls": 0,
            "total_llm_cache_hits": 0,
            "total_llm_repairs": 0,
            "total_llm_parse_errors": 0,
            "total_non_llm_estimated_ms": 0,
            "slowest_runs": [],
            "node_aggregates": {},
        }

    wall_times = []
    node_aggregate_totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "calls": 0,
            "total_duration_ms": 0,
            "max_duration_ms": 0,
            "error_count": 0,
        }
    )
    slowest_runs = []
    total_wall_ms = 0
    total_node_duration_ms = 0
    total_llm_latency_ms = 0
    total_llm_calls = 0
    total_llm_cache_hits = 0
    total_llm_repairs = 0
    total_llm_parse_errors = 0
    total_non_llm_estimated_ms = 0

    for result in timed_results:
        timing = result.get("timing") or {}
        wall_ms = _coerce_int_ms(timing.get("total_wall_ms"))
        llm_ms = _coerce_int_ms(timing.get("llm_total_ms"))
        node_total_ms = _coerce_int_ms(timing.get("node_total_ms"))
        non_llm_ms = _coerce_int_ms(timing.get("non_llm_estimated_ms"))
        wall_times.append(wall_ms)
        total_wall_ms += wall_ms
        total_node_duration_ms += node_total_ms
        total_llm_latency_ms += llm_ms
        total_llm_calls += _coerce_int_ms(timing.get("llm_calls"))
        total_llm_cache_hits += _coerce_int_ms(timing.get("llm_cache_hits"))
        total_llm_repairs += _coerce_int_ms(timing.get("llm_repairs"))
        total_llm_parse_errors += _coerce_int_ms(timing.get("llm_parse_errors"))
        total_non_llm_estimated_ms += non_llm_ms

        for span in timing.get("node_spans") or timing.get("_node_spans") or []:
            if not isinstance(span, dict):
                continue
            node = _strip(span.get("node")) or "unknown"
            duration_ms = _coerce_int_ms(span.get("duration_ms"))
            node_summary = node_aggregate_totals[node]
            node_summary["calls"] += 1
            node_summary["total_duration_ms"] += duration_ms
            node_summary["max_duration_ms"] = max(
                node_summary["max_duration_ms"], duration_ms
            )
            if _strip(span.get("status")).casefold() == "error":
                node_summary["error_count"] += 1

        slowest_nodes = timing.get("slowest_nodes") or []
        slowest_node = ""
        slowest_node_duration_ms = 0
        if slowest_nodes:
            first_slowest_node = slowest_nodes[0]
            if isinstance(first_slowest_node, dict):
                slowest_node = _strip(first_slowest_node.get("node"))
                slowest_node_duration_ms = _coerce_int_ms(
                    first_slowest_node.get("duration_ms")
                )
        slowest_runs.append(
            {
                "trial": _strip(result.get("trial")) or _strip(result.get("id")),
                "outcome": _strip(result.get("outcome"))
                or _strip(result.get("outcome_code")),
                "cohort": _strip(result.get("cohort")) or "unspecified",
                "total_wall_ms": wall_ms,
                "llm_total_ms": llm_ms,
                "non_llm_estimated_ms": non_llm_ms,
                "llm_calls": _coerce_int_ms(timing.get("llm_calls")),
                "llm_cache_hits": _coerce_int_ms(timing.get("llm_cache_hits")),
                "slowest_node": slowest_node,
                "slowest_node_duration_ms": slowest_node_duration_ms,
            }
        )

    slowest_runs.sort(key=lambda item: (-item["total_wall_ms"], item["trial"]))
    ordered_node_aggregates = {
        node: {
            "calls": data["calls"],
            "total_duration_ms": data["total_duration_ms"],
            "mean_duration_ms": int(round(data["total_duration_ms"] / data["calls"]))
            if data["calls"]
            else 0,
            "max_duration_ms": data["max_duration_ms"],
            "error_count": data["error_count"],
        }
        for node, data in sorted(
            node_aggregate_totals.items(),
            key=lambda item: (-item[1]["total_duration_ms"], item[0]),
        )
    }

    return {
        "evaluated_runs": len(timed_results),
        "total_wall_ms": total_wall_ms,
        "mean_wall_ms": int(round(mean(wall_times))) if wall_times else 0,
        "median_wall_ms": int(round(median(wall_times))) if wall_times else 0,
        "total_node_duration_ms": total_node_duration_ms,
        "total_llm_latency_ms": total_llm_latency_ms,
        "total_llm_calls": total_llm_calls,
        "total_llm_cache_hits": total_llm_cache_hits,
        "total_llm_repairs": total_llm_repairs,
        "total_llm_parse_errors": total_llm_parse_errors,
        "total_non_llm_estimated_ms": total_non_llm_estimated_ms,
        "slowest_runs": slowest_runs[:5],
        "node_aggregates": ordered_node_aggregates,
    }


def summarize_benchmark(results) -> dict:
    summary = _summarize_results_subset(results)
    cohorts: dict[str, list[dict]] = {}
    for result in results:
        cohort = _strip(result.get("cohort")) or "unspecified"
        cohorts.setdefault(cohort, []).append(result)
    summary["cohorts"] = {
        cohort: _summarize_results_subset(items)
        for cohort, items in sorted(cohorts.items())
    }
    summary["timing"] = _summarize_timing_results(results)
    return summary


def _results_by_id(results: list[dict]) -> dict[str, dict]:
    return {
        _strip(result.get("id")) or f"{_strip(result.get('trial'))}:{_strip(result.get('outcome_code'))}": result
        for result in results
    }


def compare_benchmark_runs(before: dict, after: dict) -> dict[str, Any]:
    before_summary = before.get("summary") or {}
    after_summary = after.get("summary") or {}
    fields = sorted(
        set((before_summary.get("agreement_rates") or {}))
        | set((after_summary.get("agreement_rates") or {}))
    )
    field_deltas = {}
    for field in fields:
        before_counts = (before_summary.get("agreement_counts") or {}).get(field, {})
        after_counts = (after_summary.get("agreement_counts") or {}).get(field, {})
        before_rate = float((before_summary.get("agreement_rates") or {}).get(field, 0))
        after_rate = float((after_summary.get("agreement_rates") or {}).get(field, 0))
        field_deltas[field] = {
            "before_rate": before_rate,
            "after_rate": after_rate,
            "rate_delta": after_rate - before_rate,
            "before_matches": int(before_counts.get("matches", 0)),
            "after_matches": int(after_counts.get("matches", 0)),
            "match_delta": int(after_counts.get("matches", 0))
            - int(before_counts.get("matches", 0)),
            "before_total": int(before_counts.get("total", 0)),
            "after_total": int(after_counts.get("total", 0)),
        }

    before_results = _results_by_id(before.get("results") or [])
    after_results = _results_by_id(after.get("results") or [])
    case_deltas = []
    for result_id in sorted(set(before_results) & set(after_results)):
        before_comparison = before_results[result_id].get("comparison") or {}
        after_comparison = after_results[result_id].get("comparison") or {}
        for field in AGREEMENT_FIELDS:
            if field not in before_comparison or field not in after_comparison:
                continue
            before_value = before_comparison[field]
            after_value = after_comparison[field]
            if before_value == after_value:
                continue
            if before_value is False and after_value is True:
                direction = "improved"
            elif before_value is True and after_value is False:
                direction = "harmed"
            else:
                direction = "changed"
            case_deltas.append(
                {
                    "id": result_id,
                    "field": field,
                    "before": before_value,
                    "after": after_value,
                    "direction": direction,
                }
            )

    return {"field_deltas": field_deltas, "case_deltas": case_deltas}


def write_benchmark_report(results, summary, output_path):
    output_path = Path(output_path)
    report_path = output_path.parent / "benchmark_report.md"
    json_path = output_path.parent / "benchmark_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(
        json.dumps(
            {
                "results": [_public_result(result) for result in results],
                "summary": summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    fields = list(AGREEMENT_FIELDS)
    has_meaningful_cohort = any(
        (_strip(result.get("cohort")) or "unspecified") != "unspecified"
        for result in results
    )
    lines = [
        "# Benchmark Report",
        "",
        f"- Trials evaluated: {summary.get('evaluated_trials', 0)}",
        "",
        "## Summary Agreement",
        "",
        "| Field | Agreement |",
        "| --- | ---: |",
    ]
    for field in fields:
        counts = summary.get("agreement_counts", {}).get(
            field, {"matches": 0, "total": 0}
        )
        rate = summary.get("agreement_rates", {}).get(field, 0.0) * 100
        lines.append(
            f"| {field} | {rate:.1f}% ({counts['matches']}/{counts['total']}) |"
        )

    if has_meaningful_cohort and summary.get("cohorts"):
        lines.extend(
            [
                "",
                "## Cohort Agreement",
                "",
                "| Cohort | Field | Agreement |",
                "| --- | --- | ---: |",
            ]
        )
        for cohort, cohort_summary in summary["cohorts"].items():
            for field in fields:
                counts = cohort_summary.get("agreement_counts", {}).get(
                    field, {"matches": 0, "total": 0}
                )
                rate = cohort_summary.get("agreement_rates", {}).get(field, 0.0) * 100
                lines.append(
                    f"| {cohort} | {field} | {rate:.1f}% ({counts['matches']}/{counts['total']}) |"
                )

    timing = summary.get("timing") or {}
    if timing:
        lines.extend(
            [
                "",
                "## Timing Summary",
                "",
                f"- Evaluated runs: {timing.get('evaluated_runs', 0)}",
                f"- Total wall time: {_format_seconds(timing.get('total_wall_ms', 0))}",
                f"- Mean wall time per run: {_format_seconds(timing.get('mean_wall_ms', 0))}",
                f"- Median wall time per run: {_format_seconds(timing.get('median_wall_ms', 0))}",
                f"- Total LLM latency: {_format_seconds(timing.get('total_llm_latency_ms', 0))}",
                f"- Total LLM calls: {timing.get('total_llm_calls', 0)}",
                f"- Total cache hits: {timing.get('total_llm_cache_hits', 0)}",
                "",
                "### Slowest Runs",
                "",
                "| Trial | Outcome | Wall Time | LLM Time | Estimated Non-LLM | LLM Calls | Cache Hits | Slowest Node |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for run in timing.get("slowest_runs") or []:
            slowest_node = _strip(run.get("slowest_node")) or "-"
            if slowest_node != "-":
                slowest_node = f"{slowest_node} ({_format_seconds(run.get('slowest_node_duration_ms', 0))})"
            lines.append(
                "| "
                + " | ".join(
                    [
                        _strip(run.get("trial")) or "-",
                        _strip(run.get("outcome")) or "-",
                        _format_seconds(run.get("total_wall_ms", 0)),
                        _format_seconds(run.get("llm_total_ms", 0)),
                        _format_seconds(run.get("non_llm_estimated_ms", 0)),
                        str(run.get("llm_calls", 0)),
                        str(run.get("llm_cache_hits", 0)),
                        slowest_node,
                    ]
                )
                + " |"
            )

        lines.extend(
            [
                "",
                "### Node Timing",
                "",
                "| Node | Calls | Total Time | Mean Time | Max Time | Errors |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for node, node_summary in (timing.get("node_aggregates") or {}).items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        node,
                        str(node_summary.get("calls", 0)),
                        _format_seconds(node_summary.get("total_duration_ms", 0)),
                        _format_seconds(node_summary.get("mean_duration_ms", 0)),
                        _format_seconds(node_summary.get("max_duration_ms", 0)),
                        str(node_summary.get("error_count", 0)),
                    ]
                )
                + " |"
            )

    lines.extend(["", "## Per-Trial Details", ""])
    if has_meaningful_cohort:
        lines.extend(
            [
                "| Trial | Outcome | Cohort | D1 | D2 | D3 | D4 | D5 | Official Overall | Benchmark Overall | Notes |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
    else:
        lines.extend(
            [
                "| Trial | Outcome | D1 | D2 | D3 | D4 | D5 | Official Overall | Benchmark Overall | Notes |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
    for result in results:
        comparison = result.get("comparison") or {}

        def mark(field: str) -> str:
            value = comparison.get(field)
            if value is True:
                return "Y"
            if value is False:
                return "N"
            return "-"

        notes = _strip(result.get("notes"))
        if not notes and result.get("error"):
            notes = _strip(result.get("error"))
        notes = notes[:80]

        row = [
            _strip(result.get("id")) or _strip(result.get("trial")),
            _strip(result.get("outcome")) or _strip(result.get("outcome_code")),
        ]
        if has_meaningful_cohort:
            row.append(_strip(result.get("cohort")) or "unspecified")
        row.extend(
            [
                mark("D1"),
                mark("D2"),
                mark("D3"),
                mark("D4"),
                mark("D5"),
                mark("Overall"),
                mark("Benchmark Overall"),
                notes or "-",
            ]
        )

        lines.append("| " + " | ".join(row) + " |")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_benchmark_delta_report(delta: dict, output_path: Path) -> None:
    lines = [
        "# Benchmark Delta Report",
        "",
        "## Agreement Deltas",
        "",
        "| Field | Before | After | Delta | Match Delta |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for field, item in delta.get("field_deltas", {}).items():
        lines.append(
            f"| {field} | {item['before_rate'] * 100:.1f}% | "
            f"{item['after_rate'] * 100:.1f}% | {item['rate_delta'] * 100:+.1f} pp | "
            f"{item['match_delta']:+d} |"
        )

    lines.extend(
        [
            "",
            "## Case Changes",
            "",
            "| Case | Field | Before | After | Direction |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for item in delta.get("case_deltas", []):
        lines.append(
            f"| {item['id']} | {item['field']} | {item['before']} | "
            f"{item['after']} | {item['direction']} |"
        )
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
