import csv
import hashlib
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from rob2_pipeline.ingestion.assessment import AssessmentIngestionResult
from rob2_pipeline.judges.overall import judge_overall
from rob2_pipeline.pipeline import run_assessment


LOGGER = logging.getLogger(__name__)
DOMAINS = ("D1", "D2", "D3", "D4", "D5")
REFERENCE_FIELDS = {"Trial", *DOMAINS, "Overall Risk"}
ADJUDICATION_NODE_PREFIX = "sq_support_adjudication"
OUTCOME_LABELS = {
    "OS": "Overall Survival",
    "PFS": "Progression-Free Survival",
    "AE": "Adverse Events",
}
JUDGMENT_ORDER = ("Low", "Some concerns", "High")
MISMATCH_CATEGORIES = (
    "parse",
    "retrieval",
    "packet",
    "SQ",
    "judge",
    "reference_ambiguity",
    "blocked_incomplete",
)


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


def _normalize_sq_answer(value: Any) -> str:
    raw = _strip(value).upper()
    return raw if raw in {"Y", "PY", "PN", "N", "NI", "NA"} else raw


def _sq_id_from_reference_field(field: object) -> str:
    text = _strip(field)
    if not text:
        return ""
    normalized = text.replace("_", ".")
    match = re.fullmatch(r"(?i)(?:SQ[\s.]*)?(\d+(?:\.\d+)*)", normalized)
    return match.group(1) if match else ""


def _reference_sq_answers(row: dict[str, Any]) -> dict[str, str]:
    answers: dict[str, str] = {}
    for field, value in row.items():
        if field in REFERENCE_FIELDS:
            continue
        sq_id = _sq_id_from_reference_field(field)
        answer = _normalize_sq_answer(value)
        if sq_id and answer:
            answers[sq_id] = answer
    return answers


def _pipeline_sq_answer(sq_answers: object, sq_id: str) -> str:
    if not isinstance(sq_answers, dict):
        return ""
    raw = sq_answers.get(sq_id)
    if isinstance(raw, dict):
        return _normalize_sq_answer(raw.get("answer"))
    return _normalize_sq_answer(raw)


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


def _coerce_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_seconds(value_ms: object) -> str:
    return f"{_coerce_int_ms(value_ms) / 1000:.1f}s"


def _is_adjudication_node(node: object) -> bool:
    return _strip(node).casefold().startswith(ADJUDICATION_NODE_PREFIX)


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
        "llm_input_tokens": 0,
        "llm_output_tokens": 0,
        "llm_cost_usd": 0.0,
        "adjudication_llm_calls": 0,
        "adjudication_llm_total_ms": 0,
        "adjudication_llm_input_tokens": 0,
        "adjudication_llm_output_tokens": 0,
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
    llm_input_tokens = 0
    llm_output_tokens = 0
    llm_cost_usd = 0.0
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
        llm_input_tokens += _coerce_int_ms(call.get("input_tokens"))
        llm_output_tokens += _coerce_int_ms(call.get("output_tokens"))
        llm_cost_usd += _coerce_float(call.get("cost_usd"))
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

    adjudication_llm_calls = [
        call for call in llm_calls if _is_adjudication_node(call.get("node"))
    ]
    adjudication_llm_total_ms = sum(
        _coerce_int_ms(call.get("latency_ms")) for call in adjudication_llm_calls
    )
    adjudication_llm_input_tokens = sum(
        _coerce_int_ms(call.get("input_tokens")) for call in adjudication_llm_calls
    )
    adjudication_llm_output_tokens = sum(
        _coerce_int_ms(call.get("output_tokens")) for call in adjudication_llm_calls
    )

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
            "llm_input_tokens": llm_input_tokens,
            "llm_output_tokens": llm_output_tokens,
            "llm_cost_usd": llm_cost_usd,
            "adjudication_llm_calls": len(adjudication_llm_calls),
            "adjudication_llm_total_ms": adjudication_llm_total_ms,
            "adjudication_llm_input_tokens": adjudication_llm_input_tokens,
            "adjudication_llm_output_tokens": adjudication_llm_output_tokens,
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_file_reference(path: object) -> dict[str, Any]:
    path_text = _strip(path)
    reference: dict[str, Any] = {
        "path": path_text,
        "exists": False,
        "sha256": "",
        "size_bytes": 0,
    }
    if not path_text:
        return reference
    candidate = Path(path_text)
    try:
        if candidate.exists() and candidate.is_file():
            reference.update(
                {
                    "path": str(candidate.resolve()),
                    "exists": True,
                    "sha256": _sha256_file(candidate),
                    "size_bytes": candidate.stat().st_size,
                }
            )
    except OSError as exc:
        reference["error"] = str(exc)
    return reference


def _hash_directory_reference(path: Path) -> dict[str, Any]:
    reference: dict[str, Any] = {
        "path": str(path.resolve()),
        "exists": path.exists(),
        "sha256": "",
        "file_count": 0,
    }
    if not path.exists() or not path.is_dir():
        return reference

    digest = hashlib.sha256()
    file_count = 0
    for candidate in sorted(item for item in path.rglob("*") if item.is_file()):
        try:
            relative = candidate.relative_to(path).as_posix()
            file_digest = _sha256_file(candidate)
        except OSError:
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\0")
        file_count += 1
    reference["sha256"] = digest.hexdigest()
    reference["file_count"] = file_count
    return reference


def _assessment_artifact_paths(result: dict[str, Any]) -> dict[str, str]:
    artifacts = {
        key: _strip(value)
        for key, value in (result.get("assessment_artifacts") or {}).items()
        if _strip(value)
    }
    assessment_output_dir = _strip(result.get("assessment_output_dir"))
    trial = _strip(result.get("trial"))
    if assessment_output_dir and trial:
        defaults = {
            "rob2_data_json": f"{trial}_rob2_data.json",
            "trace_json": f"{trial}_trace.json",
            "report_markdown": f"{trial}_rob2_report.md",
        }
        for key, filename in defaults.items():
            artifacts.setdefault(key, str(Path(assessment_output_dir) / filename))
    return artifacts


def _artifact_manifest(
    results: list[dict[str, Any]], workspace_path: Path
) -> dict[str, Any]:
    assessments = []
    for result in results:
        artifact_paths = _assessment_artifact_paths(result)
        artifacts = {
            key: _hash_file_reference(path)
            for key, path in sorted(artifact_paths.items())
        }
        assessments.append(
            {
                "id": _strip(result.get("id")),
                "trial": _strip(result.get("trial")),
                "outcome_code": _strip(result.get("outcome_code")),
                "primary_pdf": _hash_file_reference(result.get("pdf_path")),
                "supplements": [
                    _hash_file_reference(path)
                    for path in result.get("supplementary_paths") or []
                ],
                "assessment_output_dir": _strip(result.get("assessment_output_dir")),
                "artifacts": artifacts,
            }
        )
    return {
        "workspace": _hash_directory_reference(workspace_path),
        "assessments": assessments,
    }


def _parser_metrics_from_timing(
    timing: dict[str, Any], schema_failures: list[dict[str, Any]]
) -> dict[str, int]:
    return {
        "llm_repairs": _coerce_int_ms(timing.get("llm_repairs")),
        "llm_parse_errors": _coerce_int_ms(timing.get("llm_parse_errors")),
        "schema_validation_failures": len(schema_failures),
    }


def _packet_statuses(packet_quality: object) -> dict[str, dict[str, Any]]:
    if not isinstance(packet_quality, dict):
        return {}
    statuses: dict[str, dict[str, Any]] = {}
    for key, packet in sorted(packet_quality.items()):
        if not isinstance(packet, dict):
            continue
        readiness = packet.get("packet_readiness") or {}
        status = ""
        if isinstance(readiness, dict):
            status = _strip(readiness.get("status"))
        status = status or _strip(packet.get("status")) or _strip(packet.get("readiness"))
        raw_grade = packet.get("grade")
        packet_grade = packet.get("packet_grade")
        if not raw_grade and isinstance(packet_grade, dict):
            raw_grade = packet_grade.get("grade")
        elif not raw_grade:
            raw_grade = packet_grade
        grade = _strip(raw_grade)
        statuses[_strip(key)] = {
            "status": status or "unknown",
            "grade": grade or "unknown",
        }
    return statuses


def _quote_traceability_diagnostics(support_constraints: object) -> dict[str, Any]:
    traceability_types = {
        "quote_untraceable",
        "semantic_support_conflict",
        "missing_required_evidence",
        "wrong_outcome_context",
    }
    counts = {
        "quote_untraceable": 0,
        "semantic_support_conflict": 0,
    }
    failures = []
    if not isinstance(support_constraints, list):
        return {**counts, "failures": failures}
    for constraint in support_constraints:
        if not isinstance(constraint, dict):
            continue
        constraint_type = _strip(constraint.get("constraint_type"))
        if constraint_type not in traceability_types:
            continue
        if constraint_type in counts:
            counts[constraint_type] += 1
        failures.append(
            {
                "constraint_type": constraint_type,
                "sq_id": _strip(constraint.get("sq_id")),
                "domain": _strip(constraint.get("domain")).upper(),
                "reason": _strip(constraint.get("reason")),
            }
        )
    return {**counts, "failures": failures}


def _llm_usage_from_timing(timing: dict[str, Any]) -> dict[str, int]:
    return {
        "input_tokens": _coerce_int_ms(timing.get("llm_input_tokens")),
        "output_tokens": _coerce_int_ms(timing.get("llm_output_tokens")),
    }


def _cost_metadata_from_timing(timing: dict[str, Any]) -> dict[str, int | float | None]:
    usage = _llm_usage_from_timing(timing)
    cost = _coerce_float(timing.get("llm_cost_usd"))
    return {
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "estimated_cost_usd": cost if cost else None,
    }


def _benchmark_assessment_record(result: dict[str, Any]) -> dict[str, Any]:
    timing = _timing_without_private_fields(result.get("timing") or {})
    schema_failures = result.get("schema_failures") or []
    return {
        "id": _strip(result.get("id")),
        "trial": _strip(result.get("trial")),
        "outcome_code": _strip(result.get("outcome_code")),
        "outcome": _strip(result.get("outcome")),
        "cohort": _strip(result.get("cohort")) or "unspecified",
        "status": {
            "skipped": bool(result.get("skipped")),
            "error": result.get("error"),
            "notes": _strip(result.get("notes")),
        },
        "agreement": {
            "reference": result.get("reference") or {},
            "pipeline": result.get("pipeline") or {},
            "comparison": result.get("comparison") or {},
            "audit_caught_mismatches": result.get("audit_caught_mismatches") or {},
        },
        "packet_quality": result.get("packet_quality") or {},
        "schema_failures": schema_failures,
        "artifacts": _assessment_artifact_paths(result),
        "diagnostics": {
            "timing": timing,
            "mismatch_classification": result.get("mismatch_classification") or {},
            "parser_metrics": _parser_metrics_from_timing(timing, schema_failures),
            "cache_reuse": {
                "llm_cache_hits": _coerce_int_ms(timing.get("llm_cache_hits")),
            },
            "packet_statuses": _packet_statuses(result.get("packet_quality")),
            "quote_traceability": _quote_traceability_diagnostics(
                result.get("support_constraints")
            ),
            "schema_validation_failures": schema_failures,
            "llm_latency": {
                "llm_calls": _coerce_int_ms(timing.get("llm_calls")),
                "llm_total_ms": _coerce_int_ms(timing.get("llm_total_ms")),
            },
            "llm_usage": _llm_usage_from_timing(timing),
            "cost_metadata": _cost_metadata_from_timing(timing),
        },
    }


def _summarize_engineering_diagnostics(results: list[dict[str, Any]]) -> dict[str, Any]:
    parser_metrics = {
        "llm_repairs": 0,
        "llm_parse_errors": 0,
        "schema_validation_failures": 0,
    }
    cache_reuse = {"llm_cache_hits": 0}
    packet_status_counts: dict[str, int] = {}
    packet_grade_counts: dict[str, int] = {}
    quote_traceability = {
        "quote_untraceable": 0,
        "semantic_support_conflict": 0,
        "failure_count": 0,
    }
    llm_usage = {"input_tokens": 0, "output_tokens": 0}
    total_cost = 0.0

    for result in results:
        timing = result.get("timing") or {}
        if not isinstance(timing, dict):
            timing = {}
        schema_failures = result.get("schema_failures") or []
        metrics = _parser_metrics_from_timing(timing, schema_failures)
        parser_metrics["llm_repairs"] += metrics["llm_repairs"]
        parser_metrics["llm_parse_errors"] += metrics["llm_parse_errors"]
        parser_metrics["schema_validation_failures"] += metrics[
            "schema_validation_failures"
        ]
        cache_reuse["llm_cache_hits"] += _coerce_int_ms(timing.get("llm_cache_hits"))

        for packet in _packet_statuses(result.get("packet_quality")).values():
            status = _strip(packet.get("status")) or "unknown"
            grade = _strip(packet.get("grade")) or "unknown"
            packet_status_counts[status] = packet_status_counts.get(status, 0) + 1
            packet_grade_counts[grade] = packet_grade_counts.get(grade, 0) + 1

        traceability = _quote_traceability_diagnostics(
            result.get("support_constraints")
        )
        quote_traceability["quote_untraceable"] += _coerce_int_ms(
            traceability.get("quote_untraceable")
        )
        quote_traceability["semantic_support_conflict"] += _coerce_int_ms(
            traceability.get("semantic_support_conflict")
        )
        quote_traceability["failure_count"] += len(
            traceability.get("failures") or []
        )

        usage = _llm_usage_from_timing(timing)
        llm_usage["input_tokens"] += usage["input_tokens"]
        llm_usage["output_tokens"] += usage["output_tokens"]
        total_cost += _coerce_float(timing.get("llm_cost_usd"))

    return {
        "classification": "engineering_only",
        "parser_metrics": parser_metrics,
        "cache_reuse": cache_reuse,
        "packet_statuses": {
            "by_status": dict(sorted(packet_status_counts.items())),
            "by_grade": dict(sorted(packet_grade_counts.items())),
        },
        "quote_traceability": quote_traceability,
        "llm_usage": llm_usage,
        "cost_metadata": {
            "estimated_cost_usd": total_cost if total_cost else None,
        },
    }


def _benchmark_schema_envelope(
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    workspace_path: Path,
) -> dict[str, Any]:
    return {
        "schema": {
            "schema_name": "auto_rob2_benchmark_result",
            "schema_version": 1,
            "sections": {
                "aggregate": "Agreement, confusion, audit, timing, adjudication, and engineering diagnostic summaries across evaluated assessments.",
                "assessments": "Per-assessment agreement, artifacts, packet quality, schema failures, and engineering diagnostics.",
                "artifact_manifest": "SHA-256 references for benchmark workspace and assessment artifacts.",
            },
            "diagnostics": {
                "classification": "engineering_only",
                "fields": [
                    "timing",
                    "mismatch_classification",
                    "parser_metrics",
                    "cache_reuse",
                    "packet_statuses",
                    "quote_traceability",
                    "schema_validation_failures",
                    "llm_latency",
                    "llm_usage",
                    "cost_metadata",
                ],
            },
        },
        "artifact_manifest": _artifact_manifest(results, workspace_path),
        "aggregate": summary,
        "assessments": [_benchmark_assessment_record(result) for result in results],
        "results": [_public_result(result) for result in results],
        "summary": summary,
    }


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
            reference_row = {
                "D1": _strip(row.get("D1")),
                "D2": _strip(row.get("D2")),
                "D3": _strip(row.get("D3")),
                "D4": _strip(row.get("D4")),
                "D5": _strip(row.get("D5")),
                "Overall Risk": _strip(row.get("Overall Risk")),
            }
            sq_answers = _reference_sq_answers(row)
            if sq_answers:
                reference_row["sq_answers"] = sq_answers
            references[trial] = reference_row
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
    reference_sq_answers = reference.get("sq_answers") or {}
    if isinstance(reference_sq_answers, dict) and reference_sq_answers:
        pipeline_sq_answers = pipeline.get("sq_answers") or {}
        result["SQ"] = {
            sq_id: _pipeline_sq_answer(pipeline_sq_answers, sq_id).casefold()
            == _normalize_sq_answer(reference_answer).casefold()
            for sq_id, reference_answer in sorted(reference_sq_answers.items())
        }
    return result


def _iter_domain_records(records: object):
    if not isinstance(records, dict):
        return
    for domain, items in records.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                yield _strip(domain), item


def _support_level(answer: object) -> str:
    if not isinstance(answer, dict):
        return ""
    return _strip(answer.get("support_level")).casefold()


def _derive_initial_domain_judgments(pipeline_output: dict[str, Any]) -> dict[str, str]:
    initial = dict(pipeline_output.get("domain_judgments") or {})
    for domain, attempt in _iter_domain_records(
        pipeline_output.get("sq_support_adjudications")
    ):
        if domain not in DOMAINS:
            continue
        impact = attempt.get("domain_impact") or {}
        original = _normalize_judgment(impact.get("original_domain_judgment"))
        if original:
            initial[domain] = original
    return initial


def _overall_from_domains(domain_judgments: dict[str, str]) -> str:
    if not domain_judgments:
        return ""
    overall, _rationale = judge_overall(domain_judgments)
    return overall


def _summarize_adjudication_metrics(
    pipeline_output: dict[str, Any],
    initial_domain_judgments: dict[str, str],
    final_domain_judgments: dict[str, str],
    initial_overall_judgment: str,
    final_overall_judgment: str,
) -> dict[str, Any]:
    pivotality_total = 0
    pivotality_pivotal = 0
    weak_sq_answers = 0
    unsupported_sq_answers = 0
    for _domain, test in _iter_domain_records(pipeline_output.get("pivotality_tests")):
        pivotality_total += 1
        if test.get("pivotal"):
            pivotality_pivotal += 1
        support_level = _strip(test.get("support_level")).casefold()
        if support_level == "weak":
            weak_sq_answers += 1
        elif support_level == "unsupported":
            unsupported_sq_answers += 1

    adjudication_total = 0
    changed_answer = 0
    changed_support = 0
    changed_answer_or_support = 0
    for _domain, attempt in _iter_domain_records(
        pipeline_output.get("sq_support_adjudications")
    ):
        adjudication_total += 1
        initial_answer = attempt.get("initial_answer") or {}
        adjudicated_answer = attempt.get("adjudicated_answer") or {}
        answer_changed = initial_answer.get("answer") != adjudicated_answer.get(
            "answer"
        )
        support_changed = _support_level(initial_answer) != _support_level(
            adjudicated_answer
        )
        changed_answer += 1 if answer_changed else 0
        changed_support += 1 if support_changed else 0
        changed_answer_or_support += 1 if answer_changed or support_changed else 0

    domain_deltas = {}
    for domain in DOMAINS:
        initial_value = _normalize_judgment(initial_domain_judgments.get(domain, ""))
        final_value = _normalize_judgment(final_domain_judgments.get(domain, ""))
        if initial_value and final_value and initial_value != final_value:
            domain_deltas[domain] = {"initial": initial_value, "final": final_value}

    overall_delta = None
    if (
        initial_overall_judgment
        and final_overall_judgment
        and initial_overall_judgment != final_overall_judgment
    ):
        overall_delta = {
            "initial": initial_overall_judgment,
            "final": final_overall_judgment,
        }

    return {
        "weak_sq_answers": weak_sq_answers,
        "unsupported_sq_answers": unsupported_sq_answers,
        "pivotality_tests": {
            "total": pivotality_total,
            "pivotal": pivotality_pivotal,
            "non_pivotal": pivotality_total - pivotality_pivotal,
        },
        "sq_support_adjudications": {
            "total": adjudication_total,
            "changed_answer": changed_answer,
            "changed_support": changed_support,
            "changed_answer_or_support": changed_answer_or_support,
        },
        "initial_final_deltas": {
            "domain_judgments": domain_deltas,
            "overall_judgment": overall_delta,
        },
    }


def _audit_limited_domains(pipeline_output: dict[str, Any]) -> set[str]:
    domains = set()
    for domain, test in _iter_domain_records(pipeline_output.get("pivotality_tests")):
        if domain in DOMAINS and test.get("acceptance_status") == "audit_limited":
            domains.add(domain)
    return domains


def _audit_caught_mismatches(
    comparison: dict[str, bool],
    audit_limited_domains: set[str],
    human_review_priority: object,
) -> dict[str, bool]:
    high_priority = _strip(human_review_priority).casefold() == "high"
    caught = {}
    for field in [*DOMAINS, "Overall"]:
        if comparison.get(field) is not False:
            continue
        audit_limited = (
            field in audit_limited_domains
            if field in DOMAINS
            else bool(audit_limited_domains)
        )
        caught[field] = high_priority or audit_limited
    return caught


def _domain_for_sq_id(sq_id: object) -> str:
    text = _strip(sq_id)
    if not text:
        return ""
    prefix = text.split(".", 1)[0]
    return f"D{prefix}" if prefix in {"1", "2", "3", "4", "5"} else ""


def _schema_failure_domains(schema_failures: object) -> set[str]:
    domains = set()
    if not isinstance(schema_failures, list):
        return domains
    for failure in schema_failures:
        if not isinstance(failure, dict):
            continue
        domain = _strip(failure.get("domain")).upper()
        node = _strip(failure.get("node")).casefold()
        if domain in DOMAINS:
            domains.add(domain)
            continue
        for candidate in DOMAINS:
            if candidate.casefold() in node:
                domains.add(candidate)
    return domains


def _packet_signals(packet: object) -> list[str]:
    if not isinstance(packet, dict):
        return []
    signals = []
    grade = _strip(packet.get("packet_grade") or packet.get("grade")).casefold()
    if grade in {"insufficient", "weak", "missing", "failed"}:
        signals.append(f"packet_grade:{grade}")
    if packet.get("missing_evidence"):
        signals.append("missing_evidence")
    if packet.get("negative_flags"):
        signals.append("negative_flags")
    confidence = _strip(packet.get("retrieval_confidence")).casefold()
    if confidence in {"low", "none"}:
        signals.append(f"retrieval_confidence:{confidence}")
    return signals


def _sq_signals_for_domain(pipeline: dict[str, Any], domain: str) -> list[str]:
    signals = []
    sq_answers = pipeline.get("sq_answers") or {}
    if not isinstance(sq_answers, dict):
        return signals
    for sq_id, answer in sq_answers.items():
        if _domain_for_sq_id(sq_id) != domain or not isinstance(answer, dict):
            continue
        support = _support_level(answer)
        answer_code = _normalize_sq_answer(answer.get("answer"))
        quote = _strip(answer.get("quote"))
        if support in {"weak", "unsupported"}:
            signals.append(f"support_level:{support}")
        if answer_code == "NI":
            signals.append("answer:NI")
        if not quote:
            signals.append("quote_missing")
        if answer.get("uncertain") or answer.get("uncertainty"):
            signals.append("uncertainty_flag")
    return signals


def _classify_domain_mismatch(
    result: dict[str, Any],
    field: str,
    schema_failure_domains: set[str],
) -> dict[str, Any] | None:
    if result.get("error") or result.get("skipped"):
        return {"category": "blocked_incomplete", "signals": ["assessment_error"]}

    comparison = result.get("comparison") or {}
    if comparison.get(field) is not False:
        return None

    audit_caught = result.get("audit_caught_mismatches") or {}
    if audit_caught.get(field):
        return {
            "category": "reference_ambiguity",
            "signals": ["audit_caught_mismatch"],
        }

    if field in schema_failure_domains:
        return {"category": "parse", "signals": ["schema_failure"]}

    packet_signals = _packet_signals((result.get("packet_quality") or {}).get(field))
    if packet_signals:
        category = (
            "retrieval"
            if any(
                signal.startswith("retrieval_confidence") for signal in packet_signals
            )
            else "packet"
        )
        return {"category": category, "signals": packet_signals}

    sq_signals = _sq_signals_for_domain(result.get("pipeline") or {}, field)
    if any(signal == "quote_missing" for signal in sq_signals):
        return {"category": "retrieval", "signals": sq_signals}
    if sq_signals:
        return {"category": "SQ", "signals": sq_signals}

    return {"category": "judge", "signals": ["judgment_label_mismatch"]}


def classify_mismatches(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    schema_failure_domains = _schema_failure_domains(result.get("schema_failures"))
    classifications: dict[str, dict[str, Any]] = {}
    for field in [*DOMAINS, "Overall"]:
        classification = _classify_domain_mismatch(
            result,
            field,
            schema_failure_domains,
        )
        if classification:
            classifications[field] = classification
    return classifications


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
            trial_result["mismatch_classification"] = classify_mismatches(trial_result)
            results.append(trial_result)
            continue

        assessment_output_dir = output_dir_path / f"{pdf_path.stem}_{code.lower()}"
        trial_result["assessment_output_dir"] = str(assessment_output_dir)
        trial_result["assessment_artifacts"] = {
            "rob2_data_json": str(
                assessment_output_dir / f"{pdf_path.stem}_rob2_data.json"
            ),
            "trace_json": str(assessment_output_dir / f"{pdf_path.stem}_trace.json"),
            "report_markdown": str(
                assessment_output_dir / f"{pdf_path.stem}_rob2_report.md"
            ),
        }
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
            trial_result["mismatch_classification"] = classify_mismatches(trial_result)
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
            final_domain_judgments = pipeline_output.get("domain_judgments") or {}
            final_overall_judgment = _normalize_judgment(
                pipeline_output.get("overall_judgment")
            )
            trial_result["packet_quality"] = pipeline_output.get("packet_grades") or {}
            trial_result["schema_failures"] = (
                pipeline_output.get("schema_failures")
                or pipeline_output.get("schema_validation_failures")
                or []
            )
            trial_result["support_constraints"] = (
                pipeline_output.get("support_constraints") or []
            )
            initial_domain_judgments = _derive_initial_domain_judgments(pipeline_output)
            initial_overall_judgment = _normalize_judgment(
                pipeline_output.get("initial_overall_judgment")
            ) or _overall_from_domains(initial_domain_judgments)
            trial_result["pipeline"] = {
                "domain_judgments": final_domain_judgments,
                "overall_judgment": final_overall_judgment,
                "sq_answers": pipeline_output.get("sq_answers") or {},
                "initial_domain_judgments": initial_domain_judgments,
                "initial_overall_judgment": initial_overall_judgment,
                "human_review_priority": pipeline_output.get("human_review_priority"),
            }
            trial_result["adjudication_metrics"] = _summarize_adjudication_metrics(
                pipeline_output,
                initial_domain_judgments,
                final_domain_judgments,
                initial_overall_judgment,
                final_overall_judgment,
            )
            trial_result["comparison"] = compare_judgments(
                trial_result["pipeline"], reference_row_entry["row"]
            )
            trial_result["audit_caught_mismatches"] = _audit_caught_mismatches(
                trial_result["comparison"],
                _audit_limited_domains(pipeline_output),
                pipeline_output.get("human_review_priority"),
            )
            trial_result["mismatch_classification"] = classify_mismatches(trial_result)
        except Exception as exc:  # noqa: BLE001
            trial_result["error"] = str(exc)
            trial_result["notes"] = str(exc)
            trial_result["comparison"] = {}
            trial_result["mismatch_classification"] = classify_mismatches(trial_result)
            LOGGER.exception("run_assessment failed for trial %s", trial_name)

        results.append(trial_result)

    return results


def _empty_confusion() -> dict[str, dict[str, int]]:
    return {row: {col: 0 for col in JUDGMENT_ORDER} for row in JUDGMENT_ORDER}


def _summarize_results_subset(results) -> dict:
    fields = [*DOMAINS, "Overall"]
    counts = {field: {"matches": 0, "total": 0} for field in fields}
    sq_counts: dict[str, dict[str, int]] = {}
    audit_caught = {field: {"caught": 0, "total": 0} for field in fields}
    mismatch_categories = {category: 0 for category in MISMATCH_CATEGORIES}
    confusion = {field: _empty_confusion() for field in fields}

    evaluated_trials = 0
    for result in results:
        for classification in (result.get("mismatch_classification") or {}).values():
            if not isinstance(classification, dict):
                continue
            category = _strip(classification.get("category"))
            if category:
                mismatch_categories.setdefault(category, 0)
                mismatch_categories[category] += 1

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
        sq_comparison = comparison.get("SQ") or {}
        if isinstance(sq_comparison, dict):
            for sq_id, matched in sq_comparison.items():
                sq_summary = sq_counts.setdefault(
                    _strip(sq_id), {"matches": 0, "total": 0}
                )
                sq_summary["total"] += 1
                sq_summary["matches"] += 1 if matched else 0
        overall_ref = _normalize_judgment(reference.get("Overall Risk", ""))
        overall_pred = _normalize_judgment(pipeline.get("overall_judgment", ""))
        if overall_ref in JUDGMENT_ORDER and overall_pred in JUDGMENT_ORDER:
            confusion["Overall"][overall_ref][overall_pred] += 1

        caught_mismatches = result.get("audit_caught_mismatches") or {}
        for field in fields:
            if comparison.get(field) is False:
                audit_caught[field]["total"] += 1
                if caught_mismatches.get(field):
                    audit_caught[field]["caught"] += 1

    rates = {}
    for field, field_counts in counts.items():
        total = field_counts["total"]
        rates[field] = (field_counts["matches"] / total) if total else 0.0
    sq_rates = {
        sq_id: (field_counts["matches"] / field_counts["total"])
        if field_counts["total"]
        else 0.0
        for sq_id, field_counts in sorted(sq_counts.items())
    }

    return {
        "evaluated_trials": evaluated_trials,
        "agreement_counts": counts,
        "agreement_rates": rates,
        "sq_agreement_counts": dict(sorted(sq_counts.items())),
        "sq_agreement_rates": sq_rates,
        "audit_caught_mismatches": audit_caught,
        "mismatch_classification": {"categories": mismatch_categories},
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
            "total_adjudication_llm_calls": 0,
            "total_adjudication_llm_latency_ms": 0,
            "total_adjudication_llm_input_tokens": 0,
            "total_adjudication_llm_output_tokens": 0,
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
    total_adjudication_llm_calls = 0
    total_adjudication_llm_latency_ms = 0
    total_adjudication_llm_input_tokens = 0
    total_adjudication_llm_output_tokens = 0
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
        total_adjudication_llm_calls += _coerce_int_ms(
            timing.get("adjudication_llm_calls")
        )
        total_adjudication_llm_latency_ms += _coerce_int_ms(
            timing.get("adjudication_llm_total_ms")
        )
        total_adjudication_llm_input_tokens += _coerce_int_ms(
            timing.get("adjudication_llm_input_tokens")
        )
        total_adjudication_llm_output_tokens += _coerce_int_ms(
            timing.get("adjudication_llm_output_tokens")
        )
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
        "total_adjudication_llm_calls": total_adjudication_llm_calls,
        "total_adjudication_llm_latency_ms": total_adjudication_llm_latency_ms,
        "total_adjudication_llm_input_tokens": total_adjudication_llm_input_tokens,
        "total_adjudication_llm_output_tokens": total_adjudication_llm_output_tokens,
        "total_non_llm_estimated_ms": total_non_llm_estimated_ms,
        "slowest_runs": slowest_runs[:5],
        "node_aggregates": ordered_node_aggregates,
    }


def _empty_adjudication_metrics() -> dict[str, Any]:
    return {
        "weak_sq_answers": 0,
        "unsupported_sq_answers": 0,
        "pivotality_tests": {"total": 0, "pivotal": 0, "non_pivotal": 0},
        "sq_support_adjudications": {
            "total": 0,
            "changed_answer": 0,
            "changed_support": 0,
            "changed_answer_or_support": 0,
        },
        "initial_final_deltas": {
            "domain_judgments": {},
            "overall_judgment": None,
        },
    }


def _summarize_adjudication_results(results) -> dict[str, Any]:
    summary = _empty_adjudication_metrics()
    for result in results:
        if result.get("error") or result.get("skipped"):
            continue
        metrics = result.get("adjudication_metrics") or {}
        summary["weak_sq_answers"] += _coerce_int_ms(metrics.get("weak_sq_answers"))
        summary["unsupported_sq_answers"] += _coerce_int_ms(
            metrics.get("unsupported_sq_answers")
        )

        for key in ("total", "pivotal", "non_pivotal"):
            summary["pivotality_tests"][key] += _coerce_int_ms(
                (metrics.get("pivotality_tests") or {}).get(key)
            )
        for key in (
            "total",
            "changed_answer",
            "changed_support",
            "changed_answer_or_support",
        ):
            summary["sq_support_adjudications"][key] += _coerce_int_ms(
                (metrics.get("sq_support_adjudications") or {}).get(key)
            )

        deltas = metrics.get("initial_final_deltas") or {}
        for domain, delta in (deltas.get("domain_judgments") or {}).items():
            domain_summary = summary["initial_final_deltas"][
                "domain_judgments"
            ].setdefault(domain, {})
            if isinstance(delta, dict):
                key = (
                    _normalize_judgment(delta.get("initial")),
                    _normalize_judgment(delta.get("final")),
                )
                domain_summary[f"{key[0]} -> {key[1]}"] = (
                    domain_summary.get(f"{key[0]} -> {key[1]}", 0) + 1
                )

        overall_delta = deltas.get("overall_judgment")
        if isinstance(overall_delta, dict):
            key = (
                _normalize_judgment(overall_delta.get("initial")),
                _normalize_judgment(overall_delta.get("final")),
            )
            overall_summary = summary["initial_final_deltas"].setdefault(
                "overall_judgment_counts", {}
            )
            overall_summary[f"{key[0]} -> {key[1]}"] = (
                overall_summary.get(f"{key[0]} -> {key[1]}", 0) + 1
            )
    return summary


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
    summary["adjudication_metrics"] = _summarize_adjudication_results(results)
    summary["diagnostics"] = _summarize_engineering_diagnostics(results)
    return summary


def write_benchmark_report(results, summary, output_path):
    output_path = Path(output_path)
    report_path = output_path.parent / "benchmark_report.md"
    json_path = output_path.parent / "benchmark_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(
        json.dumps(
            _benchmark_schema_envelope(results, summary, output_path.parent),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    fields = [*DOMAINS, "Overall"]
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

    sq_counts = summary.get("sq_agreement_counts") or {}
    if sq_counts:
        lines.extend(
            [
                "",
                "## SQ Agreement",
                "",
                "| SQ | Agreement |",
                "| --- | ---: |",
            ]
        )
        for sq_id, counts in sq_counts.items():
            rate = summary.get("sq_agreement_rates", {}).get(sq_id, 0.0) * 100
            lines.append(
                f"| {sq_id} | {rate:.1f}% ({counts['matches']}/{counts['total']}) |"
            )

    audit_caught = summary.get("audit_caught_mismatches") or {}
    if any(counts.get("total", 0) for counts in audit_caught.values()):
        lines.extend(
            [
                "",
                "## Audit-Caught Mismatches",
                "",
                "| Field | Audit-caught label mismatches |",
                "| --- | ---: |",
            ]
        )
        for field in fields:
            counts = audit_caught.get(field, {"caught": 0, "total": 0})
            total = counts.get("total", 0)
            caught = counts.get("caught", 0)
            rate = (caught / total * 100) if total else 0.0
            lines.append(f"| {field} | {rate:.1f}% ({caught}/{total}) |")

    mismatch_categories = (summary.get("mismatch_classification") or {}).get(
        "categories"
    ) or {}
    if any(mismatch_categories.values()):
        lines.extend(
            [
                "",
                "## Mismatch Classification",
                "",
                "| Category | Count |",
                "| --- | ---: |",
            ]
        )
        for category in MISMATCH_CATEGORIES:
            lines.append(f"| {category} | {mismatch_categories.get(category, 0)} |")

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

    adjudication = summary.get("adjudication_metrics") or {}
    if adjudication:
        pivotality = adjudication.get("pivotality_tests") or {}
        sq_adjudications = adjudication.get("sq_support_adjudications") or {}
        lines.extend(
            [
                "",
                "## Adjudication Summary",
                "",
                f"- Weak SQ answers: {adjudication.get('weak_sq_answers', 0)}",
                f"- Unsupported SQ answers: {adjudication.get('unsupported_sq_answers', 0)}",
                "- Pivotality tests: "
                f"{pivotality.get('total', 0)} total; "
                f"{pivotality.get('pivotal', 0)} pivotal; "
                f"{pivotality.get('non_pivotal', 0)} non-pivotal",
                "- SQ support adjudications: "
                f"{sq_adjudications.get('total', 0)} total; "
                f"{sq_adjudications.get('changed_answer', 0)} changed answer; "
                f"{sq_adjudications.get('changed_support', 0)} changed support",
            ]
        )
        if timing:
            lines.append(
                "- Adjudication LLM calls: "
                f"{timing.get('total_adjudication_llm_calls', 0)} "
                f"({_format_seconds(timing.get('total_adjudication_llm_latency_ms', 0))} latency; "
                f"{timing.get('total_adjudication_llm_input_tokens', 0)} input tokens; "
                f"{timing.get('total_adjudication_llm_output_tokens', 0)} output tokens)"
            )

        delta_counts = adjudication.get("initial_final_deltas", {}).get(
            "domain_judgments", {}
        )
        overall_counts = adjudication.get("initial_final_deltas", {}).get(
            "overall_judgment_counts", {}
        )
        if delta_counts or overall_counts:
            lines.extend(
                [
                    "",
                    "### Initial-vs-Final Deltas",
                    "",
                    "| Field | Initial | Final | Count |",
                    "| --- | --- | --- | ---: |",
                ]
            )
            for domain in DOMAINS:
                for delta, count in (delta_counts.get(domain) or {}).items():
                    initial, separator, final = delta.partition(" -> ")
                    if separator:
                        lines.append(f"| {domain} | {initial} | {final} | {count} |")
            for delta, count in overall_counts.items():
                initial, separator, final = delta.partition(" -> ")
                if separator:
                    lines.append(f"| Overall | {initial} | {final} | {count} |")

    lines.extend(["", "## Per-Trial Details", ""])
    if has_meaningful_cohort:
        lines.extend(
            [
                "| Trial | Outcome | Cohort | D1 | D2 | D3 | D4 | D5 | Overall | Notes |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
    else:
        lines.extend(
            [
                "| Trial | Outcome | D1 | D2 | D3 | D4 | D5 | Overall | Notes |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
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
                notes or "-",
            ]
        )

        lines.append("| " + " | ".join(row) + " |")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
