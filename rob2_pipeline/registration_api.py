"""Fetch trial registration data from ClinicalTrials.gov API v2."""

import hashlib
import json
import re
from datetime import date
from pathlib import Path
from typing import Optional

from rob2_pipeline.types import SourceDocument

CTGOV_BASE = "https://clinicaltrials.gov/api/v2/studies"
CACHE_DIR = Path(".rob2_cache/ctgov")
_CACHE_VERSION = "v2"

_FIELDS = ",".join(
    [
        "OutcomesModule",
        "IdentificationModule",
        "DesignModule",
        "DescriptionModule",
        "OversightModule",
        "SponsorCollaboratorsModule",
        "ParticipantFlowModule",
    ]
)


def _cache_path(nct_id: str) -> Path:
    return CACHE_DIR / f"{nct_id}_{_CACHE_VERSION}.json"


def fetch_registration(nct_id: str, use_cache: bool = True) -> Optional[dict]:
    """Fetch study modules for an NCT trial. Returns None on failure."""
    nct_id = nct_id.upper().strip()
    if not re.match(r"NCT\d{8}", nct_id):
        return None
    cache_file = _cache_path(nct_id)
    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text())
    url = f"{CTGOV_BASE}/{nct_id}"
    params = {"fields": _FIELDS}
    try:
        import httpx

        resp = httpx.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(data))
    return data


def api_response_hash(registration_data: dict) -> str:
    """Return a stable hash for the ClinicalTrials.gov response payload."""
    payload = json.dumps(registration_data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def registry_source_document(nct_id: str, registration_data: dict) -> SourceDocument:
    nct_id = nct_id.upper().strip()
    return SourceDocument(
        document_id=f"registry:{nct_id}",
        document_name=f"ClinicalTrials.gov {nct_id}",
        document_role="registry",
        source_kind="ctgov",
        path=f"https://clinicaltrials.gov/study/{nct_id}",
        is_primary=False,
        status="parsed",
        retrieval_date=date.today().isoformat(),
        api_response_hash=api_response_hash(registration_data),
    )


def extract_outcomes(registration_data: dict) -> dict:
    """Extract primary and secondary outcomes from ClinicalTrials.gov response."""
    outcomes: dict = {"primary": [], "secondary": [], "other": []}
    try:
        module = registration_data.get("protocolSection", {}).get("outcomesModule", {})
        for key, label in [
            ("primaryOutcomes", "primary"),
            ("secondaryOutcomes", "secondary"),
            ("otherOutcomes", "other"),
        ]:
            for item in module.get(key, []):
                measure = item.get("measure", "").strip()
                if measure:
                    outcomes[label].append(measure)
    except Exception:
        pass
    return outcomes


def format_outcomes_for_prompt(outcomes: dict) -> str:
    """Format extracted outcomes into a text block for LLM prompts."""

    def _xml_safe(text: str) -> str:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    lines = ["Registered outcomes from ClinicalTrials.gov:"]
    if outcomes["primary"]:
        lines.append(
            "PRIMARY: " + "; ".join(_xml_safe(value) for value in outcomes["primary"])
        )
    if outcomes["secondary"]:
        lines.append(
            "SECONDARY: "
            + "; ".join(_xml_safe(value) for value in outcomes["secondary"])
        )
    if outcomes["other"]:
        lines.append(
            "EXPLORATORY/OTHER: "
            + "; ".join(_xml_safe(value) for value in outcomes["other"])
        )
    if not any(outcomes.values()):
        lines.append("(No outcomes retrieved from registration)")
    return "\n".join(lines)


def extract_design_info(registration_data: dict) -> dict:
    """Extract design metadata from a ClinicalTrials.gov response."""
    try:
        protocol = registration_data.get("protocolSection", {})
        design_module = protocol.get("designModule", {})
        if not design_module:
            return {}

        design_info = design_module.get("designInfo", {})
        masking_info = design_info.get("maskingInfo", {})
        oversight = protocol.get("oversightModule", {})
        lead_sponsor = protocol.get("sponsorCollaboratorsModule", {}).get(
            "leadSponsor", {}
        )
        enrollment = design_module.get("enrollmentInfo", {})
        phases = design_module.get("phases", [])
        if phases and not isinstance(phases, list):
            phases = [phases]

        return {
            "allocation": design_info.get("allocationType")
            or design_info.get("allocation", ""),
            "intervention_model": design_info.get("interventionModel", ""),
            "primary_purpose": design_info.get("primaryPurpose", ""),
            "masking": masking_info.get("masking", ""),
            "who_masked": masking_info.get("whoMasked", []),
            "has_dmc": oversight.get("oversightHasDmc"),
            "sponsor_name": lead_sponsor.get("name", ""),
            "sponsor_class": lead_sponsor.get("class", ""),
            "enrollment": enrollment.get("count", ""),
            "phases": phases or [],
        }
    except Exception:
        return {}


def format_design_for_prompt(design: dict) -> str:
    """Format design metadata as a labeled prompt block."""
    if not design:
        return "(No design metadata retrieved from ClinicalTrials.gov)"

    lines = ["Authoritative ClinicalTrials.gov registry design metadata:"]
    if design.get("allocation"):
        lines.append(f"  Allocation type: {design['allocation']}")
    if design.get("intervention_model"):
        lines.append(f"  Intervention model: {design['intervention_model']}")
    if design.get("masking"):
        who = (
            ", ".join(design["who_masked"])
            if design.get("who_masked")
            else "not specified"
        )
        lines.append(f"  Masking: {design['masking']} (masked parties: {who})")
    if design.get("has_dmc") is not None:
        lines.append(
            f"  Data monitoring committee (DMC): {'Yes' if design['has_dmc'] else 'No'}"
        )
    if design.get("sponsor_name"):
        lines.append(
            f"  Lead sponsor: {design['sponsor_name']} (class: {design.get('sponsor_class', 'unknown')})"
        )
    if design.get("phases"):
        lines.append(f"  Phase: {', '.join(design['phases'])}")
    if design.get("enrollment"):
        lines.append(f"  Total enrolled: {design['enrollment']}")
    return "\n".join(lines)


def extract_description(registration_data: dict) -> str:
    """Extract brief summary and detailed description from registration data."""
    try:
        desc_module = registration_data.get("protocolSection", {}).get(
            "descriptionModule", {}
        )
        brief = (desc_module.get("briefSummary") or "").strip()
        detailed = (desc_module.get("detailedDescription") or "").strip()
        parts = []
        if brief:
            parts.append(f"Brief summary: {brief}")
        if detailed:
            parts.append(
                "Detailed description (registry objectives and details):\n" + detailed
            )
        return "\n\n".join(parts)
    except Exception:
        return ""


def format_description_for_prompt(description: str) -> str:
    """Format registration description as a labeled prompt block."""
    if not description:
        return "(No description retrieved from ClinicalTrials.gov)"
    return "Authoritative ClinicalTrials.gov registry description:\n" + description


def extract_participant_flow(registration_data: dict) -> str:
    """Extract arm-level participant flow and withdrawal data."""
    try:
        flow_module = registration_data.get("resultsSection", {}).get(
            "participantFlowModule", {}
        )
        if not flow_module:
            return ""

        lines = ["Participant flow (from ClinicalTrials.gov posted results):"]
        recruitment = (flow_module.get("recruitmentDetails") or "").strip()
        if recruitment:
            lines.append(f"  Recruitment: {recruitment}")

        groups = {
            group["id"]: group["title"]
            for group in flow_module.get("groups", [])
            if "id" in group and "title" in group
        }
        for period in flow_module.get("periods", []):
            for milestone in period.get("milestones", []):
                label = milestone.get("title") or milestone.get("type", "")
                counts = [
                    f"{groups.get(item.get('groupId'), item.get('groupId', '?'))}: {item.get('numSubjects', '?')}"
                    for item in milestone.get("achievements", [])
                ]
                if counts:
                    lines.append(f"  {label}: {', '.join(counts)}")
            for withdrawal in period.get("dropWithdraws", []):
                reason = withdrawal.get("type", "Unknown")
                counts = [
                    f"{groups.get(item.get('groupId'), item.get('groupId', '?'))}: {item.get('numSubjects', '?')}"
                    for item in withdrawal.get("reasons", [])
                ]
                if counts:
                    lines.append(f"  Withdrawal - {reason}: {', '.join(counts)}")
        return "\n".join(lines)
    except Exception:
        return ""


def format_flow_for_prompt(flow: str) -> str:
    """Format participant flow as a labeled prompt block."""
    if not flow:
        return "(No participant flow data retrieved from ClinicalTrials.gov)"
    return flow
