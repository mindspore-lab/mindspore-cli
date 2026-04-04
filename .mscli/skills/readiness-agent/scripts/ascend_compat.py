#!/usr/bin/env python3
import re
from typing import Dict, List, Optional, Tuple


MINDSPORE_COMPAT_ROWS = [
    {
        "cann": "8.5.0",
        "mindspore": "2.8.0",
        "python": "3.9-3.12",
        "note": "Latest published MindSpore row from the official versions page.",
    },
    {
        "cann": "8.5.0",
        "mindspore": "2.7.2",
        "python": "3.9-3.12",
        "note": "Stable 8.5.0 line from the official versions page.",
    },
    {
        "cann": "8.3.RC1",
        "mindspore": "2.8.0",
        "python": "3.9-3.12",
        "note": "Current RC line from the official versions page.",
    },
    {
        "cann": "8.3.RC1",
        "mindspore": "2.7.1",
        "python": "3.9-3.11",
        "note": "Current RC line from the official versions page.",
    },
    {
        "cann": "8.2.RC1",
        "mindspore": "2.7.0",
        "python": "3.9-3.11",
        "note": "Current RC line from the official versions page.",
    },
    {
        "cann": "8.2.RC1",
        "mindspore": "2.7.0-rc1",
        "python": "3.9-3.11",
        "note": "Current RC line from the official versions page.",
    },
]

PTA_COMPAT_ROWS = [
    {
        "cann": "8.5.0",
        "torch": "2.9.0",
        "torch_npu": "2.9.0",
        "python": "3.9-3.11",
        "branch": "v2.9.0-7.3.0",
        "note": "Latest published PTA line in the upstream compatibility table.",
    },
    {
        "cann": "8.5.0",
        "torch": "2.8.0",
        "torch_npu": "2.8.0.post2",
        "python": "3.9-3.11",
        "branch": "v2.8.0-7.3.0",
        "note": "Newer Ascend stacks.",
    },
    {
        "cann": "8.5.0",
        "torch": "2.7.1",
        "torch_npu": "2.7.1.post2",
        "python": "3.9-3.11",
        "branch": "v2.7.1-7.3.0",
        "note": "Newer Ascend stacks.",
    },
    {
        "cann": "8.5.0",
        "torch": "2.6.0",
        "torch_npu": "2.6.0.post5",
        "python": "3.9-3.11",
        "branch": "v2.6.0-7.3.0",
        "note": "Newer Ascend stacks.",
    },
    {
        "cann": "8.3.RC1",
        "torch": "2.8.0",
        "torch_npu": "2.8.0",
        "python": "3.9-3.11",
        "branch": "v2.8.0-7.2.0",
        "note": "Current RC line.",
    },
    {
        "cann": "8.3.RC1",
        "torch": "2.7.1",
        "torch_npu": "2.7.1",
        "python": "3.9-3.11",
        "branch": "v2.7.1-7.2.0",
        "note": "Current RC line.",
    },
    {
        "cann": "8.3.RC1",
        "torch": "2.6.0",
        "torch_npu": "2.6.0.post3",
        "python": "3.9-3.11",
        "branch": "v2.6.0-7.2.0",
        "note": "Current RC line.",
    },
    {
        "cann": "8.3.RC1",
        "torch": "2.1.0",
        "torch_npu": "2.1.0.post17",
        "python": "3.8-3.11",
        "branch": "v2.1.0-7.2.0",
        "note": "Legacy compatibility.",
    },
    {
        "cann": "8.2.RC1",
        "torch": "2.6.0",
        "torch_npu": "2.6.0",
        "python": "3.9-3.11",
        "branch": "v2.6.0-7.1.0",
        "note": "Transitional line.",
    },
    {
        "cann": "8.2.RC1",
        "torch": "2.5.1",
        "torch_npu": "2.5.1.post1",
        "python": "3.9-3.11",
        "branch": "v2.5.1-7.1.0",
        "note": "Transitional line.",
    },
    {
        "cann": "8.2.RC1",
        "torch": "2.1.0",
        "torch_npu": "2.1.0.post13",
        "python": "3.8-3.11",
        "branch": "v2.1.0-7.1.0",
        "note": "Legacy compatibility.",
    },
    {
        "cann": "8.1.RC1",
        "torch": "2.5.1",
        "torch_npu": "2.5.1",
        "python": "3.9-3.11",
        "branch": "v2.5.1-7.0.0",
        "note": "Common production baseline.",
    },
    {
        "cann": "8.1.RC1",
        "torch": "2.4.0",
        "torch_npu": "2.4.0.post4",
        "python": "3.8-3.11",
        "branch": "v2.4.0-7.0.0",
        "note": "Common production baseline.",
    },
    {
        "cann": "8.1.RC1",
        "torch": "2.3.1",
        "torch_npu": "2.3.1.post6",
        "python": "3.8-3.11",
        "branch": "v2.3.1-7.0.0",
        "note": "Common production baseline.",
    },
    {
        "cann": "8.1.RC1",
        "torch": "2.1.0",
        "torch_npu": "2.1.0.post12",
        "python": "3.8-3.11",
        "branch": "v2.1.0-7.0.0",
        "note": "Older but still common.",
    },
]

CANN_VERSION_PATTERN = re.compile(r"(?i)(\d+\.\d+(?:\.\d+)?(?:\.rc\d+)?)")


def normalize_cann_version(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    match = CANN_VERSION_PATTERN.search(str(value).strip())
    if not match:
        return None
    token = match.group(1)
    return re.sub(r"(?i)\.rc", ".RC", token)


def parse_python_version(value: Optional[str]) -> Optional[Tuple[int, int]]:
    if not value:
        return None
    match = re.search(r"(\d+)\.(\d+)", str(value).strip())
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def python_version_in_range(version: Optional[str], supported_range: str) -> bool:
    parsed_version = parse_python_version(version)
    if not parsed_version:
        return False

    parts = [item.strip() for item in supported_range.split("-", 1)]
    if not parts or not parts[0]:
        return False

    lower = parse_python_version(parts[0])
    upper = parse_python_version(parts[1] if len(parts) > 1 else parts[0])
    if not lower or not upper:
        return False
    return lower <= parsed_version <= upper


def normalize_torch_version(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    token = str(value).strip().lstrip("v")
    return token.split("+", 1)[0]


def normalize_torch_npu_version(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    token = str(value).strip().lstrip("v")
    return token.split("+", 1)[0]


def normalize_mindspore_version(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    token = str(value).strip().lstrip("v")
    token = token.split("+", 1)[0]
    token = re.sub(r"(?i)\.rc", "-rc", token)
    token = re.sub(r"(?i)(\d)rc", r"\1-rc", token)
    return token


def _mindspore_package_specs(row: dict) -> List[str]:
    return [f"mindspore=={row['mindspore']}"]


def _pta_package_specs(row: dict) -> List[str]:
    return [f"torch=={row['torch']}", f"torch_npu=={row['torch_npu']}"]


def _resolve_rows(framework_path: str, cann_version: Optional[str], python_version: Optional[str]) -> dict:
    normalized_cann = normalize_cann_version(cann_version)
    payload = {
        "source": "local-ascend-compat",
        "framework_path": framework_path,
        "cann_version": normalized_cann,
        "python_version": python_version,
        "status": "unresolved",
        "reason": None,
        "selected_row": None,
        "compatible_rows": [],
        "package_specs": [],
    }

    if framework_path == "mindspore":
        rows = MINDSPORE_COMPAT_ROWS
    elif framework_path == "pta":
        rows = PTA_COMPAT_ROWS
    else:
        payload["status"] = "unsupported"
        payload["reason"] = f"framework path {framework_path!r} does not use the local Ascend compatibility table"
        return payload

    if not normalized_cann:
        payload["status"] = "cann_version_unknown"
        payload["reason"] = "CANN version is unresolved, so the local compatibility table cannot select framework packages."
        return payload

    cann_rows = [row for row in rows if row["cann"] == normalized_cann]
    if not cann_rows:
        payload["status"] = "cann_version_unmapped"
        payload["reason"] = f"CANN version {normalized_cann} is not present in the local {framework_path} compatibility table."
        return payload

    if not parse_python_version(python_version):
        payload["status"] = "python_version_unknown"
        payload["reason"] = "Selected Python version is unresolved, so the local compatibility table cannot select framework packages."
        return payload

    compatible_rows = [row for row in cann_rows if python_version_in_range(python_version, row["python"])]
    if not compatible_rows:
        payload["status"] = "python_version_incompatible"
        payload["reason"] = (
            f"Python {python_version} is outside the supported range for the local {framework_path} rows "
            f"for CANN {normalized_cann}."
        )
        return payload

    selected_row = compatible_rows[0]
    payload["status"] = "resolved"
    payload["selected_row"] = selected_row
    payload["compatible_rows"] = compatible_rows
    if framework_path == "mindspore":
        payload["package_specs"] = _mindspore_package_specs(selected_row)
    else:
        payload["package_specs"] = _pta_package_specs(selected_row)
    payload["reason"] = f"Selected the first compatible local {framework_path} row for CANN {normalized_cann} and Python {python_version}."
    return payload


def resolve_framework_compatibility(
    framework_path: str,
    cann_version: Optional[str],
    python_version: Optional[str],
) -> dict:
    return _resolve_rows(framework_path, cann_version, python_version)


def assess_installed_framework_compatibility(
    framework_path: str,
    cann_version: Optional[str],
    python_version: Optional[str],
    installed_versions: Optional[Dict[str, Optional[str]]],
) -> dict:
    reference = _resolve_rows(framework_path, cann_version, python_version)
    versions = dict(installed_versions or {})
    payload = {
        "source": "local-ascend-compat",
        "framework_path": framework_path,
        "cann_version": reference.get("cann_version"),
        "python_version": python_version,
        "installed_versions": versions,
        "recommended_package_specs": reference.get("package_specs", []),
        "reference_status": reference.get("status"),
        "status": "unresolved",
        "reason": reference.get("reason"),
        "matched_row": None,
    }

    if reference.get("status") != "resolved":
        return payload

    compatible_rows = reference.get("compatible_rows") or []
    if framework_path == "mindspore":
        installed_mindspore = normalize_mindspore_version(versions.get("mindspore"))
        if not installed_mindspore:
            payload["reason"] = "Installed MindSpore version is unavailable, so compatibility cannot be confirmed."
            return payload
        for row in compatible_rows:
            if installed_mindspore == normalize_mindspore_version(row.get("mindspore")):
                payload["status"] = "compatible"
                payload["matched_row"] = row
                payload["reason"] = (
                    f"Installed MindSpore {installed_mindspore} matches a local compatibility row "
                    f"for CANN {payload['cann_version']} and Python {python_version}."
                )
                return payload
        payload["status"] = "incompatible"
        payload["reason"] = (
            f"Installed MindSpore {installed_mindspore} does not match any local compatibility row "
            f"for CANN {payload['cann_version']} and Python {python_version}."
        )
        return payload

    if framework_path == "pta":
        installed_torch = normalize_torch_version(versions.get("torch"))
        installed_torch_npu = normalize_torch_npu_version(versions.get("torch_npu"))
        if not installed_torch or not installed_torch_npu:
            payload["reason"] = "Installed torch or torch_npu version is unavailable, so compatibility cannot be confirmed."
            return payload
        for row in compatible_rows:
            if (
                installed_torch == normalize_torch_version(row.get("torch"))
                and installed_torch_npu == normalize_torch_npu_version(row.get("torch_npu"))
            ):
                payload["status"] = "compatible"
                payload["matched_row"] = row
                payload["reason"] = (
                    f"Installed torch {installed_torch} and torch_npu {installed_torch_npu} match a local compatibility row "
                    f"for CANN {payload['cann_version']} and Python {python_version}."
                )
                return payload
        payload["status"] = "incompatible"
        payload["reason"] = (
            f"Installed torch {installed_torch} and torch_npu {installed_torch_npu} do not match any local compatibility row "
            f"for CANN {payload['cann_version']} and Python {python_version}."
        )
        return payload

    payload["status"] = "unsupported"
    payload["reason"] = f"framework path {framework_path!r} does not use installed-version compatibility validation"
    return payload
