#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple


WORKSPACE_ENV_CANDIDATES = (
    ".venv",
    "venv",
    ".env",
    "env",
)

PYTHON_RELATIVE_CANDIDATES = (
    Path("bin/python"),
    Path("bin/python3"),
    Path("Scripts/python.exe"),
    Path("Scripts/python"),
)

MIN_HELPER_PYTHON = (3, 9)


def resolve_optional_path(value: Optional[str], root: Path) -> Optional[Path]:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path


def python_in_env(env_root: Path) -> Optional[Path]:
    for candidate in PYTHON_RELATIVE_CANDIDATES:
        python_path = env_root / candidate
        if python_path.exists() and python_path.is_file():
            return python_path
    return None


def derive_env_root_from_python(python_path: Path) -> Optional[Path]:
    parent_name = python_path.parent.name.lower()
    if parent_name in {"bin", "scripts"}:
        return python_path.parent.parent
    return None


def inspect_python(python_path: Path) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    command = [
        str(python_path),
        "-c",
        (
            "import json, platform, sys; "
            "print(json.dumps({"
            "'version_info': list(sys.version_info[:3]), "
            "'version': platform.python_version()"
            "}))"
        ),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, str(exc)

    try:
        payload = json.loads(completed.stdout.strip() or "{}")
    except json.JSONDecodeError:
        return None, "python probe returned non-JSON output"

    if not isinstance(payload, dict):
        return None, "python probe returned a non-object payload"

    return payload, None


def _selection_result(
    *,
    root: Path,
    python_path: Optional[Path],
    env_root: Optional[Path],
    source: str,
    status: str,
    reason: str,
    python_version: Optional[str] = None,
    version_info: Optional[Tuple[int, int, int]] = None,
) -> Dict[str, object]:
    helper_compatible = bool(version_info and tuple(version_info) >= MIN_HELPER_PYTHON)
    selected_python = str(python_path) if python_path else None
    selected_env_root = str(env_root) if env_root else None
    payload: Dict[str, object] = {
        "working_dir": str(root),
        "selected_python": selected_python,
        "selected_env_root": selected_env_root,
        "selection_source": source,
        "selection_status": status,
        "selection_reason": reason,
        "python_version": python_version,
        "python_version_info": list(version_info) if version_info else None,
        "helper_python_compatible": helper_compatible,
    }
    return payload


def inspect_candidate(root: Path, python_path: Path, source: str, env_root: Optional[Path]) -> Dict[str, object]:
    payload, error = inspect_python(python_path)
    if error:
        return _selection_result(
            root=root,
            python_path=python_path,
            env_root=env_root,
            source=source,
            status="invalid",
            reason=error,
        )

    version_info_raw = payload.get("version_info") or []
    version_info: Optional[Tuple[int, int, int]] = None
    if isinstance(version_info_raw, list) and len(version_info_raw) >= 3:
        try:
            version_info = (
                int(version_info_raw[0]),
                int(version_info_raw[1]),
                int(version_info_raw[2]),
            )
        except (TypeError, ValueError):
            version_info = None

    python_version = payload.get("version")
    if not isinstance(python_version, str):
        python_version = None

    if not version_info:
        return _selection_result(
            root=root,
            python_path=python_path,
            env_root=env_root,
            source=source,
            status="invalid",
            reason="python probe did not return a usable version",
            python_version=python_version,
        )

    if version_info < MIN_HELPER_PYTHON:
        return _selection_result(
            root=root,
            python_path=python_path,
            env_root=env_root,
            source=source,
            status="unsupported",
            reason="selected python is below the helper minimum version 3.9",
            python_version=python_version,
            version_info=version_info,
        )

    return _selection_result(
        root=root,
        python_path=python_path,
        env_root=env_root,
        source=source,
        status="selected",
        reason="selected python is usable for readiness-agent helpers",
        python_version=python_version,
        version_info=version_info,
    )


def resolve_selected_python(
    root: Path,
    selected_python: Optional[str] = None,
    selected_env_root: Optional[str] = None,
) -> Dict[str, object]:
    explicit_python = resolve_optional_path(selected_python, root)
    if explicit_python:
        return inspect_candidate(
            root,
            explicit_python,
            "explicit_python",
            derive_env_root_from_python(explicit_python),
        )

    explicit_env = resolve_optional_path(selected_env_root, root)
    if explicit_env:
        env_python = python_in_env(explicit_env)
        if env_python:
            return inspect_candidate(root, env_python, "explicit_env", explicit_env)
        return _selection_result(
            root=root,
            python_path=None,
            env_root=explicit_env,
            source="explicit_env",
            status="missing",
            reason="selected environment root does not contain a Python executable",
        )

    for candidate_name in WORKSPACE_ENV_CANDIDATES:
        env_root = root / candidate_name
        env_python = python_in_env(env_root)
        if env_python:
            return inspect_candidate(root, env_python, "workspace_env", env_root)

    return _selection_result(
        root=root,
        python_path=None,
        env_root=None,
        source="workspace_env",
        status="missing",
        reason="no selected python was provided and no workspace virtual environment was found",
    )
