#!/usr/bin/env python3
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ascend_compat import normalize_cann_version


ASCEND_ENV_HINT_VARS = (
    "ASCEND_HOME_PATH",
    "ASCEND_TOOLKIT_HOME",
    "ASCEND_TOOLKIT_PATH",
)
BOUNDED_SEARCH_ROOT_ENV_VARS = ("HOME", "USERPROFILE")
BOUNDED_SEARCH_STATIC_ROOTS = (
    Path("/usr/local/Ascend"),
    Path("/opt/Ascend"),
)
BOUNDED_SEARCH_MAX_DEPTH = 5
BOUNDED_SEARCH_MAX_CANDIDATES = 8
SKIP_SEARCH_DIRS = {
    ".git",
    "__pycache__",
    ".cache",
    ".conda",
    ".local",
    "node_modules",
}

CANN_VERSION_FILE_NAMES = ("version.cfg", "version.info")
CANN_VERSION_LINE_PATTERN = re.compile(r"(?im)^\s*(?:version|cann(?:_version)?)\s*[:=]\s*([^\s#]+)")


def environment_has_ascend_runtime(environ: Optional[Dict[str, str]] = None) -> bool:
    env = environ or os.environ
    home_value = env.get("ASCEND_HOME_PATH") or env.get("ASCEND_TOOLKIT_HOME") or env.get("ASCEND_TOOLKIT_PATH")
    opp_value = env.get("ASCEND_OPP_PATH")
    if home_value and opp_value:
        return True

    for key in ("LD_LIBRARY_PATH", "PYTHONPATH", "PATH", "ASCEND_OPP_PATH", "TBE_IMPL_PATH"):
        value = env.get(key)
        if value and "ascend" in value.lower():
            return True

    return False


def add_candidate_path(path: Path, seen: set, candidates: List[Path]) -> None:
    normalized = str(path)
    if normalized in seen:
        return
    seen.add(normalized)
    candidates.append(path)


def normalize_cann_path(value: Optional[str]) -> List[Path]:
    if not value:
        return []
    path = Path(value).expanduser()
    candidates: List[Path] = []
    if path.name == "set_env.sh":
        candidates.append(path)
    else:
        candidates.extend(
            [
                path / "set_env.sh",
                path / "ascend-toolkit" / "set_env.sh",
                path / "latest" / "set_env.sh",
            ]
        )
    return candidates


def derive_current_env_script_candidates() -> List[Path]:
    candidates: List[Path] = []
    seen = set()

    for var_name in ASCEND_ENV_HINT_VARS:
        value = os.environ.get(var_name)
        if not value:
            continue
        for candidate in normalize_cann_path(value):
            if candidate.exists():
                add_candidate_path(candidate, seen, candidates)

    return sorted(candidates, key=rank_ascend_env_script)


def search_root_for_ascend_env_scripts(root: Path, limit: int) -> List[Path]:
    if not root.exists() or not root.is_dir() or limit <= 0:
        return []

    candidates: List[Path] = []
    seen = set()
    for candidate in normalize_cann_path(str(root)):
        if candidate.exists():
            add_candidate_path(candidate, seen, candidates)
            if len(candidates) >= limit:
                return sorted(candidates, key=rank_ascend_env_script)

    root_depth = len(root.resolve().parts)
    for current_root, dirnames, filenames in os.walk(root):
        current_path = Path(current_root)
        try:
            depth = len(current_path.resolve().parts) - root_depth
        except OSError:
            continue
        if depth > BOUNDED_SEARCH_MAX_DEPTH:
            dirnames[:] = []
            continue

        dirnames[:] = [
            name
            for name in dirnames
            if name not in SKIP_SEARCH_DIRS
        ]

        if "set_env.sh" not in filenames:
            continue
        candidate = current_path / "set_env.sh"
        lowered = str(candidate).replace("\\", "/").lower()
        if "ascend" not in lowered and "cann" not in lowered:
            continue
        add_candidate_path(candidate, seen, candidates)
        if len(candidates) >= limit:
            break

    return sorted(candidates, key=rank_ascend_env_script)


def bounded_search_roots(cann_path: Optional[str]) -> List[Path]:
    roots: List[Path] = []
    seen = set()

    def add_root(path: Path) -> None:
        normalized = str(path)
        if normalized in seen:
            return
        seen.add(normalized)
        roots.append(path)

    if cann_path:
        add_root(Path(cann_path).expanduser())

    for var_name in ASCEND_ENV_HINT_VARS:
        value = os.environ.get(var_name)
        if value:
            add_root(Path(value).expanduser())

    for var_name in BOUNDED_SEARCH_ROOT_ENV_VARS:
        value = os.environ.get(var_name)
        if value:
            add_root(Path(value).expanduser())

    for root in BOUNDED_SEARCH_STATIC_ROOTS:
        add_root(root)

    return roots


def candidate_ascend_env_scripts(cann_path: Optional[str] = None) -> Tuple[List[Path], str]:
    current_candidates = derive_current_env_script_candidates()
    if environment_has_ascend_runtime():
        return current_candidates, "current_environment"

    candidates: List[Path] = []
    seen = set()
    for root in bounded_search_roots(cann_path):
        remaining = BOUNDED_SEARCH_MAX_CANDIDATES - len(candidates)
        if remaining <= 0:
            break
        for candidate in search_root_for_ascend_env_scripts(root, remaining):
            add_candidate_path(candidate, seen, candidates)
    return candidates, "bounded_search"


def rank_ascend_env_script(path: Path) -> Tuple[int, int, str]:
    text = str(path).replace("\\", "/").lower()
    if text.endswith("/ascend-toolkit/set_env.sh"):
        return (0, len(path.parts), text)
    if "/ascend-toolkit/latest/" in text:
        return (1, len(path.parts), text)
    if "/cann-" in text:
        return (2, len(path.parts), text)
    return (10, len(path.parts), text)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def parse_cann_version_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    match = CANN_VERSION_LINE_PATTERN.search(text)
    if match:
        version = normalize_cann_version(match.group(1))
        if version:
            return version
    return normalize_cann_version(text)


def add_candidate_version_path(path: Path, seen: set, candidates: List[Path]) -> None:
    normalized = str(path)
    if normalized in seen:
        return
    seen.add(normalized)
    candidates.append(path)


def extend_version_paths(base: Path, seen: set, candidates: List[Path]) -> None:
    path = Path(base)
    if path.name == "set_env.sh":
        parents = [path.parent, path.parent.parent]
    else:
        parents = [path]

    for parent in parents:
        for file_name in CANN_VERSION_FILE_NAMES:
            add_candidate_version_path(parent / file_name, seen, candidates)
            add_candidate_version_path(parent / "latest" / file_name, seen, candidates)
            add_candidate_version_path(parent / "ascend-toolkit" / file_name, seen, candidates)
            add_candidate_version_path(parent / "ascend-toolkit" / "latest" / file_name, seen, candidates)


def candidate_cann_version_files(
    cann_path: Optional[str] = None,
    script_path: Optional[str] = None,
    environ: Optional[Dict[str, str]] = None,
) -> List[Path]:
    env = environ or os.environ
    candidates: List[Path] = []
    seen = set()

    if cann_path:
        extend_version_paths(Path(cann_path).expanduser(), seen, candidates)
    if script_path:
        extend_version_paths(Path(script_path).expanduser(), seen, candidates)

    for var_name in ASCEND_ENV_HINT_VARS:
        value = env.get(var_name)
        if value:
            extend_version_paths(Path(value).expanduser(), seen, candidates)

    return candidates


def detect_cann_version(
    cann_path: Optional[str] = None,
    script_path: Optional[str] = None,
    environ: Optional[Dict[str, str]] = None,
) -> dict:
    env = environ or os.environ

    for candidate in candidate_cann_version_files(cann_path, script_path, env):
        if not candidate.exists() or not candidate.is_file():
            continue
        version = parse_cann_version_from_text(read_text(candidate))
        if version:
            return {
                "cann_version": version,
                "cann_version_source": "version_file",
                "cann_version_file": str(candidate),
            }

    for source_name, raw_value in (
        ("cann_path", cann_path),
        ("ascend_env_script", script_path),
        *[(f"env:{name}", env.get(name)) for name in ASCEND_ENV_HINT_VARS],
    ):
        version = normalize_cann_version(raw_value)
        if version:
            return {
                "cann_version": version,
                "cann_version_source": source_name,
                "cann_version_file": None,
            }

    return {
        "cann_version": None,
        "cann_version_source": "unresolved",
        "cann_version_file": None,
    }


def detect_ascend_runtime(target: Optional[dict] = None) -> dict:
    cann_path = None
    if isinstance(target, dict):
        cann_path = target.get("cann_path")
    candidates, selection_source = candidate_ascend_env_scripts(cann_path)
    script_path = str(candidates[0]) if candidates else None
    return {
        "requires_ascend": True,
        "device_paths_present": any(Path("/dev").glob("davinci*")),
        "ascend_env_script_present": bool(script_path),
        "ascend_env_script_path": script_path,
        "ascend_env_candidate_paths": [str(path) for path in candidates[:10]],
        "ascend_env_selection_source": selection_source,
        "cann_path_input": cann_path,
        "ascend_env_active": environment_has_ascend_runtime(),
    }


def source_environment_from_script(script_path: str) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    command = [
        "bash",
        "-lc",
        "set -e; source {script} >/dev/null; env -0".format(script=shlex.quote(script_path)),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            timeout=15,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace").strip()
        stdout = (exc.stdout or b"").decode("utf-8", errors="replace").strip()
        return None, stderr or stdout or f"failed to source {script_path}"
    except (OSError, subprocess.SubprocessError) as exc:
        return None, str(exc)

    payload = completed.stdout or b""
    env: Dict[str, str] = {}
    for item in payload.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        env[key.decode("utf-8", errors="replace")] = value.decode("utf-8", errors="replace")

    if not env:
        return None, "sourced environment payload was empty"

    return env, None


def resolve_runtime_environment(system_layer: dict) -> Tuple[Dict[str, str], str, Optional[str]]:
    if system_layer.get("ascend_env_active"):
        return dict(os.environ), "current_environment", None

    script_path = system_layer.get("ascend_env_script_path")
    if script_path:
        sourced_env, error = source_environment_from_script(str(script_path))
        if sourced_env is not None:
            if environment_has_ascend_runtime(sourced_env):
                return sourced_env, "sourced_script", None
            return (
                dict(os.environ),
                "sourced_script_invalid",
                "sourced Ascend environment did not activate required runtime variables",
            )
        return dict(os.environ), "sourced_script_failed", error

    return dict(os.environ), "current_environment", None
