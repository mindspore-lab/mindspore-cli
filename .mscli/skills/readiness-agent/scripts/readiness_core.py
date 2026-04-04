#!/usr/bin/env python3
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from ascend_compat import assess_installed_framework_compatibility, resolve_framework_compatibility
from python_selection import python_in_env, resolve_selected_python
from runtime_env import detect_ascend_runtime, detect_cann_version, resolve_runtime_environment


DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
DEFAULT_PIP_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"
FALLBACK_PIP_INDEX_URL = "https://mirrors.aliyun.com/pypi/simple/"
SUPPORTED_PIP_INDEX_URLS = (DEFAULT_PIP_INDEX_URL, FALLBACK_PIP_INDEX_URL)
SKIP_DIRS = {"__pycache__", "node_modules", "venv", "env", "readiness-output", "huggingface-cache", "hf_cache"}
RUNTIME_IMPORT_CANDIDATES = {
    "mindspore",
    "torch",
    "torch_npu",
    "transformers",
    "datasets",
    "tokenizers",
    "accelerate",
    "safetensors",
    "diffusers",
    "peft",
    "trl",
    "evaluate",
    "sentencepiece",
}
FRAMEWORK_IMPORTS = {
    "mindspore": ["mindspore"],
    "pta": ["torch", "torch_npu"],
    "mixed": ["mindspore", "torch", "torch_npu"],
}
EXAMPLE_RECIPES = [
    {
        "id": "qwen3-training",
        "target_type": "training",
        "model_hub_id": "Qwen/Qwen3-0.6B",
        "dataset_hub_id": "karthiksagarn/astro_horoscope",
        "entry_script": "train.py",
        "template_path": Path(__file__).resolve().parents[1] / "examples" / "qwen3_0_6b_training_example.py",
    }
]
PROBE_CODE = """
import importlib.util
import json
import sys

mode = sys.argv[1]
payload = json.loads(sys.argv[2])

if mode == "import":
    packages = payload.get("packages", [])
    print(json.dumps({name: importlib.util.find_spec(name) is not None for name in packages}))
elif mode == "package_versions":
    try:
        from importlib import metadata as importlib_metadata
    except ImportError:
        import importlib_metadata
    packages = payload.get("packages", [])
    result = {"versions": {}, "errors": {}}
    for name in packages:
        try:
            module = __import__(name)
            version = getattr(module, "__version__", None)
            if version is None:
                candidates = [name]
                dashed_name = name.replace("_", "-")
                if dashed_name not in candidates:
                    candidates.append(dashed_name)
                for candidate in candidates:
                    try:
                        version = importlib_metadata.version(candidate)
                        break
                    except Exception:
                        continue
            result["versions"][name] = version
        except Exception as exc:
            result["versions"][name] = None
            result["errors"][name] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(result))
elif mode == "framework_smoke":
    framework_path = payload.get("framework_path")
    result = {"success": False, "details": [], "error": None}
    try:
        if framework_path == "mindspore":
            import mindspore as ms
            _ = getattr(ms, "Tensor", None)
            result["details"].append("mindspore import ok")
            result["success"] = True
        elif framework_path == "pta":
            import torch
            import torch_npu
            _ = getattr(torch, "Tensor", None)
            result["details"].extend(["torch import ok", "torch_npu import ok"])
            result["success"] = True
        elif framework_path == "mixed":
            import mindspore as ms
            import torch
            import torch_npu
            _ = getattr(ms, "Tensor", None)
            _ = getattr(torch, "Tensor", None)
            result["details"].extend(["mindspore import ok", "torch import ok", "torch_npu import ok"])
            result["success"] = True
        else:
            result["error"] = f"unsupported framework path: {framework_path}"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(result))
else:
    print(json.dumps({"error": f"unknown mode: {mode}"}))
"""


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def make_check(
    check_id: str,
    status: str,
    summary: str,
    evidence: Optional[List[str]] = None,
    **extra: object,
) -> dict:
    payload = {
        "id": check_id,
        "status": status,
        "summary": summary,
        "evidence": evidence or [],
    }
    for key, value in extra.items():
        if value is not None:
            payload[key] = value
    return payload


def resolve_optional_path(value: Optional[str], root: Path) -> Optional[Path]:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path


def should_skip_dirname(name: str) -> bool:
    return name.startswith(".") or name in SKIP_DIRS


def list_files(root: Path, max_depth: int = 2) -> List[Path]:
    files: List[Path] = []
    if not root.exists():
        return files
    root_depth = len(root.resolve().parts)
    for current_root, dirnames, filenames in os.walk(root):
        current_path = Path(current_root)
        try:
            depth = len(current_path.resolve().parts) - root_depth
        except OSError:
            continue
        if depth > max_depth:
            dirnames[:] = []
            continue
        dirnames[:] = [name for name in dirnames if not should_skip_dirname(name)]
        for name in filenames:
            files.append(current_path / name)
    return files


def extract_runtime_imports(entry_script: Optional[Path]) -> List[str]:
    if not entry_script or not entry_script.exists():
        return []
    text = read_text(entry_script)
    found: List[str] = []
    for name in sorted(RUNTIME_IMPORT_CANDIDATES):
        if f"import {name}" in text or f"from {name}" in text:
            found.append(name)
    return found


def infer_framework_from_text(text: str) -> Tuple[Optional[str], bool, List[str]]:
    lowered = text.lower()
    has_mindspore = "import mindspore" in lowered or "from mindspore" in lowered
    has_torch = "import torch" in lowered or "from torch" in lowered
    has_torch_npu = "import torch_npu" in lowered or "from torch_npu" in lowered

    evidence: List[str] = []
    if has_mindspore:
        evidence.append("mindspore imports detected")
    if has_torch or has_torch_npu:
        evidence.append("pta imports detected")

    if has_mindspore and (has_torch or has_torch_npu):
        return "mixed", True, evidence
    if has_mindspore:
        return "mindspore", True, evidence
    if has_torch or has_torch_npu:
        return "pta", True, evidence
    return None, False, evidence


def infer_target_type(
    requested_target: Optional[str],
    entry_script: Optional[Path],
    config_path: Optional[Path],
    dataset_path: Optional[Path],
    dataset_hub_id: Optional[str],
) -> Tuple[str, bool, List[str]]:
    if requested_target in {"training", "inference"}:
        return requested_target, True, [f"explicit target={requested_target}"]

    training_score = 0
    inference_score = 0
    evidence: List[str] = []

    if dataset_path or dataset_hub_id:
        training_score += 2
        evidence.append("dataset evidence suggests training")

    if entry_script:
        name = entry_script.name.lower()
        text = read_text(entry_script).lower()
        if any(token in name for token in ("train", "finetune", "pretrain")):
            training_score += 2
            evidence.append(f"entry script name suggests training: {entry_script.name}")
        if any(token in name for token in ("infer", "predict", "serve")):
            inference_score += 2
            evidence.append(f"entry script name suggests inference: {entry_script.name}")
        if "trainingarguments" in text or "trainer(" in text or "load_dataset(" in text:
            training_score += 1
            evidence.append("training APIs detected")
        if "generate(" in text or "pipeline(" in text:
            inference_score += 1
            evidence.append("inference APIs detected")

    if config_path and config_path.exists():
        text = read_text(config_path).lower()
        if any(token in text for token in ("epoch", "optimizer", "train_dataset", "dataset")):
            training_score += 1
            evidence.append("config suggests training")
        if any(token in text for token in ("max_new_tokens", "generation", "prompt")):
            inference_score += 1
            evidence.append("config suggests inference")

    if training_score > inference_score:
        return "training", inference_score == 0, evidence
    if inference_score > training_score:
        return "inference", training_score == 0, evidence
    return "inference", False, evidence or ["target remained ambiguous and defaulted to inference"]


def match_example_recipe(target: dict) -> Optional[dict]:
    for recipe in EXAMPLE_RECIPES:
        if (
            recipe["target_type"] == target.get("target_type")
            and recipe["model_hub_id"] == target.get("model_hub_id")
            and recipe["dataset_hub_id"] == target.get("dataset_hub_id")
        ):
            return dict(recipe)
    return None


def discover_execution_target(root: Path, args: object) -> dict:
    files = list_files(root)
    explicit_entry = resolve_optional_path(getattr(args, "entry_script", None), root)
    explicit_config = resolve_optional_path(getattr(args, "config_path", None), root)
    explicit_model = resolve_optional_path(getattr(args, "model_path", None), root)
    explicit_dataset = resolve_optional_path(getattr(args, "dataset_path", None), root)
    explicit_checkpoint = resolve_optional_path(getattr(args, "checkpoint_path", None), root)

    entry_script = explicit_entry
    if not entry_script:
        for candidate_name in ("train.py", "finetune.py", "infer.py", "predict.py", "run.py"):
            candidate = root / candidate_name
            if candidate.exists():
                entry_script = candidate
                break
        if not entry_script:
            entry_script = next((path for path in files if path.suffix == ".py"), None)

    config_path = explicit_config
    if not config_path:
        config_path = next((path for path in files if path.suffix.lower() in {".yaml", ".yml", ".json"}), None)

    model_path = explicit_model
    if not model_path and (root / "model").exists():
        model_path = root / "model"

    dataset_path = explicit_dataset
    if not dataset_path and (root / "dataset").exists():
        dataset_path = root / "dataset"

    checkpoint_path = explicit_checkpoint
    if not checkpoint_path:
        checkpoint_path = next((path for path in files if path.suffix.lower() in {".ckpt", ".pt", ".bin"}), None)

    target_type, target_stable, target_evidence = infer_target_type(
        getattr(args, "target", None),
        entry_script,
        config_path,
        dataset_path,
        getattr(args, "dataset_hub_id", None),
    )
    framework_hint = getattr(args, "framework_hint", None)
    if framework_hint in {"mindspore", "pta", "mixed"}:
        framework_path = framework_hint
        framework_stable = True
        framework_evidence = [f"explicit framework_hint={framework_hint}"]
    else:
        text = ""
        if entry_script:
            text += read_text(entry_script)
        if config_path:
            text += "\n" + read_text(config_path)
        framework_path, framework_stable, framework_evidence = infer_framework_from_text(text)

    target = {
        "working_dir": str(root),
        "target_type": target_type,
        "target_stable": target_stable,
        "target_evidence": target_evidence,
        "entry_script": str(entry_script) if entry_script else None,
        "config_path": str(config_path) if config_path else None,
        "model_path": str(model_path) if model_path else None,
        "dataset_path": str(dataset_path) if dataset_path else None,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "model_hub_id": getattr(args, "model_hub_id", None),
        "dataset_hub_id": getattr(args, "dataset_hub_id", None),
        "dataset_split": getattr(args, "dataset_split", None),
        "framework_path": framework_path,
        "framework_stable": framework_stable,
        "framework_evidence": framework_evidence,
        "framework_hint": framework_hint,
        "cann_path": getattr(args, "cann_path", None),
        "task_smoke_cmd": getattr(args, "task_smoke_cmd", None),
        "selected_python": getattr(args, "selected_python", None),
        "allow_network": bool(getattr(args, "allow_network", False)),
    }
    recipe = match_example_recipe(target)
    if recipe:
        target["example_recipe_id"] = recipe["id"]
        target["example_template_path"] = str(recipe["template_path"])
        if not target["entry_script"]:
            target["entry_script"] = str(root / recipe["entry_script"])
    return target


def run_json_probe_with_python(
    python_path: Path,
    mode: str,
    payload: dict,
    probe_env: Optional[Dict[str, str]] = None,
) -> Tuple[dict, Optional[str]]:
    probe_launcher = f"exec({PROBE_CODE!r})"
    try:
        completed = subprocess.run(
            [str(python_path), "-c", probe_launcher, mode, json.dumps(payload)],
            check=True,
            text=True,
            capture_output=True,
            timeout=10,
            env=probe_env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {}, str(exc)

    try:
        result = json.loads(completed.stdout.strip() or "{}")
    except json.JSONDecodeError:
        return {}, "probe returned non-JSON output"
    if not isinstance(result, dict):
        return {}, "probe returned a non-object payload"
    return result, None


def probe_imports(
    packages: List[str],
    python_path: Optional[str],
    probe_env: Optional[Dict[str, str]],
) -> Tuple[Dict[str, bool], Optional[str]]:
    if not packages:
        return {}, None
    if not python_path:
        return {package: False for package in packages}, "selected python is unavailable"
    result, error = run_json_probe_with_python(Path(python_path), "import", {"packages": packages}, probe_env)
    if error:
        return {package: False for package in packages}, error
    return {package: bool(result.get(package)) for package in packages}, None


def probe_package_versions(
    packages: List[str],
    python_path: Optional[str],
    probe_env: Optional[Dict[str, str]],
) -> Tuple[Dict[str, Optional[str]], Dict[str, str], Optional[str]]:
    if not packages or not python_path:
        return {}, {}, None
    result, error = run_json_probe_with_python(Path(python_path), "package_versions", {"packages": packages}, probe_env)
    if error:
        return {}, {}, error
    versions = result.get("versions") if isinstance(result.get("versions"), dict) else {}
    errors = result.get("errors") if isinstance(result.get("errors"), dict) else {}
    normalized_versions = {
        package: str(versions.get(package)).strip() if isinstance(versions.get(package), str) and str(versions.get(package)).strip() else None
        for package in packages
    }
    normalized_errors = {str(key): str(value) for key, value in errors.items()}
    return normalized_versions, normalized_errors, None


def probe_framework_smoke(
    framework_path: Optional[str],
    python_path: Optional[str],
    probe_env: Optional[Dict[str, str]],
) -> dict:
    if framework_path not in {"mindspore", "pta", "mixed"}:
        return {"status": "skipped", "details": [], "error": "framework path is unresolved"}
    if not python_path:
        return {"status": "skipped", "details": [], "error": "selected python is unavailable"}
    result, error = run_json_probe_with_python(Path(python_path), "framework_smoke", {"framework_path": framework_path}, probe_env)
    if error:
        return {"status": "failed", "details": [], "error": error}
    if result.get("success"):
        return {"status": "passed", "details": result.get("details") or [], "error": None}
    return {"status": "failed", "details": result.get("details") or [], "error": result.get("error")}


def head_line(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    return stripped.splitlines()[0]


def format_command(command: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def run_script_parse_smoke(
    entry_script: Optional[Path],
    python_path: Optional[str],
    root: Path,
    probe_env: Optional[Dict[str, str]],
) -> dict:
    if not python_path:
        return make_check("runtime-smoke-script-parse", "skipped", "Selected Python is unavailable for script parsing.")
    if not entry_script or not entry_script.exists():
        return make_check("runtime-smoke-script-parse", "skipped", "Entry script is unavailable for script parsing.")
    if entry_script.suffix.lower() != ".py":
        return make_check("runtime-smoke-script-parse", "skipped", "Entry script parsing is only supported for Python files.")

    command = [python_path, "-m", "py_compile", str(entry_script)]
    try:
        completed = subprocess.run(
            command,
            cwd=str(root),
            text=True,
            capture_output=True,
            timeout=10,
            env=probe_env,
        )
    except subprocess.TimeoutExpired as exc:
        return make_check(
            "runtime-smoke-script-parse",
            "block",
            "Entry script parse timed out in the selected environment.",
            evidence=[f"entry_script={entry_script.name}"],
            category_hint="workspace",
            remediable=False,
            remediation_owner="workspace",
            revalidation_scope=["workspace-assets", "runtime-smoke"],
            command_preview=format_command(command),
            timed_out=True,
            stdout_head=head_line(exc.stdout),
            stderr_head=head_line(exc.stderr),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return make_check(
            "runtime-smoke-script-parse",
            "block",
            "Entry script parse failed to start in the selected environment.",
            evidence=[f"error={exc}"],
            category_hint="workspace",
            remediable=False,
            remediation_owner="workspace",
            revalidation_scope=["workspace-assets", "runtime-smoke"],
            command_preview=format_command(command),
            timed_out=False,
        )

    if completed.returncode == 0:
        return make_check(
            "runtime-smoke-script-parse",
            "ok",
            "Entry script parses successfully in the selected environment.",
            evidence=[f"entry_script={entry_script.name}"],
            command_preview=format_command(command),
            exit_code=completed.returncode,
            stdout_head=head_line(completed.stdout),
            stderr_head=head_line(completed.stderr),
            timed_out=False,
        )

    return make_check(
        "runtime-smoke-script-parse",
        "block",
        "Entry script parse failed in the selected environment.",
        evidence=[head_line(completed.stderr) or head_line(completed.stdout) or "py_compile failed"],
        category_hint="workspace",
        remediable=False,
        remediation_owner="workspace",
        revalidation_scope=["workspace-assets", "runtime-smoke"],
        command_preview=format_command(command),
        exit_code=completed.returncode,
        stdout_head=head_line(completed.stdout),
        stderr_head=head_line(completed.stderr),
        timed_out=False,
    )


def build_smoke_command(smoke_cmd: str, python_path: str) -> List[str]:
    parts = shlex.split(smoke_cmd)
    if not parts:
        return []
    if parts[0] in {"python", "python3"}:
        parts[0] = python_path
    return parts


def run_explicit_task_smoke(
    target: dict,
    python_path: Optional[str],
    root: Path,
    probe_env: Optional[Dict[str, str]],
    timeout_seconds: int,
) -> dict:
    smoke_cmd = target.get("task_smoke_cmd")
    if not smoke_cmd:
        return make_check("task-smoke-executed", "skipped", "No explicit task smoke command was requested.")
    if not python_path:
        return make_check("task-smoke-executed", "skipped", "Explicit task smoke is skipped because selected Python is unavailable.")
    command = build_smoke_command(str(smoke_cmd), python_path)
    if not command:
        return make_check(
            "task-smoke-executed",
            "block",
            "Explicit task smoke command is empty after parsing.",
            evidence=[f"task_smoke_cmd={smoke_cmd}"],
            category_hint="workspace",
            remediable=False,
            remediation_owner="workspace",
            revalidation_scope=["task-smoke", "target"],
        )
    try:
        completed = subprocess.run(
            command,
            cwd=str(root),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            env=probe_env,
        )
    except subprocess.TimeoutExpired as exc:
        return make_check(
            "task-smoke-executed",
            "block",
            "Explicit task smoke command timed out.",
            evidence=[f"command={format_command(command)}"],
            category_hint="workspace",
            remediable=False,
            remediation_owner="workspace",
            revalidation_scope=["task-smoke", "target"],
            timed_out=True,
            stdout_head=head_line(exc.stdout),
            stderr_head=head_line(exc.stderr),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return make_check(
            "task-smoke-executed",
            "block",
            "Explicit task smoke command failed to start.",
            evidence=[f"command={format_command(command)}", f"error={exc}"],
            category_hint="workspace",
            remediable=False,
            remediation_owner="workspace",
            revalidation_scope=["task-smoke", "target"],
        )

    if completed.returncode == 0:
        return make_check(
            "task-smoke-executed",
            "ok",
            "Explicit task smoke command completed successfully.",
            evidence=[f"command={format_command(command)}"],
            exit_code=completed.returncode,
            stdout_head=head_line(completed.stdout),
            stderr_head=head_line(completed.stderr),
            timed_out=False,
        )

    return make_check(
        "task-smoke-executed",
        "block",
        "Explicit task smoke command failed.",
        evidence=[head_line(completed.stderr) or head_line(completed.stdout) or "task smoke failed"],
        category_hint="workspace",
        remediable=False,
        remediation_owner="workspace",
        revalidation_scope=["task-smoke", "target"],
        exit_code=completed.returncode,
        stdout_head=head_line(completed.stdout),
        stderr_head=head_line(completed.stderr),
        timed_out=False,
    )


def normalize_hf_endpoint(value: Optional[str]) -> str:
    endpoint = (value or DEFAULT_HF_ENDPOINT).strip()
    if "://" not in endpoint:
        endpoint = f"https://{endpoint}"
    return endpoint.rstrip("/")


def resolve_hf_cache_layout(root: Path) -> dict:
    explicit_hub = os.environ.get("HUGGINGFACE_HUB_CACHE")
    explicit_datasets = os.environ.get("HF_DATASETS_CACHE")
    explicit_hf_home = os.environ.get("HF_HOME")

    if explicit_hub or explicit_datasets:
        hub_cache = Path(explicit_hub or (root / "huggingface-cache" / "hub")).resolve()
        datasets_cache = Path(explicit_datasets or (root / "huggingface-cache" / "datasets")).resolve()
        hf_home = Path(explicit_hf_home).resolve() if explicit_hf_home else None
        source = "explicit_cache_env"
    elif explicit_hf_home:
        hf_home = Path(explicit_hf_home).resolve()
        hub_cache = (hf_home / "hub").resolve()
        datasets_cache = (hf_home / "datasets").resolve()
        source = "explicit_hf_home"
    else:
        hf_home = (root / "huggingface-cache").resolve()
        hub_cache = (hf_home / "hub").resolve()
        datasets_cache = (hf_home / "datasets").resolve()
        source = "working_dir_default"

    return {
        "source": source,
        "hf_home": str(hf_home) if hf_home else None,
        "hub_cache": str(hub_cache),
        "datasets_cache": str(datasets_cache),
        "hub_cache_writable": hub_cache.parent.exists() or hub_cache.parent.parent.exists(),
        "datasets_cache_writable": datasets_cache.parent.exists() or datasets_cache.parent.parent.exists(),
    }


def probe_hf_endpoint(endpoint: str) -> Tuple[bool, Optional[str]]:
    parsed = urlparse(endpoint)
    if not parsed.scheme or not parsed.netloc:
        return False, "HF endpoint is missing a host"

    probe_urls = [
        endpoint,
        urljoin(endpoint + "/", "api/models/Qwen/Qwen3-0.6B"),
    ]
    errors: List[str] = []
    for attempt in range(3):
        for probe_url in probe_urls:
            try:
                request = Request(probe_url, method="HEAD", headers={"User-Agent": "readiness-agent/0.1"})
                with urlopen(request, timeout=5) as response:
                    status = getattr(response, "status", None) or response.getcode()
                    if status and int(status) < 500:
                        return True, None
                    errors.append(f"attempt {attempt + 1} {probe_url}: unexpected status {status}")
            except Exception as exc:
                errors.append(f"attempt {attempt + 1} {probe_url}: {exc}")
    if errors:
        return False, "; ".join(errors[-3:])
    return False, "HF endpoint probe did not return a successful HTTP response"


def build_remote_assets(root: Path, target: dict) -> dict:
    assets = {}
    if target.get("model_hub_id"):
        assets["model_path"] = {
            "repo_id": target.get("model_hub_id"),
            "local_path": target.get("model_path") or str((root / "model").resolve()),
        }
    if target.get("dataset_hub_id"):
        assets["dataset_path"] = {
            "repo_id": target.get("dataset_hub_id"),
            "split": target.get("dataset_split") or "train",
            "local_path": target.get("dataset_path") or str((root / "dataset").resolve()),
        }
    endpoint = normalize_hf_endpoint(os.environ.get("HF_ENDPOINT"))
    reachable = None
    error = None
    if assets:
        reachable, error = probe_hf_endpoint(endpoint)
    return {
        "assets": assets,
        "hf_endpoint": endpoint,
        "hf_endpoint_source": "env" if os.environ.get("HF_ENDPOINT") else "default",
        "endpoint_reachable": reachable,
        "endpoint_error": error,
        "cache_layout": resolve_hf_cache_layout(root),
    }


def framework_package_specs(framework_path: Optional[str], compatibility: dict) -> List[str]:
    if compatibility.get("status") == "resolved" and compatibility.get("package_specs"):
        return [str(item) for item in compatibility["package_specs"]]
    if framework_path == "mindspore":
        return ["mindspore"]
    if framework_path == "pta":
        return ["torch", "torch_npu"]
    if framework_path == "mixed":
        return ["mindspore", "torch", "torch_npu"]
    return []


def build_workspace_asset_states(root: Path, target: dict, remote_assets: dict) -> dict:
    entry_script = resolve_optional_path(target.get("entry_script"), root)
    model_path = resolve_optional_path(target.get("model_path"), root)
    dataset_path = resolve_optional_path(target.get("dataset_path"), root)
    checkpoint_path = resolve_optional_path(target.get("checkpoint_path"), root)
    recipe_available = bool(target.get("example_recipe_id"))
    remote_map = remote_assets.get("assets") or {}
    remote_reachable = bool(remote_assets.get("endpoint_reachable"))

    def asset_state(path: Optional[Path], required: bool, remote_key: Optional[str] = None) -> dict:
        exists = bool(path and path.exists())
        remote_available = bool(remote_key and remote_map.get(remote_key) and remote_reachable)
        return {
            "path": str(path) if path else None,
            "required": required,
            "exists": exists,
            "remote_available": remote_available,
            "satisfied": exists or remote_available or not required,
        }

    return {
        "entry_script": {
            "path": str(entry_script) if entry_script else None,
            "required": True,
            "exists": bool(entry_script and entry_script.exists()),
            "recipe_available": recipe_available,
            "satisfied": bool(entry_script and entry_script.exists()),
        },
        "model_path": asset_state(model_path, True, "model_path"),
        "dataset_path": asset_state(dataset_path, target.get("target_type") == "training", "dataset_path"),
        "checkpoint_path": asset_state(checkpoint_path, False, None),
    }


def build_dependency_closure(root: Path, target: dict, args: object) -> dict:
    selection = resolve_selected_python(root, getattr(args, "selected_python", None), None)
    system_layer = detect_ascend_runtime({"cann_path": target.get("cann_path")})
    system_layer.update(
        detect_cann_version(
            target.get("cann_path"),
            system_layer.get("ascend_env_script_path"),
        )
    )
    probe_env, probe_source, probe_error = resolve_runtime_environment(system_layer)
    system_layer["probe_env_source"] = probe_source
    system_layer["probe_env_error"] = probe_error

    python_path = selection.get("selected_python")
    python_version = selection.get("python_version")
    compatibility = resolve_framework_compatibility(target.get("framework_path"), system_layer.get("cann_version"), python_version)
    required_framework_imports = FRAMEWORK_IMPORTS.get(target.get("framework_path") or "", [])
    framework_imports, framework_probe_error = probe_imports(required_framework_imports, python_path, probe_env)
    installed_versions, version_errors, version_probe_error = probe_package_versions(required_framework_imports, python_path, probe_env)
    installed_compatibility = assess_installed_framework_compatibility(
        target.get("framework_path"),
        system_layer.get("cann_version"),
        python_version,
        installed_versions,
    )
    framework_smoke = probe_framework_smoke(target.get("framework_path"), python_path, probe_env)

    runtime_imports = [
        name for name in extract_runtime_imports(resolve_optional_path(target.get("entry_script"), root))
        if name not in required_framework_imports
    ]
    runtime_import_probes, runtime_probe_error = probe_imports(runtime_imports, python_path, probe_env)
    remote_assets = build_remote_assets(root, target)
    workspace_assets = build_workspace_asset_states(root, target, remote_assets)

    return {
        "working_dir": str(root),
        "target_type": target.get("target_type"),
        "layers": {
            "system": system_layer,
            "python_environment": {
                "selection_status": selection.get("selection_status"),
                "selection_source": selection.get("selection_source"),
                "selection_reason": selection.get("selection_reason"),
                "selected_env_root": selection.get("selected_env_root"),
                "probe_python_path": selection.get("selected_python"),
                "python_version": selection.get("python_version"),
                "helper_python_compatible": selection.get("helper_python_compatible"),
            },
            "framework": {
                "framework_path": target.get("framework_path"),
                "required_packages": required_framework_imports,
                "recommended_package_specs": framework_package_specs(target.get("framework_path"), compatibility),
                "import_probes": framework_imports,
                "import_probe_error": framework_probe_error,
                "installed_package_versions": installed_versions,
                "version_probe_errors": version_errors,
                "version_probe_error": version_probe_error,
                "compatibility": compatibility,
                "installed_compatibility": installed_compatibility,
                "framework_smoke": framework_smoke,
            },
            "runtime_dependencies": {
                "required_imports": runtime_imports,
                "import_probes": runtime_import_probes,
                "import_probe_error": runtime_probe_error,
            },
            "remote_assets": remote_assets,
            "workspace_assets": workspace_assets,
        },
    }


def collect_checks(target: dict, closure: dict, timeout_seconds: int) -> List[dict]:
    root = Path(target["working_dir"]).resolve()
    layers = closure.get("layers", {})
    system_layer = layers.get("system", {})
    python_layer = layers.get("python_environment", {})
    framework_layer = layers.get("framework", {})
    runtime_layer = layers.get("runtime_dependencies", {})
    remote_assets = layers.get("remote_assets", {})
    workspace_assets = layers.get("workspace_assets", {})

    checks: List[dict] = []
    checks.append(
        make_check(
            "target-stability",
            "ok" if target.get("target_stable") else "warn",
            "Execution target is stable." if target.get("target_stable") else "Execution target was inferred with limited confidence.",
            evidence=target.get("target_evidence") or [],
            category_hint="workspace",
            remediable=False,
        )
    )

    framework_path = target.get("framework_path")
    checks.append(
        make_check(
            "framework-selection",
            "ok" if framework_path else "warn",
            f"Framework path resolved to {framework_path}." if framework_path else "Framework path is still unresolved.",
            evidence=target.get("framework_evidence") or [],
            category_hint="framework" if framework_path else "workspace",
            remediable=not bool(framework_path),
            remediation_owner="readiness-agent" if not framework_path else None,
            revalidation_scope=["framework"] if not framework_path else None,
        )
    )

    if python_layer.get("selection_status") == "selected":
        checks.append(
            make_check(
                "python-selected-env",
                "ok",
                "A workspace-local Python environment is selected.",
                evidence=[f"selected_env_root={python_layer.get('selected_env_root')}"] if python_layer.get("selected_env_root") else [],
                category_hint="env",
                remediable=True,
                remediation_owner="readiness-agent",
                revalidation_scope=["python-environment", "framework"],
            )
        )
        checks.append(
            make_check(
                "python-selected-python",
                "ok",
                "Selected Python is usable for readiness checks.",
                evidence=[f"selected_python={python_layer.get('probe_python_path')}"] if python_layer.get("probe_python_path") else [],
                category_hint="env",
                remediable=True,
                remediation_owner="readiness-agent",
                revalidation_scope=["python-environment", "framework"],
            )
        )
    else:
        checks.append(
            make_check(
                "python-selected-env",
                "block",
                "No usable workspace-local Python environment is selected.",
                evidence=[python_layer.get("selection_reason")] if python_layer.get("selection_reason") else [],
                category_hint="env",
                remediable=True,
                remediation_owner="readiness-agent",
                revalidation_scope=["python-environment", "framework", "runtime-smoke"],
            )
        )

    entry_asset = workspace_assets.get("entry_script", {})
    if entry_asset.get("exists"):
        checks.append(
            make_check(
                "workspace-entry-script",
                "ok",
                "Entry script is present.",
                evidence=[f"path={entry_asset.get('path')}"],
                category_hint="asset",
                remediable=False,
            )
        )
    elif entry_asset.get("recipe_available"):
        checks.append(
            make_check(
                "workspace-entry-script",
                "block",
                "Entry script is missing, but a bundled example can be scaffolded in fix mode.",
                evidence=[f"target_recipe={target.get('example_recipe_id')}"],
                category_hint="asset",
                remediable=True,
                remediation_owner="readiness-agent",
                revalidation_scope=["workspace-assets", "runtime-smoke"],
            )
        )
    else:
        checks.append(
            make_check(
                "workspace-entry-script",
                "block",
                "Entry script is missing from the workspace.",
                category_hint="workspace",
                remediable=False,
                remediation_owner="workspace",
                revalidation_scope=["workspace-assets", "runtime-smoke"],
            )
        )

    for key in ("model_path", "dataset_path", "checkpoint_path"):
        asset = workspace_assets.get(key, {})
        if not asset or not asset.get("required", False):
            continue
        label = key.replace("_", "-")
        if asset.get("exists"):
            checks.append(make_check(f"workspace-{label}", "ok", f"{key} is present.", evidence=[f"path={asset.get('path')}"], category_hint="asset"))
        elif asset.get("remote_available"):
            checks.append(
                make_check(
                    f"workspace-{label}",
                    "ok",
                    f"{key} is resolvable from the configured Hugging Face source.",
                    evidence=[f"repo_id={(remote_assets.get('assets') or {}).get(key, {}).get('repo_id')}"],
                    category_hint="asset",
                )
            )
        else:
            remote_meta = (remote_assets.get("assets") or {}).get(key)
            remediable = bool(remote_meta and target.get("allow_network"))
            checks.append(
                make_check(
                    f"workspace-{label}",
                    "block",
                    f"{key} is missing.",
                    evidence=[remote_assets.get("endpoint_error")] if remote_assets.get("endpoint_error") else [],
                    category_hint="asset" if remediable else "workspace",
                    remediable=remediable,
                    remediation_owner="readiness-agent" if remediable else "workspace",
                    revalidation_scope=["workspace-assets"],
                )
            )

    if framework_path:
        import_probes = framework_layer.get("import_probes") or {}
        if import_probes and all(import_probes.values()):
            checks.append(
                make_check(
                    "framework-importability",
                    "ok",
                    "Required framework packages are importable.",
                    evidence=[f"{name}=ok" for name in sorted(import_probes)],
                    category_hint="framework",
                )
            )
        else:
            missing = [name for name, ok in sorted(import_probes.items()) if not ok]
            checks.append(
                make_check(
                    "framework-importability",
                    "block",
                    "Required framework packages are missing from the selected environment.",
                    evidence=missing or ([framework_layer.get("import_probe_error")] if framework_layer.get("import_probe_error") else []),
                    category_hint="framework",
                    remediable=True,
                    remediation_owner="readiness-agent",
                    revalidation_scope=["framework", "runtime-smoke"],
                )
            )

        compatibility = framework_layer.get("installed_compatibility") or {}
        compatibility_status = compatibility.get("status")
        if compatibility_status == "compatible":
            checks.append(make_check("framework-compatibility", "ok", "Installed framework versions match the local Ascend compatibility table."))
        elif compatibility_status == "incompatible":
            checks.append(
                make_check(
                    "framework-compatibility",
                    "block",
                    compatibility.get("reason") or "Installed framework versions are incompatible with the local Ascend compatibility table.",
                    category_hint="framework",
                    remediable=True,
                    remediation_owner="readiness-agent",
                    revalidation_scope=["framework"],
                )
            )
        elif compatibility.get("reference_status") not in {None, "unsupported"} and has_ascend_runtime_evidence(system_layer):
            checks.append(
                make_check(
                    "framework-compatibility",
                    "warn",
                    compatibility.get("reason") or "Framework compatibility could not be fully confirmed from the local Ascend table.",
                    category_hint="framework",
                    remediable=False,
                )
            )

        framework_smoke = framework_layer.get("framework_smoke") or {}
        if framework_smoke.get("status") == "passed":
            checks.append(
                make_check(
                    "runtime-smoke-framework",
                    "ok",
                    "Framework runtime smoke passed in the selected environment.",
                    evidence=framework_smoke.get("details") or [],
                    category_hint="framework",
                )
            )
        elif framework_smoke.get("status") == "failed":
            checks.append(
                make_check(
                    "runtime-smoke-framework",
                    "block",
                    "Framework runtime smoke failed in the selected environment.",
                    evidence=[framework_smoke.get("error")] if framework_smoke.get("error") else [],
                    category_hint="framework",
                    remediable=True,
                    remediation_owner="readiness-agent",
                    revalidation_scope=["framework", "runtime-smoke"],
                )
            )

    script_parse = run_script_parse_smoke(
        resolve_optional_path(target.get("entry_script"), root),
        python_layer.get("probe_python_path"),
        root,
        dict(os.environ) if system_layer.get("probe_env_source") == "current_environment" else None,
    )
    checks.append(script_parse)

    runtime_status = "warn"
    runtime_summary = "Runtime smoke did not gather enough evidence to certify this workspace yet."
    runtime_evidence: List[str] = []
    framework_smoke_check = next((item for item in checks if item.get("id") == "runtime-smoke-framework"), None)
    script_parse_status = str(script_parse.get("status") or "").lower()
    script_parse_summary = str(script_parse.get("summary") or "")

    if framework_path and framework_smoke_check and framework_smoke_check.get("status") == "block":
        runtime_status = "block"
        runtime_summary = "Runtime smoke failed because framework smoke did not pass."
    elif framework_path and not framework_smoke_check:
        runtime_status = "block"
        runtime_summary = "Runtime smoke could not verify framework imports in the selected workspace environment."
    elif script_parse_status == "block":
        runtime_status = "block"
        runtime_summary = "Runtime smoke failed because the entry script does not parse."
    elif script_parse_status == "skipped":
        if script_parse_summary in {
            "Selected Python is unavailable for script parsing.",
            "Entry script is unavailable for script parsing.",
        }:
            runtime_status = "block"
            runtime_summary = "Runtime smoke could not run because entry script parsing prerequisites are unresolved."
        else:
            runtime_status = "warn"
            runtime_summary = "Runtime smoke could not parse the entry script in this workspace."
    elif framework_path:
        runtime_status = "ok"
        runtime_summary = "Runtime smoke passed."
    else:
        runtime_status = "ok"
        runtime_summary = "Runtime smoke passed."
        runtime_evidence.append("framework unresolved; runtime smoke is based on entry script parsing only")
    if framework_path and system_layer.get("cann_version") and framework_layer.get("installed_compatibility", {}).get("status") == "incompatible":
        runtime_status = "block"
        runtime_summary = "Runtime smoke failed because framework compatibility is incompatible with the detected CANN version."
    checks.append(
        make_check(
            "runtime-smoke",
            runtime_status,
            runtime_summary,
            evidence=runtime_evidence,
            category_hint="framework" if runtime_status == "block" else None,
            remediable=True if runtime_status == "block" and framework_path else False,
            remediation_owner="readiness-agent" if runtime_status == "block" and framework_path else None,
            revalidation_scope=["runtime-smoke", "framework"] if runtime_status == "block" and framework_path else None,
        )
    )

    checks.append(
        run_explicit_task_smoke(
            target,
            python_layer.get("probe_python_path"),
            root,
            dict(os.environ) if system_layer.get("probe_env_source") == "current_environment" else None,
            timeout_seconds,
        )
    )

    if runtime_layer.get("required_imports"):
        missing_runtime = [name for name, ok in sorted((runtime_layer.get("import_probes") or {}).items()) if not ok]
        if missing_runtime:
            checks.append(
                make_check(
                    "runtime-dependencies",
                    "block",
                    "Runtime dependencies imported by the entry script are missing.",
                    evidence=missing_runtime,
                    category_hint="framework",
                    remediable=True,
                    remediation_owner="readiness-agent",
                    revalidation_scope=["framework", "runtime-smoke"],
                )
            )
        else:
            checks.append(
                make_check(
                    "runtime-dependencies",
                    "ok",
                    "Runtime dependencies imported by the entry script are available.",
                    evidence=runtime_layer.get("required_imports"),
                    category_hint="framework",
                )
            )

    return checks


def normalize_findings(checks: List[dict]) -> dict:
    category_map = {
        "env": ("env_remediable", True, "readiness-agent"),
        "framework": ("framework_remediable", True, "readiness-agent"),
        "asset": ("asset_remediable", True, "readiness-agent"),
        "workspace": ("workspace_manual", False, "workspace"),
        "system": ("system_fatal", False, "system"),
    }
    blockers_detailed = []
    warnings_detailed = []

    for item in checks:
        status = str(item.get("status") or "").lower()
        if status not in {"block", "warn"}:
            continue
        hint = str(item.get("category_hint") or "workspace")
        mapped_category, default_remediable, default_owner = category_map.get(hint, ("unknown", False, "workspace"))
        normalized = {
            "id": item.get("id"),
            "summary": item.get("summary"),
            "evidence": item.get("evidence") or [],
            "category": mapped_category,
            "severity": item.get("severity") or ("high" if status == "block" else "medium"),
            "remediable": bool(item.get("remediable")) if item.get("remediable") is not None else default_remediable,
            "remediation_owner": item.get("remediation_owner") or default_owner,
            "revalidation_scope": item.get("revalidation_scope") or [],
        }
        if status == "block":
            blockers_detailed.append(normalized)
        else:
            warnings_detailed.append(normalized)

    return {
        "blockers": [item["summary"] for item in blockers_detailed],
        "warnings": [item["summary"] for item in warnings_detailed],
        "blockers_detailed": blockers_detailed,
        "warnings_detailed": warnings_detailed,
    }


def has_ascend_runtime_evidence(system_layer: dict) -> bool:
    return any(
        [
            system_layer.get("cann_path_input"),
            system_layer.get("cann_version"),
            system_layer.get("ascend_env_active"),
            system_layer.get("ascend_env_script_present"),
            system_layer.get("device_paths_present"),
        ]
    )


def package_base_name(package_name: str) -> str:
    token = package_name.strip()
    token = token.split("[", 1)[0]
    for marker in ("==", ">=", "<=", "~=", "!=", ">", "<"):
        if marker in token:
            token = token.split(marker, 1)[0]
            break
    return token.strip().replace("_", "-").lower()


def default_uv_bin_dir() -> Path:
    return Path.home() / ".local" / "bin"


def resolve_uv_executable() -> Optional[Path]:
    uv_path = shutil.which("uv")
    if uv_path:
        return Path(uv_path)

    candidates = [default_uv_bin_dir() / "uv"]
    if os.name == "nt":
        candidates.extend([default_uv_bin_dir() / "uv.exe", default_uv_bin_dir() / "uv.cmd"])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def ensure_uv_available() -> Tuple[bool, str, Optional[Path]]:
    uv_path = resolve_uv_executable()
    if uv_path:
        return True, "uv is already available", uv_path

    command = [sys.executable, "-m", "pip", "install", "--user", "uv"]
    try:
        subprocess.run(command, check=True, text=True, capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc), None

    uv_path = resolve_uv_executable()
    if uv_path:
        return True, "uv installed into the user environment", uv_path
    return False, "uv installation completed but the executable is still unavailable", None


def preferred_pip_index_urls() -> List[str]:
    explicit_index = os.environ.get("READINESS_PIP_INDEX_URL") or os.environ.get("PIP_INDEX_URL")
    if explicit_index:
        ordered = [explicit_index]
        ordered.extend(url for url in SUPPORTED_PIP_INDEX_URLS if url != explicit_index)
        return ordered
    return list(SUPPORTED_PIP_INDEX_URLS)


def run_install_command(command: List[str]) -> Tuple[bool, str]:
    try:
        subprocess.run(command, check=True, text=True, capture_output=True, timeout=300)
    except subprocess.CalledProcessError as exc:
        return False, exc.stderr.strip() or exc.stdout.strip() or "install command failed"
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    return True, ""


def build_uv_pip_install_command(uv_path: Path, python_path: Path, packages: List[str], index_url: Optional[str] = None) -> List[str]:
    command = [str(uv_path), "pip", "install", "--python", str(python_path)]
    if index_url:
        command.extend(["--index-url", index_url])
    command.extend(packages)
    return command


def install_packages(python_path: Path, packages: List[str]) -> Tuple[bool, str]:
    if not packages:
        return True, ""
    uv_ok, message, uv_path = ensure_uv_available()
    if not uv_ok or not uv_path:
        return False, message

    last_error = "install command failed"
    for index_url in preferred_pip_index_urls():
        command = build_uv_pip_install_command(uv_path, python_path, packages, index_url=index_url)
        ok, error = run_install_command(command)
        if ok:
            return True, f"installed packages via {index_url}"
        last_error = error
    return False, last_error


def ensure_workspace_env_actions(actions: List[dict], root: Path) -> None:
    if not any(item.get("action_type") == "install_uv" for item in actions) and not resolve_uv_executable():
        actions.append(
            {
                "id": "install-uv",
                "action_type": "install_uv",
                "summary": "Install uv into the user environment.",
                "revalidation_scope": ["python-environment"],
            }
        )

    if any(item.get("action_type") == "create_or_select_env" for item in actions):
        return

    actions.append(
        {
            "id": "create-workspace-env",
            "action_type": "create_or_select_env",
            "summary": "Create a workspace-local Python environment at .venv.",
            "env_root": str((root / ".venv").resolve()),
            "revalidation_scope": ["python-environment", "framework", "runtime-smoke"],
        }
    )


def scaffold_example_entry_script(template_path: Path, destination_path: Path) -> Tuple[bool, str]:
    try:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_text(template_path.read_text(encoding="utf-8"), encoding="utf-8")
    except OSError as exc:
        return False, str(exc)
    return True, f"scaffolded {destination_path.name} from bundled example"


def download_huggingface_model_asset(python_path: Path, repo_id: str, local_path: Path, remote_assets: dict) -> Tuple[bool, str]:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["HF_ENDPOINT"] = remote_assets.get("hf_endpoint") or DEFAULT_HF_ENDPOINT
    cache_layout = remote_assets.get("cache_layout") or {}
    if cache_layout.get("hf_home") and not env.get("HF_HOME"):
        env["HF_HOME"] = str(cache_layout["hf_home"])
    code = (
        "from huggingface_hub import snapshot_download; "
        "import sys; "
        "snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2], local_dir_use_symlinks=False)"
    )
    try:
        subprocess.run([str(python_path), "-c", code, repo_id, str(local_path)], check=True, text=True, capture_output=True, env=env, timeout=600)
    except subprocess.CalledProcessError as exc:
        return False, exc.stderr.strip() or exc.stdout.strip() or "model download failed"
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    return True, f"downloaded model asset {repo_id}"


def download_huggingface_dataset_asset(
    python_path: Path,
    repo_id: str,
    split: Optional[str],
    local_path: Path,
    remote_assets: dict,
) -> Tuple[bool, str]:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["HF_ENDPOINT"] = remote_assets.get("hf_endpoint") or DEFAULT_HF_ENDPOINT
    cache_layout = remote_assets.get("cache_layout") or {}
    if cache_layout.get("hf_home") and not env.get("HF_HOME"):
        env["HF_HOME"] = str(cache_layout["hf_home"])
    code = (
        "from datasets import load_dataset; "
        "import sys; "
        "dataset = load_dataset(sys.argv[1], split=sys.argv[2]); "
        "dataset.save_to_disk(sys.argv[3])"
    )
    split_token = split or "train"
    try:
        subprocess.run([str(python_path), "-c", code, repo_id, split_token, str(local_path)], check=True, text=True, capture_output=True, env=env, timeout=600)
    except subprocess.CalledProcessError as exc:
        return False, exc.stderr.strip() or exc.stdout.strip() or "dataset download failed"
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    return True, f"downloaded dataset asset {repo_id}:{split_token}"


def build_fix_actions(target: dict, closure: dict, normalized: dict, allow_network: bool) -> List[dict]:
    actions: List[dict] = []
    root = Path(target["working_dir"]).resolve()
    layers = closure.get("layers", {})
    python_layer = layers.get("python_environment", {})
    framework_layer = layers.get("framework", {})
    runtime_layer = layers.get("runtime_dependencies", {})
    workspace_assets = layers.get("workspace_assets", {})
    remote_assets = layers.get("remote_assets", {})

    if python_layer.get("selection_status") != "selected":
        ensure_workspace_env_actions(actions, root)

    missing_framework_imports = [name for name, ok in sorted((framework_layer.get("import_probes") or {}).items()) if not ok]
    install_python = selected_python_for_execution(root, target, closure)
    framework_packages = []
    if framework_layer.get("framework_path"):
        if missing_framework_imports:
            framework_packages = framework_layer.get("recommended_package_specs") or missing_framework_imports
        elif not install_python:
            framework_packages = framework_layer.get("recommended_package_specs") or (framework_layer.get("required_packages") or [])
    if framework_packages:
        if not install_python:
            ensure_workspace_env_actions(actions, root)
        actions.append(
            {
                "id": "install-framework-packages",
                "action_type": "install_framework_packages",
                "summary": "Install framework packages required by the selected target.",
                "package_specs": framework_packages,
                "revalidation_scope": ["framework", "runtime-smoke"],
            }
        )

    missing_runtime_imports = [name for name, ok in sorted((runtime_layer.get("import_probes") or {}).items()) if not ok]
    runtime_packages = missing_runtime_imports or ([] if install_python else sorted(runtime_layer.get("required_imports") or []))
    if runtime_packages:
        if not install_python:
            ensure_workspace_env_actions(actions, root)
        actions.append(
            {
                "id": "install-runtime-dependencies",
                "action_type": "install_runtime_dependencies",
                "summary": "Install runtime dependencies imported by the entry script.",
                "package_specs": runtime_packages,
                "revalidation_scope": ["framework", "runtime-smoke"],
            }
        )

    entry_asset = layers.get("workspace_assets", {}).get("entry_script", {})
    if not entry_asset.get("exists") and target.get("example_template_path"):
        actions.append(
            {
                "id": "scaffold-example-entry",
                "action_type": "scaffold_example_entry",
                "summary": "Scaffold a bundled example entry script into the workspace.",
                "template_path": target.get("example_template_path"),
                "destination_path": target.get("entry_script"),
                "revalidation_scope": ["workspace-assets", "runtime-smoke"],
            }
        )

    model_asset = workspace_assets.get("model_path", {})
    if (
        not model_asset.get("satisfied")
        and allow_network
        and (remote_assets.get("assets") or {}).get("model_path")
    ):
        actions.append(
            {
                "id": "download-model-asset",
                "action_type": "download_model_asset",
                "summary": "Download the requested model asset into the workspace.",
                "repo_id": remote_assets["assets"]["model_path"]["repo_id"],
                "destination_path": remote_assets["assets"]["model_path"]["local_path"],
                "revalidation_scope": ["workspace-assets"],
            }
        )

    dataset_asset = workspace_assets.get("dataset_path", {})
    if (
        not dataset_asset.get("satisfied")
        and allow_network
        and (remote_assets.get("assets") or {}).get("dataset_path")
    ):
        actions.append(
            {
                "id": "download-dataset-asset",
                "action_type": "download_dataset_asset",
                "summary": "Download the requested dataset asset into the workspace.",
                "repo_id": remote_assets["assets"]["dataset_path"]["repo_id"],
                "split": remote_assets["assets"]["dataset_path"].get("split"),
                "destination_path": remote_assets["assets"]["dataset_path"]["local_path"],
                "revalidation_scope": ["workspace-assets"],
            }
        )

    return actions


def selected_workspace_python(root: Path, closure: dict) -> Optional[Path]:
    workspace_uv_python = python_in_env(root / ".venv")
    if workspace_uv_python:
        return workspace_uv_python

    python_layer = closure.get("layers", {}).get("python_environment", {})
    selected_env_root = python_layer.get("selected_env_root")
    probe_python_path = python_layer.get("probe_python_path")
    if not selected_env_root:
        return None

    env_root = Path(selected_env_root)
    if not path_is_within(env_root, root):
        return None

    if probe_python_path:
        probe_python = Path(probe_python_path)
        if probe_python.exists() and path_is_within(probe_python, env_root):
            return probe_python

    return python_in_env(env_root)


def selected_python_for_execution(root: Path, target: dict, closure: dict) -> Optional[Path]:
    workspace_python = selected_workspace_python(root, closure)
    if workspace_python:
        return workspace_python

    explicit = resolve_optional_path(target.get("selected_python"), root)
    if explicit and path_is_within(explicit, root):
        return explicit
    return None


def execute_fix_actions(target: dict, closure: dict, actions: List[dict], execute: bool) -> dict:
    root = Path(target["working_dir"]).resolve()
    results = []
    executed_actions = []
    failed_actions = []
    needs_revalidation: List[str] = []

    if not execute:
        return {
            "execute": False,
            "planned_actions": actions,
            "results": results,
            "executed_actions": executed_actions,
            "failed_actions": failed_actions,
            "needs_revalidation": needs_revalidation,
        }

    for action in actions:
        action_id = action["id"]
        action_type = action["action_type"]
        ok = False
        message = ""

        if action_type == "install_uv":
            ok, message, _ = ensure_uv_available()
        elif action_type == "create_or_select_env":
            uv_ok, uv_message, uv_path = ensure_uv_available()
            if not uv_ok or not uv_path:
                ok = False
                message = uv_message
            else:
                env_root = Path(action["env_root"])
                command = [str(uv_path), "venv", str(env_root)]
                ok, error = run_install_command(command)
                message = "workspace environment created" if ok else error
        elif action_type in {"install_framework_packages", "install_runtime_dependencies"}:
            python_path = selected_python_for_execution(root, target, closure)
            if not python_path:
                ok = False
                message = "selected python is unavailable for package installation"
            else:
                ok, message = install_packages(python_path, [str(item) for item in action.get("package_specs", [])])
        elif action_type == "scaffold_example_entry":
            ok, message = scaffold_example_entry_script(Path(action["template_path"]), Path(action["destination_path"]))
        elif action_type == "download_model_asset":
            python_path = selected_python_for_execution(root, target, closure)
            if not python_path:
                ok = False
                message = "selected python is unavailable for model download"
            else:
                ok, message = download_huggingface_model_asset(
                    python_path,
                    str(action["repo_id"]),
                    Path(action["destination_path"]),
                    closure.get("layers", {}).get("remote_assets", {}),
                )
        elif action_type == "download_dataset_asset":
            python_path = selected_python_for_execution(root, target, closure)
            if not python_path:
                ok = False
                message = "selected python is unavailable for dataset download"
            else:
                ok, message = download_huggingface_dataset_asset(
                    python_path,
                    str(action["repo_id"]),
                    action.get("split"),
                    Path(action["destination_path"]),
                    closure.get("layers", {}).get("remote_assets", {}),
                )
        else:
            ok = False
            message = f"unsupported action type: {action_type}"

        results.append({"action_id": action_id, "status": "executed" if ok else "failed", "message": message})
        if ok:
            executed_actions.append(action_id)
            for scope in action.get("revalidation_scope") or []:
                if scope not in needs_revalidation:
                    needs_revalidation.append(scope)
        else:
            failed_actions.append(action_id)

    return {
        "execute": True,
        "planned_actions": actions,
        "results": results,
        "executed_actions": executed_actions,
        "failed_actions": failed_actions,
        "needs_revalidation": needs_revalidation,
    }


def build_readiness_env_payload(root: Path, target: dict, closure: dict) -> dict:
    layers = closure.get("layers", {})
    system_layer = layers.get("system", {})
    python_layer = layers.get("python_environment", {})
    remote_assets = layers.get("remote_assets", {})
    cache_layout = remote_assets.get("cache_layout") or {}

    return {
        "working_dir": str(root),
        "ascend_env_script": system_layer.get("ascend_env_script_path"),
        "hf_endpoint": remote_assets.get("hf_endpoint") or DEFAULT_HF_ENDPOINT,
        "selected_python": python_layer.get("probe_python_path"),
        "selected_env_root": python_layer.get("selected_env_root"),
        "hf_home": cache_layout.get("hf_home"),
        "huggingface_hub_cache": os.environ.get("HUGGINGFACE_HUB_CACHE"),
        "hf_datasets_cache": os.environ.get("HF_DATASETS_CACHE"),
    }


def shell_export(name: str, value: Optional[str]) -> Optional[str]:
    if value in {None, ""}:
        return None
    return f"export {name}={shlex.quote(str(value))}"


def write_readiness_env_file(path: Path, root: Path, target: dict, closure: dict) -> dict:
    payload = build_readiness_env_payload(root, target, closure)
    lines = [
        "#!/usr/bin/env bash",
        "# Generated by readiness-agent. Source this file before running training or inference in this workspace.",
        "",
    ]
    if payload.get("ascend_env_script"):
        lines.append(f"source {shlex.quote(str(payload['ascend_env_script']))}")
        lines.append("")
    for line in (
        shell_export("READINESS_WORKING_DIR", payload.get("working_dir")),
        shell_export("HF_ENDPOINT", payload.get("hf_endpoint")),
        shell_export("HF_HOME", payload.get("hf_home")),
        shell_export("HUGGINGFACE_HUB_CACHE", payload.get("huggingface_hub_cache")),
        shell_export("HF_DATASETS_CACHE", payload.get("hf_datasets_cache")),
        shell_export("READINESS_SELECTED_ENV_ROOT", payload.get("selected_env_root")),
        shell_export("READINESS_SELECTED_PYTHON", payload.get("selected_python")),
    ):
        if line:
            lines.append(line)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return payload


def build_state(args: object, root: Path) -> dict:
    target = discover_execution_target(root, args)
    closure = build_dependency_closure(root, target, args)
    checks = collect_checks(target, closure, int(getattr(args, "timeout_seconds", 10) or 10))
    normalized = normalize_findings(checks)
    return {
        "target": target,
        "closure": closure,
        "checks": checks,
        "normalized": normalized,
    }
