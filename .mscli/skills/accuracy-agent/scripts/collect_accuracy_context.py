#!/usr/bin/env python3
"""Collect a minimal accuracy-context snapshot for later diagnosis."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path


PACKAGE_ALIASES = {
    "mindspore": ["mindspore"],
    "torch": ["torch"],
    "torch_npu": ["torch_npu", "torch-npu"],
}


def _read_version(name: str) -> dict[str, object]:
    dist_names = PACKAGE_ALIASES[name]
    version = None
    for dist_name in dist_names:
        try:
            version = importlib.metadata.version(dist_name)
            break
        except importlib.metadata.PackageNotFoundError:
            continue

    module_name = "torch_npu" if name == "torch_npu" else name
    spec = importlib.util.find_spec(module_name)
    return {
        "installed": version is not None or spec is not None,
        "version": version,
        "module_found": spec is not None,
    }


def _safe_runtime_probe() -> tuple[dict[str, object], list[str]]:
    runtime: dict[str, object] = {}
    notes: list[str] = []

    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover - import behavior is env-specific
        runtime["torch_import_error"] = repr(exc)
        return runtime, notes

    try:
        importlib.import_module("torch_npu")
        runtime["torch_npu_importable"] = True
    except Exception as exc:  # pragma: no cover - import behavior is env-specific
        runtime["torch_npu_importable"] = False
        runtime["torch_npu_import_error"] = repr(exc)

    if hasattr(torch, "cuda"):
        try:
            runtime["torch_cuda_available"] = bool(torch.cuda.is_available())
        except Exception as exc:  # pragma: no cover - env-specific
            runtime["torch_cuda_available_error"] = repr(exc)

    if hasattr(torch, "npu"):
        try:
            runtime["torch_npu_available"] = bool(torch.npu.is_available())
            runtime["torch_npu_device_count"] = int(torch.npu.device_count())
        except Exception as exc:  # pragma: no cover - env-specific
            runtime["torch_npu_available_error"] = repr(exc)

    if runtime.get("torch_npu_importable") is False:
        notes.append(
            "torch is importable but torch_npu is not; do not trust a PyTorch-on-Ascend baseline until that is resolved."
        )

    try:
        ms = importlib.import_module("mindspore")
        runtime["mindspore_importable"] = True
        get_context = getattr(getattr(ms, "context", None), "get_context", None)
        if callable(get_context):
            try:
                runtime["mindspore_device_target"] = get_context("device_target")
            except Exception as exc:  # pragma: no cover - env-specific
                runtime["mindspore_device_target_error"] = repr(exc)
    except Exception as exc:  # pragma: no cover - import behavior is env-specific
        runtime["mindspore_importable"] = False
        runtime["mindspore_import_error"] = repr(exc)

    return runtime, notes


def main() -> None:
    cwd = Path(os.getcwd()).resolve()
    framework_versions = {
        name: _read_version(name) for name in ("mindspore", "torch", "torch_npu")
    }
    runtime, notes = _safe_runtime_probe()
    data = {
        "working_dir": str(cwd),
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "framework_versions": framework_versions,
        "runtime": runtime,
        "env": {
            "device": {
                key: os.environ.get(key, "")
                for key in (
                    "ASCEND_VISIBLE_DEVICES",
                    "CUDA_VISIBLE_DEVICES",
                    "DEVICE_ID",
                )
            },
            "determinism": {
                key: os.environ.get(key, "")
                for key in (
                    "HCCL_DETERMINISTIC",
                    "ASCEND_LAUNCH_BLOCKING",
                    "CUBLAS_WORKSPACE_CONFIG",
                )
            },
        },
        "candidate_metric_files": sorted(
            str(p)
            for p in list(cwd.glob("**/*.json")) + list(cwd.glob("**/*.log"))
            if p.is_file()
        )[:50],
        "notes": notes,
    }
    print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
