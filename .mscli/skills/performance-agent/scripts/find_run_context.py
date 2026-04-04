#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Optional


SCRIPT_HINTS = ("train", "main", "run", "infer", "launch")
SCRIPT_SUFFIXES = {".py", ".sh"}
CONFIG_SUFFIXES = {".yaml", ".yml", ".json", ".toml", ".ini"}
CKPT_SUFFIXES = {".ckpt", ".pt", ".pth", ".safetensors"}
LOG_SUFFIXES = {".log", ".out", ".txt"}
TRACE_HINTS = (
    "prof",
    "msprof",
    "trace",
    "timeline",
    "hotspot",
    "mindstudio_profiler_output",
    "ascend",
)
IGNORE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
    "site-packages",
    "scripts",
    "references",
    "tests",
    "doc",
    "docs",
}


def recent_files(root: Path, limit: int) -> list[Path]:
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and not any(part in IGNORE_DIRS for part in path.parts)
    ]
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return files[:limit]


def classify(path: Path) -> Optional[str]:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix == ".sh":
        return "script"
    if suffix == ".py" and any(token in name for token in SCRIPT_HINTS):
        return "script"
    if any(token in name for token in TRACE_HINTS):
        return "trace"
    if suffix in CONFIG_SUFFIXES:
        return "config"
    if suffix in CKPT_SUFFIXES:
        return "checkpoint"
    if suffix in LOG_SUFFIXES:
        return "log"
    return None


def read_text(path: Path, limit: int = 12000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return ""


def detect_stack(text: str) -> Optional[str]:
    lower = text.lower()
    ms_hits = sum(token in lower for token in ("mindspore", "msrun", ".ckpt"))
    pta_hits = sum(token in lower for token in ("torch_npu", "torchrun", ".pt", ".pth"))
    if ms_hits and ms_hits >= pta_hits:
        return "ms"
    if pta_hits:
        return "pta"
    return None


def detect_workload(text: str) -> Optional[str]:
    lower = text.lower()
    training_hits = sum(token in lower for token in ("train", "epoch", "loss", "backward", "optimizer"))
    inference_hits = sum(token in lower for token in ("infer", "inference", "evaluate", "eval", "generation"))
    if training_hits and training_hits >= inference_hits:
        return "training"
    if inference_hits:
        return "inference"
    return None


def detect_scale(text: str) -> Optional[str]:
    lower = text.lower()
    if any(token in lower for token in ("rank", "world size", "hccl", "distributed", "allreduce")):
        return "distributed"
    return None


def detect_metric_focus(text: str) -> Optional[str]:
    lower = text.lower()
    scores = {
        "throughput": sum(token in lower for token in ("throughput", "samples/s", "steps/s", "fps", "img/s")),
        "latency": sum(token in lower for token in ("latency", "p95", "p99", "step time", "step_time", "ms/step")),
        "memory": sum(token in lower for token in ("memory", "peak memory", "max memory")),
    }
    focus, score = max(scores.items(), key=lambda item: item[1])
    return focus if score else None


def extract_metric_lines(text: str) -> dict[str, str]:
    token_map = {
        "throughput": ("throughput", "samples/s", "steps/s", "fps", "img/s"),
        "latency": ("latency", "p95", "p99", "step time", "step_time", "ms/step"),
        "memory": ("peak memory", "max memory", "memory"),
    }
    found: dict[str, str] = {}
    for line in text.splitlines():
        lower = line.lower()
        if not re.search(r"\d", lower):
            continue
        for key, tokens in token_map.items():
            if key in found:
                continue
            if any(token in lower for token in tokens):
                found[key] = line.strip()[:200]
    return found


def summarize(root: Path, limit: int) -> dict:
    candidates = {"script": [], "config": [], "checkpoint": [], "log": [], "trace": []}
    signal_text = []

    for path in recent_files(root, limit):
        kind = classify(path)
        if not kind:
            continue
        rel = str(path.relative_to(root))
        candidates[kind].append(rel)
        if kind in {"script", "log", "trace"}:
            signal_text.append(read_text(path))

    joined = "\n".join(signal_text)
    recovered = {
        "stack": detect_stack(joined),
        "workload": detect_workload(joined),
        "scale": detect_scale(joined),
        "metric_focus": detect_metric_focus(joined),
        "artifacts_exist": bool(candidates["trace"]),
        "metrics_found": extract_metric_lines(joined),
    }

    filled = sum(bool(value) for key, value in recovered.items() if key != "metrics_found")
    if recovered["metrics_found"]:
        filled += 1

    return {
        "root": str(root),
        "candidates": {key: values[:5] for key, values in candidates.items() if values},
        "recovered_context": recovered,
        "confidence": "moderate" if filled >= 3 else "weak",
        "next_action": (
            "Confirm the recovered baseline with the user before choosing the next performance step."
            if filled
            else "No usable run record found. Ask whether to rerun and confirm script, config, checkpoint, and output path first."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan a prepared workspace for prior Ascend run context.")
    parser.add_argument("root_path", nargs="?", help="optional positional workspace root to scan")
    parser.add_argument("--root", dest="root_flag", help="workspace root to scan")
    parser.add_argument("--working-dir", dest="working_dir_flag", help="alias of --root")
    parser.add_argument("--limit", type=int, default=200, help="maximum recent files to inspect")
    args = parser.parse_args()

    root_arg = args.root_flag or args.working_dir_flag or args.root_path or "."
    report = summarize(Path(root_arg).resolve(), args.limit)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
