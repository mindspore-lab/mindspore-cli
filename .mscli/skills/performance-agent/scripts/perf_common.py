#!/usr/bin/env python3
import csv
import json
import re
from pathlib import Path
from typing import Optional


TRACE_PATTERNS = {
    "step_trace_time": ["**/ASCEND_PROFILER_OUTPUT/step_trace_time.csv"],
    "kernel_details": ["**/ASCEND_PROFILER_OUTPUT/kernel_details.csv"],
    "trace_view": ["**/ASCEND_PROFILER_OUTPUT/trace_view.json"],
    "communication": ["**/ASCEND_PROFILER_OUTPUT/communication.json"],
    "communication_matrix": ["**/ASCEND_PROFILER_OUTPUT/communication_matrix.json"],
    "memory_record": ["**/ASCEND_PROFILER_OUTPUT/memory_record.csv"],
    "operator_memory": ["**/ASCEND_PROFILER_OUTPUT/operator_memory.csv"],
    "module_memory": ["**/ASCEND_PROFILER_OUTPUT/npu_module_mem.csv"],
    "dataset": ["**/ASCEND_PROFILER_OUTPUT/dataset.csv"],
    "minddata_pipeline_csv": ["**/ASCEND_PROFILER_OUTPUT/minddata_pipeline_summary_*.csv"],
    "minddata_pipeline_json": ["**/ASCEND_PROFILER_OUTPUT/minddata_pipeline_summary_*.json"],
    "op_summary": ["**/mindstudio_profiler_output/op_summary_*.csv"],
    "task_time": ["**/mindstudio_profiler_output/task_time_*.csv"],
    "hotspot_summary_json": ["**/hotspot_summary.json"],
    "hotspot_summary_md": ["**/hotspot_summary.md"],
}


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def parse_number(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []
        return [dict(row) for row in reader]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def profiler_root_from_path(path: Path) -> Path:
    current = path.resolve()
    if current.is_file():
        current = current.parent

    if current.name == "ASCEND_PROFILER_OUTPUT":
        return current.parent
    if current.name == "mindstudio_profiler_output":
        return current.parent.parent if current.parent.name.startswith("PROF_") else current.parent

    while True:
        if (current / "ASCEND_PROFILER_OUTPUT").exists():
            return current
        if (current / "profiler_metadata.json").exists():
            return current
        if any(current.glob("profiler_info_*.json")):
            return current
        if current.name.startswith("PROF_") and (current / "mindstudio_profiler_output").exists():
            return current.parent
        if current.parent == current:
            return path.resolve() if path.resolve().is_dir() else path.resolve().parent
        current = current.parent


def trace_file_inventory(root: Path) -> dict[str, list[Path]]:
    inventory: dict[str, list[Path]] = {}
    for key, patterns in TRACE_PATTERNS.items():
        matches: list[Path] = []
        for pattern in patterns:
            matches.extend(sorted(root.glob(pattern)))
        deduped = []
        seen = set()
        for match in matches:
            resolved = match.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            deduped.append(resolved)
        if deduped:
            inventory[key] = deduped
    return inventory


def score_trace_inventory(inventory: dict[str, list[Path]]) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    weights = {
        "step_trace_time": 30,
        "kernel_details": 25,
        "trace_view": 20,
        "communication": 20,
        "communication_matrix": 10,
        "memory_record": 15,
        "operator_memory": 10,
        "dataset": 10,
        "minddata_pipeline_csv": 10,
        "op_summary": 10,
        "hotspot_summary_json": 12,
    }
    for key, weight in weights.items():
        if inventory.get(key):
            score += weight
            reasons.append(f"found {key}")
    return score, reasons


def confidence_from_score(score: int) -> str:
    if score >= 80:
        return "strong"
    if score >= 45:
        return "moderate"
    if score > 0:
        return "weak"
    return "none"


def infer_stack_from_root(root: Path) -> Optional[str]:
    text = root.name.lower()
    if text.endswith("_ascend_ms") or "_ascend_ms_" in text:
        return "ms"
    if text.endswith("_ascend_pt") or "_ascend_pt_" in text:
        return "pta"
    return None


def relpath(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def first_file(root: Path, inventory: dict[str, list[Path]], key: str) -> Optional[str]:
    files = inventory.get(key, [])
    if not files:
        return None
    return relpath(files[0], root)


def list_files(root: Path, inventory: dict[str, list[Path]], key: str) -> list[str]:
    return [relpath(path, root) for path in inventory.get(key, [])]


def stage_to_domain(stage_name: str) -> Optional[str]:
    mapping = {
        "communication": "communication",
        "input_pipeline": "input_pipeline",
        "memory_pressure": "memory",
        "host_overhead": "host_framework_overhead",
        "idle_gap": "host_framework_overhead",
        "graph_compile": "graph_compile",
        "compute": "compute",
        "operator_hotspot": "operator_hotspot",
    }
    return mapping.get(stage_name)


def load_optional_json(path_str: Optional[str]):
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        return None
    return read_json(path)
