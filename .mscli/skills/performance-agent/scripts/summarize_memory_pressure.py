#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Optional

from perf_common import normalize_key, parse_number, write_json


NAME_KEYS = {"op_name", "operator_name", "operator", "name", "kernel_name", "module_name"}
MEMORY_KEYS = {
    "memory_mb",
    "peak_memory_mb",
    "peak_mem_mb",
    "allocated_mb",
    "reserved_mb",
    "memory",
    "peak_memory",
    "peak_mem",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []
        return [dict(row) for row in reader]


def detect_name_and_memory_fields(
    rows: list[dict[str, str]]
) -> tuple[Optional[str], Optional[str]]:
    if not rows:
        return None, None
    normalized = {field: normalize_key(field) for field in rows[0].keys()}
    name_field = next((field for field, key in normalized.items() if key in NAME_KEYS), None)
    memory_field = next((field for field, key in normalized.items() if key in MEMORY_KEYS), None)
    if memory_field is None:
        memory_field = next((field for field, key in normalized.items() if "mem" in key or "memory" in key), None)
    return name_field, memory_field


def summarize_operator_memory(rows: list[dict[str, str]]) -> list[dict]:
    name_field, memory_field = detect_name_and_memory_fields(rows)
    if not name_field or not memory_field:
        return []

    ranked = []
    for row in rows:
        name = str(row.get(name_field) or "").strip()
        value = parse_number(row.get(memory_field))
        if not name or value is None:
            continue
        ranked.append({"name": name, "memory_mb": value})
    ranked.sort(key=lambda item: item["memory_mb"], reverse=True)
    total = sum(item["memory_mb"] for item in ranked) or 1.0
    return [
        {
            "name": item["name"],
            "memory_mb": round(item["memory_mb"], 3),
            "share_percent": round(item["memory_mb"] / total * 100, 2),
        }
        for item in ranked[:5]
    ]


def summarize_peak_memory(
    rows: list[dict[str, str]]
) -> tuple[Optional[float], Optional[str]]:
    if not rows:
        return None, None

    best_field = None
    peak_value = None
    for field in rows[0].keys():
        key = normalize_key(field)
        if "mem" not in key and "memory" not in key:
            continue
        for row in rows:
            value = parse_number(row.get(field))
            if value is None:
                continue
            if peak_value is None or value > peak_value:
                peak_value = value
                best_field = field
    return peak_value, best_field


def default_paths(
    trace_root: Path,
) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
    base = trace_root / "ASCEND_PROFILER_OUTPUT"
    files = (
        base / "memory_record.csv",
        base / "operator_memory.csv",
        base / "npu_module_mem.csv",
    )
    return tuple(path if path.exists() else None for path in files)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize memory pressure from Ascend profiler CSV exports")
    parser.add_argument("--trace-root", help="profiler export root")
    parser.add_argument("--memory-record-csv", help="explicit memory_record.csv path")
    parser.add_argument("--operator-memory-csv", help="explicit operator_memory.csv path")
    parser.add_argument("--module-memory-csv", help="explicit npu_module_mem.csv path")
    parser.add_argument("--output-json", required=True, help="path to write the memory summary JSON")
    args = parser.parse_args()

    memory_record = Path(args.memory_record_csv).resolve() if args.memory_record_csv else None
    operator_memory = Path(args.operator_memory_csv).resolve() if args.operator_memory_csv else None
    module_memory = Path(args.module_memory_csv).resolve() if args.module_memory_csv else None
    if args.trace_root:
        inferred_memory_record, inferred_operator_memory, inferred_module_memory = default_paths(Path(args.trace_root).resolve())
        memory_record = memory_record or inferred_memory_record
        operator_memory = operator_memory or inferred_operator_memory
        module_memory = module_memory or inferred_module_memory

    if not any(path and path.exists() for path in (memory_record, operator_memory, module_memory)):
        raise SystemExit("No memory profiler files were found. Provide explicit memory CSV paths or a trace root.")

    record_rows = load_rows(memory_record) if memory_record and memory_record.exists() else []
    operator_rows = load_rows(operator_memory) if operator_memory and operator_memory.exists() else []
    module_rows = load_rows(module_memory) if module_memory and module_memory.exists() else []

    peak_memory_mb, peak_source_field = summarize_peak_memory(record_rows or module_rows)
    top_operators = summarize_operator_memory(operator_rows)
    top_modules = summarize_operator_memory(module_rows)

    pressure = "low"
    if top_operators and top_operators[0]["share_percent"] >= 35:
        pressure = "high"
    elif peak_memory_mb is not None:
        pressure = "moderate"

    report = {
        "source_files": {
            "memory_record_csv": str(memory_record) if memory_record and memory_record.exists() else None,
            "operator_memory_csv": str(operator_memory) if operator_memory and operator_memory.exists() else None,
            "module_memory_csv": str(module_memory) if module_memory and module_memory.exists() else None,
        },
        "peak_memory_mb": round(peak_memory_mb, 3) if peak_memory_mb is not None else None,
        "peak_source_field": peak_source_field,
        "memory_pressure": pressure,
        "top_operators": top_operators,
        "top_modules": top_modules,
        "likely_domains": ["memory"] if peak_memory_mb is not None else [],
        "next_action": (
            "Validate peak memory, memory-heavy stage, and batch-size headroom after the first memory-focused change."
            if peak_memory_mb is not None
            else "Collect stronger memory evidence before choosing a memory optimization."
        ),
    }
    write_json(Path(args.output_json), report)
    print(json.dumps({"peak_memory_mb": report["peak_memory_mb"], "pressure": report["memory_pressure"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
