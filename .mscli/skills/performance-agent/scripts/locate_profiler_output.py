#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Optional

from perf_common import (
    confidence_from_score,
    first_file,
    infer_stack_from_root,
    list_files,
    profiler_root_from_path,
    score_trace_inventory,
    trace_file_inventory,
    write_json,
)


DISCOVERY_PATTERNS = [
    "**/profiler_metadata.json",
    "**/profiler_info_*.json",
    "**/ASCEND_PROFILER_OUTPUT/step_trace_time.csv",
    "**/ASCEND_PROFILER_OUTPUT/trace_view.json",
    "**/ASCEND_PROFILER_OUTPUT/communication.json",
    "**/ASCEND_PROFILER_OUTPUT/memory_record.csv",
    "**/mindstudio_profiler_output/op_summary_*.csv",
    "**/hotspot_summary.json",
]


def discover_candidate_roots(working_dir: Path) -> list[Path]:
    seen: set[Path] = set()
    candidates: list[Path] = []
    for pattern in DISCOVERY_PATTERNS:
        for path in sorted(working_dir.glob(pattern)):
            root = profiler_root_from_path(path)
            resolved = root.resolve()
            if resolved in seen or not resolved.exists():
                continue
            seen.add(resolved)
            candidates.append(resolved)
    return candidates


def describe_candidate(root: Path, explicit: bool = False) -> dict:
    inventory = trace_file_inventory(root)
    score, reasons = score_trace_inventory(inventory)
    if explicit:
        score += 40
        reasons = ["explicit trace path"] + reasons
    return {
        "root": str(root),
        "score": score,
        "confidence": confidence_from_score(score),
        "stack": infer_stack_from_root(root),
        "reasons": reasons,
        "files": {
            "step_trace_time": first_file(root, inventory, "step_trace_time"),
            "kernel_details": first_file(root, inventory, "kernel_details"),
            "trace_view": first_file(root, inventory, "trace_view"),
            "communication": first_file(root, inventory, "communication"),
            "communication_matrix": first_file(root, inventory, "communication_matrix"),
            "memory_record": first_file(root, inventory, "memory_record"),
            "operator_memory": first_file(root, inventory, "operator_memory"),
            "dataset": first_file(root, inventory, "dataset"),
            "minddata_pipeline_csv": first_file(root, inventory, "minddata_pipeline_csv"),
            "op_summary": list_files(root, inventory, "op_summary"),
            "hotspot_summary_json": first_file(root, inventory, "hotspot_summary_json"),
        },
    }


def build_report(working_dir: Path, trace_path: Optional[Path]) -> dict:
    candidates: list[dict] = []

    if trace_path is not None and trace_path.exists():
        explicit_root = profiler_root_from_path(trace_path)
        candidates.append(describe_candidate(explicit_root, explicit=True))

    for root in discover_candidate_roots(working_dir):
        if any(item["root"] == str(root) for item in candidates):
            continue
        candidates.append(describe_candidate(root))

    candidates.sort(key=lambda item: item["score"], reverse=True)
    selected = candidates[0] if candidates else None

    if not selected:
        return {
            "working_dir": str(working_dir),
            "trace_requested_path": str(trace_path) if trace_path else None,
            "selected_root": None,
            "confidence": "none",
            "stack": None,
            "candidates": [],
            "next_action": "No profiler export root was found. Provide a trace path or generate an Ascend profiler export first.",
        }

    return {
        "working_dir": str(working_dir),
        "trace_requested_path": str(trace_path) if trace_path else None,
        "selected_root": selected["root"],
        "confidence": selected["confidence"],
        "stack": selected["stack"],
        "selected_files": selected["files"],
        "candidates": candidates,
        "next_action": (
            "Use the selected profiler root for structured summaries."
            if selected["confidence"] in {"strong", "moderate"}
            else "Profiler root was recovered weakly. Confirm the selected export before relying on the summaries."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Locate the best profiler export root under a workspace")
    parser.add_argument("working_dir_positional", nargs="?", help="optional positional workspace root to search")
    parser.add_argument("--working-dir", dest="working_dir_flag", help="workspace root to search")
    parser.add_argument("--root", dest="root_flag", help="alias of --working-dir")
    parser.add_argument("--trace-path", help="explicit trace file or export directory")
    parser.add_argument("--output-json", help="optional path to write the locator report JSON")
    args = parser.parse_args()

    working_dir_arg = args.working_dir_flag or args.root_flag or args.working_dir_positional or "."
    working_dir = Path(working_dir_arg).resolve()
    trace_path = Path(args.trace_path).resolve() if args.trace_path else None

    report = build_report(working_dir, trace_path)
    if args.output_json:
        write_json(Path(args.output_json), report)
        print(json.dumps({"selected_root": report["selected_root"], "confidence": report["confidence"]}, indent=2))
    else:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
