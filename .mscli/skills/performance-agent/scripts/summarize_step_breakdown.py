#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Optional

from perf_common import normalize_key, parse_number, stage_to_domain, write_json


STEP_ID_KEYS = {"step_id", "step", "iteration", "iter", "id"}
IGNORE_KEYS = {"rank", "stream", "index", "count"}
STAGE_RULES = [
    ("step_total", ("step_time", "steptime", "iteration_interval", "total_time", "iter_time", "step_total")),
    ("graph_compile", ("compile", "graph_build", "build")),
    ("input_pipeline", ("data", "dataset", "minddata", "loader", "input")),
    ("communication", ("comm", "allreduce", "reduce", "gather", "broadcast", "hccl")),
    ("host_overhead", ("host", "launch", "dispatch", "cpu")),
    ("idle_gap", ("idle", "gap", "wait", "tail", "bubble")),
    ("compute", ("compute", "forward", "backward", "kernel", "fp", "bp")),
]


def classify_stage(header: str) -> Optional[str]:
    key = normalize_key(header)
    if key in STEP_ID_KEYS or key in IGNORE_KEYS:
        return None
    for stage, tokens in STAGE_RULES:
        if any(token in key for token in tokens):
            return stage
    if key.endswith("_ms") or key.endswith("_us") or "time" in key:
        return "other"
    return None


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []
        return [dict(row) for row in reader]


def build_summary(path: Path) -> dict:
    rows = load_rows(path)
    if not rows:
        raise SystemExit(f"No rows were found in {path}")

    stage_totals: dict[str, float] = {}
    step_totals: list[float] = []

    for row in rows:
        per_row_total = 0.0
        total_from_explicit_column = None
        for header, value in row.items():
            stage = classify_stage(header)
            if not stage:
                continue
            number = parse_number(value)
            if number is None:
                continue
            stage_totals[stage] = stage_totals.get(stage, 0.0) + number
            if stage == "step_total":
                total_from_explicit_column = number
            else:
                per_row_total += number
        step_totals.append(total_from_explicit_column if total_from_explicit_column is not None else per_row_total)

    steps_analyzed = len(rows)
    dominant_name = None
    dominant_total = 0.0
    base_total = stage_totals.get("step_total") or sum(
        value for key, value in stage_totals.items() if key != "step_total"
    )
    for stage, total in stage_totals.items():
        if stage == "step_total":
            continue
        if total > dominant_total:
            dominant_name = stage
            dominant_total = total

    if base_total <= 0:
        base_total = dominant_total or 1.0

    mean_step = sum(step_totals) / max(len(step_totals), 1)
    variance = (
        sum((value - mean_step) ** 2 for value in step_totals) / max(len(step_totals), 1)
        if step_totals
        else 0.0
    )
    coefficient_of_variation = (variance ** 0.5) / mean_step if mean_step else 0.0

    dominant_stage = None
    if dominant_name is not None:
        dominant_stage = {
            "name": dominant_name,
            "domain": stage_to_domain(dominant_name),
            "total_ms": round(dominant_total, 3),
            "avg_ms": round(dominant_total / steps_analyzed, 3),
            "share_percent": round(dominant_total / base_total * 100, 2),
        }

    return {
        "source_file": str(path),
        "steps_analyzed": steps_analyzed,
        "average_step_time_ms": round(mean_step, 3),
        "coefficient_of_variation": round(coefficient_of_variation, 4),
        "consistency": "stable" if coefficient_of_variation <= 0.15 else "variable",
        "stage_totals_ms": {key: round(value, 3) for key, value in sorted(stage_totals.items())},
        "stage_avg_ms": {
            key: round(value / steps_analyzed, 3) for key, value in sorted(stage_totals.items())
        },
        "dominant_stage": dominant_stage,
        "likely_domains": [dominant_stage["domain"]] if dominant_stage and dominant_stage["domain"] else [],
        "next_action": (
            f"Validate the {dominant_stage['name']} stage against other summaries before choosing the first optimization."
            if dominant_stage
            else "Collect stronger step-level evidence before classifying the bottleneck."
        ),
    }


def default_step_trace_path(trace_root: Path) -> Path:
    candidate = trace_root / "ASCEND_PROFILER_OUTPUT" / "step_trace_time.csv"
    if candidate.exists():
        return candidate
    matches = sorted(trace_root.glob("**/ASCEND_PROFILER_OUTPUT/step_trace_time.csv"))
    if matches:
        return matches[0]
    raise SystemExit(f"step_trace_time.csv was not found under {trace_root}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize step-level stage dominance from step_trace_time.csv")
    parser.add_argument("--trace-root", help="profiler export root")
    parser.add_argument("--input-csv", help="explicit step_trace_time.csv path")
    parser.add_argument("--output-json", required=True, help="path to write the step summary JSON")
    args = parser.parse_args()

    if not args.trace_root and not args.input_csv:
        raise SystemExit("Either --trace-root or --input-csv is required.")

    input_path = Path(args.input_csv).resolve() if args.input_csv else default_step_trace_path(Path(args.trace_root).resolve())
    summary = build_summary(input_path)
    write_json(Path(args.output_json), summary)
    print(json.dumps({"dominant_stage": summary["dominant_stage"], "steps": summary["steps_analyzed"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
