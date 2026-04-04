#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from perf_common import normalize_key, parse_number, read_json, stage_to_domain, write_json


CATEGORY_RULES = [
    ("graph_compile", ("compile", "graph_build", "recompile", "build_graph")),
    ("input_pipeline", ("dataset", "minddata", "dataloader", "getnext", "input")),
    ("communication", ("allreduce", "reduce", "allgather", "broadcast", "hccl", "communication")),
    ("host_overhead", ("host", "launch", "dispatch", "cpu")),
    ("idle_gap", ("idle", "gap", "wait", "bubble", "stall")),
    ("compute", ("matmul", "conv", "compute", "kernel", "forward", "backward")),
]


def iter_events(node) -> list[dict]:
    if isinstance(node, list):
        events: list[dict] = []
        for item in node:
            events.extend(iter_events(item))
        return events
    if isinstance(node, dict):
        if "traceEvents" in node:
            return iter_events(node["traceEvents"])
        if "events" in node:
            return iter_events(node["events"])
        name = node.get("name") or node.get("op_name") or node.get("event")
        duration = None
        for key in ("duration_ms", "dur_ms", "time_ms", "elapsed_ms", "duration"):
            if duration is not None:
                break
            duration = parse_number(node.get(key))
        if name and duration is not None:
            return [{"name": str(name), "duration_ms": duration}]
        events: list[dict] = []
        for value in node.values():
            events.extend(iter_events(value))
        return events
    return []


def classify_event(name: str) -> str:
    normalized = normalize_key(name)
    for category, tokens in CATEGORY_RULES:
        if any(token in normalized for token in tokens):
            return category
    return "other"


def summarize_events(events: list[dict]) -> dict:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    total_duration_ms = 0.0

    for event in events:
        category = classify_event(event["name"])
        totals[category] = totals.get(category, 0.0) + event["duration_ms"]
        counts[category] = counts.get(category, 0) + 1
        total_duration_ms += event["duration_ms"]

    dominant_name = None
    dominant_total = 0.0
    for category, duration in totals.items():
        if category == "other":
            continue
        if duration > dominant_total:
            dominant_name = category
            dominant_total = duration

    dominant_category = None
    if dominant_name is not None and total_duration_ms > 0:
        dominant_category = {
            "name": dominant_name,
            "domain": stage_to_domain(dominant_name),
            "total_ms": round(dominant_total, 3),
            "share_percent": round(dominant_total / total_duration_ms * 100, 2),
            "event_count": counts.get(dominant_name, 0),
        }

    likely_domains = []
    if dominant_category and dominant_category.get("domain"):
        likely_domains.append(dominant_category["domain"])

    return {
        "event_count": len(events),
        "total_duration_ms": round(total_duration_ms, 3),
        "category_totals_ms": {key: round(value, 3) for key, value in sorted(totals.items())},
        "category_event_counts": dict(sorted(counts.items())),
        "dominant_category": dominant_category,
        "likely_domains": likely_domains,
        "next_action": (
            f"Validate the {dominant_name} trace-gap signature against step and hotspot summaries."
            if dominant_name is not None
            else "Trace events were parsed, but no dominant gap category was identified."
        ),
    }


def default_trace_view_path(trace_root: Path) -> Path:
    candidate = trace_root / "ASCEND_PROFILER_OUTPUT" / "trace_view.json"
    if candidate.exists():
        return candidate
    matches = sorted(trace_root.glob("**/ASCEND_PROFILER_OUTPUT/trace_view.json"))
    if matches:
        return matches[0]
    raise SystemExit(f"trace_view.json was not found under {trace_root}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize timeline gaps and host-side trace signatures from trace_view.json")
    parser.add_argument("--trace-root", help="profiler export root")
    parser.add_argument("--trace-json", help="explicit trace_view.json path")
    parser.add_argument("--output-json", required=True, help="path to write the trace-gap summary JSON")
    args = parser.parse_args()

    if not args.trace_root and not args.trace_json:
        raise SystemExit("Either --trace-root or --trace-json is required.")

    trace_path = Path(args.trace_json).resolve() if args.trace_json else default_trace_view_path(Path(args.trace_root).resolve())
    events = iter_events(read_json(trace_path))
    summary = {
        "source_file": str(trace_path),
        **summarize_events(events),
    }
    write_json(Path(args.output_json), summary)
    print(json.dumps({"dominant_category": summary["dominant_category"], "event_count": summary["event_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
