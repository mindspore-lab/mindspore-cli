#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Optional

from perf_common import load_optional_json, write_json


def candidate(
    name: str,
    confidence: float,
    evidence: list[str],
    validation_checks: list[str],
    optimization_hints: list[str],
) -> dict:
    return {
        "name": name,
        "confidence": round(confidence, 3),
        "evidence": evidence,
        "validation_checks": validation_checks,
        "optimization_hints": optimization_hints,
    }


def add_candidate(candidates_by_name: dict[str, dict], item: dict) -> None:
    current = candidates_by_name.get(item["name"])
    if not current:
        candidates_by_name[item["name"]] = item
        return
    current["confidence"] = round(max(current["confidence"], item["confidence"]), 3)
    for key in ("evidence", "validation_checks", "optimization_hints"):
        merged = current[key] + [value for value in item[key] if value not in current[key]]
        current[key] = merged


def classify(
    profile: dict,
    step: Optional[dict],
    communication: Optional[dict],
    memory: Optional[dict],
    input_summary: Optional[dict],
    trace_gaps: Optional[dict],
    hotspot: Optional[dict],
) -> list[dict]:
    candidates_by_name: dict[str, dict] = {}

    if communication and communication.get("dominant_collective"):
        pressure = communication.get("communication_pressure")
        base = 0.8 if pressure == "high" else 0.6
        evidence = [
            f"dominant collective: {communication['dominant_collective']['name']}",
            f"communication pressure: {pressure}",
        ]
        if communication.get("matrix_imbalance_ratio") is not None:
            evidence.append(f"matrix imbalance ratio: {communication['matrix_imbalance_ratio']}")
        add_candidate(
            candidates_by_name,
            candidate(
                "communication",
                base,
                evidence,
                ["compare collective time share", "compare collective count", "compare exposed step tail"],
                ["check overlap", "check bucket or fusion settings", "remove unnecessary synchronization"],
            ),
        )

    if input_summary and input_summary.get("bottleneck_detected"):
        evidence = []
        if input_summary.get("queue_empty_percent") is not None:
            evidence.append(f"queue empty percent: {input_summary['queue_empty_percent']}")
        if input_summary.get("warning"):
            evidence.append(f"pipeline warning: {input_summary['warning']}")
        add_candidate(
            candidates_by_name,
            candidate(
                "input_pipeline",
                0.7,
                evidence,
                ["compare pre-compute idle time", "compare queue-empty rate", "compare end-to-end throughput"],
                ["increase pipeline parallelism", "reduce decode or transform cost", "check prefetch or caching"],
            ),
        )

    if memory and memory.get("peak_memory_mb") is not None:
        evidence = [f"peak memory: {memory['peak_memory_mb']} MB"]
        if memory.get("top_operators"):
            top = memory["top_operators"][0]
            evidence.append(f"top memory operator: {top['name']} ({top['share_percent']}%)")
        add_candidate(
            candidates_by_name,
            candidate(
                "memory",
                0.75 if memory.get("memory_pressure") == "high" else 0.55,
                evidence,
                ["compare peak memory", "compare top memory-consuming stage", "compare batch-size headroom"],
                ["review recomputation", "reduce temporary tensors", "review precision and layout"],
            ),
        )

    if trace_gaps and trace_gaps.get("dominant_category"):
        dominant = trace_gaps["dominant_category"]["name"]
        share = float(trace_gaps["dominant_category"].get("share_percent") or 0.0)
        evidence = [f"dominant trace category: {dominant}", f"share: {round(share, 2)}%"]
        if dominant in {"host_overhead", "idle_gap"}:
            add_candidate(
                candidates_by_name,
                candidate(
                    "host_framework_overhead",
                    0.72 if share >= 20 else 0.6,
                    evidence,
                    ["compare trace idle gap", "compare host launch duration", "compare kernel launch density"],
                    ["reduce host-side per-step work", "remove unnecessary syncs", "increase graph-heavy execution"],
                ),
            )
        elif dominant == "graph_compile":
            add_candidate(
                candidates_by_name,
                candidate(
                    "graph_compile",
                    0.72 if share >= 20 else 0.6,
                    evidence,
                    ["compare compile duration", "compare compile count", "compare warmup versus steady-state latency"],
                    ["stabilize shapes", "reduce recompilation triggers", "separate warmup compile from steady-state measurement"],
                ),
            )
        elif dominant == "communication":
            add_candidate(
                candidates_by_name,
                candidate(
                    "communication",
                    0.7 if share >= 20 else 0.58,
                    evidence,
                    ["compare communication slices in the trace", "compare overlap between communication and compute", "compare exposed step tail"],
                    ["check overlap", "check bucket or fusion settings", "remove unnecessary synchronization"],
                ),
            )
        elif dominant == "input_pipeline":
            add_candidate(
                candidates_by_name,
                candidate(
                    "input_pipeline",
                    0.68 if share >= 20 else 0.55,
                    evidence,
                    ["compare queue-empty rate", "compare dataset stage time", "compare pre-compute idle time"],
                    ["increase pipeline parallelism", "reduce decode or transform cost", "check prefetch or caching"],
                ),
            )

    if step and step.get("dominant_stage"):
        dominant = step["dominant_stage"]["name"]
        share = step["dominant_stage"]["share_percent"]
        if dominant in {"host_overhead", "idle_gap"}:
            add_candidate(
                candidates_by_name,
                candidate(
                    "host_framework_overhead",
                    0.68,
                    [f"dominant stage: {dominant}", f"share: {share}%"],
                    ["compare host idle gap", "compare kernel launch density", "compare utilization trend"],
                    ["reduce host-side per-step work", "remove unnecessary syncs", "increase graph-heavy execution"],
                ),
            )
        elif dominant == "graph_compile":
            add_candidate(
                candidates_by_name,
                candidate(
                    "graph_compile",
                    0.66,
                    [f"dominant stage: {dominant}", f"share: {share}%"],
                    ["compare compile time", "compare compile count", "compare steady-state latency after warmup"],
                    ["stabilize shapes", "reduce recompilation triggers", "separate first-run compile from steady state"],
                ),
            )
        elif dominant == "compute":
            add_candidate(
                candidates_by_name,
                candidate(
                    "compute",
                    0.52,
                    [f"dominant stage: {dominant}", f"share: {share}%"],
                    ["compare operator time share", "compare end-to-end step time"],
                    ["focus on top compute operators", "review fusion opportunities", "review backend kernel path"],
                ),
            )

    if hotspot and hotspot.get("top_operators"):
        lead = hotspot["top_operators"][0]
        confidence = 0.7 if float(lead.get("share_percent") or 0.0) >= 35 else 0.5
        hints = ["focus on the top operator first", "review fusion or kernel path", "avoid spreading effort across the long tail"]
        if lead.get("category") == "communication":
            hints = ["check overlap", "check collective fusion", "validate communication-heavy update or step tail"]
        add_candidate(
            candidates_by_name,
            candidate(
                "operator_hotspot",
                confidence,
                [f"lead operator: {lead['operator']}", f"time share: {lead['share_percent']}%"],
                ["compare dominant operator share", "compare end-to-end step time or latency"],
                hints,
            ),
        )

    candidates = list(candidates_by_name.values())
    if not candidates:
        candidates.append(
            candidate(
                "inconclusive",
                0.2,
                ["structured summaries are insufficient for a bottleneck claim"],
                ["collect step, operator, and trace summaries"],
                ["collect stronger profiler evidence before changing the workload"],
            )
        )

    candidates.sort(key=lambda item: item["confidence"], reverse=True)
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify ranked bottleneck candidates from structured summaries")
    parser.add_argument("--profile-json", required=True, help="performance profile JSON path")
    parser.add_argument("--step-json", help="step summary JSON path")
    parser.add_argument("--communication-json", help="communication summary JSON path")
    parser.add_argument("--memory-json", help="memory summary JSON path")
    parser.add_argument("--input-json", help="input summary JSON path")
    parser.add_argument("--trace-gaps-json", help="trace-gap summary JSON path")
    parser.add_argument("--hotspot-json", help="hotspot summary JSON path")
    parser.add_argument("--output-json", required=True, help="path to write the bottleneck classification JSON")
    args = parser.parse_args()

    profile = json.loads(Path(args.profile_json).read_text(encoding="utf-8"))
    step = load_optional_json(args.step_json)
    communication = load_optional_json(args.communication_json)
    memory = load_optional_json(args.memory_json)
    input_summary = load_optional_json(args.input_json)
    trace_gaps = load_optional_json(args.trace_gaps_json)
    hotspot = load_optional_json(args.hotspot_json)

    ranked = classify(profile, step, communication, memory, input_summary, trace_gaps, hotspot)
    report = {
        "schema_version": "performance-agent/0.1",
        "skill": "performance-agent",
        "primary_candidate": ranked[0],
        "ranked_candidates": ranked,
        "next_action": (
            "Apply one targeted optimization and collect a before/after comparison for the primary candidate."
            if ranked[0]["name"] != "inconclusive"
            else "Collect stronger profiler evidence before choosing the first optimization."
        ),
    }
    write_json(Path(args.output_json), report)
    print(json.dumps({"primary_candidate": report["primary_candidate"]["name"], "confidence": report["primary_candidate"]["confidence"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
