#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Optional

from perf_common import load_optional_json, stage_to_domain, write_json


def detect_workload(user_problem: str, locate_report: Optional[dict]) -> Optional[str]:
    text = user_problem.lower()
    if any(token in text for token in ("inference", "latency", "p95", "p99", "serving")):
        return "inference"
    if any(token in text for token in ("training", "epoch", "step", "throughput", "loss", "optimizer")):
        return "training"
    if locate_report:
        candidate = locate_report.get("selected_root") or ""
        if "infer" in candidate.lower():
            return "inference"
    return None


def detect_metric_focus(user_problem: str, summaries: dict[str, dict]) -> Optional[str]:
    text = user_problem.lower()
    if "memory" in text or "batch size" in text:
        return "memory"
    if any(token in text for token in ("latency", "p95", "p99", "ms/step", "step time")):
        return "latency"
    if any(token in text for token in ("throughput", "fps", "samples/s", "steps/s")):
        return "throughput"
    if summaries.get("memory") and summaries["memory"].get("peak_memory_mb") is not None:
        return "memory"
    if summaries.get("step") and summaries["step"].get("dominant_stage"):
        stage = summaries["step"]["dominant_stage"]["name"]
        if stage in {"idle_gap", "host_overhead"}:
            return "latency"
    if summaries.get("trace_gaps") and summaries["trace_gaps"].get("dominant_category"):
        category = summaries["trace_gaps"]["dominant_category"]["name"]
        if category in {"idle_gap", "host_overhead", "graph_compile"}:
            return "latency"
    return None


def symptom_from_summaries(
    user_problem: str, metric_focus: Optional[str], summaries: dict[str, dict]
) -> str:
    text = user_problem.lower()
    if any(token in text for token in ("allreduce", "all reduce", "communication", "collective", "hccl")):
        return "communication overhead"
    if any(token in text for token in ("memory", "oom", "batch size", "peak memory")):
        return "memory bottleneck"
    if summaries.get("communication") and summaries["communication"].get("communication_pressure") in {"moderate", "high"}:
        return "communication overhead"
    if summaries.get("memory") and summaries["memory"].get("memory_pressure") in {"moderate", "high"}:
        return "memory bottleneck"
    if summaries.get("input") and summaries["input"].get("bottleneck_detected"):
        return "dataloader stall"
    if summaries.get("trace_gaps") and summaries["trace_gaps"].get("dominant_category"):
        dominant = summaries["trace_gaps"]["dominant_category"]["name"]
        if dominant in {"idle_gap", "host_overhead"}:
            return "host launch overhead"
        if dominant == "graph_compile":
            return "latency bottleneck"
        if dominant == "communication":
            return "communication overhead"
    if summaries.get("step") and summaries["step"].get("dominant_stage"):
        dominant = summaries["step"]["dominant_stage"]["name"]
        if dominant in {"idle_gap", "host_overhead"}:
            return "host launch overhead"
        if dominant == "graph_compile":
            return "latency bottleneck"
        if dominant == "compute":
            return "throughput bottleneck"
    if metric_focus == "memory":
        return "memory bottleneck"
    if metric_focus == "latency":
        return "latency bottleneck"
    return "throughput bottleneck"


def score_domains(summaries: dict[str, dict]) -> list[dict]:
    scores: dict[str, float] = {}

    step = summaries.get("step")
    if step and step.get("dominant_stage"):
        stage = step["dominant_stage"]["name"]
        share = float(step["dominant_stage"].get("share_percent") or 0.0)
        domain = stage_to_domain(stage)
        if domain:
            scores[domain] = scores.get(domain, 0.0) + share / 100.0

    communication = summaries.get("communication")
    if communication and communication.get("dominant_collective"):
        score = 0.6 if communication.get("communication_pressure") == "high" else 0.35
        scores["communication"] = scores.get("communication", 0.0) + score

    memory = summaries.get("memory")
    if memory and memory.get("peak_memory_mb") is not None:
        score = 0.6 if memory.get("memory_pressure") == "high" else 0.35
        scores["memory"] = scores.get("memory", 0.0) + score

    input_summary = summaries.get("input")
    if input_summary and input_summary.get("bottleneck_detected"):
        scores["input_pipeline"] = scores.get("input_pipeline", 0.0) + 0.55

    trace_gaps = summaries.get("trace_gaps")
    if trace_gaps and trace_gaps.get("dominant_category"):
        dominant = trace_gaps["dominant_category"]
        domain = dominant.get("domain")
        if domain:
            scores[domain] = scores.get(domain, 0.0) + float(dominant.get("share_percent") or 0.0) / 100.0

    hotspot = summaries.get("hotspot")
    if hotspot and hotspot.get("top_operators"):
        lead_share = float(hotspot["top_operators"][0].get("share_percent") or 0.0)
        scores["operator_hotspot"] = scores.get("operator_hotspot", 0.0) + lead_share / 100.0

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [{"domain": name, "score": round(score, 3)} for name, score in ranked]


def derive_confidence(locate_report: Optional[dict], summaries: dict[str, dict]) -> str:
    summary_count = sum(1 for item in summaries.values() if item)
    locate_confidence = (locate_report or {}).get("confidence")
    if locate_confidence == "strong" and summary_count >= 2:
        return "strong"
    if locate_confidence in {"strong", "moderate"} and summary_count >= 1:
        return "moderate"
    if summary_count >= 1:
        return "weak"
    return "none"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a structured performance profile from summary artifacts")
    parser.add_argument("--working-dir", default=".", help="workspace root")
    parser.add_argument("--user-problem", default="", help="user problem description")
    parser.add_argument("--locate-json", help="locator report JSON")
    parser.add_argument("--step-json", help="step summary JSON")
    parser.add_argument("--communication-json", help="communication summary JSON")
    parser.add_argument("--memory-json", help="memory summary JSON")
    parser.add_argument("--input-json", help="input-pipeline summary JSON")
    parser.add_argument("--trace-gaps-json", help="trace-gap summary JSON")
    parser.add_argument("--hotspot-json", help="hotspot summary JSON")
    parser.add_argument("--output-json", required=True, help="path to write the performance profile JSON")
    args = parser.parse_args()

    locate_report = load_optional_json(args.locate_json)
    summaries = {
        "step": load_optional_json(args.step_json),
        "communication": load_optional_json(args.communication_json),
        "memory": load_optional_json(args.memory_json),
        "input": load_optional_json(args.input_json),
        "trace_gaps": load_optional_json(args.trace_gaps_json),
        "hotspot": load_optional_json(args.hotspot_json),
    }

    workload_type = detect_workload(args.user_problem, locate_report)
    metric_focus = detect_metric_focus(args.user_problem, summaries)
    likely_domains = score_domains(summaries)
    symptom = symptom_from_summaries(args.user_problem, metric_focus, summaries)
    confidence = derive_confidence(locate_report, summaries)

    profile = {
        "schema_version": "performance-agent/0.1",
        "skill": "performance-agent",
        "working_dir": str(Path(args.working_dir).resolve()),
        "trace_root": (locate_report or {}).get("selected_root"),
        "stack": (locate_report or {}).get("stack"),
        "workload_type": workload_type,
        "metric_focus": metric_focus,
        "primary_symptom": symptom,
        "confidence": confidence,
        "likely_domains": likely_domains,
        "available_artifacts": {
            "step_summary": bool(summaries["step"]),
            "communication_summary": bool(summaries["communication"]),
            "memory_summary": bool(summaries["memory"]),
            "input_summary": bool(summaries["input"]),
            "trace_gap_summary": bool(summaries["trace_gaps"]),
            "hotspot_summary": bool(summaries["hotspot"]),
        },
        "summary_refs": {
            "step": args.step_json,
            "communication": args.communication_json,
            "memory": args.memory_json,
            "input": args.input_json,
            "trace_gaps": args.trace_gaps_json,
            "hotspot": args.hotspot_json,
        },
        "evidence_level": "trace" if (locate_report or {}).get("selected_root") else "workspace_context",
        "user_problem": args.user_problem,
        "next_action": (
            "Classify ranked bottleneck candidates from the strongest available summaries."
            if likely_domains
            else "Collect at least one structured profiler summary before classifying bottlenecks."
        ),
    }
    write_json(Path(args.output_json), profile)
    print(json.dumps({"primary_symptom": profile["primary_symptom"], "confidence": profile["confidence"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
