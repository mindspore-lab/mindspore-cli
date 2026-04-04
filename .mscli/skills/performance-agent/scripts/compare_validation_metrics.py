#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from perf_common import parse_number, read_json, write_json


LOWER_IS_BETTER = {
    "latency",
    "step_time",
    "peak_memory",
    "communication_time",
    "idle_gap",
    "operator_share",
}
HIGHER_IS_BETTER = {
    "throughput",
    "utilization",
    "batch_size_headroom",
}


def normalize_metrics(payload) -> dict[str, float]:
    if isinstance(payload, dict) and isinstance(payload.get("metrics"), dict):
        payload = payload["metrics"]
    metrics: dict[str, float] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            number = parse_number(value)
            if number is not None:
                metrics[key] = number
    return metrics


def classify_direction(name: str) -> str:
    lowered = name.lower()
    if lowered in LOWER_IS_BETTER:
        return "down"
    if lowered in HIGHER_IS_BETTER:
        return "up"
    if "latency" in lowered or "time" in lowered or "memory" in lowered or "gap" in lowered or "share" in lowered:
        return "down"
    return "up"


def compare(before: dict[str, float], after: dict[str, float]) -> dict:
    keys = sorted(set(before) & set(after))
    comparisons = []
    improved = 0
    regressed = 0
    for key in keys:
        before_value = before[key]
        after_value = after[key]
        delta = after_value - before_value
        percent_change = (delta / before_value * 100) if before_value else None
        direction = classify_direction(key)
        if direction == "down":
            outcome = "improved" if after_value < before_value else "regressed" if after_value > before_value else "unchanged"
        else:
            outcome = "improved" if after_value > before_value else "regressed" if after_value < before_value else "unchanged"
        if outcome == "improved":
            improved += 1
        elif outcome == "regressed":
            regressed += 1
        comparisons.append(
            {
                "metric": key,
                "before": before_value,
                "after": after_value,
                "delta": round(delta, 6),
                "percent_change": round(percent_change, 3) if percent_change is not None else None,
                "desired_direction": direction,
                "outcome": outcome,
            }
        )

    overall = "mixed"
    if improved and not regressed:
        overall = "improved"
    elif regressed and not improved:
        overall = "regressed"
    elif not improved and not regressed:
        overall = "unchanged"

    return {
        "metrics_compared": comparisons,
        "improved_count": improved,
        "regressed_count": regressed,
        "overall_result": overall,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare before/after validation metrics for a chosen optimization")
    parser.add_argument("--before-json", required=True, help="before metrics JSON path")
    parser.add_argument("--after-json", required=True, help="after metrics JSON path")
    parser.add_argument("--output-json", required=True, help="path to write the comparison JSON")
    args = parser.parse_args()

    before = normalize_metrics(read_json(Path(args.before_json)))
    after = normalize_metrics(read_json(Path(args.after_json)))
    if not before or not after:
        raise SystemExit("Both before and after metric files must contain numeric metrics.")

    report = compare(before, after)
    report["before_ref"] = str(Path(args.before_json).resolve())
    report["after_ref"] = str(Path(args.after_json).resolve())
    write_json(Path(args.output_json), report)
    print(json.dumps({"overall_result": report["overall_result"], "metrics": len(report["metrics_compared"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
