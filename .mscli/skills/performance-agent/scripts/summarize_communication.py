#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Optional

from perf_common import normalize_key, parse_number, read_json, write_json


NAME_KEYS = {"name", "op_name", "operator", "operator_name", "collective", "task_name"}
TIME_KEYS = {"time", "time_ms", "duration", "duration_ms", "total_time", "total_time_ms", "elapse_time"}
COUNT_KEYS = {"count", "calls", "op_count"}
SIZE_KEYS = {"size_mb", "data_size_mb", "msg_size_mb", "message_size_mb"}


def flatten_records(node) -> list[dict]:
    if isinstance(node, list):
        records: list[dict] = []
        for item in node:
            records.extend(flatten_records(item))
        return records
    if isinstance(node, dict):
        normalized = {normalize_key(key): value for key, value in node.items()}
        name_key = next((key for key in normalized if key in NAME_KEYS), None)
        time_key = next((key for key in normalized if key in TIME_KEYS), None)
        if name_key and time_key:
            name = str(normalized[name_key]).strip()
            time_value = parse_number(normalized[time_key])
            if name and time_value is not None:
                count = None
                size_mb = None
                for key in normalized:
                    if count is None and key in COUNT_KEYS:
                        count = parse_number(normalized[key])
                    if size_mb is None and key in SIZE_KEYS:
                        size_mb = parse_number(normalized[key])
                return [
                    {
                        "name": name,
                        "time_ms": time_value,
                        "count": int(count) if count is not None else 1,
                        "size_mb": size_mb,
                    }
                ]
        records: list[dict] = []
        for value in node.values():
            records.extend(flatten_records(value))
        return records
    return []


def matrix_stats(node) -> dict:
    values: list[float] = []
    if isinstance(node, list):
        for item in node:
            stats = matrix_stats(item)
            values.extend(stats.get("values", []))
    elif isinstance(node, dict):
        if "values" in node:
            stats = matrix_stats(node["values"])
            values.extend(stats.get("values", []))
        else:
            for value in node.values():
                stats = matrix_stats(value)
                values.extend(stats.get("values", []))
    else:
        number = parse_number(node)
        if number is not None:
            values.append(number)

    positives = [value for value in values if value > 0]
    if not positives:
        return {"values": [], "imbalance_ratio": None}
    return {
        "values": positives,
        "imbalance_ratio": round(max(positives) / min(positives), 3) if min(positives) > 0 else None,
    }


def summarize_records(records: list[dict]) -> dict:
    totals: dict[str, dict] = {}
    total_time = 0.0
    total_calls = 0
    for record in records:
        total_time += record["time_ms"]
        total_calls += record["count"]
        current = totals.setdefault(
            record["name"],
            {"name": record["name"], "time_ms": 0.0, "count": 0, "size_mb": 0.0, "size_samples": 0},
        )
        current["time_ms"] += record["time_ms"]
        current["count"] += record["count"]
        if record["size_mb"] is not None:
            current["size_mb"] += record["size_mb"]
            current["size_samples"] += 1

    ranked = sorted(totals.values(), key=lambda item: item["time_ms"], reverse=True)
    top_collectives = []
    for item in ranked[:5]:
        share = item["time_ms"] / total_time * 100 if total_time else 0.0
        top_collectives.append(
            {
                "name": item["name"],
                "time_ms": round(item["time_ms"], 3),
                "share_percent": round(share, 2),
                "count": item["count"],
                "avg_size_mb": round(item["size_mb"] / item["size_samples"], 3) if item["size_samples"] else None,
            }
        )

    pressure = "low"
    if top_collectives and top_collectives[0]["share_percent"] >= 40:
        pressure = "high"
    elif top_collectives and top_collectives[0]["share_percent"] >= 20:
        pressure = "moderate"

    return {
        "records_used": len(records),
        "collective_count": total_calls,
        "total_time_ms": round(total_time, 3),
        "communication_pressure": pressure,
        "top_collectives": top_collectives,
        "dominant_collective": top_collectives[0] if top_collectives else None,
    }


def default_comm_paths(trace_root: Path) -> tuple[Optional[Path], Optional[Path]]:
    comm = trace_root / "ASCEND_PROFILER_OUTPUT" / "communication.json"
    matrix = trace_root / "ASCEND_PROFILER_OUTPUT" / "communication_matrix.json"
    return (comm if comm.exists() else None, matrix if matrix.exists() else None)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize communication overhead from profiler JSON exports")
    parser.add_argument("--trace-root", help="profiler export root")
    parser.add_argument("--communication-json", help="explicit communication.json path")
    parser.add_argument("--matrix-json", help="explicit communication_matrix.json path")
    parser.add_argument("--output-json", required=True, help="path to write the communication summary JSON")
    args = parser.parse_args()

    comm_path = Path(args.communication_json).resolve() if args.communication_json else None
    matrix_path = Path(args.matrix_json).resolve() if args.matrix_json else None
    if args.trace_root:
        inferred_comm, inferred_matrix = default_comm_paths(Path(args.trace_root).resolve())
        comm_path = comm_path or inferred_comm
        matrix_path = matrix_path or inferred_matrix

    if not comm_path or not comm_path.exists():
        raise SystemExit("communication.json was not found. Provide --communication-json or --trace-root with communication artifacts.")

    comm_records = flatten_records(read_json(comm_path))
    summary = summarize_records(comm_records)
    matrix_payload = None
    imbalance_ratio = None
    if matrix_path and matrix_path.exists():
        matrix_payload = read_json(matrix_path)
        matrix_result = matrix_stats(matrix_payload)
        imbalance_ratio = matrix_result["imbalance_ratio"]

    report = {
        "source_files": {
            "communication_json": str(comm_path),
            "matrix_json": str(matrix_path) if matrix_path and matrix_path.exists() else None,
        },
        **summary,
        "matrix_imbalance_ratio": imbalance_ratio,
        "likely_domains": ["communication"] if summary["top_collectives"] else [],
        "next_action": (
            "Validate communication overlap, collective count, and step tail before changing compute kernels."
            if summary["communication_pressure"] in {"moderate", "high"}
            else "Communication does not currently dominate the exported evidence."
        ),
    }
    write_json(Path(args.output_json), report)
    print(json.dumps({"dominant_collective": report["dominant_collective"], "pressure": report["communication_pressure"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
