#!/usr/bin/env python3
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional


NAME_KEYS = {
    "op_name",
    "operator",
    "operator_name",
    "name",
    "kernel_name",
    "task_name",
}

TIME_KEYS = {
    "total_time",
    "total_time_us",
    "total_time_ms",
    "duration",
    "duration_us",
    "duration_ms",
    "avg_time",
    "avg_time_us",
    "execution_time",
    "execution_time_us",
    "self_time",
}

COMM_PATTERNS = (
    "allreduce",
    "all_reduce",
    "allgather",
    "all_gather",
    "reducescatter",
    "reduce_scatter",
    "broadcast",
    "hccl",
)


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def parse_number(value: str) -> Optional[float]:
    text = value.strip()
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


def classify_op(name: str) -> str:
    lower = name.lower()
    if any(token in lower for token in COMM_PATTERNS):
        return "communication"
    return "computation_or_other"


def load_csv_rows(path: Path) -> tuple[list[dict], Optional[str], Optional[str]]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return rows, None, None
        normalized = {field: normalize_key(field) for field in reader.fieldnames}
        name_field = next((field for field, key in normalized.items() if key in NAME_KEYS), None)
        time_field = next((field for field, key in normalized.items() if key in TIME_KEYS), None)
        if not name_field or not time_field:
            return rows, None, None
        for raw in reader:
            name = (raw.get(name_field) or "").strip()
            time_value = parse_number(raw.get(time_field) or "")
            if not name or time_value is None:
                continue
            rows.append({"name": name, "time": time_value})
    return rows, name_field, time_field


def find_best_source(input_dir: Path) -> tuple[Optional[Path], list[dict]]:
    best_path = None
    best_rows: list[dict] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".csv", ".txt"}:
            continue
        rows, _, _ = load_csv_rows(path)
        if len(rows) > len(best_rows):
            best_rows = rows
            best_path = path
    return best_path, best_rows


def build_report(rows: list[dict], source_path: Path, top_n: int) -> dict:
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        totals[row["name"]] += row["time"]
        counts[row["name"]] += 1

    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    grand_total = sum(time for _, time in ranked) or 1.0
    top_items = []
    cumulative = 0.0
    for name, total_time in ranked[:top_n]:
        cumulative += total_time
        top_items.append(
            {
                "operator": name,
                "total_time": total_time,
                "share_percent": round(total_time / grand_total * 100, 2),
                "count": counts[name],
                "category": classify_op(name),
            }
        )

    priority = []
    for item in top_items[:3]:
        priority.append(
            {
                "operator": item["operator"],
                "reason": f"{item['operator']} share is {item['share_percent']}% and should be investigated before lower-cost operators.",
                "category": item["category"],
            }
        )

    return {
        "source_file": str(source_path),
        "rows_used": len(rows),
        "unique_operators": len(ranked),
        "top_operators": top_items,
        "priority_list": priority,
        "summary": {
            "top_n_cumulative_share_percent": round(cumulative / grand_total * 100, 2),
            "first_focus": [item["operator"] for item in top_items[:3]],
        },
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# msprof Hotspot Summary",
        "",
        f"- source file: `{report['source_file']}`",
        f"- rows used: `{report['rows_used']}`",
        f"- unique operators: `{report['unique_operators']}`",
        f"- top-N cumulative share: `{report['summary']['top_n_cumulative_share_percent']}%`",
        "",
        "## Priority List",
        "",
    ]
    for idx, item in enumerate(report["priority_list"], 1):
        lines.append(
            f"{idx}. `{item['operator']}` ({item['category']}): {item['reason']}"
        )
    lines.extend(["", "## Top Operators", ""])
    for item in report["top_operators"]:
        lines.append(
            f"- `{item['operator']}`: total_time={item['total_time']}, share={item['share_percent']}%, count={item['count']}, category={item['category']}"
        )
    lines.extend(
        [
            "",
            "## Suggested Use",
            "",
            "- Focus on the top 1 to top 3 time-consuming operators instead of spreading effort evenly.",
            "- If the leading operator is in the communication category, check overlap, bucket/fusion settings, and synchronization points first.",
            "- If the leading operator is in the computation_or_other category, check the operator itself, fusion opportunities, graph execution shape, and redundant upstream compute first.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize msprof operator hotspots from an output directory")
    parser.add_argument("--input-dir", required=True, help="msprof output directory")
    parser.add_argument("--output-md", required=True, help="path to write markdown summary")
    parser.add_argument("--output-json", required=True, help="path to write json summary")
    parser.add_argument("--top-n", type=int, default=10, help="number of top operators to keep")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_md = Path(args.output_md)
    output_json = Path(args.output_json)

    source_path, rows = find_best_source(input_dir)
    if not source_path or not rows:
        raise SystemExit("No operator time table with recognizable name/time columns was found under the input directory.")

    report = build_report(rows, source_path, args.top_n)
    output_md.write_text(render_markdown(report), encoding="utf-8")
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
