#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def default_direction(category: str) -> str:
    if category == "communication":
        return "Check communication overlap, bucket/fusion settings, synchronization points, and collective count first."
    return "Check the hotspot operator itself, fusion opportunities, graph execution shape, and redundant upstream compute first."


def default_rerun_metrics(category: str) -> list[str]:
    if category == "communication":
        return [
            "communication time share",
            "collective count",
            "step tail",
            "step time",
        ]
    return [
        "hotspot operator time share",
        "step time or latency",
        "kernel launch density",
    ]


def build_brief(summary: dict, top_k: int) -> dict:
    top_ops = summary.get("top_operators", [])[:top_k]
    priority = []
    for idx, item in enumerate(top_ops, 1):
        category = item.get("category", "computation_or_other")
        priority.append(
            {
                "rank": idx,
                "operator": item["operator"],
                "share_percent": item["share_percent"],
                "category": category,
                "why_priority": f"{item['operator']} currently takes {item['share_percent']}% of the measured time and should be handled before lower-share operators.",
                "first_optimization_direction": default_direction(category),
                "rerun_metrics": default_rerun_metrics(category),
            }
        )

    return {
        "source_file": summary.get("source_file"),
        "top_n_cumulative_share_percent": summary.get("summary", {}).get(
            "top_n_cumulative_share_percent"
        ),
        "primary_focus": priority[0]["operator"] if priority else None,
        "priority_queue": priority,
        "agent_notes": [
            "Focus on the top 1 to top 3 operators instead of expanding evenly into the long tail.",
            "Explain why the top-ranked operator deserves attention before giving the first optimization direction.",
            "After rerun, check only the key metrics that match the operator category.",
        ],
    }


def render_markdown(brief: dict) -> str:
    lines = [
        "# Hotspot Brief",
        "",
        f"- source file: `{brief.get('source_file')}`",
        f"- top-N cumulative share: `{brief.get('top_n_cumulative_share_percent')}%`",
        f"- primary focus: `{brief.get('primary_focus')}`",
        "",
        "## Priority Queue",
        "",
    ]
    for item in brief.get("priority_queue", []):
        lines.append(f"{item['rank']}. `{item['operator']}`")
        lines.append(f"   - share: `{item['share_percent']}%`")
        lines.append(f"   - category: `{item['category']}`")
        lines.append(f"   - why: {item['why_priority']}")
        lines.append(f"   - first direction: {item['first_optimization_direction']}")
        lines.append(
            "   - rerun metrics: " + ", ".join(f"`{metric}`" for metric in item["rerun_metrics"])
        )
    lines.extend(["", "## Agent Notes", ""])
    for note in brief.get("agent_notes", []):
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a compact hotspot brief from hotspot_summary.json")
    parser.add_argument("--input-json", required=True, help="hotspot_summary.json path")
    parser.add_argument("--output-json", required=True, help="brief json path")
    parser.add_argument("--output-md", required=True, help="brief markdown path")
    parser.add_argument("--top-k", type=int, default=3, help="number of prioritized operators")
    args = parser.parse_args()

    input_json = Path(args.input_json)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)

    summary = json.loads(input_json.read_text(encoding="utf-8"))
    brief = build_brief(summary, args.top_k)
    output_json.write_text(json.dumps(brief, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.write_text(render_markdown(brief), encoding="utf-8")
    print(json.dumps({"primary_focus": brief["primary_focus"], "top_k": len(brief["priority_queue"])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
