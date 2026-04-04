#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from readiness_core import build_fix_actions, build_state, execute_fix_actions, write_readiness_env_file
from readiness_report import build_report, write_report_artifacts


VALUE_FLAGS = {
    "--working-dir",
    "--output-dir",
    "--target",
    "--framework-hint",
    "--cann-path",
    "--mode",
    "--entry-script",
    "--selected-python",
    "--config-path",
    "--model-path",
    "--model-hub-id",
    "--dataset-path",
    "--dataset-hub-id",
    "--dataset-split",
    "--checkpoint-path",
    "--task-smoke-cmd",
    "--timeout-seconds",
}
BOOL_FLAGS = {"--check", "--fix", "--allow-network", "--verbose"}
HELP_FLAGS = {"-h", "--help"}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sanitize_cli_args(raw_args: List[str]) -> Tuple[List[str], List[dict]]:
    sanitized: List[str] = []
    ignored: List[dict] = []
    index = 0

    while index < len(raw_args):
        token = raw_args[index]
        if token in HELP_FLAGS:
            sanitized.append(token)
            index += 1
            continue

        if token.startswith("--"):
            flag, has_inline_value, inline_value = token.partition("=")

            if flag in BOOL_FLAGS:
                sanitized.append(flag)
                if has_inline_value:
                    ignored.append({"token": token, "reason": "bool_flag_inline_value_ignored"})
                index += 1
                continue

            if flag in VALUE_FLAGS:
                if has_inline_value:
                    if inline_value:
                        sanitized.extend([flag, inline_value])
                    else:
                        ignored.append({"token": token, "reason": "missing_value"})
                    index += 1
                    continue

                if index + 1 < len(raw_args) and not raw_args[index + 1].startswith("-"):
                    sanitized.extend([flag, raw_args[index + 1]])
                    index += 2
                    continue

                ignored.append({"token": token, "reason": "missing_value"})
                index += 1
                continue

            ignored.append({"token": token, "reason": "unknown_flag"})
            if index + 1 < len(raw_args) and not raw_args[index + 1].startswith("-"):
                ignored.append({"token": raw_args[index + 1], "reason": "unknown_flag_value"})
                index += 2
            else:
                index += 1
            continue

        ignored.append({"token": token, "reason": "orphan_value"})
        index += 1

    return sanitized, ignored


def detect_removed_mode_usage(raw_args: List[str]) -> Optional[str]:
    index = 0
    while index < len(raw_args):
        token = raw_args[index]
        if token == "--auto":
            return "auto mode was removed; use --fix for readiness remediation."
        if token == "--mode":
            if index + 1 < len(raw_args) and raw_args[index + 1] == "auto":
                return "mode=auto was removed; use --mode fix for readiness remediation."
            index += 2
            continue
        if token.startswith("--mode=") and token.partition("=")[2] == "auto":
            return "mode=auto was removed; use --mode fix for readiness remediation."
        index += 1
    return None


def normalize_mode_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> str:
    alias_modes = [mode for mode, enabled in (("check", args.check), ("fix", args.fix)) if enabled]
    if len(alias_modes) > 1:
        parser.error("use at most one of --check or --fix")
    alias_mode = alias_modes[0] if alias_modes else None
    explicit_mode = args.mode
    if alias_mode and explicit_mode and explicit_mode != alias_mode:
        parser.error("--mode conflicts with the requested alias flag")
    if alias_mode:
        return alias_mode
    if explicit_mode:
        return explicit_mode
    return "check"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the streamlined readiness-agent check or fix workflow", allow_abbrev=False)
    parser.add_argument("--working-dir", help="workspace root (defaults to the current shell path)")
    parser.add_argument("--output-dir", help="output directory for readiness artifacts (defaults to <working_dir>/readiness-output)")
    parser.add_argument("--target", default="auto", help="training, inference, or auto")
    parser.add_argument("--framework-hint", help="explicit framework preference such as mindspore or pta")
    parser.add_argument("--cann-path", help="explicit CANN root or set_env.sh path")
    parser.add_argument("--mode", choices=("check", "fix"), help="check or fix")
    parser.add_argument("--check", action="store_true", help="alias for --mode check")
    parser.add_argument("--fix", action="store_true", help="alias for --mode fix")
    parser.add_argument("--verbose", action="store_true", help="accepted for caller compatibility; currently no-op")
    parser.add_argument("--entry-script", help="explicit entry script path")
    parser.add_argument("--selected-python", help="explicit Python interpreter for the workspace")
    parser.add_argument("--config-path", help="explicit training or inference config path")
    parser.add_argument("--model-path", help="explicit model directory or model artifact path")
    parser.add_argument("--model-hub-id", help="explicit Hugging Face model repo ID")
    parser.add_argument("--dataset-path", help="explicit dataset path")
    parser.add_argument("--dataset-hub-id", help="explicit Hugging Face dataset repo ID")
    parser.add_argument("--dataset-split", help="optional dataset split")
    parser.add_argument("--checkpoint-path", help="explicit checkpoint path")
    parser.add_argument("--task-smoke-cmd", help="optional explicit smoke command")
    parser.add_argument("--allow-network", action="store_true", help="allow network-dependent remediation")
    parser.add_argument("--timeout-seconds", type=int, default=10, help="timeout for explicit smoke execution")
    return parser


def main() -> int:
    parser = build_parser()
    raw_cli_args = sys.argv[1:]
    removed_mode_error = detect_removed_mode_usage(raw_cli_args)
    if removed_mode_error:
        print(json.dumps({"error": removed_mode_error}, indent=2), file=sys.stderr)
        return 2

    sanitized_cli_args, ignored_cli_args = sanitize_cli_args(raw_cli_args)
    args = parser.parse_args(sanitized_cli_args)
    args.mode = normalize_mode_args(parser, args)

    working_dir = Path(args.working_dir or ".").resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else working_dir / "readiness-output"
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    inputs_snapshot = {
        "working_dir": str(working_dir),
        "output_dir": str(output_dir),
        "target": args.target,
        "framework_hint": args.framework_hint,
        "cann_path": args.cann_path,
        "mode": args.mode,
        "entry_script": args.entry_script,
        "selected_python": args.selected_python,
        "config_path": args.config_path,
        "model_path": args.model_path,
        "model_hub_id": args.model_hub_id,
        "dataset_path": args.dataset_path,
        "dataset_hub_id": args.dataset_hub_id,
        "dataset_split": args.dataset_split,
        "checkpoint_path": args.checkpoint_path,
        "task_smoke_cmd": args.task_smoke_cmd,
        "allow_network": args.allow_network,
        "timeout_seconds": args.timeout_seconds,
        "raw_cli_args": raw_cli_args,
        "ignored_cli_args": ignored_cli_args,
    }
    write_json(meta_dir / "inputs.json", inputs_snapshot)

    initial_state = build_state(args, working_dir)
    actions = build_fix_actions(initial_state["target"], initial_state["closure"], initial_state["normalized"], args.allow_network)
    fix_applied = execute_fix_actions(initial_state["target"], initial_state["closure"], actions, args.mode == "fix")

    final_state = build_state(args, working_dir) if fix_applied.get("executed_actions") else initial_state
    readiness_env_path = (working_dir / ".readiness.env").resolve()
    write_readiness_env_file(readiness_env_path, working_dir, final_state["target"], final_state["closure"])

    env_snapshot = {
        "mode": args.mode,
        "pipeline_passes": 2 if fix_applied.get("executed_actions") else 1,
        "control_python": sys.executable,
        "initial_selection": initial_state["closure"]["layers"]["python_environment"],
        "final_selection": final_state["closure"]["layers"]["python_environment"],
        "fix_execute": bool(fix_applied.get("execute")),
        "executed_actions": fix_applied.get("executed_actions", []),
        "failed_actions": fix_applied.get("failed_actions", []),
        "needs_revalidation": fix_applied.get("needs_revalidation", []),
        "readiness_env_path": str(readiness_env_path),
    }
    write_json(meta_dir / "env.json", env_snapshot)

    verdict = build_report(
        final_state["target"],
        final_state["normalized"],
        final_state["checks"],
        final_state["closure"],
        fix_applied,
    )
    write_report_artifacts(output_dir, verdict)

    print(
        json.dumps(
            {
                "status": verdict["status"],
                "target": verdict["target"],
                "can_run": verdict["can_run"],
                "next_action": verdict["next_action"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
