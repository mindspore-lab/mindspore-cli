#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  collect_msprof.sh --stack <ms|pta> --script <entry.py> --output-dir <dir> [--python <bin>] [--inject-only] [-- <script args...>]

Purpose:
  Create a copied `*-perf.py` entry script, inject deterministic profiler hooks
  for MindSpore or PTA, optionally execute the copied script, and recover the
  generated profiler root plus any available structured summaries.

Important:
  - The original training script is not modified.
  - The caller must specify `--stack ms` or `--stack pta`.
  - Execution defaults to running the copied script. Use `--inject-only` to
    stop after generating the instrumented `*-perf.py`.
  - Extra CLI arguments after `--` are passed to the copied script unchanged.

Example:
  collect_msprof.sh --stack pta --script inference.py --output-dir /tmp/msprof-run -- --model-path ./ckpt

Outputs:
  - copied perf entry script with injected profiler hooks
  - collect_metadata.json
  - inject_metadata.json
  - locator.json when a profiler root is found or searched
  - summary JSON files when the corresponding profiler exports exist
EOF
}

OUTPUT_DIR=""
STACK=""
SCRIPT_PATH=""
PYTHON_BIN="${PYTHON:-python}"
RUN_MODE="run"
SCRIPT_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stack)
      STACK="${2:-}"
      shift 2
      ;;
    --script)
      SCRIPT_PATH="${2:-}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --python)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --inject-only)
      RUN_MODE="inject_only"
      shift
      ;;
    --)
      shift
      SCRIPT_ARGS=("$@")
      break
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
  echo "--output-dir is required" >&2
  usage >&2
  exit 2
fi

if [[ -z "$STACK" ]]; then
  echo "--stack is required" >&2
  usage >&2
  exit 2
fi

if [[ "$STACK" != "ms" && "$STACK" != "pta" ]]; then
  echo "--stack must be either 'ms' or 'pta'" >&2
  exit 2
fi

if [[ -z "$SCRIPT_PATH" ]]; then
  echo "--script is required" >&2
  usage >&2
  exit 2
fi

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "Training entry script does not exist: $SCRIPT_PATH" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

SCRIPT_DIRNAME="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
SCRIPT_BASENAME="$(basename "$SCRIPT_PATH")"
SCRIPT_STEM="${SCRIPT_BASENAME%.py}"
PERF_SCRIPT_PATH="$SCRIPT_DIRNAME/${SCRIPT_STEM}-perf.py"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INJECT_SCRIPT="$SCRIPT_DIR/inject_profiler.py"
LOCATOR_SCRIPT="$SCRIPT_DIR/locate_profiler_output.py"
STEP_SCRIPT="$SCRIPT_DIR/summarize_step_breakdown.py"
COMM_SCRIPT="$SCRIPT_DIR/summarize_communication.py"
MEMORY_SCRIPT="$SCRIPT_DIR/summarize_memory_pressure.py"
INPUT_SCRIPT="$SCRIPT_DIR/summarize_input_pipeline.py"
TRACE_GAP_SCRIPT="$SCRIPT_DIR/summarize_trace_gaps.py"
SUMMARY_SCRIPT="$SCRIPT_DIR/summarize_msprof_hotspots.py"
BRIEF_SCRIPT="$SCRIPT_DIR/build_hotspot_brief.py"

if [[ ! -f "$INJECT_SCRIPT" ]]; then
  echo "Profiler injection script not found: $INJECT_SCRIPT" >&2
  exit 2
fi

INJECT_METADATA="$OUTPUT_DIR/inject_metadata.json"
LOCATOR_JSON="$OUTPUT_DIR/locator.json"
COLLECT_METADATA="$OUTPUT_DIR/collect_metadata.json"

echo "Injecting profiler hooks into copied script:"
echo "  $SCRIPT_PATH -> $PERF_SCRIPT_PATH"
"$PYTHON_BIN" "$INJECT_SCRIPT" \
  --stack "$STACK" \
  --input-script "$SCRIPT_PATH" \
  --output-script "$PERF_SCRIPT_PATH" \
  --trace-dir "$OUTPUT_DIR" \
  --metadata-json "$INJECT_METADATA"

RUN_STATUS="not_run"
RUN_EXIT_CODE=0
if [[ "$RUN_MODE" == "run" ]]; then
  echo "Running copied profiler script..."
  pushd "$SCRIPT_DIRNAME" >/dev/null
  if "$PYTHON_BIN" "$PERF_SCRIPT_PATH" "${SCRIPT_ARGS[@]}"; then
    RUN_STATUS="completed"
  else
    RUN_EXIT_CODE=$?
    RUN_STATUS="failed"
  fi
  popd >/dev/null
  if [[ "$RUN_STATUS" == "failed" ]]; then
    echo "Profiler run failed with exit code $RUN_EXIT_CODE" >&2
  fi
fi

cat > "$COLLECT_METADATA" <<EOF
{
  "stack": "$STACK",
  "output_dir": "$OUTPUT_DIR",
  "source_script": "$SCRIPT_PATH",
  "perf_script": "$PERF_SCRIPT_PATH",
  "python_bin": "$PYTHON_BIN",
  "run_mode": "$RUN_MODE",
  "run_status": "$RUN_STATUS",
  "run_exit_code": $RUN_EXIT_CODE
}
EOF

echo "Locating profiler output under $OUTPUT_DIR ..."
"$PYTHON_BIN" "$LOCATOR_SCRIPT" --working-dir "$OUTPUT_DIR" --output-json "$LOCATOR_JSON"

SELECTED_ROOT="$("$PYTHON_BIN" - "$LOCATOR_JSON" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload.get("selected_root") or "")
PY
)"

if [[ -n "$SELECTED_ROOT" ]]; then
  echo "Recovered profiler root:"
  echo "  $SELECTED_ROOT"

  if [[ -f "$STEP_SCRIPT" ]]; then
    "$PYTHON_BIN" "$STEP_SCRIPT" --trace-root "$SELECTED_ROOT" --output-json "$OUTPUT_DIR/step-summary.json" || true
  fi
  if [[ -f "$COMM_SCRIPT" ]]; then
    "$PYTHON_BIN" "$COMM_SCRIPT" --trace-root "$SELECTED_ROOT" --output-json "$OUTPUT_DIR/communication-summary.json" || true
  fi
  if [[ -f "$MEMORY_SCRIPT" ]]; then
    "$PYTHON_BIN" "$MEMORY_SCRIPT" --trace-root "$SELECTED_ROOT" --output-json "$OUTPUT_DIR/memory-summary.json" || true
  fi
  if [[ -f "$INPUT_SCRIPT" ]]; then
    "$PYTHON_BIN" "$INPUT_SCRIPT" --trace-root "$SELECTED_ROOT" --output-json "$OUTPUT_DIR/input-summary.json" || true
  fi
  if [[ -f "$TRACE_GAP_SCRIPT" ]]; then
    "$PYTHON_BIN" "$TRACE_GAP_SCRIPT" --trace-root "$SELECTED_ROOT" --output-json "$OUTPUT_DIR/trace-gaps-summary.json" || true
  fi
else
  echo "No profiler root was recovered. Keep locator.json and verify the runtime generated Ascend profiler files." >&2
fi

if [[ -f "$SUMMARY_SCRIPT" ]]; then
  echo "Attempting to summarize msprof hotspots from collected artifacts, if any..."
  if "$PYTHON_BIN" "$SUMMARY_SCRIPT" \
    --input-dir "$OUTPUT_DIR" \
    --output-md "$OUTPUT_DIR/hotspot_summary.md" \
    --output-json "$OUTPUT_DIR/hotspot_summary.json"; then
    echo "Hotspot summary written to:"
    echo "  $OUTPUT_DIR/hotspot_summary.md"
    echo "  $OUTPUT_DIR/hotspot_summary.json"
    if [[ -f "$BRIEF_SCRIPT" ]]; then
      echo "Building hotspot brief..."
      if "$PYTHON_BIN" "$BRIEF_SCRIPT" \
        --input-json "$OUTPUT_DIR/hotspot_summary.json" \
        --output-json "$OUTPUT_DIR/hotspot_brief.json" \
        --output-md "$OUTPUT_DIR/hotspot_brief.md"; then
        echo "Hotspot brief written to:"
        echo "  $OUTPUT_DIR/hotspot_brief.md"
        echo "  $OUTPUT_DIR/hotspot_brief.json"
      else
        echo "Hotspot brief generation failed." >&2
      fi
    fi
  else
    echo "Hotspot summary was not generated. No recognizable operator time table was found." >&2
  fi
else
  echo "Hotspot summary script not found: $SUMMARY_SCRIPT" >&2
fi

if [[ "$RUN_STATUS" == "failed" ]]; then
  exit "$RUN_EXIT_CODE"
fi
