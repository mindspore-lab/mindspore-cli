# Context Recovery

Use this reference before Step 1 when the workspace is already prepared and the
user likely ran the model before invoking `performance-agent`.

## Goal

Recover the minimum useful baseline from local artifacts first, then ask the
user to confirm it. Do not open with a broad questionnaire if the workspace
already contains enough evidence to propose a candidate baseline.

## What To Look For

Search for the smallest set of artifacts that can anchor a real baseline:

- Python entry scripts such as `train.py`, `main.py`, `run_*.py`, `infer.py`,
  `launch.py`, or copied `*-perf.py` files
- launcher shell scripts such as `run.sh`, `train.sh`, `launch.sh`, `msrun*.sh`,
  or `torchrun*.sh`
- model configs in `yaml`, `yml`, `json`, `toml`, or `ini`
- checkpoints such as `*.ckpt`, `*.pt`, `*.pth`, `*.safetensors`
- logs with step time, throughput, latency, memory, rank size, or device count
- profiler exports such as `PROF_*`, `mindstudio_profiler_output`, `msprof`,
  `ascend_*`, `hotspot_summary.json`, or operator tables

Prefer recent files and directories, but do not assume newest means correct.

## Recovery Procedure

1. Scan the current working directory first.
2. Use `scripts/find_run_context.py --root <workdir>` for an initial pass when
   available.
3. Inspect the most relevant hits yourself and extract tentative fields:
   - stack: `ms` or `pta`
   - training or inference
   - single-card or distributed
   - primary metric focus if visible
   - entry script
   - config path
   - checkpoint path
   - profiler export path
   - end-to-end baseline evidence such as throughput, latency, or step time
4. Summarize the candidate baseline in a compact form and ask the user to
   confirm or correct it.

## Confirmation Rules

- If one plausible baseline is found, show one compact candidate and ask the
  user to confirm it.
- If multiple plausible baselines are found, show the top 1 to top 3 candidates
  and ask which one matches the run they want to optimize.
- If the linkage between artifacts is weak, say so explicitly.
- Never silently choose a baseline when multiple runs are plausible.

## If Nothing Usable Is Found

Ask whether the user wants to rerun a comparable baseline.

Before rerunning, confirm:

- model entry script
- model config
- checkpoint
- training or inference
- single-card or distributed scale
- primary metric focus if known
- output directory for new logs or profiler artifacts

Keep the rerun comparable unless the user explicitly wants a changed setup.

## Output Pattern

Use a short recovery summary before asking for confirmation:

- what was found
- what was inferred
- what is still unknown
- the one confirmation question needed next
