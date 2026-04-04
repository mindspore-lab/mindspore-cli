# Profiler Output Layout

Read this file when the user provides a generated profiler directory and you
need to understand which files are most useful for bottleneck analysis.

This reference is based on the official Ascend MindStudio / MindSpore profiler
documentation for MindSpore output layout, and the `torch_npu` output is
treated as layout-compatible for the same analysis flow:

- MindSpore profiler output structure:
  [official documentation](https://www.hiascend.com/document/detail/zh/mindstudio/81RC1/msquickstart/atlasquick_train_0017.html?framework=mindspore)
- `torch_npu` collection guide:
  [official documentation](https://www.hiascend.com/document/detail/zh/mindstudio/81RC1/msquickstart/atlasquick_train_0018.html?framework=mindspore)

## Goal

Do not scan the whole profiler directory blindly.

Start from the smallest set of high-signal files that answer:

1. where step time is spent
2. which operators dominate
3. whether host, input pipeline, or communication causes idle gaps
4. whether memory pressure is a primary limiter

## Root Layout

MindSpore runs produce a root directory shaped like:

```text
{worker_name}_{timestamp}_ascend_ms/
├── profiler_info_{Rank_ID}.json
├── profiler_metadata.json
├── ASCEND_PROFILER_OUTPUT/
├── FRAMEWORK/
└── PROF_xxx.../
```

The official MindSpore docs describe this as the full collected and parsed
result layout. Important details from the docs:

- `ASCEND_PROFILER_OUTPUT/` contains the integrated and parsed files that the
  skill should prefer first.
- `FRAMEWORK/` is raw framework-side data and usually does not need direct
  inspection for first-pass diagnosis.
- `PROF_xxx.../mindstudio_profiler_output/` contains parsed CANN-side output
  that can provide fallback detail when the integrated files are insufficient.

`torch_npu` output is similar enough that the same first-pass file priority can
be used unless the current environment shows a different concrete layout.

## First-Pass File Priority

Read files in this order when they exist:

1. `ASCEND_PROFILER_OUTPUT/step_trace_time.csv`
2. `ASCEND_PROFILER_OUTPUT/kernel_details.csv`
3. `ASCEND_PROFILER_OUTPUT/trace_view.json`
4. communication files:
   - `ASCEND_PROFILER_OUTPUT/communication.json`
   - `ASCEND_PROFILER_OUTPUT/communication_matrix.json`
5. memory files:
   - `ASCEND_PROFILER_OUTPUT/memory_record.csv`
   - `ASCEND_PROFILER_OUTPUT/operator_memory.csv`
   - `ASCEND_PROFILER_OUTPUT/npu_module_mem.csv`
6. dataset files when input stall is suspected:
   - `ASCEND_PROFILER_OUTPUT/dataset.csv`
   - `ASCEND_PROFILER_OUTPUT/minddata_pipeline_summary_{Rank_ID}.csv`
   - `ASCEND_PROFILER_OUTPUT/minddata_pipeline_summary_{Rank_ID}.json`
7. summarized operator files when present in parsed CANN output:
   - `PROF_xxx.../mindstudio_profiler_output/op_summary_*.csv`
   - `PROF_xxx.../mindstudio_profiler_output/task_time_*.csv`

If a `hotspot_summary.json` or `hotspot_summary.md` already exists in the skill
run output, use that before re-reading raw operator tables.

## File Meaning by Bottleneck Type

### Step or Stage Dominance

Read first:

- `step_trace_time.csv`

Use it to decide:

- whether the issue is concentrated in step tail, host wait, or a stable heavy
  stage
- whether the step-level slowdown is consistent enough to compare before/after

### Operator Hotspots

Read first:

- `kernel_details.csv`
- fallback: `op_summary_*.csv`

Use them to decide:

- which kernels or operators dominate execution time
- whether the top hotspots are compute-heavy or likely communication-heavy
- whether `Step Id` exists and helps map hotspots to specific schedule windows

The official MindSpore docs note that `kernel_details.csv` contains all
operators executed on NPU, and `Step Id` appears when `schedule` plus `step`
marking is used.

### Timeline and Idle Gaps

Read first:

- `trace_view.json`

Use it to decide:

- whether device idle gaps are caused by host launch overhead
- whether input stalls or synchronization gaps dominate
- whether communication overlaps with computation or serializes the step

### Communication Overhead

Read first:

- `communication.json`
- `communication_matrix.json`

Use them to decide:

- whether collective communication is the dominant overhead
- whether bandwidth, size, or rank-level imbalance is visible
- whether the issue belongs in communication optimization before compute tuning

### Memory Pressure

Read first:

- `memory_record.csv`
- `operator_memory.csv`
- `npu_module_mem.csv`

Use them to decide:

- whether peak memory is tied to a small set of operators
- whether memory pressure or fragmentation is limiting batch size
- which operator or stage should be the first memory optimization target

### Input Pipeline Stall

Read first:

- `dataset.csv`
- `minddata_pipeline_summary_{Rank_ID}.csv`
- `minddata_pipeline_summary_{Rank_ID}.json`

Use them to decide:

- whether data pipeline work is slower than device consumption
- whether queue utilization or queue-empty frequency points to dataloader stall
- whether CPU-side preprocessing is a more plausible first bottleneck than NPU
  kernels

The official MindSpore docs explicitly note that the minddata pipeline summary
files include bottleneck warnings and suggestions.

## Practical Reading Rules

- Prefer `ASCEND_PROFILER_OUTPUT/` over raw `FRAMEWORK/` and raw `PROF`
  subdirectories for first-pass analysis.
- Use parsed CANN files as fallback detail, not the default starting point.
- If only one file can be requested from the user, ask for:
  - `step_trace_time.csv` when the symptom is generic slowness
  - `kernel_details.csv` when the symptom is operator cost
  - `trace_view.json` when the symptom is idle gaps or launch overhead
  - memory files when the symptom is batch-size pressure
  - communication files when the symptom is multi-card slowdown
  - dataset files when the symptom is input stall
- Do not ask for the whole export directory if one or two files can answer the
  current classification question.

## Notes

- This file is for runtime interpretation of profiler outputs, not for changing
  collection behavior.
- When the environment diverges from the official layout, trust the actual
  generated file names in the user’s profiler directory and adapt the same
  priority order.
