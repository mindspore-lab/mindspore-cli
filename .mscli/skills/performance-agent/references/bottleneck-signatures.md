# Bottleneck Signatures

Read only the branch that matches the dominant pattern in the trace.

## Branch A: Communication Dominates

Use when the trace shows:

- large `AllReduce`, `ReduceScatter`, `AllGather`, or similar collective slices
- long waits near backward end or update
- step tail growing with distributed scale

Primary suspects:

- communication not overlapped with computation
- too many small collectives
- poor bucketization or fusion
- synchronization inserted in the wrong place

Recommended sequence:

1. Confirm whether the cost is inside backward, update, or step tail.
2. Check whether collectives are serialized or overlapped.
3. Recommend one communication-focused change first.
4. Validate by comparing collective count, collective time, and step tail.

## Branch B: Host Launch or Idle Gap Dominates

Use when the trace shows:

- large gaps between kernels
- low device utilization with active host-side work
- many small launches with poor packing

Primary suspects:

- Python-side per-step overhead
- unnecessary sync points
- too much PyNative-style host dispatch
- fragmented execution instead of graph-heavy execution

Recommended sequence:

1. Identify where the idle gap occurs.
2. Distinguish host work from input stalls.
3. Recommend one host-overhead reduction first.
4. Validate by comparing idle gap and kernel launch density.

## Branch C: Input Pipeline Stalls

Use when the trace shows:

- device idle periods before compute starts
- dataloader or input transform time dominating the step gap

Primary suspects:

- insufficient parallel loading
- slow decode or transform path
- missing prefetch or caching

Recommended sequence:

1. Confirm that compute is waiting on input.
2. Keep the first fix in the input path, not in kernels.
3. Validate with shorter pre-compute idle time and better utilization.

## Branch D: Graph Build or Recompilation Dominates

Use when the trace shows:

- large compile sections
- repeated graph build or recompile behavior
- latency dominated by setup rather than steady-state kernels

Primary suspects:

- unstable shapes
- control-flow changes that force recompilation
- repeated graph construction in the serving loop

Recommended sequence:

1. Separate first-run compile cost from steady-state latency.
2. Confirm whether recompilation repeats or only happens once.
3. Recommend one graph-stabilization change first.
4. Validate by comparing compile count and compile time.

## Branch E: Compute Operator Hotspot

Use when the trace shows:

- one or a few operators dominating execution time
- little evidence of communication or input stalls

Primary suspects:

- inefficient operator choice
- missing fusion
- backend kernel path not ideal
- redundant computation

Recommended sequence:

1. Name the dominant operator or operator family.
2. Confirm it is truly dominant in time share.
3. Recommend one operator- or fusion-focused change first.
4. Validate by comparing operator time share after rerun.

## Branch F: Memory Pressure or Memory-Heavy Operators

Use when the trace shows:

- one or a few operators dominating peak memory
- memory spikes that cap batch size
- allocation pressure causing secondary slowdown

Primary suspects:

- activation-heavy modules
- excessive temporary tensors
- precision choice
- layout or batch-size effects
- missing recomputation or checkpointing where appropriate

Recommended sequence:

1. Name the memory-heavy operator or stage.
2. Distinguish steady high memory from short spikes.
3. Recommend one memory-focused change first.
4. Validate by comparing peak memory and the top memory-heavy stage.

## Branch G: Optimizer or Update Dominates

Use when the trace shows:

- update phase dominating the step
- regression appearing after distributed enablement
- optimizer time inflated by sync or communication

Primary suspects:

- communication surfacing inside update
- too many small reductions
- update-side synchronization
- genuine optimizer compute cost only after communication is ruled out

Recommended sequence:

1. Decide whether update cost is communication-heavy or compute-heavy.
2. Prefer a communication-focused fix first if distributed enablement caused
   the regression.
3. Validate with update-time share, collective time, and step tail.

Common Ascend dual-stack note:

- If the regression appears only after multi-card enablement in MindSpore or
  torch_npu, assume communication-heavy until the trace proves otherwise.
