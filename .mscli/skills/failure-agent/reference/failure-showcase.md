# Failure Showcase (Known-Issue Transition Reference)

This file is a local fallback reference for `failure-agent` when Factory cards
are unavailable or unreadable.

It is intentionally written as Markdown, but each entry mirrors the core
semantics of a Factory `known_issue` card so the content can be migrated later
with minimal reshaping.

Defaults used in this file unless a line says otherwise:

- `kind: known_issue`
- `symptom: failure`
- `lifecycle_state: stable`
- `source_kind: bootstrap`
- `confidence_level: bootstrap`

Do not auto-mutate Factory from this file.

## Shared Failures

### Missing CANN Environment

- kind: known_issue
- symptom: failure
- id_hint: missing-cann-environment
- severity: high
- lifecycle_state: stable
- source_kind: bootstrap
- confidence_level: bootstrap
- tags: [shared, ascend, cann, environment, import, setup]
- affects_platforms: [ascend]
- detection_pattern: "libascendcl.so not found|libhccl.so not found|ASCEND_OPP_PATH|cannot find CANN"
- description: Runtime initialization fails because required CANN libraries or environment variables are missing, unset, or point to an incomplete installation.
- fix_summary: Source the Ascend toolkit environment script, confirm CANN is installed, and re-check the required Ascend environment variables before retrying.
- validation: Re-run the failing import or startup command and confirm the missing-library or missing-environment error no longer appears.

### Out of Memory

- kind: known_issue
- symptom: failure
- id_hint: device-out-of-memory
- severity: high
- lifecycle_state: stable
- source_kind: bootstrap
- confidence_level: bootstrap
- tags: [shared, memory, oom, ascend, gpu, allocation]
- affects_platforms: [ascend, gpu]
- detection_pattern: "EL0004|FAIL_TO_ALLOCATE_MEMORY|out of memory|device memory exhausted|CUDA out of memory"
- description: Device memory is exhausted by model size, batch size, fragmentation, or concurrent workloads on the same accelerator.
- fix_summary: Reduce memory pressure first by lowering batch size, enabling checkpointing or recompute, and clearing stale cached memory before deeper analysis.
- validation: Re-run the same step with reduced memory pressure and confirm the allocation failure disappears or moves to a later, larger workload.

### HCCL or Communication Timeout

- kind: known_issue
- symptom: failure
- id_hint: distributed-communication-timeout
- severity: high
- lifecycle_state: stable
- source_kind: bootstrap
- confidence_level: bootstrap
- tags: [shared, ascend, distributed, hccl, timeout, communication]
- affects_platforms: [ascend]
- detection_pattern: "HCCL|EI0002|EI0006|107020|times out|notify wait|socket build"
- description: Distributed execution stalls or aborts because ranks do not make matching progress, network setup is broken, or the communication layer times out during collective execution.
- fix_summary: Verify rank configuration, startup ordering, and network reachability first, then re-check timeout settings and whether one rank exits early.
- validation: Launch a minimal distributed job and confirm all ranks enter and complete the same collective without timeout or mismatch logs.

### NCCL Communication Error

- kind: known_issue
- symptom: failure
- id_hint: nccl-communication-error
- severity: high
- lifecycle_state: stable
- source_kind: bootstrap
- confidence_level: bootstrap
- tags: [shared, gpu, distributed, nccl, timeout, communication]
- affects_platforms: [gpu]
- detection_pattern: "NCCL error|unhandled system error|NCCL.*timeout|distributed.*GPU"
- description: GPU distributed execution fails because the NCCL communication layer cannot establish or complete the expected collective path across ranks.
- fix_summary: Check topology visibility, NCCL version compatibility, firewall or network restrictions, and whether all ranks enter the same collective sequence.
- validation: Re-run a minimal NCCL distributed case with debug logs enabled and confirm the same collective completes across all ranks.

### Feature or Operator Not Supported

- kind: known_issue
- symptom: failure
- id_hint: feature-or-operator-not-supported
- severity: medium
- lifecycle_state: stable
- source_kind: bootstrap
- confidence_level: bootstrap
- tags: [shared, operator, backend, unsupported, fallback]
- affects_platforms: [ascend, gpu, cpu]
- detection_pattern: "feature not supported|operator not supported|not implemented|backend.*not supported"
- description: The requested operator, feature, or dispatch path is not available on the current backend or software-version combination.
- fix_summary: Confirm backend support and version compatibility, then switch to a supported operator path, upgrade the stack, or use a safer fallback backend if available.
- validation: Check the operator or feature against the current stack version and confirm a supported alternative path runs without the same backend-support failure.

### Version Mismatch

- kind: known_issue
- symptom: failure
- id_hint: stack-version-mismatch
- severity: high
- lifecycle_state: stable
- source_kind: bootstrap
- confidence_level: bootstrap
- tags: [shared, ascend, version, compatibility, abi]
- affects_platforms: [ascend]
- detection_pattern: "symbol not found|ABI mismatch|version mismatch|compatibility|import fails after upgrade"
- description: Runtime components were installed from incompatible version sets, so symbols, kernels, or dispatch registrations expected by one layer are missing in another.
- fix_summary: Rebuild or reinstall the stack with an explicitly compatible version matrix instead of mixing independently upgraded components.
- validation: Print the effective component versions, align them to a known-compatible set, and confirm the original import or launch failure is gone.

### Device Task Abort or Heartbeat Lost

- kind: known_issue
- symptom: failure
- id_hint: device-task-abort-heartbeat-lost
- severity: critical
- lifecycle_state: stable
- source_kind: bootstrap
- confidence_level: bootstrap
- tags: [shared, ascend, hardware, heartbeat, abort, device]
- affects_platforms: [ascend]
- detection_pattern: "107010|507010|FORCE STOP|lost heartbeat|device task abort"
- description: The accelerator stops making forward progress because the device enters an unhealthy state, loses heartbeat, or aborts an in-flight task.
- fix_summary: Check device health first, isolate whether the failure is repeatable on the same card only, and reset or replace the device before debugging higher layers.
- validation: Use device-health tools to inspect the card, then re-run on a healthy card and confirm the failure does not recur immediately.

### HBM ECC Error

- kind: known_issue
- symptom: failure
- id_hint: hbm-ecc-error
- severity: critical
- lifecycle_state: stable
- source_kind: bootstrap
- confidence_level: bootstrap
- tags: [shared, ascend, hardware, ecc, hbm, memory]
- affects_platforms: [ascend]
- detection_pattern: "507054|HBM ECC|multi-bit|memory fault|hardware error"
- description: Ascend device memory reports an ECC hardware fault, which usually means the failure is at the device-health layer rather than in model logic or operator implementation.
- fix_summary: Treat the card as unhealthy, inspect device-health telemetry, and retry on a different device before spending time on framework or operator debugging.
- validation: Confirm the ECC signature in device-health output and verify the workload succeeds on a known-good device.

### AI Core Execution Timeout

- kind: known_issue
- symptom: failure
- id_hint: ai-core-execution-timeout
- severity: high
- lifecycle_state: stable
- source_kind: bootstrap
- confidence_level: bootstrap
- tags: [shared, ascend, aicore, timeout, runtime, execution]
- affects_platforms: [ascend]
- detection_pattern: "507014|AICORE_TIMEOUT|AI Core timeout|execution timeout"
- description: An Ascend operator execution path exceeds the runtime timeout window, often because the kernel is stuck, input size is pathological, or the device is already unhealthy.
- fix_summary: Identify the failing operator from runtime logs, reduce the input size to narrow the trigger, and check whether the issue tracks a specific device or stack version.
- validation: Re-run the minimal failing step with runtime logs enabled and confirm the same operator remains the timeout source.

## MindSpore-Focused Failures

### Device Target Mismatch

- kind: known_issue
- symptom: failure
- id_hint: ms-device-target-mismatch
- severity: medium
- lifecycle_state: stable
- source_kind: bootstrap
- confidence_level: bootstrap
- tags: [ms, device_target, context, backend, configuration]
- affects_platforms: [ascend, gpu, cpu]
- detection_pattern: "device_target|should be one of|Ascend GPU CPU|invalid device"
- description: MindSpore is configured to use a device target that does not match the available hardware or uses an invalid target string.
- fix_summary: Correct the `set_context` device target and verify that the requested backend is actually available on the host.
- validation: Print the effective context settings and confirm a minimal MindSpore tensor program runs on the intended backend.

### Graph Compilation Error

- kind: known_issue
- symptom: failure
- id_hint: ms-graph-compilation-error
- severity: high
- lifecycle_state: stable
- source_kind: bootstrap
- confidence_level: bootstrap
- tags: [ms, graph, compile, infer, frontend, pynative]
- affects_platforms: [ascend, gpu, cpu]
- detection_pattern: "graph compile|infer failed|abstract type|compile failed|GRAPH_MODE"
- description: GRAPH_MODE compilation fails because the program uses unsupported control flow, unsupported Python constructs, or incompatible abstract inference paths.
- fix_summary: Narrow the failing construct in a minimal example, compare behavior in PyNative mode, and replace unsupported graph patterns with supported equivalents.
- validation: Re-run the minimal case in PyNative mode and confirm the error is specific to graph compilation rather than a general runtime failure.

### Shape Inference Failure

- kind: known_issue
- symptom: failure
- id_hint: ms-shape-inference-failure
- severity: medium
- lifecycle_state: stable
- source_kind: bootstrap
- confidence_level: bootstrap
- tags: [ms, shape, infer, dimensions, rank]
- affects_platforms: [ascend, gpu, cpu]
- detection_pattern: "shape|infer|dimensions|rank mismatch|expected"
- description: Input tensor shapes or ranks violate operator constraints, causing shape inference to fail before or during execution.
- fix_summary: Inspect the concrete tensor shapes flowing into the failing operator and correct reshape, transpose, broadcast, or rank assumptions.
- validation: Print the intermediate tensor shapes at the failing boundary and confirm they now match the operator contract.

### Dtype Mismatch

- kind: known_issue
- symptom: failure
- id_hint: ms-dtype-mismatch
- severity: medium
- lifecycle_state: stable
- source_kind: bootstrap
- confidence_level: bootstrap
- tags: [ms, dtype, cast, precision, validation]
- affects_platforms: [ascend, gpu, cpu]
- detection_pattern: "TypeError|dtype|Float32|Float16|type not match|cast"
- description: Tensor types or mixed-precision settings do not satisfy operator requirements, so MindSpore rejects the call or fails later in execution.
- fix_summary: Align the input and parameter dtypes explicitly and verify whether mixed-precision settings or implicit casts are creating the mismatch.
- validation: Print the failing tensors' dtypes and confirm the operator runs after explicit casting to a consistent supported type set.

### Context Empty

- kind: known_issue
- symptom: failure
- id_hint: ms-context-empty
- severity: high
- lifecycle_state: stable
- source_kind: bootstrap
- confidence_level: bootstrap
- tags: [ms, ascend, context, initialization, runtime]
- affects_platforms: [ascend]
- detection_pattern: "107002|context is empty|aclrtSetContext|aclrtSetDevice"
- description: Ascend runtime APIs are called before MindSpore has established a valid device context for the current process.
- fix_summary: Ensure context initialization happens before tensor or operator execution and verify device setup runs exactly once in the expected startup path.
- validation: Add a minimal startup check before the first operator call and confirm the context-empty error no longer appears.

### TBE Operator Compilation Error

- kind: known_issue
- symptom: failure
- id_hint: ms-tbe-operator-compilation-error
- severity: high
- lifecycle_state: stable
- source_kind: bootstrap
- confidence_level: bootstrap
- tags: [ms, ascend, tbe, compile, kernel, ub]
- affects_platforms: [ascend]
- detection_pattern: "TBE|compile failed|E9[0-9A-Z]+|EB[0-9A-Z]+|UB overflow|operator compilation"
- description: MindSpore's Ascend backend fails while compiling a TBE operator because the input shape, dtype, or backend toolchain path violates kernel-generation constraints.
- fix_summary: Recheck the operator's input shape and dtype against backend limits, then validate whether the current CANN stack supports that compilation path before forcing fallback behavior.
- validation: Reproduce the failure with the same operator signature and confirm whether the error disappears after aligning shape, dtype, or stack compatibility.

### mint View Op in GRAPH_MODE or JIT

- kind: known_issue
- symptom: failure
- id_hint: ms-mint-view-op-graph-mode
- severity: medium
- lifecycle_state: stable
- source_kind: bootstrap
- confidence_level: bootstrap
- tags: [ms, mint, graph, jit, view, reshape]
- affects_platforms: [ascend, gpu, cpu]
- detection_pattern: "jit_view_unsupported|view|squeeze|flatten|reshape|GRAPH_MODE|graph compile"
- description: Some `mint` view-style APIs are not safe in GRAPH_MODE or JIT lowering, so code that works in eager-style execution can fail or behave incorrectly when compiled.
- fix_summary: Narrow the failing view op, compare with PyNative behavior, and replace the unsupported `mint` view path with a safer `ops.*` path or an explicitly materialized tensor.
- validation: Re-run the same logic in PyNative mode and confirm the failure is tied to the graph-compiled view path rather than to the underlying data itself.

### mint.nn Layer Parameter Validation

- kind: known_issue
- symptom: failure
- id_hint: ms-mint-nn-layer-parameter-validation
- severity: medium
- lifecycle_state: stable
- source_kind: bootstrap
- confidence_level: bootstrap
- tags: [ms, mint, nn, validation, conv, batchnorm, groups]
- affects_platforms: [ascend, gpu, cpu]
- detection_pattern: "Validator|in_channels|out_channels|groups|mint\\.nn\\.Conv|mint\\.nn\\.BatchNorm"
- description: `mint.nn` layers reject invalid parameter combinations early, so the reported runtime failure is often strict argument validation rather than a backend execution bug.
- fix_summary: Check layer parameter invariants directly, especially channel and group relationships, before investigating backend kernels or graph compilation.
- validation: Print the layer configuration and confirm the failure disappears once the documented parameter constraints are satisfied.

### mint.distributed init_process_group Failure

- kind: known_issue
- symptom: failure
- id_hint: ms-mint-distributed-init-process-group-failure
- severity: high
- lifecycle_state: stable
- source_kind: bootstrap
- confidence_level: bootstrap
- tags: [ms, mint, distributed, init_process_group, tcpstore, networking]
- affects_platforms: [ascend, gpu, cpu]
- detection_pattern: "init_process_group|TCPStore|connection refused|MASTER_ADDR|MASTER_PORT|mint\\.distributed"
- description: Distributed startup fails before training begins because the rendezvous or TCP store settings are inconsistent, unreachable, or launched in the wrong order.
- fix_summary: Verify the rendezvous address, port, rank, and world-size settings first, then confirm the master process is reachable before debugging collective kernels.
- validation: Launch a minimal distributed startup test with the same rendezvous settings and confirm all ranks can initialize successfully.

## PTA / torch_npu-Focused Failures

### Delayed Execution Warning

- kind: known_issue
- symptom: failure
- id_hint: pta-delayed-execution-warning
- severity: medium
- lifecycle_state: stable
- source_kind: bootstrap
- confidence_level: bootstrap
- tags: [pta, torch_npu, async, debug, stacktrace]
- affects_platforms: [ascend]
- detection_pattern: "operator is called asynchronously|stacktrace may be inaccurate|ASCEND_LAUNCH_BLOCKING"
- description: torch_npu launches work asynchronously, so the visible Python stack often points to the launch site rather than the true failing device operation.
- fix_summary: Switch to a synchronous debug mode or inspect lower-level runtime logs before assuming the top stack frame is the real failure site.
- validation: Re-run with synchronous launch settings and confirm the reported failure point becomes stable and actionable.
- notes: This is often a diagnosis aid rather than the final root cause.

### CANN Inner Error

- kind: known_issue
- symptom: failure
- id_hint: pta-cann-inner-error
- severity: high
- lifecycle_state: stable
- source_kind: bootstrap
- confidence_level: bootstrap
- tags: [pta, cann, internal, runtime, kernel]
- affects_platforms: [ascend]
- detection_pattern: "EZ9999|E[1-9A-Z]9999|CANN Inner Error"
- description: A lower-level CANN runtime or kernel path fails internally, often because an operator path, shape, dtype, or version combination is not supported as exercised.
- fix_summary: Capture the failing operator context, review the CANN logs, and validate stack compatibility before treating it as an isolated backend bug.
- validation: Reproduce the failure with the smallest operator-level case possible and confirm the same CANN error still appears with the same stack versions.

### AI Core Overflow

- kind: known_issue
- symptom: failure
- id_hint: pta-ai-core-overflow
- severity: high
- lifecycle_state: stable
- source_kind: bootstrap
- confidence_level: bootstrap
- tags: [pta, ascend, overflow, fp16, mixed-precision, nan]
- affects_platforms: [ascend]
- detection_pattern: "207003|ACL_ERROR_RT_AICORE_OVER_FLOW|overflow|loss becomes NaN|mixed precision"
- description: Mixed-precision or fp16 execution overflows numerically, which can surface directly as an AI Core error or indirectly as downstream instability.
- fix_summary: Stabilize the hot path by casting sensitive operations to fp32, tuning loss scaling, and checking gradients for inf or nan before continuing.
- validation: Re-run the same step with the sensitive path forced to fp32 and confirm the overflow signature disappears.

### Stream Not in Current Context

- kind: known_issue
- symptom: failure
- id_hint: pta-stream-not-in-current-context
- severity: medium
- lifecycle_state: stable
- source_kind: bootstrap
- confidence_level: bootstrap
- tags: [pta, ascend, stream, context, synchronization]
- affects_platforms: [ascend]
- detection_pattern: "stream not in current context|current stream|aclrtSynchronizeStream|aclrtSetCurrentContext"
- description: A stream created under one device context is later synchronized or reused while a different context is current.
- fix_summary: Trace stream creation and reuse carefully, then ensure the correct device and context are active at every synchronization boundary.
- validation: Add logging around device and stream ownership and confirm the same stream is no longer reused across incompatible contexts.

### aten::_convert_weight_to_int4pack Not Implemented for NPU

- kind: known_issue
- symptom: failure
- id_hint: pta-int4pack-not-implemented-for-npu
- severity: medium
- lifecycle_state: stable
- source_kind: bootstrap
- confidence_level: bootstrap
- tags: [pta, torch_npu, quantization, int4, dispatcher, aclnn]
- affects_platforms: [ascend]
- affects_operators: [_convert_weight_to_int4pack]
- detection_pattern: "_convert_weight_to_int4pack|PrivateUse1 dispatcher not registered|int4 quantization|weight packing"
- description: The requested int4 weight-packing path is not wired up for NPU, so PyTorch dispatch falls back into an unsupported path instead of finding an NPU-specific implementation.
- fix_summary: Treat this as an operator-support gap first: use the NPU-specific weight-packing path if one exists, or fall back to a supported quantization workflow instead of assuming a generic CANN runtime bug.
- validation: Check whether the call site can be switched to a supported NPU quantization or packing path and confirm the same dispatcher error disappears.
