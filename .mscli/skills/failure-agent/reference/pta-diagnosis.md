# PTA Diagnosis Reference

Use this file when stack is `pta` and the failure sits near torch_npu operator
support, ERR codes, device/runtime integration, distributed failures, or
version compatibility.

## Current High-Value Card Matches

- `107002`, `context is empty` -> `ms-context-empty`
- `EL0004`, `200000`, `207018`, `CUDA out of memory` -> `device-out-of-memory`
- `EI0002`, `EI0006`, `107020`, `HCCL` timeout -> `distributed-communication-timeout`
- `symbol not found`, ABI mismatch after upgrade -> `stack-version-mismatch`
- `TBE`, `E9xxxx`, `EBxxxx`, operator compile failure -> `ms-tbe-operator-compilation-error`

## Environment Normalization

Collect these before committing to an operator-level or backend-level root
cause:

- PyTorch version
- torch_npu version
- CANN version
- device placement of the failing tensors
- whether the failure happens in eager op execution, autograd, or a higher-level module path
- whether async execution could be hiding the first real stack frame

If the user only gives a downstream stack, re-run with `ASCEND_LAUNCH_BLOCKING=1`
or another blocking mode before blaming registration or ACLNN.

## torch_npu ERR Format

Format: `ERR<SubModule><ErrorCode>`

SubModule IDs:

- `00`: PTA core framework
- `01`: operator execution
- `02`: distributed / HCCL
- `03`: graph / compilation
- `04`: profiler

Common error suffixes:

- `001`: invalid parameter
- `002`: invalid type
- `003`: invalid value
- `006`: memory error
- `007`: feature not supported
- `011`: timeout
- `100`: ACL API call failed
- `200`: HCCL API call failed
- `300`: GE API call failed
- `999`: unhandled exception

## Python Exception Types

- `RuntimeError`: runtime, operator, graph, or backend execution failure
- `ValueError`: shape, value, or config validation issue
- `TypeError`: dtype or argument-type mismatch
- `NotImplementedError`: feature or operator not supported on current backend
- `ImportError`: missing shared libs, env setup issue, or version mismatch

## Ascend Runtime and CANN Families

General interpretation:

- `1xxxxx`: environment or logic errors
- `2xxxxx`: resource exhaustion or unsupported feature
- `3xxxxx`: business, queue, or storage class errors
- `5xxxxx`: internal software or hardware fault

Useful runtime families:

- `107xxx`: parameter, context, or runtime control path
- `207xxx`: resource, memory, or feature support path
- `507xxx`: internal, hardware, timeout, or device-health path
- `161xxx`, `361xxx`, `561xxx`: ACLNN parameter, runtime, or internal path
- `E*` alphanumeric codes: TBE, runtime, HCCL, AICPU, profiling, or inner CANN errors

Important direct mappings:

- `107002`: context missing or not initialized
- `107003`: stream not in current context
- `107020`: task timeout
- `207003`: AI Core overflow
- `207018`: device OOM
- `507010`: heartbeat lost or device abort path
- `507014`: AI Core timeout
- `507054`: HBM ECC error
- `561003`: kernel not found
- `561107`: OPP path not found
- `EI0002`: HCCL notify wait timeout
- `EI0006`: HCCL socket build timeout
- `EL0004`: OOM

## Registration and Support Checks

When an operator is reported as missing or unsupported, check in this order:

1. is the operator expected to exist on NPU at all
2. is the dispatcher or registration path present
3. is there an NPU implementation or fallback path
4. is the current PyTorch + torch_npu + CANN combination compatible

High-value routing checks:

- import failure or `symbol not found` after upgrade
  - treat as `stack-version-mismatch` first
- HCCL timeout or collective stall
  - check `distributed-communication-timeout` signals before operator logic
- CANN environment import or startup failure
  - check `missing-cann-environment` first
- OOM or allocator failure
  - treat as `device-out-of-memory` first

## Common Failure Classes

- operator registration gap
- dispatcher gap or `PrivateUse1` mismatch
- dtype or shape constraint mismatch
- stream or context mismatch
- backend feature gap rather than a generic runtime bug
- version skew across PyTorch, torch_npu, and CANN

## PTA Misleading Patterns

Use these checks to avoid over-calling a torch_npu operator bug:

- `Expected all tensors to be on the same device`
  - usually device placement or mixed CPU/NPU input state, not missing operator support
- `ACLNN operator not found: aclnnXxx`
  - usually CANN capability or version floor mismatch before dispatcher corruption
- shape mismatch on NPU with otherwise normal registration
  - usually broadcasting or caller preconditions, not an op-registration gap
- failures that only appear through autograd or a module wrapper
  - compare the explicit op call path before assuming the base kernel is broken
- stream or context style failures after reuse across devices
  - treat as context-lifetime or async-path issues before inferring op logic defects

## Quick Contrast Checks

- explicit op call vs higher-level module or composite path
  - isolates wrapper behavior from the raw operator path
- NPU vs CPU reference behavior
  - checks whether the issue is backend-specific or a generic usage problem
- eager execution vs backward or autograd path
  - distinguishes forward-kernel issues from gradient or composite-path issues
- blocking vs default async execution
  - improves stack accuracy for delayed kernel-launch failures

## Upstream PyTorch Baseline

When a PTA failure depends on expected upstream operator behavior, use PyTorch
itself as the clean reference point before blaming torch_npu:

- confirm what behavior PyTorch expects from the operator on CPU or CUDA
- decide whether the failure is an NPU feature gap or a general operator misuse
- inspect the authoritative upstream signature and dispatch definition

Primary upstream sources:

- `native_functions.yaml`
- ATen native CPU implementation
- ATen CUDA implementation
- public PyTorch operator documentation

Good first comparison targets:

- `InstanceNorm`
- `GroupNorm`
- `LayerNorm`
- `Convolution`
- `Linear`

When comparing upstream expectations against NPU behavior, check:

- input rank and shape constraints
- dtype rules
- device expectations
- return type and inplace or view behavior
- backward or gradient expectations

Useful upstream search patterns:

```bash
rg -n "<op_name>" <pytorch_repo>/aten/src/ATen/native/native_functions.yaml
rg -n "<OpName>" <pytorch_repo>/aten/src/ATen/native/
rg -n "<OpName>" <pytorch_repo>/aten/src/ATen/native/cuda/
```

If the failure only appears on NPU while upstream PyTorch behavior is stable,
keep investigating torch_npu registration, support, or backend behavior rather
than re-litigating the upstream operator contract.

## Searching for Specific Operators

Useful search patterns when source is available:

```bash
find <torch_npu_repo> -name "npu_native_functions.yaml" -exec grep "<op_name>" {} +
grep -r "<OpName>" <op_plugin_repo> --include="*KernelNpuOpApi.cpp"
grep -r "<OpName>" <op_plugin_repo> --include="*KernelNpu.cpp"
```

What to look for:

- registration entry exists
- implementation exists
- op-plugin or fallback path exists
- backward rule exists when failure is grad-only

## Debugging Operator Issues

- test whether the operator exists before assuming a runtime-only issue
- inspect CANN logs for the exact failing operator path
- use `ASCEND_LAUNCH_BLOCKING=1` for accurate stack traces
- compare explicit operator API behavior vs autograd or composite path behavior

Common PTA-specific traps:

- CUDA-only test helpers or `_cuda_*` assumptions running on NPU
- feature gaps where an operator exists upstream but not on NPU yet
- NPU-specific alternate APIs for features like packing or quantization
- stream or context mismatch after cross-device or cross-context reuse

## GPU, CUDA, and NCCL Hints

- `CUDA out of memory`: GPU memory pressure
- `illegal address`, `launch failure`: kernel or indexing path
- `unhandled system error`, `remote process exited`, `timeout`: NCCL or distributed path

## Routing Hint

- hardware, ECC, heartbeat, or link errors -> Platform
- code, shape, or dtype misuse -> Scripts or Framework
- backend numeric or alphanumeric codes -> Backend, CANN, or ACLNN
- if a code cleanly matches a current real Factory card, prefer the card over a
  generic code-only explanation
