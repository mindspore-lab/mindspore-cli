# MindSpore API Diagnosis Notes

Use this file when stack is `ms` and the failure sits near API usage,
execution mode, backend dispatch, or operator support.

## API Layer Hierarchy

Read the issue from top to bottom:

- `mindspore.mint`
- `mindspore.ops`
- `mindspore.nn`
- backend registration and kernel support

If the failure only appears at one layer, do not immediately assume the lower
layer is broken.

## Execution Modes

### GRAPH_MODE

Typical failure classes:

- graph compile failure
- abstract/type inference failure
- unsupported control flow or graph-only restriction
- backend compile failures surfaced during graph build

### PYNATIVE_MODE

Typical failure classes:

- eager runtime errors
- parameter or dtype validation issues
- backend runtime failure on the first real operator launch

High-value comparison:

- if `GRAPH_MODE` fails and `PYNATIVE_MODE` works, narrow to frontend,
  infer, graph lowering, or compile-time backend path first.

## Context and Device Management

Check these first on Ascend:

- `device_target`
- execution mode
- device initialization order
- backend-specific context config

Current high-value routing checks:

- `107002` / `context is empty`
  - check `set_context` ordering and device initialization first
  - usually aligns with `ms-context-empty`, not an operator bug
- `device_target` mismatch
  - verify the requested backend is available and spelled correctly
- `TBE` compile failure on Ascend
  - check shape, dtype, and stack compatibility before blaming user code
  - often aligns with `ms-tbe-operator-compilation-error`

## High-Value Misreads

Use these sanity checks before calling a failure a backend bug:

- `allclose` fails and gradients become zero
  - route first to `bprop` or graph structure, not forward-kernel precision
  - check whether forward results still look normal and whether the failure is backward-only
- `AbstractProblem` or `Invalid abstract`
  - route first to graph compile or IR cleanup, not plain shape rules
  - check whether static shape also fails and whether the issue is `GRAPH_MODE`-only
- small numerical drift after a recent upgrade
  - route first to CANN version or benchmark-version drift, not operator logic
  - check whether the deviation is small and whether the stack changed since the last known good run
- `DID NOT RAISE`, missing `TypeError`, or list/tuple ambiguity
  - route first to API/signature validation paths, especially Graph vs PyNative divergence
  - check whether argument normalization or deprecated paths are bypassing validation
- import or callable errors after refactor
  - route first to API packaging or runtime import-path changes, not installation damage
  - check whether the module exists but the old import path or callable shape changed

## High-Frequency Operator Signals

These are triage hints only. They help choose the first checks, not the final fix.

- `pow` / `select`
  - zero gradient or branch-sensitive failure usually points to `bprop` or graph structure before forward compute
  - shape or broadcast wording may still mask a control-flow or IR issue
- `matmul`
  - small drift after CANN changes usually points to backend-version or accumulation behavior
  - intermittent multi-card crashes point to runtime or communication interactions before pure math bugs
- `trunc` / `fix`
  - Ascend results near `2.14748e+09` or `-2.14748e+09` are strong signs of ACLNN or platform-specific behavior
  - compare CPU and torch_npu behavior before blaming MindSpore API wiring
- `adam` / `adamw`
  - framework mismatch against TensorFlow or PyTorch often starts with baseline version drift
  - `NoneType` or state-init failures often point to optimizer state setup before kernel issues
- `sort` / `argsort`
  - output-order or checksum instability may be nondeterminism, not a correctness regression
  - check whether the comparison method assumes deterministic ordering

## mint / ops / nn Pitfalls

Common pitfalls worth checking before deep debugging:

- view vs copy semantics in graph-sensitive code paths
- return-type differences between APIs that look similar
- experimental APIs changing across versions
- strict parameter validation in `mint.nn` and `nn` modules
- graph-only vs backend-only operator support differences

## Operator Debugging Checklist

- confirm the first failure point instead of a downstream runtime error
- print effective context, mode, target, and key tensor dtypes/shapes
- compare `GRAPH_MODE` vs `PYNATIVE_MODE`
- compare Ascend vs CPU when possible
- verify the operator is supported on the requested backend
- check whether the failure is registration/dispatch, infer, compile, or runtime

Useful commands and patterns:

```bash
rg -l "^{op_name}:" mindspore/mindspore/ops/op_def/yaml/
rg -l "class {OpName}FuncImpl" mindspore/mindspore/ops/infer/
rg "FACTORY_REG.*{OpName}" mindspore/mindspore/ops/kernel/
```

## When to Go Deeper

If the issue still looks like a MindSpore operator or backend implementation
problem after these checks, move to [mindspore-dianosis](mindspore-dianosis.md).
