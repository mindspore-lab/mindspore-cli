# MindSpore Deep Debug Reference

Use this reference when a MindSpore runtime failure has already been scoped by
`failure-agent`, but the next step now requires source-level investigation,
historical issue mining, fix validation, or test planning.

This file is a local workflow aid. It does not authorize automatic code edits,
Factory writes, or test changes by `failure-agent`.

Before using this guide, first check whether the issue already aligns with a
current real Factory `known_issue` card such as `missing-cann-environment`,
`device-out-of-memory`, `distributed-communication-timeout`,
`ms-context-empty`, `ms-tbe-operator-compilation-error`, or
`stack-version-mismatch`.

## When to use

Use this guide when the current issue needs one or more of these:

- source-level navigation in MindSpore code
- historical issue mining beyond lightweight fallback lookup
- deeper operator or backend boundary confirmation
- regression validation planning after a likely root cause is found
- test-scope planning for an operator or runtime bug

## Deep-Debug Workflow

### 1. Reconfirm scope before going deep

Before reading source code, restate:

- failing stack: `ms`
- platform: `ascend`, `gpu`, or `cpu`
- first visible failure point
- failing operator or component if known
- whether the symptom looks like:
  - `API/mode misuse`
  - `unsupported/missing op`
  - `graph compile/frontend issue`
  - `runtime/backend issue`
  - `distributed/communication issue`
  - `numerical/precision symptom` inside a runtime failure

If this scope is still unstable, return to lightweight triage instead of going
deeper.

### 2. Run contrast experiments to narrow the layer

Prefer low-cost contrast checks before assuming a kernel bug:

- `GRAPH_MODE` vs `PYNATIVE_MODE`
- target backend vs `CPU`
- `float16` vs `float32`
- static shape vs dynamic shape
- single-card vs distributed

Interpretation shortcuts:

- only `GRAPH_MODE` fails: inspect frontend, infer, or graph-lowering path
- only `Ascend` fails while `CPU` passes: inspect backend, kernel, or ACLNN path
- only low precision fails: inspect dtype constraints or numerical instability
- only distributed fails: inspect communication ordering, rank config, or sync path

## Environment Normalization

Before chasing component-specific theories, normalize these facts into one short
snapshot:

- MindSpore version or commit
- CANN version
- Python version
- device target and device generation when known, such as `910A` vs `910B`
- execution mode: `GRAPH_MODE` or `PYNATIVE_MODE`
- optimization level or graph-build level when the failure appears compile-sensitive

If one of these changes between the last known good run and the failing run,
keep version or environment drift in the top hypothesis set.

## Contrast Checks

Prefer a tiny contrast matrix before reading source:

- `GRAPH_MODE` vs `PYNATIVE_MODE`
  - separates compile or graph-lowering issues from eager runtime issues
- `Ascend` vs `CPU`
  - separates backend-dispatch issues from common API or shape mistakes
- `float16` vs `float32`
  - separates numerical fragility from logic defects
- static shape vs dynamic shape
  - separates dynamic-shape support gaps from general infer bugs
- MindSpore vs PyTorch or TensorFlow baseline
  - separates MindSpore-only behavior from benchmark or expectation drift
- blocking vs default async execution
  - helps verify whether the visible stack is delayed by async launch behavior

## Triage-Safe Routing Aids

Use the sections below to stabilize the component route before source reading.
They are still triage-safe because they only classify the likely layer and the
next validation check.

## Quick Component Routing

Route by the strongest stable signal first:

- precision or numerical keywords such as `allclose`, `NaN`, `Inf`, `data_me_error`
  - first route: numerical, `bprop`, or backend-version drift
  - first checks: forward vs backward, `float16` vs `float32`, recent CANN or baseline changes
- API and signature keywords such as `takes N positional arguments`, `unexpected keyword`, `DID NOT RAISE`
  - first route: API validation, signature drift, or Graph vs PyNative path differences
  - first checks: minimal reproducer, Graph vs PyNative, list vs tuple normalization
- shape, broadcast, `AbstractProblem`, or `Invalid abstract`
  - first route: infer or graph compile
  - first checks: static vs dynamic shape, Graph-only vs both modes, whether `DeadNode` or `ValueProblem` appears
- `DeadNode`, `keyword_arg`, `FakeBprop`, or `control_node_parser`
  - first route: compiler or IR
  - first checks: IR export, pass ordering symptoms, control-flow transforms
- `segmentation fault`, `core dump`, `FAILED:`, or custom-op build errors
  - first route: kernel implementation, CANN compatibility, or threading/runtime
  - first checks: full stack trace, single-thread vs concurrent run, CANN symbol or version checks
- gradient-only failures such as zero grad, `GradOf...`, or `scalar type invalid`
  - first route: `bprop`, helper dtype coverage, or backward graph structure
  - first checks: forward vs backward split, PyTorch backward comparison, dtype-specific repro
- `device address`, `output addr`, import-path, or `module not callable` failures
  - first route: runtime, context, or packaging
  - first checks: context ordering, device target, import path, platform-specific execution path

## Stack-Frame Routing Hints

When a Python or C++ stack is available, use the first meaningful frame:

- `ops_func_impl`, `InferShape`, or `InferType`
  - route to infer or abstract propagation
- `AclnnKernelMod`, `LAUNCH_ACLNN`, or `aclnn`
  - route to Ascend kernel dispatch or CANN behavior
- `BpropBuilder`, `grad_ops`, or `GradOf`
  - route to backward graph or `bprop`
- `FrontendOptimize`, `AbstractSpecialize`, `make_keyword_arg`, or `extract_keyword_arg`
  - route to compiler frontend or IR transforms
- `control_node_parser` or `graph_scheduler`
  - route to runtime scheduling or graph control flow
- `PyBoost` or `OpRunner`
  - route to dispatch-path differences between generated wrappers and backend execution

## Source Navigation

If a local MindSpore source checkout is available, start with targeted search.

### Operator and API lookup

```bash
rg -l "^{op_name}:" mindspore/mindspore/ops/op_def/yaml/
rg -l "class {OpName}FuncImpl" mindspore/mindspore/ops/infer/
rg "FACTORY_REG.*{OpName}" mindspore/mindspore/ops/kernel/
rg 'REG_BPROP_BUILDER\("{OpName}"\)' mindspore/mindspore/ccsrc/frontend/expander/
rg -l "{OpName}" mindspore/mindspore/python/mindspore/ops/
```

### Layer-specific hints

- YAML / op definition:
  - check signature, attrs, dtype and shape constraints
- infer implementation:
  - check abstract inference, shape rules, rank rules, dtype propagation
- kernel registration:
  - check whether the backend kernel exists for the requested target
- bprop registration:
  - check gradient path when the failure is backward-only or grad-only
- Python API:
  - check wrapper validation, mode restrictions, and argument normalization

## Historical Issue Mining

When lightweight lookup is not enough, search for similar failures using:

- operator name
- error code
- first failing symbol
- stable error keywords
- backend-specific signatures such as `AbstractProblem`, `DeadNode`, `acl`, `aclnn`, `kernel not found`, or `dtype`

Good search targets:

- issue trackers or local issue exports
- prior bug summaries
- internal case indexes
- test names only after a stronger error signature is already known

Do not treat a superficially similar issue as a match unless the first failure
point, backend, and operator context are materially aligned.

## Misleading Patterns

Before concluding the root cause, explicitly check for common misreads:

- a downstream runtime error hiding an earlier frontend or context error
- a precision symptom being misread as a kernel registration failure
- a missing-op report that is actually a mode, shape, or dtype precondition issue
- a backend-only failure caused by version mismatch rather than by operator logic
- a gradient failure caused by bprop path differences rather than forward kernel behavior

Concrete high-value patterns:

- `allclose` failure plus zero gradients
  - likely route: backward graph structure or `bprop`, not forward precision
  - validate by comparing forward correctness separately from backward behavior
- `AbstractProblem` or `Invalid abstract` with static shape inputs
  - likely route: compiler or IR cleanup, not ordinary shape inference alone
  - validate by checking whether the issue is Graph-only and whether IR contains `DeadNode` or `ValueProblem`
- small drift after stack upgrade
  - likely route: CANN or benchmark-version drift, not logic regression
  - validate by checking version deltas and whether the deviation stays small and stable
- `DID NOT RAISE` or silent parameter acceptance
  - likely route: API or signature validation bypass, often PyNative-specific
  - validate by comparing Graph vs PyNative and checking list or tuple normalization
- intermittent `core dump` with STL container frames
  - likely route: threading or runtime races, not immediately math-kernel correctness
  - validate by reducing concurrency and checking whether the failure disappears or shifts
- import-path or callable-shape errors after refactor
  - likely route: packaging or runtime import changes, not device setup
  - validate by checking the effective import target and whether only one code path uses the stale symbol
- Ascend-only special-value anomalies such as `nan+nanj`, preserved `NaN`, or exact `2.14748e+09` boundaries
  - likely route: ACLNN or CANN implementation behavior, not API wiring
  - validate by comparing CPU behavior and, when available, torch_npu behavior on the same host

## Regression Validation Checklist

Once a likely root cause is identified, define validation before discussing any
fix as credible:

- reproduce the original failure on demand
- confirm the first failure point is stable
- rerun the minimal reproducer after the proposed change or workaround
- compare at least one nearby non-failing control case
- verify the relevant mode, backend, and dtype combination
- for distributed issues, rerun with the same world size and topology assumptions

Do not mark a source-level hypothesis as confirmed without a concrete
reproducer-level validation step.

## Test-Scope Checklist

If the bug would require a test in a deeper workflow, scope it like this:

- original repro case
- nearest boundary case
- dtype coverage when dtype is part of the symptom
- backend coverage when behavior differs across targets
- mode coverage when Graph and PyNative diverge
- backward-path coverage when the issue is gradient or bprop specific

Avoid broad “test everything” plans. Add the narrowest test set that would have
caught the observed bug and a closely related regression.

## ACLNN and Ascend Notes

For Ascend-specific runtime or operator failures, verify:

- CANN version compatibility
- whether the operator path is native kernel, ACLNN path, or fallback path
- dtype and shape constraints on the failing path
- whether the failure happens only on one device generation or software stack

If the same logical op works on CPU but fails only on Ascend, do not jump
directly to “kernel bug” before checking version and dispatch-path alignment.
