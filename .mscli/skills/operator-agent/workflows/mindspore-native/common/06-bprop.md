# Workflow 6: BPROP Registration

## Goal

Implement the backward graph in the bprop builder.

## Inputs

- **PTA `derivatives.yaml` analysis**: which inputs are differentiable and how gradient-function arguments are passed
- **Backward operator YAML definition**: backward operator parameters

## Outputs

- **BPROP implementation code**: added to the appropriate `grad_*ops.cc`

---

## Steps

The bprop c++ implementation can be found in `mindspore/ccsrc/frontend/expander/grad/grad_xxx_ops.cc`.

### Step 1: Basic Wiring (`reference.md#bprop-reference`)

- Build the backward subgraph only for inputs that actually require gradients
- Return zero-gradient placeholders for non-Tensor inputs or inputs that do not require gradients
- Use `need_compute_grad_out()` to decide whether gradient computation is necessary

### Step 2: I/O Count Rules (`reference.md#bprop-io-rules`)

- backward inputs = number of forward inputs + 2 (`out` and `dout`)
- backward outputs = number of forward inputs (one gradient per forward input)
- when the forward output is multi-output, `out` is usually a tuple on the backward side -> use `TupleGetItem`

### Step 3: Advanced Notes (`reference.md#bprop-advanced-notes`)

| Scenario | Handling |
| --- | --- |
| Non-differentiable input | `ib->OutZeros(x)` |
| All inputs non-differentiable | `ReturnZeros` |
| Theoretical gradient is zero | `ib->ZerosLikeExt()` |
| Inplace backward | If input and output are the same object, **as long as one is used in backward, it must not be added to `SetUnusedInputs`**; if the backward logic needs the pre-update `self`, register **`CloneInplaceInput`** (see `reference.md#bprop-advanced-notes`) |
| KBK dynamic-shape inplace | `ib->Depend(target, inplace_call)` |
| `str` parameter gradient slot | Returning `OutZeros` for a `str` position may break KBK backward with dynamic shape; follow real framework behavior (see `reference.md#bprop-advanced-notes`) |

### Step 4: `SetUnusedInputs` (`reference.md#bprop-set-unused-inputs`)

If backward does not depend on the tensor values of certain inputs, mark them as unused so memory can be released earlier.

See the code skeleton in `reference.md#bprop-builder-skeleton`.

### Step 5: Dynamic Inputs In Graph Mode (`reference.md#bprop-dynamic-inputs`)

> In Graph mode (KBK), the **value or shape** of forward inputs may be unknown at graph-compile time.
> Any shape calculation or control-flow branch in the bprop builder that depends on forward inputs must be deferrable to runtime.
> **If this is not handled, backward compilation in Graph mode may fail or produce wrong results.**

You must check the following scenarios and apply the correct handling:

| Scenario | How To Check | Handling |
| --- | --- | --- |
| Scalar input value is unknown | `GetScalarValue<>()->has_value()` | If known, branch in C++; if unknown, build a runtime branch with `Conditional` |
| Input shape is dynamic | `IsDynamicRank()` / `IsDynamicShape()` | Use `DEF_PURE_SHAPE_CALC` + `ib->ShapeCalc` for shape-dependent calculations |
| Control flow depends on runtime values | compile-time value may change | Use `ib->Conditional(cond, true_br, false_br)` instead of raw C++ `if/else` |

> 🚫 **Forbidden anti-pattern**:
>
> If `GetScalarValue<>()->has_value()` returns false, you must **not**
> immediately throw `MS_EXCEPTION(ValueError)`.
> That effectively gives up Graph-mode dynamic-input support.
>
> **Wrong pattern (forbidden)**:
> ```cpp
> p = p_node->BuildValue();
> if (!GetScalarValue<float>(p)->has_value()) {
>   MS_EXCEPTION(ValueError) << "p must be constant!";  // ❌ forbidden
> }
> ```
>
> **Correct pattern (required)**:
> ```cpp
> p = p_node->BuildValue();
> p_opt = GetScalarValue<float>(p);
> if (p_opt->has_value()) {
>   // known at graph compile time -> optimize with C++ branching
>   auto p_val = p_opt.value();
>   // ... branch on the value
>   if (p_val...) 
> } else {
>   // unknown at graph compile time -> build a runtime branch with Conditional
>   auto true_branch = [&ib](...) { ... };
>   auto false_branch = [&ib](...) { ... };
>   result = ib->Conditional(cond, true_branch, false_branch);
> }
> ```
>
> If the backward logic truly cannot be inferred when the value is unknown, which is rare, or if you cannot handle KBK dynamic input support, you must **record the reason explicitly in the validation report and ask the user to confirm**, rather than silently throwing.

**Reference implementations** (typical repository patterns):
- `ReduceStd` bprop: uses `Conditional` when `keep_dims` and `unbiased` are unknown
- `MatMulExt` bprop: uses a dedicated dynamic path when `IsDynamicRank(x_shape) || IsDynamicShape(w_shape)`

---

## Success Criteria

- [ ] BPROP registration code has been added
- [ ] Backward I/O counts match the forward operator
- [ ] Non-differentiable inputs return zero-gradient placeholders
- [ ] The differentiable input list is aligned with PTA `derivatives.yaml`
- [ ] **Graph-mode dynamic input cases are handled** (unknown scalar -> `Conditional`; dynamic shape -> `ShapeCalc` / dynamic path)

---

## Common Problems

1. **Directly throwing when a bprop scalar is unknown**: using `MS_EXCEPTION` when `ContainsValueAny()` is true instead of building a runtime branch.
   -> You must use `ib->Conditional(cond, true_branch, false_branch)`.
   If the value is known at graph compile time, branch in C++; if unknown, branch at runtime with `Conditional`.
   See `ReduceStd` bprop for the pattern. Details are in Step 5 of `workflows/06-bprop.md`.
