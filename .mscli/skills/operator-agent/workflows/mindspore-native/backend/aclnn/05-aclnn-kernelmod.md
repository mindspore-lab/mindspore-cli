# Workflow 5: Aclnn Kernelmod (Graph, C++)

Aclnn Kernelmod is the aclnn operator invocation flow for GRAPH mode, a.k.a. **KBK**.

## Goal

Implement the ACLNN kernel for Graph mode.
**The workload of this step differs greatly depending on the integration path:**
- **Path 1 (auto-generated)**: `gen_ops.py` already generates the required pieces, so this step only needs **validation**
- **Path 2 (Customize)**: you must handwrite the kernel files (`GetWorkSpaceInfo` + `Launch` + registration)

## Inputs

- **Integration path**: auto / customize
- **YAML definition**: parameter list
- **PyBoost implementation**: may be used as a reference for argument handling logic (Path 2)
- **(Composite scenarios)** ACLNN call chain

## Outputs

- **Path 1**: validated auto-registration
- **Path 2**: handwritten KBK kernel files + registration
  - `op_name_aclnn_kernel.cc/.h`
  - `MS_ACLNN_KERNEL_FACTORY_REG` registration

---

## Steps

### Path 1 Branch: Validate The Auto-Registration

> If Pre-B determined Path 1 (direct parameter passthrough), **you do not need to handwrite KBK kernel files**.
> `gen_ops.py` already generates the registration code in `aclnn_kernel_register_auto.cc`.

1. **Confirm the registration exists**: search the generated registration file for the operator name and confirm that `MS_ACLNN_COMMON_KERNEL_FACTORY_REG` is present
2. After validation passes, continue directly to [Workflow 6: BPROP](../common/06-bprop.md)

> **If auto-registration is wrong**, you need to reevaluate the integration path.

### Path 2 Branch: Handwrite Kernel Files

#### Step 0: Verify The ACLNN Interface

This is the same as [04-pyboost.md Step 0](./04-aclnn-pyboost.md#step-0-verify-the-aclnn-interface). If Step 4 (PyBoost) already confirmed the interface, reuse that conclusion directly.

#### Step 1: Standard Structure (`reference.md#kbk-reference`)

- `GetWorkSpaceInfo()`: fetch arguments + call `GetWorkspaceForResize`
- `Launch()`: call `RunOp` or the equivalent execution path
- registration: `MS_ACLNN_KERNEL_FACTORY_REG`

### Step 2: Hard Constraints

- Forward and backward must use **separate files and separate registrations**
- The header and implementation namespaces must stay consistent
- Do not mutate attributes in `InferShape` (`reference.md#resize-launch-no-attr-mutation`)

### Step 3: Resize/Launch Optimization (`reference.md#resize-launch-optimization`)

- Logic that can be fixed in `Init` should stay in `Init`
- Logic strongly tied to shape should stay in `Resize`
- `Launch` should only dispatch the real execution
- For useless outputs, override `GetUseLessOutputIdx()`
- For compute-dependent outputs, allocate the largest possible output and call `SyncOutputShape`

### Step 4: Composite Operator Pattern (Meta DSL, `reference.md#composite-kbk-pattern`)

Meta DSL uses C++ graph construction instead of manual `GetWorkSpaceInfo` / `Launch` / `RunOp`, and the framework then handles type inference and autodiff automatically:
1. create a new `.cc` file under `mindspore/ccsrc/frontend/operator/meta_dsl/func_op/`
2. register the operator with `REGISTER_FUNCTION_OP(OpName)` and optionally pass a validation function
3. inside `BeginFunction(OpName, args...) { ... } EndFunction(OpName)`, compose sub-operators with `Call(Prim(SubOp), ...)`
4. the framework handles multi-platform adaptation automatically, so **no handwritten KBK kernel file is needed**

Code skeletons are available in `reference.md#kbk-skeleton` for single operators and `reference.md#composite-kbk-pattern` for Meta DSL, but **the repository's current code remains the final reference**.

### Step 5: View Host Kernel (When YAML Has `graph_view: True`, `reference.md#view-kbk-host-kernel`)

When the operator is a View operator and must support the Graph-mode View path:

1. add `graph_view: True` in YAML
2. create `{op_name}_view.cc/.h` under `ops/kernel/host/view/kernel_mod_impl/`
3. inherit from `HostKernelMod`, implement `GetWorkSpaceInfo`, and update the output `tensor_storage_info` through strides calculation
4. register it with `MS_HOST_REG_KERNEL({OpName}View, {OpName}View)`
5. **do not call ACLNN**; the host kernel operates directly on strides

> Registration macro names such as `MS_ACLNN_KERNEL_FACTORY_REG`, base classes, and workspace APIs may vary across versions.
> Always inspect the latest existing operators under `kernel_mod_impl/` first.

---

## Success Criteria

**Path 1**:
- [ ] Confirm the auto-registration exists and uses the correct operator name

**Path 2**:
- [ ] KBK forward kernel implementation is complete
- [ ] KBK backward kernel implementation is complete (if needed)
- [ ] Registration macro is correct and can dispatch in Graph mode
- [ ] In composite scenarios, either the Meta DSL composition is correct or workspace handling is correct in the legacy pattern
- [ ] Forward and backward are split across separate files

**View operators**:
- [ ] `graph_view: True` is configured in the View-specific YAML
- [ ] The host kernel is implemented correctly (strides update + `MS_HOST_REG_KERNEL` registration)

---
