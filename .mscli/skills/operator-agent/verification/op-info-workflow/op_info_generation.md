# Operator ST Test Case Generation

<a id="op-info-generation-goal"></a>
## Goal

Add Python ST coverage for a new operator, cover different cases including functionality, accuracy, and dynamic shape.

**Important:** Similar operator test cases may already exist in the repository, but their scenario coverage may be incomplete. When using similar operators as references, do not treat their coverage as the target coverage. You must strictly follow the scenario coverage requirements in this document.

<a id="op-info-generation-inputs"></a>
## Inputs

- **API name**: Use the API name to collect relevant information, such as the operator API definition YAML and operator API documentation.
- **Torch counterpart API**: the testing benchmark.

<a id="op-info-generation-outputs"></a>
## Outputs

> **⚠️ The following output is required**

| Type | File Location |
| --- | --- | --- | --- |
| **Python ST** | `tests/st/ops/share/_op_info/op_database.py` (OpInfo registration) |

---

## Basic Testing Principles

The behavior of the tested API should fully align with the benchmark API.
- If the benchmark API supports certain inputs and behaviors, the tested API should support them as well.
- If the benchmark API does not support certain inputs and behaviors, the tested API does not need to support them.

<a id="op-info-generation-steps"></a>
## Execution Steps

> The current ST uses the **op info testing framework**. The core operation is registering OpInfo in `op_database.py`. Writing separate standalone test files is not allowed. For framework details, see [`../_shared/reference.md` 8.2 ST op-info testing framework](../../_shared/reference.md#testing-st-opinfo).

**There are two scenarios:**

| Applicable Scenario | Action |
| --- | --- |
| Regular operators such as Unary/Binary/Reduction | Add OpInfo in `op_database.py` -> add it to the corresponding `xxx_op_db` -> it is automatically included in frontend parameterized test cases |
| Operators requiring custom test logic | Inherit from `OpsFactory` to build a custom test suite + create a new frontend test file |

<a id="op-info-generation-common-ops"></a>
### Regular Operators Such as Unary/Binary/Reduction

For Unary/Binary/Reduction operators, `op_info.py` already provides rich common input-generation helpers (various shape combinations, broadcasting, discontiguous tensors, special values, extreme values, and so on). Once OpInfo is registered, these scenarios are covered automatically.

1. **Determine the operator's OpInfo category**: Unary -> `UnaryOpInfo`, Binary -> `BinaryOpInfo`, Reduction -> `ReductionOpInfo`, others -> `OpInfo`.
2. **Add an OpInfo instance in `op_database.py`**: configure `name`, `op`, `ref`, `dtypes_support` (and `dtypes_grad`, `dtypes_dynamic`, and so on).
3. **Add the operator name to the corresponding `xxx_op_db` list** (for example, `binary_op_db`, `unary_op_db`).
4. **If custom input scenarios are needed**: implement `op_basic_reference_inputs_func` / `op_extra_reference_inputs_func` and return a list of `OpSampleInput`.
5. **Decide whether it should be added to `xxx_op_kbk_db`** (see the constraints below).
6. **Verify coverage**: confirm that the frontend test file (for example, `test_binary_ops.py`) includes the new operator in its parameterized cases.

> **Constraints for adding operators to KBK lists (`xxx_op_kbk_db`):**
>
> KBK scenarios are relatively time-consuming, so not every operator needs to be included. Add an operator to the corresponding `xxx_op_kbk_db` (such as `binary_op_kbk_db`, `unary_op_kbk_db`, `reduction_op_kbk_db`, and so on) only in the following cases, so that the frontend test files run KBK forward/backward/dynamic-shape cases:
>
> - The operator has **relatively complex dynamic shape inference logic** (for example, output shape depends on input values or uses multi-branch inference), which can be verified by checking the operator infer code.
> - The operator uses a **composite implementation** (multiple operator calls chained together in PyBoost/aclnn kernelmod).
> - The operator includes **frontend API overloading** (there is an API YAML definition in `mindspore/ops/api_def`).
>
>
> **Cases where it does not need to be added:**
> - Simple passthrough operators (single ACLNN call, no parameter preprocessing)
> - The KBK list already contains **an operator with the same type or implementation pattern**. For example, if `unary_op_kbk_db` already includes `mint.tanh`, then similar trigonometric operators such as `mint.cosh` do not need to be added again.

### Operators Requiring Custom Test Logic

For **other-type operators** (added to `other_op_db`), you must **write input-generation functions manually** in `op_database.py` and pass them to OpInfo through `op_basic_reference_inputs_func` and `op_extra_reference_inputs_func`.

#### Test Case Coverage Scenarios
- [ ] `[MUST]` **Default-parameter scenario validation**: call forward and backward with all default parameter values to confirm the basic path works.
- [ ] `[MUST]` **Dynamic shape self-validation**: the frontend test file calls the `test_op_dynamic` method from `OpsFactory`.
- [ ] `[MUST]` **Empty tensor input**: verify whether forward/backward with empty tensors is supported or raises the correct error.
- [ ] `[MUST]` **Full input dtype coverage**: every dtype declared as supported by the operator must have corresponding cases, including exception cases for unsupported types.
- [ ] `[MUST]` **Input dimension coverage**: include both valid dimensions (covering 0D, 8D, and one intermediate-size dimension if supported) and invalid dimensions.
- [ ] `[MUST]` **Input value range validation**: fully cover boundary values, extreme values (very large/very small), and enumerated parameters such as `margin`/`reduction`.
- [ ] `[MUST]` **Cross-input constraint validation**: shape match/mismatch, dtype same/different, and rank same/different.
- [ ] `[MUST]` **Assert exact error messages for exception cases**: exception scenarios must assert the specific message of `TypeError`/`ValueError`/`RuntimeError`.
- [ ] `[MUST]` **Multi-layout coverage**: if the operator supports multiple layouts (such as BSND/TND/PA_BSND), cover forward and backward for every layout combination.
- [ ] `[MUST]` **Discontiguous tensors**: construct discontiguous inputs using `transpose`/`permute` and verify correctness.
- [ ] `[MUST]` **Special-value robustness**: validate `inf`/`-inf`/`nan` scenarios, at minimum ensuring no crash and correct shape/flow behavior.
- [ ] `[SHOULD]` **Variable-length sequences with multiple batches**: if parameters such as `actual_seq_len` are involved, cover multiple batches plus variable-length scenarios.
- [ ] `[MUST]` **bf16 scenarios**: confirm bf16 support. If supported, test accuracy; otherwise include exception cases. Promote to `float32` before comparison.
- [ ] `[MUST]` **Implicit type conversion**: confirm whether automatic promotion is supported when input dtypes differ; if not, include exception cases.
- [ ] `[MUST]` **Broadcasting**: confirm whether shape broadcasting between inputs is supported; if not, include exception cases.
- [ ] `[MUST]` **Inconsistent dtype across multiple Tensor inputs**: confirm whether the operator supports different dtypes across multiple Tensor inputs; if not, include exception cases. This is not required for operators that do not take multiple Tensor inputs.

| Required Scenario | How to Write It | Example |
| --- | --- | --- |
| **Multiple shapes** (including 0D scalar, 1D, intermediate 2D-3D, and high-dimensional) | Multiple `yield` statements with different shapes | `make_arg(())`, `make_arg((S,))`, `make_arg((S,M,S))` |
| **Empty tensor** (one dimension is 0) | Include 0 in the shape | `make_arg((0, S))`, `make_arg((S, 0, M))` |
| **Discontiguous tensor** | Use the `discontiguous=True` parameter | `make_tensor(shape, discontiguous=True)` |
| **Boundary parameter values** | Cover extreme/boundary parameter values | `dim=0`, `dim=-1`, `dim=last dimension`; `p=1`, `p=2`, `p=inf` |
| **Large tensor** | At least one relatively large shape | `make_arg((LARGE_DIM_SIZE, M))` |

Implementation reference: follow the patterns of `basic_reference_inputs_binary_op_common_func` and `_generate_binary_op_broadcasting_and_discontiguous_tensor_inputs_func` in `op_info.py`.

If the operator supports `op_extra_reference_inputs_func` (extra accuracy scenarios) or `op_dynamic_inputs_func` (dynamic shape/rank), implement them by following similar patterns in `op_info.py`.

#### Test Matrix and Stability
- [ ] `[MUST]` **Test matrix coverage**: API form (functional/nn/Tensor) x backend x mode (Pynative/KBK) x shape type (static/dynamic).
- [ ] `[MUST]` **Backoff-disabled validation**: all cases must pass under `export MS_DISABLE_KERNEL_BACKOFF=1` to prevent fallback to non-ACLNN paths.
<!-- - [ ] `[SHOULD]` **Regression of existing tests in the test repository**: if ST cases already exist in the test repository, confirm that they all PASS. -->

<!-- #### Functional Compliance Confirmation
- [ ] `[MUST]` **Does not affect existing APIs**: adding a new operator/primitive must not cause operators or existing ops APIs to call the new primitive unless that is the intended design.
- [ ] `[SHOULD]` **AMP mixed precision**: confirm whether it is already supported or not applicable (new Primitives need attention to `amp_white/black_list`).
- [ ] `[SHOULD]` **Inconsistent dtype across multiple Tensor inputs**: for multi-input operators, confirm whether different input dtypes are supported.
- [ ] `[SHOULD]` **Whether output shape depends on computation results**: if the output is compute-dependent, a `SyncOutputShape` mechanism is required. -->


<!-- <a id="op-info-generation-bitwise-validation"></a>
### Zero-Bit Accuracy Validation ([`../_shared/reference.md` 14.1 Zero-Bit Accuracy](../../_shared/reference.md#bitwise-accuracy-validation), when needed)

- Fix the random seed and save outputs as `.npy`.
- Use `md5sum` to compare the output hashes of MS/PTA.

<a id="op-info-generation-memory-validation"></a>
### Memory Alignment Validation ([`../_shared/reference.md` 14.2 Memory Usage Alignment](../../_shared/reference.md#memory-alignment-validation), when needed)

- MS: `mindspore.runtime.max_memory_allocated()`
- PTA: `torch_npu.npu.max_memory_allocated()`
- Measure at the same stage. -->

---

<a id="op-info-generation-gate"></a>
## Mandatory Check Before Completing Step 8 (Cannot Be Skipped)

**Before marking Step 8 as complete, every item in the following checklist must be confirmed one by one:**

```text
Test Output Checklist:

Python ST (OpInfo Registration):
  - Registration file: tests/st/ops/share/_op_info/op_database.py
  - Has OpInfo been registered? ✅ Yes (operator name: ___) / ❌ No (reason: ___)
  - Has it been added to the corresponding `xxx_op_db` list? ✅ Yes / ❌ No
  - Is it covered by frontend parameterized test cases? ✅ Yes (test file: ___) / ❌ No
  - If custom inputs are required: has `inputs_func` been implemented? ✅ Yes / ⏭ Not needed
  - 🚫 Was a standalone test script created? This must be No. If one was created by mistake, delete it and migrate it into OpInfo.
```

> If the Python ST status is ❌, **you must explain the reason and pause until the user confirms before continuing**.
> Silent skipping is not allowed.

<a id="op-info-generation-success-criteria"></a>
## Success Criteria

- [ ] **Python ST OpInfo is registered and included in frontend parameterized test cases** (automatically covering multiple modes, forward accuracy, and dynamic shape)
- [ ] Covered scenarios: dynamic shape / static shape / discontiguous tensor / empty tensor / special values
