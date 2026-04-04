# Feature Document For `{OperatorName}` Operator Development

> **Note**: this document is a **required deliverable** for operator review and test handoff.


<a id="feature-background"></a>
## 1. Background `[Pre-B Stage]`

{Describe the operator background, origin, such as an NSA/DSA paper if applicable, and why MindSpore needs this operator.}

<a id="feature-benchmark-api"></a>
## 2. Benchmark And APIs `[Pre-B Stage]`

- **Reference interface**: `torch_npu.npu_xxx` / `torch.xxx`
- **Functionality**: {One-sentence description}
- **MindSpore interfaces**:
  - functional: `mindspore.ops.xxx` / `mindspore.mint.xxx`
  - nn: `mindspore.mint.nn.Xxx` (if needed)
  - Tensor: `Tensor.xxx` (if needed)

<a id="feature-task-list"></a>
## 3. Task List `[Initialized In Pre-B, Updated During Development]`

> Use the standard 13 categories and mark the status item by item.

| No. | Task Category | Subtask | Status (new/modified/no change/not involved) | Notes |
| ---- | ------ | -------- | ------------------------------ | ---- |
| 1 | Basic interface functionality | Primitive | | |
| | | functional | | |
| | | nn | | |
| | | tensor | | |
| 2 | Backend and dtype support | Ascend | | |
| | | GPU | | |
| | | CPU | | |
| 3 | Dynamic support | Dynamic shape | | |
| | | Dynamic rank | | |
| 4 | Backward support | bprop function | | |
| 5 | Supporting materials | API mapping | | |
| | | English/Chinese interface docs | | |
| 6 | Functional behavior | Empty Tensor support | | |
| | | `inf`/`nan` support | | |
| | | 0D-8D coverage | | |
| | | Other functional points | | |
| 7 | Gate test completion | UT | | |
| 9 | Safety and exceptions | Error cases and error-message conventions | | |

<a id="feature-functional-spec"></a>
## 4. Functional And API Specification `[Pre-B Stage]`

### Functional Overview

{The operator formula, semantics, and core behavior.}

### Public Interface

```python
mindspore.ops.xxx(
    param1: Tensor,      # [shape], dtype: xxx
    param2: int,         # description
    ...
) -> Tensor | Tuple[Tensor, ...]
```

### Parameter Description

| Parameter | Type | Required/Optional | Default | Description |
| ---- | ---- | -------- | ------ | ---- |
| param1 | Tensor | Required | — | {Description} |
| ... | | | | |

<a id="feature-yaml-definition"></a>
## 5. YAML Definition (Reference) `[After Step 1]`

```yaml
# operator xxx
xxx:
    args:
        # {Insert the actual YAML here}
    returns:
        # {Insert the actual YAML here}
    dispatch:
        enable: True
        Ascend: XxxAscend
```

<a id="feature-constraints"></a>
## 6. Constraints And Types `[Pre-B Stage]`

- **Input/output dtypes**: {List them}
- **Shapes and ranges**: {List the shape constraints for each input}
<!-- - **Empty Tensor**: {supported / unsupported, with explanation} -->

<a id="feature-execution-modes"></a>
## 7. Execution Modes And Adaptation `[After Step 4/5]`

### Pynative (PyBoost)
- {Implementation notes}

### Graph (KBK)
- {Implementation notes}

<a id="feature-pta-alignment"></a>
## 8. Differences From PTA And Alignment Status `[Initialized In Pre-B, Completed During Development]`

- **Functional alignment**: {How the implementation aligns with PTA}
- **Numerical accuracy**: {Comparison strategy, such as zero deviation or `rtol/atol`}
- **Differences**: {List the differences from PTA and explain the reasons}

<a id="feature-dynamic-shape"></a>
## 9. Dynamic Shape/Rank Support `[After Step 3]`

- {Dynamic-dimension / dynamic-rank inference strategy}
- {Fallback strategy when compile-time values are unknown}

<a id="feature-validation-and-errors"></a>
## 10. Validation And Error Handling `[After Step 3/4]`

### Inference Phase (Infer)
- {List checked added in inference-time}

### Runtime Phase (ACLNN)
- {List checks added in runtime}

<a id="feature-bprop"></a>
## 11. Backward (BPROP) `[After Step 6]`

- {How BPROP is registered, backward inputs/outputs, and gradient handling}
- If autodiff is used instead, state "no explicit bprop is required"

<a id="feature-test-plan"></a>
## 12. Test Plan `[After Step 8]`

### UT (C++ GeneralInfer)
- {Covered scenarios}

<a id="feature-code-change-summary"></a>
## 13. Code And File Change Summary `[After Development]`

| Category | File Path |
| ---- | -------- |
| YAML | `mindspore/ops/op_def/yaml/xxx_op.yaml` |
| Infer | `mindspore/ops/infer/ops_func_impl/xxx.cc/.h` |
| PyBoost | `mindspore/ops/kernel/.../customize/xxx.cc/.h` |
| KBK | `mindspore/ops/kernel/.../customize/xxx_aclnn_kernel.cc/.h` |
| BPROP | {Path or "not involved"} |
| API export | `mindspore/ops/api_def/xxx.yaml`, `__init__.py` |
| Docs (EN) | `mindspore/ops/op_def/yaml/doc/xxx_doc.yaml` (`_ext` style) or `api_def/function_doc/` (legacy style) |
| Docs (CN) | `docs/api/api_python/ops/mindspore.ops.xxx.rst` |
| Tests (UT) | `tests/ut/cpp/ops/test_xxx_general_infer.cc` |

<a id="feature-acceptance-report"></a>
## 14. Acceptance Report `[Fill Before Test Handoff]`

### Basic Information

- **Operator under acceptance**: `mindspore.mint.xxx`
- **Reference benchmark operator**: `torch_npu.npu_xxx`
- **Whether this is a side-effect operator**: No / Yes

### Documentation Validation

| Self-check Item | Result | Notes |
| -------- | -------- | ---- |
| New interface list provided | | |
| Typical UT cases provided | | |
| Chinese RST provided and aligned with English comments | | |
| Interface description is detailed and accurate | | |
| Interface matches PyTorch | | |
| Formula provided in the summary section | | |
| Attribute descriptions are complete and correct | | |
| Input descriptions are complete and correct | | |
| Output descriptions are complete and correct | | |
| Output size matches input when applicable | | {Explain the effect of parameters such as `reduction` on the output shape} |
| Supported platforms are fully documented | | |
| Documentation format, including examples, is correct | | |
| Example is provided | | |
| Example prints output | | |
| Example runs successfully | | |

<!-- ### Functional Validation

| Self-check Item | Result | Notes |
| -------- | -------- | ---- |
| Default-parameter case validated | | |
| Empty Tensor forward/backward validated | | |
| `inf` and `nan` validated | | |
| Supported dtypes align with the benchmark operator (PyTorch NPU/GPU/CPU) | | |
| Input value range validated | | |
| Input ranks 0D-8D covered | | |
| All supported input dtypes covered | | |
| Implicit type conversion supported | | |
| Broadcasting supported | | |
| Input constraints validated | | |
| Forward accuracy validation passed | | |
| Backward supported | | |
| Backward implemented as a single operator | | |
| Error cases validate the exact error message | | |
| Error-message whitelist provided | | |
| Dynamic shape/rank/attribute support complete | | |
| Backoff disabled validation (`export MS_DISABLE_KERNEL_BACKOFF=1`) completed | | |
| All related test-repository cases pass and there are no remaining issue tickets | | |
| `bf16` supported | | |
| Multi-input operator bprop considers selective differentiation | | |
| Output shape depends on the operator result | | |
| Non-contiguous input support validated | | |
| Zero-deviation comparison against PTA completed (attach MD5 comparison screenshot) | | |
| New primitive affects existing operator or legacy ops interface routing | | |
| For multiple Tensor inputs, mismatched Tensor dtypes are supported | | | -->


### Secure Coding Review

| Self-check Item | Result | Notes |
| -------- | -------- | ---- |
| Null pointers are checked where required | | {In Infer, `Primitive`, input/output, `GetShape`, `GetType`, and `GetValue` do not require null checks; other pointers do} |
| Pointer use does not happen before validation | | |
| No out-of-bounds access on arrays or pointers | | |
| No division by zero | | |
| No memory leaks (`new`/`malloc` allocations are released) | | |
| Exception or error branches release memory, file handles, and other resources | | |
| Objects created with `new` are declared with `nothrow` where required | | |
| Safe memory APIs are used where appropriate | | |
| Type conversions do not cause numeric truncation, overflow, or underflow | | |
| No redundant code (redundant validation, unreachable code, and so on) | | |
| No sensitive information is exposed | | |
| No weak random number generator is used | | |
