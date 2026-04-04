# PTA Source Review Report - {OpName}

> **Purpose**: record the PTA (`op-plugin`) source review results as the basis for MindSpore adaptation.
> **Document status**: local file, do not commit to Git.
> **Generated at**: {generation_time}

---

## 1. Basic Information

| Attribute | Value |
| ---- | -- |
| **PTA interface name** | `torch.xxx` |
| **MindSpore target interface** | `mindspore.mint.xxx` |

---

## 2. Forward Interface Analysis

### 2.1 Function Signature (from `op_plugin_functions.yaml`)

```yaml
# excerpted from op_plugin/config/op_plugin_functions.yaml
{paste_yaml_entry}
```

### 2.2 Parameter Details

| Parameter Name | Type | Required | Default | MindSpore Mapping | Notes |
| ------ | ---- | ---- | ------ | ------- | ---- |
| {param1} | {type} | ✅/❌ | {default} | {ms_name} | |

### 2.3 ACLNN Call Analysis (from the C++ implementation)

**File**: `op_plugin/ops/opapi/{OpName}KernelNpuOpApi.cpp`

```cpp
// key code excerpt
{key_code_snippet}
```

**ACLNN interface called**: `aclnn{XxxYyy}`
**Parameter preprocessing**: {describe_preprocessing}
**Output construction**: {describe_output_construction}
**Hard-coded parameters**: {list_hardcoded_params}

---

## 3. Backward Interface Analysis

### 3.1 Backward Registration (from `derivatives.yaml`)

```yaml
# excerpted from op_plugin/config/derivatives.yaml
{paste_derivatives_entry}
```

### 3.2 Differentiable Inputs

| Input | Differentiable | Notes |
| ---- | -------- | ---- |
| {input1} | ✅/❌ | |

### 3.3 Backward ACLNN Call Analysis

**File**: `op_plugin/ops/opapi/{OpName}GradKernelNpuOpApi.cpp`

**ACLNN interface called**: `aclnn{XxxGrad}`
**Backward outputs**: {list_grad_outputs}
**Hard-coded parameters**: {list_hardcoded_params}

---

## 4. Forward Vs Backward Differences

| # | Item | Forward | Backward | Impact On MindSpore Adaptation |
| - | ---- | ---- | ---- | ----------- |
| 1 | {diff_item} | {fwd_value} | {bwd_value} | {impact} |

---

## 5. Documentation/Code Mismatches (If Any)

| # | Item | Documentation Says | Actual Code Behavior | File/Line | Status |
| - | ---- | -------- | ------------ | --------- | ---- |
| 1 | {item} | {doc_says} | {code_does} | {file:line} | ⚠️ To be confirmed / ✅ Confirmed |

> **Handling rule**: when documentation and code disagree, ask the user to confirm with the interface owner before continuing.

---

## 6. ACLNN Call Chain (Fill For Composite Scenarios)

> Leave this section empty if PTA directly calls a single `aclnnXxx`.

```text
Forward call chain of {OpName}:
  1. aclnn{Sub1}(...) -> {intermediate_1}
  2. aclnn{Sub2}(...) -> {output}

Backward call chain of {OpName}:
  1. aclnn{SubGrad}(...) -> {grad_outputs}
```

---

## 7. MindSpore Adaptation Conclusion And Integration Plan

### 7.1 Integration Decision

| Attribute | Value | Basis |
| ---- | -- | ---- |
| **Integration type** | new primitive / reuse primitive | {Whether parameters match, whether there is only a name mapping, whether semantics differ} |
| **Integration path** | auto / customize | {Whether parameters can be passed to ACLNN unchanged} |
| **ACLNN interface** | `aclnn{Xxx}` | |
| **Composite scenario** | {yes/no} | The sub-operator list is in Section 6 |

### 7.2 Primitive / YAML Strategy

| Question | Conclusion | Explanation |
| ---- | ---- | ---- |
| **Is there already a primitive with the same name?** | {yes / no} | |
| **Can the existing primitive be reused?** | {yes / no (parameter-incompatible)} | {Why it is incompatible} |
| **YAML strategy** | {new / modify existing YAML} | |
| **`dispatch` configuration** | `enable: True` {+ `Ascend: XxxAscend` for Path 2 / omit `Ascend` for Path 1} | |

### 7.3 Parameter Preprocessing Needed For Customize (Fill For Path 2)

> Leave this section empty on Path 1.

| MindSpore Parameter | MindSpore Type | ACLNN Expects | Preprocessing Method |
| -------- | ------- | ---------- | ---------- |
| {param} | {type} | {aclnn_type} | {for example Scalar -> ScalarPtr / tuple -> vector / None -> empty tensor} |

### 7.4 Backward Strategy

| Attribute | Value |
| ---- | -- |
| **PTA backward style** | {dedicated `aclnnGrad` / autograd composition / no backward} |
| **MindSpore backward style** | {handwritten `REG_BPROP_BUILDER` / autodiff via `bprop_expander: False` / no backward} |
| **Backward ACLNN** | {`aclnnXxxGrad` / none, use existing ops composition} |
