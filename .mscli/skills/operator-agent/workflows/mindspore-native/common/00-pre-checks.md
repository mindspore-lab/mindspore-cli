# Workflow 0: Pre-Checks (Pre-A / Pre-B / Pre-C)

## Goal

Before writing code, complete the repository inventory check, reference analysis, solution design, and, for composite scenarios, a call-chain inventory.

## Inputs

- **Operator name**: the API name, Primitive name, and ACLNN interface name
- **PTA reference interface**: `torch_npu.npu_xxx` or `torch.xxx`

## Outputs

- **Inventory result**: which parts of this operator already exist or are missing in the MindSpore repository
- **Solution design document**: interface type, integration category, impact analysis, output as a Markdown file
- **ACLNN call-chain inventory** (for composite scenarios): sub-operator coverage status and rollout plan

## Constraints

- **Local source wins**: source inspection of PTA (`op-plugin`), PyTorch, and MindSpore **must be performed by searching the local workspace**.

---

## Pre-A: Inventory Check

When the user asks you to "add" or "adapt" an operator, **search first** to confirm whether the operator already exists in the repository.

### Steps

1. **Search YAML**: search for the operator name under `mindspore/ops/op_def/yaml/` and `mindspore/ops/api_def/yaml/`
2. **Search Infer**: search the corresponding Infer registration under `ops_func_impl` / `ops/infer`
3. **Search PyBoost**: search for `class OPS_ASCEND_API {OpName}`
4. **Search KBK**: search for `MS_ACLNN_KERNEL_FACTORY_REG({OpName}, ...)` and `MS_ACLNN_COMMON_KERNEL_FACTORY_REG({OpName}, ...)`
5. **Search BPROP**: search for `REG_BPROP_BUILDER("{OpName}")`
6. **Search tests**: search under `tests/st/ops` and `test/ut/cpp/ops`
7. **Search documentation**: search under `docs/api/`

### Output Template

```text
Operator inventory check: {OpName}

| Component | Status | File Path | Notes |
| ---- | ---- | -------- | ---- |
| YAML (op_def) | ✅/❌ | ... | |
| YAML (api_def) | ✅/❌ | ... | |
| Infer | ✅/❌ | ... | |
| PyBoost | ✅/❌ | ... | |
| KBK kernel | ✅/❌ | ... | |
| BPROP | ✅/❌ | ... | |
| Tests (UT) | ✅/❌ | ... | |
| Tests (ST) | ✅/❌ | ... | |
| Docs (EN) | ✅/❌ | ... | |
| Docs (CN) | ✅/❌ | ... | |

Conclusion: {brand-new development / only xxx parts need to be added}
```

---

## Pre-B: Solution Design And Reference Analysis

Analyze the differences between the MindSpore, PTA, and ACLNN interfaces, decide the primitive/interface integration strategy, **choose the integration path (Path 1 auto-generated / Path 2 Customize)**, and initialize the Feature document.

### Steps

1. **PTA source review (mandatory)**: review the three key file categories in `op-plugin` (see `reference.md#pta-source-review`)
   - `op_plugin_functions.yaml`: function signatures, parameter types/defaults
   - `derivatives.yaml`: backward registration and differentiable inputs
   - `XxxKernelNpuOpApi.cpp`: actual ACLNN call and parameter preprocessing
   - Check whether PTA has **overloads with the same name** but different signatures
   - **ACLNN interface definition**: look up the corresponding ACLNN document in `../../references/aclnn_doc` (for example `aclnnAbs.md`)
2. **Five-factor interface analysis (mandatory)** (`reference.md#api-analysis-five-factors`)
   - Check whether functionality, parameter definitions, and dtypes match
   - Decide **whether a new primitive is needed** and **whether to add a new interface or reuse an existing one**
3. **Choose the YAML strategy** (`reference.md#yaml-three-scenarios`)
   - YAML interface definitions are described in `mindspore/ops/op_def/yaml/README.md`
   - Existing YAML + reuse existing primitive -> add a `dispatch` field
   - Existing YAML + new primitive -> create a new YAML with the `_ext` suffix
   - No YAML exists -> create a new one
   > **Clarification of the relationship among YAML, primitive, and function interface:**
   > YAML is tightly bound to the primitive, and the defined interface generates the primitive interface.
   > The YAML flow can also auto-generate function interfaces that directly call the generated primitive instance, which simplifies implementation. This is controlled by the `function`-related fields in YAML.
   > Primitive and function interfaces are related, but not strictly identical.
   > When the backend interface(e.g. aclnn interface) and primitive interface match, prefer a primitive name **without** the `Ext` suffix. Use `Ext` only when a primitive with the same name already exists and cannot be reused because of interface dismatch. If a function interface with the same name already exists but no primitive does, still prefer the primitive name without `Ext`; primitives and function interfaces are distinct.
4. **Choose the integration path (core decision)** (`reference.md#dispatch-path-selection`)
   - Determine whether the MindSpore API parameters can be **passed through unchanged** to the ACLNN interface
   - **Path 1 (auto-generated)**: direct passthrough -> omit the `Ascend` field in YAML -> PyBoost and ACLNN kernelmod are auto-generated
   - **Path 2 (Customize)**: parameters require preprocessing -> write `Ascend: XxxAscend` in YAML -> PyBoost and ACLNN kernelmod must be handwritten
   - Common preprocessing cases: scalar extraction, argument reordering, manual output allocation
   - **This decision determines the implementation workload for all later steps and must be finalized in Pre-B**
   - YAML also supports `type_cast` for simple input type conversion. If the converted parameters then match the ACLNN interface, Path 1 can still be used.
5. **Produce a PTA difference record**: use the `templates/pta-analysis-report.md` template and generate a file such as `{op_name}_pta_analysis.md`

---

## 🔒 Feature Document Initialization (Must Run After Pre-B, Cannot Be Skipped)

> **This is a required review and test-handoff deliverable.** No matter the scenario, forward or backward, single operator or composite, internal or public, you must generate a Feature document. If you skip this step, later review will fail.

### Steps

1. Copy `templates/feature-document.md` and name it `{operator_name}_Feature.md`
2. Fill the following sections based on the Pre-B analysis results:
   - [1. Background](../../../templates/feature-document.md#feature-background)
   - [2. Benchmark And APIs](../../../templates/feature-document.md#feature-benchmark-api)
   - [3. Task List](../../../templates/feature-document.md#feature-task-list) (initialize the standard 13-category table)
   - [4. Functional And API Specification](../../../templates/feature-document.md#feature-functional-spec) (interface signature and parameter descriptions)
   - [6. Constraints And Types](../../../templates/feature-document.md#feature-constraints) (device, dtype, and shape constraints)
   - [8. Differences From PTA And Alignment Status](../../../templates/feature-document.md#feature-pta-alignment) (initial version)

---

## Pre-C: ACLNN Call-Chain Analysis And Sub-Operator Inventory (Mandatory For Composite Scenarios)

> Execute this only when PTA C++ uses **multiple smaller ACLNN operators chained together**.
> Skip it if PTA directly calls a single `aclnnXxx`.

### Steps

1. **Extract the ACLNN call chain**: extract all forward and backward `EXEC_NPU_CMD` / `aclnnXxx` calls from the PTA C++ code (see `reference.md#aclnn-callchain-extraction`)
2. **Inventory MindSpore coverage**: search one by one to confirm whether each sub-operator has already been integrated (`reference.md#ms-coverage-inventory`)
3. **Produce the coverage inventory** using `templates/aclnn-callchain-analysis.md`
4. **Plan the rollout order**: leaves first, composite later; follow topological order (`reference.md#callchain-rollout-order`)

---

## Success Criteria

**⛔ HARD GATE: before entering Step 1, the following two items must both be completed and delivered to the user:**
1. ✅ PTA source review report (output of Pre-B, using `templates/pta-analysis-report.md`)
2. ✅ Initialized Feature document

**Important**: "delivered to the user" means generating real `.md` files in the workspace and explicitly telling the user their file paths.
