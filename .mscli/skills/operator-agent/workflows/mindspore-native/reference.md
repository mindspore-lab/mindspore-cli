# Full ACLNN Operator Development Reference (Detailed)

This file now serves as the detailed implementation reference for the
MindSpore native workflow under `operator-agent`.
The top-level `SKILL.md` remains responsible for routing, while the concrete
implementation detail lives here so the routing docs do not become bloated.

<a id="reference-index"></a>
## Contents

- [1. Directory and File Discovery](#directory-and-file-discovery)
- [2. YAML Design Templates](#yaml-design)
- [3. Common `gen_ops.py` Troubleshooting](#gen-ops-py-troubleshooting)
- [4. GeneralInfer (C++) Conventions](#general-infer)
- [5. PyBoost (Pynative) Implementation Notes](#pyboost-reference)
- [6. KBK (Graph) Kernel Notes](#kbk-reference)
- [7. BPROP Wiring Notes](#bprop-reference)
- [8. Test Strategy (UT + ST)](#testing-reference)
- [9. Delivery / Handoff Consensus Points](#delivery-consensus)
- [10. Documentation and Material Development Notes](#documentation-reference)
- [11. Performance Self-Check Tool: `apitimewrapper`](#11-performance-self-check-tool-apitimewrapper)
- [12. Backward Implementation Notes](#bprop-advanced-notes)
- [13. Resize / Launch Optimization Notes](#resize-launch-optimization)
- [14. Bitwise Accuracy and Memory Alignment Self-Checks](#accuracy-and-memory-validation)
- [15. API Development Notes (functional / nn / Tensor)](#api-development)
- [16. When ACLNN / PTA Docs Are Incomplete: Use Probe Scripts](#16-when-aclnn--pta-docs-are-incomplete-use-probe-scripts-to-fill-the-fact-gap)
- [17. `vmap` Support (When Needed)](#17-vmap-support-when-needed)
- [18. Code Skeleton Templates](#code-skeletons)
- [19. PTA Source Review Method (Mandatory)](#pta-source-review)
- [20. InferValue Constant Folding (Optional Optimization)](#infervalue-constant-folding)
- [21. Dynamic Shape Classification and Handling Strategy](#dynamic-shape-strategy)
- [22. ACLNN Call-Chain Analysis and Sub-Operator Inventory](#aclnn-callchain-analysis)
- [23. Composite Implementation Patterns (C++ Small-Op Composition + Meta DSL)](#composite-implementation)
- [24. Feature Document (Mandatory for Review and Delivery)](#feature-document-reference)
- [25. API Overload Adaptation (Tensor / functional Same Name, Multiple Signatures)](#api-overload-adaptation)
- [26. View Operator Development (Zero-Copy Shape / Strides Transform)](#view-operator-development)

---

## 1. Directory and File Discovery (Prefer Search Over Hard-Coded Paths)

<a id="directory-and-file-discovery"></a>
MindSpore / `op-plugin` layouts may differ across branches, so search first:
- Search by strings such as `gen_ops.py`, `LAUNCH_ACLNN`,
  `MS_ACLNN_KERNEL_FACTORY_REG`, and `REG_BPROP_BUILDER`.
- Search by similar operators: classify the target operator by its traits, then
  find already integrated operators of the same kind in the repo
  (see [`2.4 Similar-Operator Search Strategy`](#similar-operator-search)).

Common target areas, only as directional hints:
- **YAML**: `mindspore/ops/op_def/yaml/`
- **Infer / meta implementation**: directories under `mindspore/` such as
  `ops`, `infer`, `ops_func_impl`, depending on the actual repo
- **Ascend kernel / PyBoost / KBK**: `mindspore/ccsrc/` and `op-plugin-*`
  subtrees containing `ascend`, `kernel`, `aclnn`, or `customize`
- **bprop**: `mindspore/ccsrc/` locations such as `bprop` or `grad_*ops.cc`
- **tests**: `tests/ut/`, `tests/st/`
- **docs**: English `function_doc` YAML and Chinese
  `docs/api/api_python/ops/*.rst`

<a id="yaml-design"></a>
## 2. YAML Design Templates (One Forward, One Backward)

<a id="yaml-minimal-consistency"></a>
### 2.1 Minimum Consistency Principle

The same parameter, for example `actual_seq_len`, must remain **consistent**
across:
- YAML (`op_def` + `api_def` + `function_doc`)
- GeneralInfer (C++ infer)
- PyBoost (Pynative call path)
- KBK (Graph kernel argument handling / launch)
- Documentation (English and Chinese)
- UT / ST (including parameter boundaries and error paths)

<a id="customize-suffix"></a>
### 2.2 `Customize` Suffix

If you are using the project’s default ACLNN kernel mechanism, you
**generally do not need to add a `Customize` suffix manually in YAML**.
The framework handles that automatically.

<a id="dispatch-path-selection"></a>
### 2.3 Two Integration Paths (The Core Decision That Drives Total Workload)

The most important ACLNN integration decision is:
can the MindSpore API parameters be passed to the ACLNN interface as-is?

That determines whether you use the **auto-generated path** or the
**manual Customize path**, which directly determines which files must be
written.

<a id="dispatch-path-1-auto-generated"></a>
#### Path 1: Auto-Generated (Direct Parameter Pass-Through, No Customize Needed)

**Applicable when**: the MindSpore API parameters exactly match the ACLNN
interface parameters.
That means the parameter count, order, type, and default values all match and
need no conversion before the call.

**Key YAML rule**: set `dispatch: enable: True` in `op_def`, and **do not**
write the `Ascend:` field.
Inside the framework, `Ascend` defaults to `'default'`, which selects the
auto-generated path.

```yaml
# Path 1 example, such as abs, mul, trunc
dispatch:
  enable: True
  # Omit Ascend -> auto-generated path
```

**Generated automatically at build time**:
- PyBoost call code
  (`pyboost_ascend_call_template.tpl` -> `LAUNCH_ACLNN(aclnnXxx, ...)`)
- KBK registration
  (`MS_ACLNN_COMMON_KERNEL_FACTORY_REG` -> `aclnn_kernel_register_auto.cc`)
- Python API wrappers such as `functional_overload.py`

**Files developers still need to write by hand**:

| File | Step |
| --- | --- |
| `op_def/yaml/xxx_op.yaml` | Step 1 |
| `api_def/xxx.yaml` | Step 1 |
| `op_def/yaml/doc/xxx_doc.yaml` (`_ext` style) or `api_def/function_doc/xxx_doc.yaml` (legacy style) | Step 1 |
| `infer/ops_func_impl/xxx.h` + `.cc` | Step 3 |
| add mapping to `aclnn_config.yaml` (edit, **Path 1 only**) | Step 2 |
| exports in `math_func.py` / `mint/__init__.py` / `tensor_method.py` (edit) | Step 7 |
| `tests/ut/cpp/ops/test_xxx.cc` | Step 8 |
| `tests/st/ops/share/_op_info/op_database.py` (OpInfo registration) + matching `test_xxx_ops.py` | Step 8 |
| English `function_doc` + Chinese RST (one per API form) | Step 9 |

**Files you do not need**: PyBoost customize files and KBK customize files.
That means **skip Step 4 and Step 5**.

**Examples**: `abs`, `mul`, `trunc`, `xlogy`, `div` (basic arithmetic).

<a id="dispatch-path-2-customize"></a>
#### Path 2: Manual Customize (Parameter Preprocessing Required)

**Applicable when**: arguments must be transformed before calling ACLNN.
Typical cases:
- `tuple[int]` -> `std::vector<int64_t>` such as `actual_seq_qlen`
- `Optional[Tensor]` needs explicit `None` semantics
- `str` -> enum / int conversion such as `layout: "BSND"`
- scalar extraction from `Value`
- multiple inputs must be reordered or merged before the ACLNN call
- output tensors must be allocated manually because output shape differs from
  input shape

**Key YAML rule**: `dispatch: enable: True` plus
`Ascend: XxxAscend`, explicitly naming the Customize class.

```yaml
# Path 2 example, such as dense_lightning_indexer_grad_kl_loss
dispatch:
  enable: True
  Ascend: DenseLightningIndexerGradKlLossAscend
```

At build time, `gen_ops.py` generates wrapper code that calls your handwritten
Customize class
(`pyboost_ascend_customize_call_template.tpl` -> `XxxAscendCustomize(...)`).

**Additional handwritten files required** beyond Path 1:

| File | Step |
| --- | --- |
| `kernel/.../pyboost_impl/customize/xxx.h` + `.cc` | Step 4 |
| `kernel/.../kernel_mod_impl/customize/xxx_aclnn_kernel.h` + `.cc` | Step 5 |
| the `_grad` versions of the files above, if backward exists | Step 4 / 5 |

**Examples**:
`dense_lightning_indexer_grad_kl_loss`, `multi_scale_deformable_attn`,
`conv2d_ext`, `add`.

<a id="dispatch-path-decision-flow"></a>
#### Path Decision Flow

```text
Compare MindSpore API parameters against ACLNN interface parameters
                │
      Can parameters be passed through directly?
       ╱                         ╲
      Yes                        No
      │                          │
  Path 1 (auto)             Path 2 (Customize)
      │                          │
  No Ascend field           Add Ascend: XxxAscend
      │                          │
  Skip Step 4/5             Must implement Step 4/5
      │                          │
  Build auto-generates      Build calls your
  PyBoost / KBK             Customize class
```

<a id="integration-type-mapping"></a>
#### Mapping from “Integration Type” to Path

| Integration type | Description | Path |
| --- | --- | --- |
| **Type 1** | API definition fully matches ACLNN | **Path 1** (auto-generated) |
| **Type 2** | Name differs but functionality matches | usually **Path 1** (use YAML `class` mapping) |
| **Type 3** | Prototype / semantics differ | **Path 2** (manual Customize required) |

> **Note**: whether Type 2 needs Customize depends on whether the difference is
> only the operator name.
> If only the name differs and parameters match exactly, Path 1 is enough.
> If parameter order or parameter types differ, it is still Path 2.

<a id="similar-operator-search"></a>
### 2.4 Similar-Operator Search Strategy (Do Not Hard-Code Operator Names)

During development, you often need similar integrated operators as references
for structure, directory layout, macros, and registration style.
**Do not start by assuming a few operator names.**
First analyze the target operator’s traits, then search the repo for matching
integrated operators.

#### Classification Dimensions (In Priority Order)

#### A. Functional / Algorithm Family

This is the most intuitive dimension. Operators in the same family often share
very similar implementation patterns.

| Family | Typical operators | Shared traits |
| --- | --- | --- |
| **Attention** | `flash_attention`, `nsa_compress_attention`, `paged_attention`, `incre_flash_attention` | TND / BSND layouts, multiple outputs (`softmax_max` / `softmax_sum`), mask / `actual_seq_len`, separate grad op |
| **Loss** | `cross_entropy`, `cosine_embedding_loss`, `ctc_loss`, `nll_loss` | forward returns loss + cached intermediates like `log_sum_exp`, `reduction` parameter, backward depends on intermediates |
| **Norm** | `layer_norm`, `group_norm`, `rms_norm`, `batch_norm` | input + weight + bias, running mean / var state, `rstd` intermediates, backward outputs `dx/dw/db` |
| **Optimizer** | `adam`, `sgd`, `lamb`, `adamw` | in-place update, scalar params like lr / beta / epsilon, multiple tensor inputs, usually no backward |
| **Activation** | `relu`, `gelu`, `silu`, `swish`, `leaky_relu` | elementwise, one input one output, simple backward, usually direct Path 1 |
| **Elementwise arithmetic** | `add`, `mul`, `div`, `eq`, `ne`, `gt` | elementwise, broadcasting, Tensor-Scalar overloads, symbolic overloads like `__add__`, polymorphic dispatch |
| **Reduce** | `sum`, `mean`, `prod`, `amax`, `argmax` | reduce over axis, `keepdim`, reduced output rank, some have backward and some do not |
| **Matrix math** | `matmul`, `bmm`, `linear`, `baddbmm` | 2D / 3D matmul, transpose params, alpha / beta, output shape follows matmul rules |
| **Index / gather** | `index_select`, `gather`, `scatter`, `embedding` | index tensor inputs, irregular infer logic, backward often follows scatter / zero-fill patterns |
| **Shape / reorder** | `reshape`, `transpose`, `permute`, `contiguous` | often pure shape changes, no ACLNN compute, backward absent or inverse |
| **Conv / pool** | `conv2d`, `avg_pool2d`, `max_pool2d` | kernel / stride / padding / dilation groups, NCHW / NHWC, separate grad ops |
| **Communication / distributed** | `all_reduce`, `all_gather`, `reduce_scatter` | collective communication, group param, side effects, usually no standard ACLNN and instead HCCL |

> **How to use it**: identify the target operator family first, then search the
> repo for already integrated operators in the same family.
> Operators in the same family often share infer logic, PyBoost / KBK call
> shape, bprop wiring, and test coverage strategy. Those are your best
> references.

#### B. Technical Implementation Traits

Use this as a second-stage filter inside the same family.

| Dimension | Typical categories | Search hints |
| --- | --- | --- |
| **input layout** | TND / BSND / BNSD / standard elementwise | grep matching shape comments in `op_def/yaml/` |
| **ACLNN integration style** | single ACLNN direct call / multi-ACLNN composition / no ACLNN (pure Python composition) | count `LAUNCH_ACLNN`; for composed ops inspect `customize` dirs |
| **has backward** | separate grad op / autodiff / no backward | grep `REG_BPROP_BUILDER` and `_grad` YAML |
| **API surface** | functional only / functional + nn / functional + tensor / symbolic overload | inspect `interface` in `api_def` YAML |
| **special parameters** | `Optional[Tensor]`, `tuple[int]`, enums like `layout` / `mode`, scalars | inspect YAML `default: None`, `type_cast`, `arg_handler` |
| **integration type** | Type 1 exact match / Type 2 name mapping / Type 3 customize | judge using [`2.3 Two Integration Paths`](#dispatch-path-selection) |

<a id="similar-operator-search-flow"></a>
#### Search Workflow

1. **Identify the functional family** of the target operator.
   - Example: `nsa_compress_attention` -> **Attention**
   - Example: `cosine_embedding_loss` -> **Loss**
   - Example: `eq` (`==` overload) -> **Elementwise arithmetic**

2. **Pick 2 to 3 technical tags** from table B to narrow candidates.
   - Example: `nsa_compress_attention` -> Attention + TND layout +
     single ACLNN direct call + separate grad + `tuple[int]`
   - Example: `cosine_embedding_loss` -> Loss + multi-ACLNN composition +
     no standalone Primitive + functional + nn + `reduction`
   - Example: `adamw` -> Optimizer + in-place side effect +
     multiple tensor inputs + no backward

3. **Search the repo** for similar cases:

   ```bash
   # By family name: search same-family operators, for example attention
   grep -rl "attention" mindspore/ops/op_def/yaml/ --include="*.yaml"

   # By layout: search operators using the same shape mode
   grep -r "TND" mindspore/ops/op_def/yaml/ --include="*.yaml" -l

   # By ACLNN composition: search customize files with multiple LAUNCH_ACLNN calls
   grep -rl "LAUNCH_ACLNN" mindspore/ops/kernel/.../customize/

   # By backward pattern: search operators with Grad YAML
   ls mindspore/ops/op_def/yaml/*_grad_op.yaml

   # By API surface: search operators that have both tensor and function interfaces
   grep -l "interface:.*tensor.*function" mindspore/ops/api_def/*.yaml

   # By reduction parameter, common in loss operators
   grep -l "reduction" mindspore/ops/op_def/yaml/*.yaml
   ```

4. **Choose the 2 to 3 closest operators**, and compare their YAML / Infer /
   PyBoost / KBK / bprop / tests / docs directory by directory.
   Prefer same-family + closest technical traits first, then different-family
   operators with highly similar integration traits.

5. **If no close match exists**, fall back to any operator with the same
   integration type from [`2.3 Two Integration Paths`](#dispatch-path-selection)
   as a style reference, and validate more carefully step by step.

> **Principle**: similar operators are references for structure and style, not
> templates to copy for business logic.
> The functional logic must follow PTA source code and ACLNN docs. Similar
> operators are only for directory layout, macro names, registration patterns,
> test style, and related structure.

<a id="dispatch-bootstrap-pattern"></a>
### 2.5 Practical Pattern: `dispatch` + “Auto-Generate First, Then Copy and Adapt”

When you need custom PyBoost / KBK code, an efficient pattern is:
1. Enable `dispatch.enable: True` in YAML.
2. **Temporarily comment out** custom entries such as
   `dispatch.Ascend: XxxAscend` so `gen_ops.py` first generates a compilable
   skeleton.
3. **Copy** the generated `.h/.cc` files from the generated directory into the
   `customize` directory or the matching custom directory.
4. Adjust inputs to match the real ACLNN signature, for example remove dtypes
   ACLNN does not need or convert tuple -> vector.
5. Rename the entry according to project convention, commonly
   `OpNameAscendCustomize` / `OpNameGradAscendCustomize`, then restore the YAML
   declaration.
6. Remove the temporary auto-generated files and keep only the custom
   implementation.

<a id="gen-ops-py-troubleshooting"></a>
## 3. Common `gen_ops.py` Troubleshooting

Typical errors and directions:
- **`keys` structure mismatch**: compare against working basic-operator YAML,
  such as `add`, and fix the field hierarchy.
- **missing `py_method`**: fill in the Python exposure fields.
- **missing `function_doc` entries**: complete the corresponding doc nodes and
  keep parameters aligned.

Tip: on Windows, avoid mixing Chinese characters into English YAML docs to
reduce encoding issues.

<a id="general-infer"></a>
## 4. GeneralInfer (C++) Conventions

<a id="general-infer-responsibilities"></a>
### 4.1 Responsibility Boundary

- Only do **shape / type inference**.
  Do not perform runtime legality checks here. Those belong to ACLNN or the
  runtime.
- Use framework exception macros for errors. Error messages should include the
  parameter name, expected value, and actual value.

<a id="general-infer-dynamic-shape-rank"></a>
### 4.2 Dynamic Shape / Dynamic Rank

> For the full three-way classification of dynamic shape, including
> `InputDynamic` and `OutputDynamic`, see
> [`21 Dynamic Shape Classification and Handling Strategy`](#dynamic-shape-strategy).
> This section focuses on quick fallback rules inside Infer.

Recommended strategy, consistent with prior development experience:
- Dynamic rank: return dynamic rank
  (`kShapeRankAny` or the equivalent constant in the repo).
- If a key infer dependency such as `block`, `stride`, or `seq_len` is unknown:
  - set the affected output dimension to dynamic
    (`kShapeDimAny` or repo equivalent)
  - infer the remaining dimensions from inputs as usual
- When all key inputs are known: return as precise a shape as possible.

<a id="general-infer-api"></a>
### 4.3 Common `InferInfo` APIs (The Header Definition Is the Source of Truth)

`InferInfo`-related interfaces are defined in
`mindspore/core/include/ops/infer_info/infer_info.h`.
The header is the final authority.

- `IsDynamic`, `IsDynamicRank`: dynamic shape / rank checks
- `GetScalarValueWithCheck<T>()`: scalar extraction with validation
- `GetArrayValue<T>()` + `HasUnknownValue()`: tuple / list extraction
- `IsNone()`: detect `None`

Do not invent APIs that do not exist in the repo.

<a id="pyboost-reference"></a>
## 5. PyBoost (Pynative) Implementation Notes

<a id="pyboost-argument-normalization"></a>
### 5.1 Input Argument Conversion

- For tuple / list arguments, convert them to `std::vector<int64_t>` before
  passing to ACLNN whenever possible.
- For optional inputs, define explicit `None` semantics and handle them
  consistently in PyBoost, Infer, and KBK.

### 5.2 Calling Convention

Follow the project’s existing ACLNN wrappers such as `LAUNCH_ACLNN` or
`RunOp`, and keep the style consistent with the repo.

<a id="kbk-reference"></a>
## 6. KBK (Graph) Kernel Notes

> For Init / Resize / Launch responsibility split, useless outputs, and
> compute-depend outputs, see
> [`13 Resize / Launch Optimization Notes`](#resize-launch-optimization).

Recommended fixed structure:
- `GetWorkSpaceInfo()`: parse arguments + call `GetWorkspaceForResize`
- `Launch()`: call `RunOp` or the equivalent execution path
- registration: `MS_ACLNN_KERNEL_FACTORY_REG` or the repo-equivalent macro

Hard constraints:
- forward and backward must use separate files and separate registration
- header / implementation namespaces must match, or you will easily hit
  “not declared / not defined” failures

<a id="kbk-auto-generated-skeleton"></a>
### 6.1 Where Auto-Generated KBK Skeletons Usually Land

From existing examples, KBK auto-generated code often appears in paths like:
- `.../ops/kernel/ascend/opapi/aclnn_auto_gen/`

The actual location depends on the repo. A common pattern is to let
`gen_ops.py` generate the initial code, then copy it into the custom directory
and adapt it
(see [`2.5 dispatch + Auto-Generate First, Then Copy and Adapt`](#dispatch-bootstrap-pattern)).

<a id="bprop-reference"></a>
## 7. BPROP Wiring Notes

> For advanced backward concerns such as `OutZeros`, `ZerosLikeExt`, inplace,
> and `Depend`, also see
> [`12 Backward Implementation Notes`](#bprop-advanced-notes).

Inside the bprop builder:
- build backward subgraphs only for inputs that actually need gradients
- return zero-gradient placeholders for non-tensor inputs or inputs that do not
  require gradients
- use `need_compute_grad_out()` or the repo-equivalent API to decide whether a
  backward branch is required

<a id="bprop-io-rules"></a>
### 7.1 Heuristic Rule for Backward Input / Output Counts

- **backward input count** =
  “forward input count + 2” (`out` and `dout`)
- **backward output count** =
  “forward input count” (one gradient per input)
- for multi-output forward ops, `out` on the backward side is usually a tuple,
  and you need `TupleGetItem` to extract individual outputs

<a id="bprop-set-unused-inputs"></a>
### 7.2 When to Use `SetUnusedInputs`

If backward does not depend on the tensor value of some input, for example it
depends only on shape / type or does not use the input at all, mark it as
unused. In Pynative async mode this can release forward-kernel memory earlier
and reduce peak memory.

<a id="bprop-dynamic-inputs"></a>
### 7.3 Dynamic Inputs in Graph Mode (Must Be Considered in BPROP)

> **Priority: Pynative correctness > KBK dynamic compatibility.**
> If you cannot implement a correct KBK dynamic version yet, keep the Pynative
> version correct first and let failing cases drive later fixes.

In Graph mode (KBK), the **value or shape** of forward inputs may be unknown at
compile time (`ValueAny`, dynamic dimension, dynamic rank).
A bprop builder written with direct C++ `if/else` and `GetValue<T>()` only
works when values are known at compile time.
If values may be unknown, you must use the framework’s **runtime deferral
mechanisms**.

#### Three Cases and the Matching Tools

| Case | Compile-time behavior | Tool | Notes |
| --- | --- | --- | --- |
| **scalar value unknown** | `GetScalarValue` result has `has_value == false` | `ib->Conditional(cond, true_br, false_br)` | Turn C++ branching into graph-time conditional subgraphs |
| **shape dimension unknown** | `IsDynamicShape(shape)` is true | `DEF_PURE_SHAPE_CALC` + `ib->ShapeCalc(calc, inputs, indices)` | Wrap shape-dependent logic in a ShapeCalc node |
| **rank unknown** | `IsDynamicRank(shape)` is true | separate dynamic-path function | Split the whole backward path into a dedicated dynamic path |

#### Typical Patterns

**Pattern A: scalar value unknown -> `Conditional`**

```cpp
auto keep_dims_value = keep_dims->BuildValue();
auto keep_dims_opt = GetScalarValue<bool>(keep_dims_value);
if (keep_dims_opt.has_value()) {
  // Compile-time known: take the direct C++ branch
  if (!keep_dims_opt.value()) { std_d = ib->Reshape(std_d, res[0]); }
} else {
  // Compile-time unknown: build a runtime conditional subgraph
  auto true_branch = [&](Emitter *e) -> NodePtrList { return {e->Reshape(std_d, res[0])}; };
  auto false_branch = [&](const Emitter *e) -> NodePtrList { return {std_d}; };
  auto cond = ib->Equal(keep_dims, ib->Value<bool>(false));
  std_d = ib->Conditional(cond, true_branch, false_branch);
}
```

> **Reference**: `ReduceStd` bprop in `grad_math_ops.cc`

**Pattern B: dynamic shape -> separate dynamic path**

```cpp
bool is_dynamic_rank = IsDynamicRank(x_shape) || IsDynamicRank(w_shape);
bool is_dynamic_shape = IsDynamicShape(x_shape) || IsDynamicShape(w_shape);
if (is_dynamic_rank || is_dynamic_shape) {
  return MatMulBackwardDynamic(ib, is_complex);  // dedicated dynamic path
}
// static path
dx = MatMulInputBackward(ib, is_complex);
```

> **Reference**: `MatMulExt` bprop in `grad_math_ops.cc`

**Pattern C: shape-dependent computation -> `ShapeCalc`**

```cpp
DEF_PURE_SHAPE_CALC(g_reduce_std)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    return ReduceStdShapeFunc(inputs.at(0), inputs.at(1));
  })
  ...
// Called inside bprop
auto res = ib->ShapeCalc(g_reduce_std, {x, axis}, {1});
```

> **Reference**: `ReduceStd` bprop in `grad_math_ops.cc`

#### Implementation Checklist

After writing bprop, check each item:

- [ ] For every scalar input used in backward (`int` / `float` / `bool`), did you check `BuildValue()->ContainsValueAny()`?
- [ ] For unknown scalar values, did you use `Conditional` instead of C++ `if`?
- [ ] For shape-dependent computation in backward, such as using `ib->GetShape()` to build `axis` / `dims`, did you handle dynamic shape?
- [ ] If rank can be dynamic, did you add `IsDynamicRank` checks and a fallback path?

<a id="testing-reference"></a>
## 8. Test Strategy (UT + ST)

<a id="testing-cpp-ut"></a>
### 8.1 C++ UT (GeneralInfer)

Typical construction, based on the repo’s existing UT helpers:
- scalar: `ShapeVector{}` + `CreateScalar<T>(value)`
- tuple: `ShapeArray{{}}` + `ValuePtrList{CreateScalar<...>(...)}`
- `None`: `kMetaTypeNone` + `kNone`
- unknown: `kValueAny` or the repo-equivalent placeholder

<a id="testing-st-opinfo"></a>
### 8.2 ST `op_info` Test Framework

> The core idea is to move from “write a test file for each operator” to
> “register a new operator into an existing parameterized test system.”
> The main code lives under `tests/st/ops/share/`, and is built from
> **OpInfo** (“what to test”) plus **OpsFactory** (“how to test it”).

#### 8.2.1 Directory Layout

```text
tests/st/ops/share/
├── _op_info/
│   ├── op_info.py        # OpInfo and derived classes
│   ├── op_database.py    # all OpInfo registration + xxx_op_db lists
│   └── op_common.py      # helpers: make_tensor / OpSampleInput / OpDynamicInput
├── _internal/
│   ├── meta.py           # OpsFactory base class (test_op_reference / test_op_dynamic etc.)
│   ├── binary_ops.py     # BinaryStdOpsFactory and related specializations
│   └── ...
├── ../op_info_tests/
|   ├── test_binary_ops.py
|   ├── test_unary_ops.py
|   └── ...
```

#### 8.2.2 OpInfo Configuration

OpInfo defines which operator is tested, which inputs are used, and what the
reference implementation is.

| Key field | Meaning |
| --- | --- |
| `name` | operator name, used as the parameterized test id |
| `op` | callable operator object such as `mint.add` |
| `ref` | reference implementation such as `numpy.add` |
| `dtypes_xxx` | dtype sets for different scenarios such as `dtypes_support`, `dtypes_grad`, `dtypes_dynamic` |
| `op_basic_reference_inputs_func` | base input generator returning a list of `OpSampleInput` |
| `op_extra_reference_inputs_func` | optional extra input scenarios |
| `compare` | compare config such as `atol` / `rtol` |

**Derived classes** with built-in default input generators:

| Class | Use case | Default inputs |
| --- | --- | --- |
| `BinaryOpInfo` | binary operators such as `add` / `mul` | two random tensors of the same shape |
| `UnaryOpInfo` | unary operators such as `abs` / `sin` | one random tensor |
| `ReductionOpInfo` | reductions such as `sum` / `mean` | one random tensor + `axis` |

**Configuration example** from `op_database.py`:

```python
# Binary operator: use BinaryOpInfo and reuse default inputs
BinaryOpInfo(
    name="mint.add",
    op=mint.add,
    ref=numpy.add,
    dtypes_support=FLOAT_TYPES | INT_TYPES,
)

# Custom input scenarios through op_basic_reference_inputs_func
OpInfo(
    name="mint.clamp",
    op=mint.clamp,
    ref=numpy.clip,
    op_basic_reference_inputs_func=clamp_inputs_func,
    dtypes_support=FLOAT_TYPES,
)
```

After registration, add the operator name to the matching list such as
`binary_op_db` or `unary_op_db`, and the frontend test files will pick it up
automatically.

#### 8.2.3 OpsFactory Test Suite

OpsFactory provides standardized test methods so developers do not have to
write test logic by hand.

| Method | Meaning |
| --- | --- |
| `test_op_reference()` | run the operator on `OpSampleInput` and compare numerically against `ref` |
| `test_op_dynamic()` | test dynamic shape (`DYNAMIC_SHAPE` / `DYNAMIC_RANK`) |
| `compare_with_torch()` | compare numerically with PyTorch |

Hierarchy:
`OpsFactory` -> `UnaryOpsFactory` / `BinaryOpsFactory` / `ReductionOpsFactory`
-> specialized `XxxStdOpsFactory`.

Execution mode is switched via `set_context_mode()`:
`pynative`, `kbk`, or `ge`.

#### 8.2.4 Frontend Test File Pattern

Frontend test files use pytest parameterization plus `arg_mark` decorators to
expand all registered operators. Each test function creates the matching
`OpsFactory` and calls the standard test method.

```python
# test_binary_ops.py (simplified example)
from tests.st.ops.share._internal.binary_ops import BinaryOpsFactory
from tests.st.ops.share._op_info.op_database import get_op_info, binary_op_db, binary_op_kbk_db

# forward accuracy (pynative)
@pytest.mark.parametrize("op_info", binary_op_db)
def test_binary_op_reference_forward(op_info):
    fact = BinaryOpsFactory(op_info=get_op_info(op_info))
    fact.set_context_mode(mode='pynative')
    fact.test_op_reference()

# backward accuracy (pynative)
@pytest.mark.parametrize("op_info", binary_op_db)
def test_binary_op_reference_backward(op_info):
    fact = BinaryOpsFactory(op_info=get_op_info(op_info))
    fact.set_context_mode(mode='pynative')
    fact.test_op_reference(grad_cmp=True)

# dynamic shape (kbk)
@pytest.mark.parametrize("op_info", binary_op_kbk_db)
def test_binary_op_dynamic_forward(op_info):
    fact = BinaryOpsFactory(op_info=get_op_info(op_info))
    fact.set_context_mode(mode='kbk')
    fact.test_op_dynamic(only_dynamic_shape=True)
    fact.test_op_dynamic(only_dynamic_rank=True)
```

#### 8.2.5 Quick Overview of Helper Tools

| Tool | Purpose |
| --- | --- |
| `make_tensor(shape, dtype)` | generate random tensors using `np.random` underneath |
| `OpSampleInput(input, args, kwargs)` | package a single test input case |
| `OpDynamicInput(...)` | package dynamic-shape test inputs, including `dynamic_level` |
| `OpErrorInput(...)` | package expected-error inputs for exception-path tests |
| `wrap_sample_inputs(func)` | wrap a simple input generator into a list of `OpSampleInput` |

<a id="delivery-consensus"></a>
## 9. Delivery / Handoff Consensus Points

### 9.1 Basic Principles for the Adaptation Plan

- **Align with PyTorch / PTA first**. If PTA does not support a behavior, you
  may choose not to develop it.
- If CANN does not support a behavior and PTA does not support it either, you
  may also choose not to develop it.
- Try to keep forward and backward using the same ACLNN / aten composition as
  PTA. That makes bitwise output alignment more achievable.

### 9.2 Impact Assessment

- **When reusing existing primitives or adapting existing interfaces**:
  you must ensure existing CPU / GPU paths do not regress and existing UT / ST
  still pass.
- **For a brand-new operator** (new primitive + new interface):
  you do not need to add CPU / GPU support. Ascend-only is acceptable.
- If existing GEOP / Lite flows may be affected, provide a mitigation plan,
  for example via a Pass or Expander.

<a id="delivery-artifacts-and-scope"></a>
### 9.3 Delivery Artifacts and Validation Scope (Summary)

Handoff validation usually needs to cover:
- API forms: `nn`, `functional`, `Tensor`
  (if fully identical, validating one entry may be enough)
- backend: Ascend
  (when reusing existing primitives / interfaces, CPU / GPU must not regress;
  for new operators, Ascend-only is acceptable)
- modes: dynamic graph, static graph, static graph KernelByKernel
- shape: dynamic and static
- dimensions: generality (`dtype` / `shape`), accuracy, and performance
  according to project thresholds

<a id="documentation-reference"></a>
## 10. Documentation and Material Development Notes

### 10.0 Basic Requirements for Documentation Export

- English `function_doc` YAML and Chinese RST must match in parameter names,
  default values, required / optional status, and examples.
- The `ops` package must explicitly export the API. Non-Ascend devices should
  have placeholder behavior with a clear error message.

<a id="documentation-general-principles"></a>
### 10.1 General Principles

- **Chinese and English must stay strictly aligned**:
  parameters, defaults, required / optional markers, constraints, examples, and
  so on must match.
- **Append API names in alphabetical order** to reduce conflicts and duplicates.
- **Filename / internal title / internal interface definition must match** or
  the generated page may fail.
- Examples must include full imports and be runnable.
  When helpful, print output values or shapes for clarity.

<a id="documentation-output-mapping"></a>
### 10.2 Common Scenarios and Output Locations (Summary)

- New `functional` API:
  English comments go in the implementation `.py`;
  Chinese docs go under `docs/api/api_python/ops/` as `func_*.rst`;
  update the API list as well.
- New `mint` API:
  update both the English / Chinese mint lists and the Chinese RST.
  If the API is just an import alias of an existing one, reuse is fine.
- New `Tensor` method:
  English docs go in `tensor.py`;
  Chinese docs go under `docs/api/api_python/mindspore/Tensor/`;
  update the list too.

## 11. Performance Self-Check Tool `apitimewrapper`

### 11.1 Purpose

Wrap MindSpore / PyTorch script APIs and collect end-to-end timing, optionally
including timing breakdown inside each API.

### 11.2 Usage Notes (Summary)

- install:
  `pip install apitimewrapper-0.0.3-py3-none-any.whl`
- whole-network instrumentation:
  start `start_hook_net(hook_inside=False)` at network entry before execution
- single API:
  you can enable both `start_hook_net` and `start_hook_torch_net`, then wrap the
  loop with `start_analysis()` / `end_analysis()`

<a id="bprop-advanced-notes"></a>
## 12. Backward Implementation Notes

> For the basic BPROP wiring rules, such as I/O count,
> `need_compute_grad_out`, and `SetUnusedInputs`, see
> [`7 BPROP Wiring Notes`](#bprop-reference).

### 12.1 Non-Differentiable Inputs

For inputs that are non-differentiable in Torch, such as `index` or `mode`,
MindSpore backward outputs must still match the input count:
- return `ib->OutZeros(x)` for each non-differentiable input
- if all inputs are non-differentiable, `ReturnZeros` may be used depending on
  the current framework behavior

### 12.2 When “Gradient Is Zero”

If a given input’s theoretical gradient is zero, prefer `ib->ZerosLikeExt()`
so execution stays on the ACLNN / backend path expected by the framework.

### 12.3 Inplace Operator Backward

- If backward needs the pre-update `self`, register `CloneInplaceInput(...)`
  so the framework preserves the old value.
- In KBK dynamic-shape scenarios, if using inplace logic in backward breaks
  execution ordering, use `ib->Depend(target, inplace_call)`.

<a id="resize-launch-optimization"></a>
## 13. Resize / Launch Optimization Notes

> For the base KBK structure and registration style, see
> [`6 KBK Kernel Notes`](#kbk-reference).

<a id="resize-launch-no-attr-mutation"></a>
### 13.1 Do Not Mutate Attributes in `InferShape`

Do not set or mutate operator attributes in `InferShape` / `InferType`.
This causes issues in Pynative mode.

### 13.2 Split Responsibilities Between Resize and Launch

- Put what can be fixed in Init into Init.
  Put shape-dependent work into Resize.
  Keep Launch focused on the actual launch / execution call.
- Do not allocate device memory at runtime, such as `cudaMalloc` / `cudaFree`.
  Let the framework manage memory through workspace.

### 13.3 Ignore Useless Outputs

For reserved or meaningless outputs, override `GetUseLessOutputIdx()` or the
repo-equivalent hook so they do not cause dump, overflow false positives, or
determinism side effects.

### 13.4 Compute-Depend Outputs

Follow framework rules:
allocate the maximum possible output first, then sync and update the real
output shape after execution, as in `NonZero`-style patterns.

<a id="accuracy-and-memory-validation"></a>
## 14. Bitwise Accuracy and Memory Alignment Self-Checks

<a id="bitwise-accuracy-validation"></a>
### 14.1 Bitwise Accuracy

When the goal is bitwise equivalence with PTA output:
- fix the random seed and save outputs as `.npy`
- compare the hashes of the two output files using `md5sum` or the equivalent

<a id="memory-alignment-validation"></a>
### 14.2 Memory Usage Alignment

The key point is to measure peak memory at the **same stage** in MindSpore and
PTA, so initialization or compilation is not mixed into the comparison.

- MS: `mindspore.runtime.max_memory_allocated()`
- PTA: `torch_npu.npu.max_memory_allocated()`

<a id="api-development"></a>
## 15. API Development Notes (functional / nn / Tensor)

### 15.1 `functional` API (Strict Constraint)

- Inside functional APIs, **always** use `_get_cache_prim` to fetch the
  Primitive instance. Avoid repeated `__init__` overhead.
- Complex interfaces may use one-to-many or composed mappings:
  choose different Primitives or compositions by parameter branch.

### 15.2 `nn` API

- `nn` APIs are `Cell` subclasses:
  initialize operators and attributes in `__init__`,
  execute logic in `construct`.
- `construct` acts like a compiler entry.
  Do not `raise` directly inside it.
  If you need compile-time validation / errors, use helper functions decorated
  with `@constexpr`.

### 15.3 `Tensor` Methods (Including GE Mapping Notes)

- Tensor methods must cover the required modes:
  PyNative / KBK and GE when the project requires it.
- In GE mode you often need:
  - mapping registration in `resource.cc`
  - implementation in `standard_method.py`
    (validation helpers there cannot take `Tensor` inputs directly, so use the
    proper wrappers)

<a id="api-integration-strategy"></a>
### 15.4 Primitive and API Integration Strategy

This must be decided during Pre-B, before YAML design starts.

<a id="api-analysis-five-factors"></a>
#### 15.4.1 Five Factors for Interface Analysis

By comparing MindSpore and PTA / torch documentation and implementation,
clarify:
1. whether the functionality is the same
2. whether the parameter definitions are the same

The input name dismatch doesn't matter here(e.g. 'x' vs 'input'), as long as the input semantics and types matches.

If both API and functionality match, reuse the existing Primitive.
Otherwise create a new Primitive such as `XXXExt`, following the
[`ops.extend` namespace](#ops-extend-namespace).

> PTA / torch may expose **same-name overloads** with different signatures.
> Analyze each one individually.

<a id="yaml-three-scenarios"></a>
#### 15.4.2 Three YAML Scenarios

| Scenario | YAML action | Example |
| --- | --- | --- |
| **existing YAML + reuse existing Primitive** | add a `dispatch` field to the current YAML | `eye`: existing Primitive, add `dispatch.Ascend: EyeAscend` |
| **existing YAML + new Primitive** | create a new YAML with `_ext` suffix | `zeros_like_ext`: existing `zeros_like` exists but parameters are incompatible |
| **no YAML exists** | create a new YAML, usually without `_ext` | brand-new operator |

**Example: reuse existing Primitive**

```yaml
# Add dispatch to the existing eye YAML
dispatch:
  enable: True
  Ascend: EyeAscend
```

**Example: new Primitive + `_ext`**

```yaml
zeros_like_ext:
    args:
        input:
            dtype: tensor
        dtype:
            dtype: TypeId
            arg_handler: dtype_to_type_id
            default: None
    returns:
        y:
            dtype: tensor
    function:
        disable: True
    dispatch:
        enable: True
        Ascend: ZerosLikeExtAscend
```

<a id="ops-extend-namespace"></a>
#### 15.4.3 `ops.extend` Namespace

> If ACLNN functionality is inconsistent with an existing `ops.xx` API and the
> existing API cannot be changed compatibly, you need a new extend interface.

MindSpore interface namespaces:
- `ops.xxx` / `ops.xxx_ext()` / `ops.auto_generate.xxx_ext()` /
  `ops.extend.xxx_ext()`
- `nn.xxx` / `nn.xxxExt()` / `nn.extend.xxx()`

<a id="existing-primitive-signature-change"></a>
#### 15.4.4 Modifying Existing Primitive Signatures and Adapting Overloads

In real development, existing Primitives often need **parameter extension**,
for example a new PTA version introduces parameters or ACLNN-specific params
must be supported.
You may also need to adapt **same-name overloads** in PTA / torch.

**Practical strategy for extending parameters**:
1. search similar operators in the MS repo and use their approach as a
   reference
2. analyze compatibility in detail:
   can the new parameter have a default value, does it affect existing callers,
   does it affect other backends
3. choose the path:
   - compatible change -> modify existing YAML + Infer + interface directly
   - incompatible change -> add a new `_ext` Primitive, or use `ops.extend`
4. ensure no existing functionality regresses:
   existing UT / ST must still pass
5. follow the review rules
   (see [`15.4.5 Review Rules`](#api-review-rules))

**Overload adaptation**:
for Tensor / functional same-name, multiple-signature scenarios
(for example Tensor-Scalar vs Tensor-Tensor, with or without keyword-only
parameters, old / new interface compatibility, and so on),
use the multi-entry `api_def` YAML mechanism.
For the full mechanism, scenario classification, deprecated YAML, and aliases,
see [`25 API Overload Adaptation`](#api-overload-adaptation).

<a id="api-review-rules"></a>
#### 15.4.5 Review Rules

| Change type | Review requirement |
| --- | --- |
| no new interface, behavior fully identical to before | no review needed |
| no new interface, but behavior is extended | review required |
| new interface | **high-priority review** |
| incompatible behavior change to an existing interface | **not allowed in principle**, only special reviewed cases |
| new operator | review required |
| incompatible behavior change to an existing operator | review required |

Any scenario requiring review **must be confirmed by the user**.

## 16. When ACLNN / PTA Docs Are Incomplete: Use Probe Scripts to Fill the Fact Gap

In reality, ACLNN / PTA docs may lag or omit detail, especially when support
changes across CANN / PTA versions. Do not guess.

### 16.1 Record the Version Matrix First

Ask the user to confirm and record these items in the Feature document,
acceptance report, or test output:
- torch version and `torch_npu` version
- CANN version, or a traceable installation path / image version
- chip model and driver info, if printable

### 16.2 Generate and Run a PTA Support Probe Script

Recommended approach:
- this skill includes a template script:
  `scripts/probe_pta_sparse_flash_attention.py`
  It uses `sparse_flash_attention` only as a **template**.
  When adapting it for another operator, copy it and modify the `run_case`
  input construction and API call.
- purpose:
  enumerate combinations of dtype / layout / key parameters, record
  success / failure and errors, and output a JSON summary.

Example usage:

```bash
python scripts/probe_pta_sparse_flash_attention.py --device npu:0 --out pta_probe.json
# Quick mode, only core combinations:
python scripts/probe_pta_sparse_flash_attention.py --device npu:0 --quick --out pta_probe_quick.json
```

Evidence you need back from the user:
- `pta_probe.json`, or at least its summary and the key error messages
- version information from the same output
  (`torch` / `torch_npu` / env / `npu-smi`)

### 16.3 Use Probe Results to Finalize API Alignment and Constraints

Use probe results to decide:
- whether `sparse_size` is fixed, for example some CANN versions requiring 2048
- whether combinations like `attention_mode`, `return_softmax_lse`, and
  `layout` are actually supported
- whether dtype support is truly only fp16 / bf16, and whether hidden
  constraints exist

Then sync the conclusions into YAML, Infer, docs, and tests.

## 17. `vmap` Support (When Needed)

> Source: `算子流程/.../4. 算子关键特性.md`.
> This skill focuses mainly on ACLNN operator forward / backward / infer /
> test / docs flow. `vmap` is listed here only as an **optional extension**.
> If the target operator does not need `vmap`, this section can be skipped.

### 17.1 When `vmap` Is Needed

- the operator must support `vmap` / `vectorize_cell`
- the project explicitly requires `vmap` coverage

### 17.2 Key Points (Summary)

- register the `vmap` rule in the framework-defined location, following the
  repo’s current registration pattern
- add dedicated `vmap` UT to validate batched shape and value correctness
- note that the `vmap` path may not use ACLNN directly and may instead fall
  back to composition or loop expansion, so confirm whether performance is
  acceptable

<a id="code-skeletons"></a>
## 18. Code Skeleton Templates (Can Be Copied and Adapted)

> These skeletons come from the real “auto-generate first, then customize”
> workflow and are only **starting references**.
> Before using them, always compare against similar operators in the actual repo
> for macro names, namespaces, and parameter lists.

<a id="yaml-skeleton"></a>
### 18.1 Minimal YAML Skeleton (`op_def` + `api_def` + `doc`)

```yaml
# ---- op_def ----
op_name: "OpName"
args:
  input:
    dtype: tensor
returns:
  output:
    dtype: tensor
dispatch:
  enable: True
  Ascend: "OpNameAscendCustomize"
# ---- api_def ----
api:
  py_method: "op_name"
  module: "mindspore.ops"
# ---- function_doc ----
function_doc:
  desc: "Brief English description of the operator."
```

<a id="pyboost-skeleton"></a>
### 18.3 PyBoost Customize Skeleton (C++)

```cpp
// op_name_ascend_customize.cc
#include "plugin/device/ascend/kernel/pyboost/customize/op_name_ascend_customize.h"
// Adjust includes to match the actual repo

namespace mindspore::kernel::pyboost {

// forward
tensor::TensorPtr OpNameAscendCustomize::Call(
    const tensor::TensorPtr &input_x,
    const std::optional<float> &scale) {
  // 1. Allocate output tensor
  auto output = std::make_shared<tensor::Tensor>(input_x->data_type(), out_shape);

  // 2. Convert arguments, such as tuple->vector or None handling
  // auto scale_val = scale.value_or(1.0f);

  // 3. Two-phase ACLNN call, following the project wrapper
  // LAUNCH_ACLNN(aclnnOpName, stream, input_x, scale_val, output);

  return output;
}

}  // namespace mindspore::kernel::pyboost
```

<a id="kbk-skeleton"></a>
### 18.4 KBK Kernel Skeleton (C++)

```cpp
// op_name_aclnn_kernel.cc
#include "plugin/device/ascend/kernel/opapi/aclnn_kernel/op_name_aclnn_kernel.h"
// Adjust includes to match the actual repo

namespace mindspore::kernel {

void OpNameAclnnKernel::GetWorkSpaceInfo(
    const std::vector<KernelTensor *> &inputs,
    const std::vector<KernelTensor *> &outputs) {
  // Parse arguments, such as scalars or tuples
  // auto scale = inputs[1]->GetValueWithCheck<float>();

  // Get workspace
  // GetWorkspaceForResize(aclnnOpNameGetWorkspaceSize, ...);
}

bool OpNameAclnnKernel::Launch(
    const std::vector<KernelTensor *> &inputs,
    const std::vector<KernelTensor *> &workspace,
    const std::vector<KernelTensor *> &outputs,
    void *stream_ptr) {
  // RunOp(aclnnOpName, stream, ...);
  return true;
}

// registration
MS_ACLNN_KERNEL_FACTORY_REG(OpName, OpNameAclnnKernel);

}  // namespace mindspore::kernel
```

<a id="bprop-builder-skeleton"></a>
### 18.5 BPROP Builder Skeleton (C++)

```cpp
// grad_xxx_ops.cc (add to the correct bprop registration file)

REG_BPROP_BUILDER("OpName").SetBody([](const BpropBuilder *ib) -> NodePtrList {
  // forward inputs
  auto input_x = ib->GetInput(kIndex0);
  auto scale = ib->GetInput(kIndex1);
  // forward outputs and upstream gradients
  auto out = ib->GetInput(kIndex2);   // forward input count + 0
  auto dout = ib->GetInput(kIndex3);  // forward input count + 1

  // backward subgraph
  NodePtr dx;
  if (ib->need_compute_grad_out(kIndex0)) {
    dx = ib->Emit("OpNameGrad", {input_x, out, dout, scale});
  } else {
    dx = ib->OutZeros(input_x);
  }

  // non-tensor params such as scale return zero-gradient placeholders
  auto d_scale = ib->OutZeros(scale);

  return {dx, d_scale};
});
```

<a id="pta-source-review"></a>
## 19. PTA Source Review Method (Mandatory)

> PTA docs may lag or omit detail.
> Before development, you **must** review the actual code in the `op-plugin`
> repo as well as the docs.
> If they disagree, do not guess. Ask the user to confirm with the ACLNN / PTA
> operator owners
> (see [`19.5 What to Do When Code and Docs Disagree`](#pta-source-doc-mismatch)).

### 19.1 Three Critical File Types to Review

| File type | Path pattern | What to extract |
| --- | --- | --- |
| **function-signature YAML** | `op_plugin/config/op_plugin_functions.yaml` | exact parameter names, types, defaults, return structure, whether it uses `op_api`, `acl_op`, or `gen_opapi` |
| **backward registration YAML** | `op_plugin/config/derivatives.yaml` | which inputs are differentiable, grad function name, parameter pass-through order, `output_differentiability` |
| **C++ implementation** | `op_plugin/ops/opapi/XxxKernelNpuOpApi.cpp` including grad variants | actual `aclnnXxx` calls, parameter preprocessing, output tensor construction, hidden hard-coded defaults |

### 19.2 Differences That Often Matter

PTA code often reveals critical details not documented elsewhere.

**1. Forward and backward parameter naming / order may differ**
- example: forward uses `actual_seq_lengths_query`, backward uses
  `actual_seq_qlen`
- example: forward has `layout_query` and `layout_kv`, backward collapses to
  one `layout`
- **impact**:
  the MS bprop builder must follow the real backward signature, not the forward
  one

**2. Hidden extra parameters in backward ACLNN calls**
- example: backward hard-codes `deterministic_const = true`
- example: backward omits `block_table` even though forward takes it
- **impact**:
  MS backward KBK / PyBoost implementations must match this hidden behavior

**3. `None` handling for optional parameters**
- example: when `query_rope` is `None`, PTA passes `at::Tensor()`
- example: gradient output for `query_rope == None` may be
  `at::empty({0}, ...)` rather than a zero tensor
- **impact**:
  MS must match these `None` semantics or ACLNN may error or outputs may diverge

**4. Output tensor count and construction**
- example: forward returns `(output, softmax_max, softmax_sum)` = 3 outputs,
  backward returns 5
- `softmax_max` / `softmax_sum` shape logic may have explicit 3D / 4D branches
  in C++
- **impact**:
  MS Infer must match the output shape logic, and bprop must pass intermediates
  correctly

**5. Gradient pass-through in `derivatives.yaml`**
- example: `result0` / `result1` / `result2` refer to forward outputs 0 / 1 / 2
- confirms which inputs are `non_differentiable` and which participate in grad
- **impact**:
  MS `GetInput` indexing and `OutZeros` placement must match

### 19.3 Review Workflow

1. **Search `op_plugin_functions.yaml` for the operator name**:
   extract the exact forward / backward signatures and record parameter
   differences.
2. **Search `derivatives.yaml` for the operator name**:
   confirm differentiable inputs and backward parameter pass-through.
3. **Open the matching C++ implementation** in `ops/opapi/` and inspect:
   - output tensor shape construction
   - `None` handling for optional parameters
   - the actual parameter list and order passed to
     `EXEC_NPU_NO_FORMAT_CHECK_CMD` / `aclnnXxx`
   - any hard-coded parameters such as `deterministic`
4. **Record the differences** as key evidence in your validation loop.
5. **If code and docs disagree, stop and ask for confirmation**
   (see 19.5).

### 19.4 Typical Difference-Record Template

```text
Operator: npu_sparse_flash_attention

Forward vs backward parameter differences:
- actual_seq_lengths_query (fwd) -> actual_seq_qlen (bwd)
- layout_query + layout_kv (fwd) -> single layout (bwd)
- block_table (fwd exists) -> not passed in bwd
- return_softmax_lse (fwd exists) -> not passed in bwd

Hidden backward behavior:
- deterministic_const = true (hard-coded)
- when query_rope is None, d_query_rope = at::empty({0}, ...)

Output structure:
- forward: (output, softmax_max, softmax_sum) = 3
- backward: (d_query, d_key, d_value, d_query_rope, d_key_rope) = 5

Differentiable inputs from derivatives.yaml:
- query, key, value, query_rope, key_rope (5 total)
- sparse_indices, block_table, etc. are non-differentiable
```

<a id="pta-source-doc-mismatch"></a>
### 19.5 What to Do When Code and Docs Disagree

> **Core principle**:
> when docs and source code agree, use both and move quickly.
> When they disagree, do not guess. Ask the user to confirm.

**If they agree**:
use docs for semantics / constraints and code for implementation detail and
hidden behavior, then proceed directly.

**If they disagree**:
1. **prepare a difference list**:
   “docs say X, code actually does Y”, with file paths and line numbers
2. **hand it to the user immediately**:
   do not decide which side wins by yourself;
   ask the user to confirm with ACLNN / PTA operator owners
3. **continue only after confirmation**:
   record the conclusion in the design doc / Feature doc, then continue the MS
   adaptation

Difference-confirmation template:

```text
⚠️ PTA code and docs disagree, confirmation is required

Difference list:
| # | Item | Doc says | Code actually does | File / line |
| - | ---- | -------- | ------------------ | ----------- |
| 1 | ...  | ...      | ...                | ...         |

Recommended owners to confirm with:
- ACLNN operator owner
- PTA operator owner

Please confirm which side is correct, and I will continue from there.
```

<a id="infervalue-constant-folding"></a>
## 20. InferValue Constant Folding (Optional Optimization)

> When all operator inputs are known at compile time, InferValue can compute the
> result directly and skip runtime execution, improving whole-graph performance.

### 20.1 Two Implementation Forms

- **Python callback** such as `concat`:
  register an InferValue callback in
  `mindspore/python/mindspore/ops/operations/manually_defined/ops_def.py`
- **C++ implementation** such as `add`:
  implement it under `mindspore/ops/infer/ops_frontend_func_impl/`
- **C++ is preferred** for performance

### 20.2 How to Validate It

- add InferValue UT for the all-constant-input case
- run the test script and inspect the IR to confirm constant folding happened,
  meaning the output node became a `ValueNode`

### 20.3 When It Applies

- when operator inputs are known at compile time, such as helper operators used
  in shape computation or type conversion
- most ACLNN compute operators get their inputs only at runtime, so
  **InferValue is usually unnecessary**

<a id="dynamic-shape-strategy"></a>
## 21. Dynamic Shape Classification and Handling Strategy

> For quick Infer fallback guidance, also see
> [`4.2 Dynamic Shape / Dynamic Rank`](#general-infer-dynamic-shape-rank).

### 21.1 Three Dynamic-Shape Types

| Type | Meaning | Typical operators | Infer strategy |
| --- | --- | --- | --- |
| **InputDynamic** | input shape is unknown at compile time | most operators | set affected output dims to -1 (`kShapeDimAny`) |
| **OutputDynamic (Input Value Depend)** | output shape depends on input values | `Std`, `Ones` | use `GetScalarValue` / `GetArrayValue`; if unknown, fall back to dynamic dims / rank |
| **OutputDynamic (Compute Depend)** | output shape requires runtime computation | `NonZero`, `UniqueConsecutive` | allocate max possible size, then call `SyncOutputShape` after execution |

### 21.2 InputDynamic Handling

- if input shape has `-1` dimensions, mirror those as `-1` in the output
- if input rank is dynamic (`-2`), fall back to dynamic rank
- if a key scalar parameter is unknown (`HasUnknownValue`), any dependent output
  dim falls back to `-1`

### 21.3 Input Value Depend Handling

The output shape depends on **input values**, either scalar or array, which may
be known or unknown at compile time.

- **scalar value dependence**:
  use `GetScalarValue<T>()`;
  if `!has_value()`, fall back to dynamic rank or dynamic dims
- **array value dependence**:
  use `GetArrayValue<T>()`;
  if the whole array is unknown, fall back to dynamic rank;
  if only some elements are unknown, set only those output dims to
  `kShapeDimAny`
- typical case A, `Std`:
  output shape depends on `dim` (array) and `keepdim` (scalar);
  if any is unknown, fall back to `kShapeRankAny`
- typical case B, `Ones`:
  output shape comes directly from the input `shape` array;
  unknown elements become `kShapeDimAny`

### 21.4 Compute Depend Handling

- allocate the maximum possible output size based on a compile-time upper bound
- after execution, use `Sync` + `SyncOutputShape` to update the real output
  shape
- override `GetUseLessOutputIdx()` to avoid dump / overflow false positives

<a id="aclnn-callchain-analysis"></a>
## 22. ACLNN Call-Chain Analysis and Sub-Operator Inventory (Composite Cases)

> A single `torch_npu.npu_xxx()` interface in PTA does not necessarily call one
> big `aclnnXxx` operator underneath.
> A common pattern is **multiple smaller ACLNN operators chained together**,
> in both forward and backward.
> In that case, MindSpore must first inventory all sub-operators, fill any
> missing ones, and then compose them in the same way.

### 22.1 When Call-Chain Analysis Is Needed

- the PTA C++ implementation contains **multiple**
  `EXEC_NPU_CMD` / `EXEC_NPU_NO_FORMAT_CHECK_CMD` calls
- the PTA C++ implementation calls other `at_npu::native::` functions
  indirectly
- ACLNN docs / headers do not expose a single one-to-one “big operator” for
  the PTA interface
- backward is not a single `aclnnXxxGrad`, but is composed from multiple
  smaller ops

<a id="aclnn-callchain-extraction"></a>
### 22.2 How to Extract the Call Chain

1. **Locate the PTA forward C++ implementation**
   (`ops/opapi/XxxKernelNpuOpApi.cpp`) and mark line by line:
   - each `EXEC_NPU_CMD(aclnnYyy, ...)` or `OpApiFunc(aclnnYyy, ...)`
   - intermediate tensor construction such as `at::empty(...)` or
     `npu_preparation::apply_tensor(...)`
   - parameter preprocessing such as type conversion, default filling, and
     `None` handling
2. **Do the same for backward C++**
   (`XxxGradKernelNpuOpApi.cpp` or the function pointed to by
   `derivatives.yaml`)
3. **Produce a call-chain diagram**, text form is enough:

```text
Forward call chain for torch_npu.npu_foo(q, k, v, scale):
  ① aclnnBarPrepare(q, k) -> intermediate_qk
  ② aclnnAttentionScore(intermediate_qk, v, scale) -> output
  ③ aclnnSoftmaxLse(output) -> softmax_lse

Backward call chain for torch_npu.npu_foo:
  ① aclnnAttentionScoreGrad(dout, q, k, v, softmax_lse) -> (dq, dk, dv)
  (backward is one large op, so no decomposition is needed)
```

<a id="ms-coverage-inventory"></a>
### 22.3 How to Inventory Coverage on the MindSpore Side

For each sub-operator in the chain, search the MS repo and confirm:

| Search target | Search keyword | What it tells you |
| --- | --- | --- |
| YAML definition | `aclnnYyy` or matching `op_name` | whether `op_def` exists |
| C++ small-op API | function names in `functions/auto_generate/functions.h` | whether PyBoost composition is available ([`23.1 PyBoost Composition`](#composite-pyboost-pattern)) |
| Meta DSL Primitive | `Prim(OpName)` or Primitive definitions in generated headers | whether KBK composition is available ([`23.2 KBK Composition`](#composite-kbk-pattern)) |
| PyBoost implementation | `LAUNCH_ACLNN(aclnnYyy` or matching customize files | whether the Pynative path exists |
| KBK kernel | `MS_ACLNN_KERNEL_FACTORY_REG` plus class name | whether the Graph path exists |
| Infer | matching `FuncImpl` class | whether infer exists |
| `aclnn_config.yaml` | operator name mapping, Path 1 only | whether dispatch mapping exists |

### 22.4 Inventory Result Template

```text
Target interface: torch_npu.npu_foo -> mindspore.ops.foo

ACLNN call-chain inventory:
| # | aclnnXxx | Purpose | MS status | Notes |
| - | -------- | ------- | --------- | ----- |
| 1 | aclnnBarPrepare | forward preprocessing | ✅ integrated | YAML / Infer / PyBoost / KBK all exist |
| 2 | aclnnAttentionScore | forward main compute | ⚠️ only YAML + Infer | missing PyBoost customize and KBK |
| 3 | aclnnSoftmaxLse | forward auxiliary output | ❌ not integrated | requires the full workflow |
| 4 | aclnnAttentionScoreGrad | backward | ✅ integrated | no extra work needed |

Execution plan:
1. fill #3 first: go through YAML -> Infer -> PyBoost -> KBK -> UT
2. then fill #2 PyBoost / KBK
3. finally compose #1 + #2 + #3 inside foo Customize
```

<a id="callchain-rollout-order"></a>
### 22.5 Rollout Order Principles

- **Leaves first, composition later**:
  implement all independent sub-operators before the composite operator
- **forward first, backward later**:
  backward often reuses forward sub-operators
- **each missing sub-operator must complete its own workflow**:
  implement it step by step using the `SKILL.md` workflow
  (usually export / docs are not needed independently, but YAML + Infer +
  PyBoost + KBK + UT usually are)
- **compose last**:
  once all sub-operators are available, implement the composition layer

<a id="composite-implementation"></a>
## 23. Composite Implementation Patterns (C++ Small-Op Composition + Meta DSL)

> When the target operator is implemented as a composition of multiple smaller
> operators, MindSpore provides two mechanisms:
> - **PyBoost (Pynative)**:
>   compose prebuilt C++ small-op APIs, including implicit type conversion,
>   op call, and autodiff
> - **KBK (Graph)**:
>   use Meta DSL graph construction, letting the framework handle type infer,
>   autodiff, and multi-platform adaptation

<a id="composite-pyboost-pattern"></a>
### 23.1 PyBoost Composition (C++ Small-Op APIs)

**Core idea**:
in PyBoost customize code, directly call existing operator C++ APIs such as
`add()`, `mul()`, or `transpose()` instead of manually calling `LAUNCH_ACLNN`.
Each API already wraps implicit type conversion, the PyBoost call, and autodiff.

**Key header**:

```cpp
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"
```

**YAML rule**:
set `bprop_expander: False`, meaning the composite op itself does not use a
backward expander. Each composed small operator provides its own autodiff.

```yaml
bprop_expander: False
dispatch:
  enable: True
  Ascend: FooAscend
```

**`RequireGradGuard` usage**:
if the composite operator already has a dedicated bprop registration and you
did not set `bprop_expander: False`, then use `RequireGradGuard(false)` so the
small ops do not each perform autodiff and duplicate backward work.

**Simplified example**, adapted from `cosine_embedding_loss.cc`:

```cpp
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"

tensor::TensorPtr FooAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                     const TensorPtr &input1, const TensorPtr &input2,
                                     const FP32ImmPtr &margin, const Int64ImmPtr &reduction) {
  // Compose logic directly from C++ small-op APIs
  auto prod = mul(input1, input2);
  auto result = sum_ext(prod, dim_tuple, std::make_shared<BoolImm>(False), std::nullopt);
  auto output = div(result, sqrt(denom));

  // reduction branch
  auto reduction_val = static_cast<Reduction>(GetValue<int64_t>(reduction));
  if (reduction_val == Reduction::MEAN) {
    output = mean_ext(output, std::nullopt, std::make_shared<BoolImm>(False), std::nullopt);
  }

  op->set_outputs({output});
  return op->output(0);
}
```

**Available APIs**:
inspect the YAML definitions under `mindspore/ops/op_def/yaml/`.
Existing operators that already have YAML + dispatch will generate matching
C++ APIs.

<a id="composite-kbk-pattern"></a>
### 23.2 KBK Composition (Meta DSL)

**Core idea**:
use `REGISTER_FUNCTION_OP` plus `BeginFunction / EndFunction` to build the op
as a graph in C++, letting the framework handle type infer, autodiff, and
multi-platform adaptation. No handwritten `GetWorkSpaceInfo / Launch / RunOp`
is needed.

**Code location**:
`mindspore/ccsrc/frontend/operator/meta_dsl/func_op/`

**Core interfaces**:

| API | Meaning | Example |
| --- | --- | --- |
| `REGISTER_FUNCTION_OP(OpName)` | register the composed operator, optionally with a validation function | `REGISTER_FUNCTION_OP(Foo, CheckFunc)` |
| `BeginFunction(Op, args...) { }` | begin the composed implementation | `BeginFunction(Foo, x, y, z) { ... }` |
| `EndFunction(Op)` | end the implementation | `EndFunction(Foo)` |
| `Prim(OpName)` | a Primitive handle | `Prim(Add)`, `Prim(SumExt)` |
| `Call(prim, args...)` | call an operator | `Call(Prim(Mul), x, y)` |
| `Value(v)` | constant value node | `Value(0)`, `Value(1.0)`, `Value(kNone)` |
| `Return(out)` | return output | `Return(output)` |
| `If(cond, true_br, false_br)` | control-flow branch | branches are lambdas |
| `Tuple(...)` / `List(...)` | create tuples / lists | `Tuple(x, y, z)` |
| `Rank(x)` / `Shape(x)` | query rank / shape | — |
| `PRIMITIVE_BPROP_REG(Op, Grad)` | define backward bprop when needed | `PRIMITIVE_BPROP_REG(Foo, FooGrad)` |

**Simplified example**, adapted from `cosine_embedding_loss.cc`:

```cpp
REGISTER_FUNCTION_OP(CosineEmbeddingLoss, CheckCosineEmbeddingLossInputs)

BeginFunction(CosineEmbeddingLoss, input1_tensor, input2_tensor, target_tensor, margin, reduction) {
  constexpr float EPSILON = 1e-12;
  auto dim_tuple_ptr = Tuple(Rank(target_tensor));
  auto prod_sum = Call(Prim(SumExt), Call(Prim(Mul), input1_tensor, input2_tensor),
                       dim_tuple_ptr, Value(false), Value(kNone));
  // mag_square + denom + cos calculation...
  auto denom = Call(Prim(Sqrt), Call(Prim(Mul), mag_square1, mag_square2));
  auto cos = Call(Prim(Div), prod_sum, denom);

  auto zeros = ZerosLike(cos);
  auto pos = Call(Prim(SubExt), OnesLike(cos), cos, Value(1));
  auto neg = Call(Prim(ClampMin), Call(Prim(SubScalar), cos, margin, Value(1)), Value(0));
  auto output_pos = Call(Prim(Select), Equal(target_tensor, Value(1)), pos, zeros);
  auto output_neg = Call(Prim(Select), Equal(target_tensor, Value(-1)), neg, zeros);
  auto output = Call(Prim(AddExt), output_pos, output_neg, Value(1));

  // nested If: first NONE, then MEAN / SUM
  auto condition_none = Equal(reduction, Value(static_cast<int64_t>(Reduction::NONE)));
  auto none_true_branch = [&]() { Return(output); };
  auto none_false_branch = [&]() {
    auto condition_mean = Equal(reduction, Value(static_cast<int64_t>(Reduction::MEAN)));
    auto mean_true_branch = [&]() { Return(Call(Prim(MeanExt), output, Value(kNone), Value(false), Value(kNone))); };
    auto mean_false_branch = [&]() { Return(Call(Prim(SumExt), output, Value(kNone), Value(false), Value(kNone))); };
    Return(If(condition_mean, mean_true_branch, mean_false_branch));
  };
  Return(If(condition_none, none_true_branch, none_false_branch));
}
EndFunction(CosineEmbeddingLoss)
```

### 23.3 YAML Configuration Notes

| Field | Value | Meaning |
| --- | --- | --- |
| `bprop_expander` | `False` | composite op does not use a backward expander; small ops handle autodiff |
| `dispatch.enable` | `True` | enable dispatch |
| `dispatch.Ascend` | `FooAscend` | point to the PyBoost customize implementation, if one exists |

> **Note**:
> once `bprop_expander: False` is set, you usually no longer need a handwritten
> `REG_BPROP_BUILDER`.
> Backward is assembled automatically from the existing bprop of the small ops.
> If you need a custom backward anyway, Meta DSL also supports
> `PRIMITIVE_BPROP_REG`.

### 23.4 Infer Notes

- Infer for a composite operator only needs to infer the **final output**
  shape / type, not every intermediate tensor.
- If the final output depends on the shape of intermediates, compute the final
  output infer directly from the known inputs.

### 23.5 Layered Validation Strategy for Composite Operators

| Stage | Validate what | How |
| --- | --- | --- |
| **sub-operator level** | each sub-operator works independently | sub-operator UT / ST |
| **composition level, intermediates** | intermediate tensors align with PTA | temporarily dump intermediates and compare step by step |
| **composition level, final output** | final output aligns with PTA | normal ST alignment flow |
| **backward** | gradient correctness | backward ST + numerical gradient check where applicable |

---

<a id="feature-document-reference"></a>
## 24. Feature Document (Mandatory for Review and Delivery)

> Source: real delivered Feature documents such as
> `==符号重载Feature.md`, `CosineEmbeddingLoss Feature.md`,
> `NsaCompressAttention_Feature_文档.md`, and `参考feature.md`.

<a id="feature-document-overview"></a>
### 24.1 What a Feature Document Is

The Feature document is the **required document** for operator review and
delivery handoff. It consolidates design, interface definition, implementation
detail, test plan, and acceptance results into one standard document.
Review committees use it to decide whether the operator can be merged.

### 24.2 Standard Feature Document Sections

| No. | Section | When to fill it | Meaning |
| ---- | ---- | -------- | ---- |
| 1 | Background | Pre-B | origin of the operator, motivation, why MindSpore needs it |
| 2 | Benchmark and APIs | Pre-B | benchmark API (PTA / Torch), MindSpore API (`functional` / `nn` / `tensor`) |
| 3 | Task list | initialize in Pre-B, then update during development | **standard 13-category table** (see [`24.3 Standard 13 Task Categories`](#feature-document-task-categories)) |
| 4 | Functional and API specification | Pre-B | formula, interface signatures, parameter descriptions |
| 5 | YAML definition | after Step 1 | `op_def` YAML content |
| 6 | Constraints and types | Pre-B | device, dtype, shape constraints, empty-tensor strategy |
| 7 | Execution modes and adaptation | after Step 4 / 5 | PyBoost / KBK implementation notes |
| 8 | Differences from PTA and alignment status | initialize in Pre-B, then complete during development | functional / accuracy / API semantic differences |
| 9 | Dynamic Shape / Rank support | after Step 3 | dynamic dim / dynamic rank infer strategy |
| 10 | Validation and errors | after Step 3 / 4 | infer-time and runtime validation |
| 11 | Backward (BPROP) | after Step 6 | bprop registration, backward interface, gradient handling |
| 12 | Test plan | after Step 8 | UT / ST / TEST_OP coverage |
| 13 | Code and file change summary | after implementation completes | full paths of all added / modified files |
| 14 | Acceptance report | before handoff | four self-check tables: documentation, function, performance, secure coding (see [`24.4 Four Acceptance Tables`](#feature-document-acceptance-tables)) |

<a id="feature-document-task-categories"></a>
### 24.3 Standard 13 Task Categories

The “task list” in a Feature document is a standardized table. Each operator
**must** fill it item by item.

| No. | Task item | Sub-items |
| ---- | ------ | ---- |
| 1 | basic API functionality | Primitive / functional / nn / tensor |
| 2 | backend and dtype support | Ascend / GPU / CPU |
| 3 | supports `vmap` | — |
| 4 | supports dynamic shape | dynamic shape / dynamic rank |
| 5 | supports backward | bprop function / complex support |
| 6 | documentation completed | API mapping / Chinese and English API docs |
| 7 | functionality | empty tensor / `inf-nan` / 0D-8D / other function points |
| 8 | gate test completion | UT / ST / TEST_OP |
| 9 | security and exceptions | exception cases and error-message standard |

Each item must be marked as one of:
`new`, `modified`, `unchanged`, or `not involved`, with a brief note.

<a id="feature-document-acceptance-tables"></a>
### 24.4 Four Acceptance Tables

#### Documentation Validation Table (17 items)

Covers:
API list, UT / ST cases, Chinese and English docs, API description, formula,
parameter descriptions, input descriptions, output descriptions, output-size
vs input relation, `Raises`, platform declaration, format checking, sample
availability, printed sample results, sample executability, API sandbox.

#### Functional Validation Table (26 items)

Covers:
default parameters, empty tensor, `inf/nan`, dtype alignment, valid value
range, dimension coverage 0D-8D, full dtype coverage, implicit type
conversion, broadcasting, input constraints, forward accuracy, backward
support, backward single-op implementation, exception message quality,
error whitelist, dynamic shape / rank, fallback-off validation, regression in
the test repo, bf16, on-demand bprop generation, output shape compute-depend,
non-contiguous inputs, PTA 0-diff, impact on existing interfaces, AMP,
multi-tensor dtype mismatch.

#### Performance Validation Table (4 items)

Covers:
broadcast-case performance, backward memory optimization via `SetUnusedInputs`,
performance for at least 3 shapes, and memory usage no worse than PTA.

#### Secure Coding Review Table (12 items)

Covers:
null pointer checks, use-before-check, bounds issues, divide by zero, memory
leaks, cleanup on exception path, `nothrow`, secure libc usage, overflow in
type conversion, dead code, sensitive information, weak randomness.

### 24.5 Feature Document Generation Workflow

```text
During Pre-B:
  1. Copy a new document from templates/feature-document.md
  2. Fill in [1. Background](templates/feature-document.md#feature-background),
     [2. Benchmark and APIs](templates/feature-document.md#feature-benchmark-api),
     [3. Task List](templates/feature-document.md#feature-task-list),
     [4. Functional and API Specification](templates/feature-document.md#feature-functional-spec),
     [6. Constraints and Types](templates/feature-document.md#feature-constraints),
     and [8. Differences from PTA and Alignment](templates/feature-document.md#feature-pta-alignment)
  3. Submit it to the review committee for solution review

During development:
  4. Backfill the corresponding sections after each workflow step
     - Step 1 -> [5. YAML Definition](templates/feature-document.md#feature-yaml-definition)
     - Step 3 -> [9. Dynamic Shape/Rank Support](templates/feature-document.md#feature-dynamic-shape),
                 [10. Validation and Errors](templates/feature-document.md#feature-validation-and-errors)
     - Step 4/5 -> [7. Execution Modes and Adaptation](templates/feature-document.md#feature-execution-modes)
     - Step 6 -> [11. Backward (BPROP)](templates/feature-document.md#feature-bprop)
     - Step 8 -> [12. Test Plan](templates/feature-document.md#feature-test-plan)

Before handoff:
  5. Complete [13. Code and File Change Summary](templates/feature-document.md#feature-code-change-summary)
  6. Fill the four self-check tables in [14. Acceptance Report](templates/feature-document.md#feature-acceptance-report)
  7. Update the final status of each item in [3. Task List](templates/feature-document.md#feature-task-list)
  8. Submit the complete Feature document together with the code PR
```

### 24.6 Feature Document Differences by Operator Type

| Scenario | Difference |
| ---- | ---- |
| **single ACLNN operator** | standard flow; [7. Execution Modes and Adaptation](templates/feature-document.md#feature-execution-modes) describes one ACLNN call in PyBoost and one in KBK |
| **composite operator** | [4. Functional and API Specification](templates/feature-document.md#feature-functional-spec) must describe the ACLNN call chain; [7. Execution Modes and Adaptation](templates/feature-document.md#feature-execution-modes) must describe multi-ACLNN composition; [12. Test Plan](templates/feature-document.md#feature-test-plan) must validate in layers |
| **symbolic overload such as `==`** | [4. Functional and API Specification](templates/feature-document.md#feature-functional-spec) must describe `MultitypeFuncGraph` adaptation; in [3. Task List](templates/feature-document.md#feature-task-list), the `functional` / `tensor` columns are usually `modified` |
| **pure Python composition, no Primitive** | in [3. Task List](templates/feature-document.md#feature-task-list), the Primitive column is `not involved`; [7. Execution Modes and Adaptation](templates/feature-document.md#feature-execution-modes) describes only the functional-layer implementation |

### 24.7 Template Location

- template file: `templates/feature-document.md`
- reference examples: existing Feature documents provided by the user
  (it is recommended to find a similar operator’s Feature doc before
  development starts)

<a id="api-overload-adaptation"></a>
## 25. API Overload Adaptation (Tensor / functional Same Name, Multiple Signatures)

> When PTA / torch exposes multiple call forms under the same API name,
> such as `div(x, y)` and `div(x, y, *, rounding_mode=None)`,
> MindSpore uses multi-entry `api_def` YAML to dispatch overloads.
> The framework matches the corresponding `op_yaml` automatically by input type
> and arity.

### 25.1 Core Mechanism: `api_def` YAML Overload Dispatch

Under `mindspore/ops/api_def/{op_name}.yaml`, a single API name can define
**multiple `op_yaml` entries**.
In dynamic graph mode, the framework checks entries in order.
In static graph mode, `deprecated` branches are matched first.

**Key fields**:

| Field | Meaning |
| --- | --- |
| `op_yaml` | mapped `op_def/yaml/` operator YAML, which decides the Primitive |
| `py_method` | Python callback in `tensor_method.py`, used when backend config is `py_method` |
| `kwonlyargs` | keyword-only parameters such as `rounding_mode`, which must match `op_yaml` exactly |
| `Ascend` / `CPU` / `GPU` | backend execution mode: `pyboost` or `py_method` |
| `interface` | interface type: `tensor`, `function`, or both |
| `disable_scalar_tensor` | parameter names that must not auto-convert scalar -> Tensor |

**Example: `less.yaml` for input-type overloads**

```yaml
less:
  - op_yaml: less_scalar_op.yaml      # Tensor-Scalar branch
    py_method: tensor_less
    Ascend: pyboost
    CPU: py_method
    GPU: py_method
    interface: tensor, function

  - op_yaml: less_op.yaml             # Tensor-Tensor branch
    py_method: tensor_less
    Ascend: pyboost
    CPU: pyboost
    GPU: pyboost
    interface: tensor, function
```

**Example: `div.yaml` for multiple signatures + keyword-only args**

```yaml
div:
  - op_yaml: divs_op.yaml             # Tensor-Scalar, no rounding_mode
    py_method: tensor_div
    Ascend: pyboost
    CPU: pyboost
    GPU: pyboost
    disable_scalar_tensor: other
    interface: tensor, function

  - op_yaml: div_op.yaml              # Tensor-Tensor, no rounding_mode
    py_method: tensor_div
    Ascend: pyboost
    CPU: pyboost
    GPU: pyboost
    interface: tensor, function

  - op_yaml: divmods_op.yaml          # Tensor-Scalar + rounding_mode
    py_method: tensor_div
    kwonlyargs: rounding_mode
    Ascend: pyboost
    CPU: pyboost
    GPU: pyboost
    disable_scalar_tensor: other
    interface: tensor, function

  - op_yaml: divmod_op.yaml           # Tensor-Tensor + rounding_mode
    py_method: tensor_div
    kwonlyargs: rounding_mode
    Ascend: pyboost
    CPU: pyboost
    GPU: pyboost
    interface: tensor, function
```

<a id="api-overload-scenarios"></a>
### 25.2 Four Typical Overload Scenarios

| Scenario | Description | Typical operators | `api_def` shape |
| --- | --- | --- | --- |
| **1. Different input types** | same function name, but Tensor / Scalar inputs map to different `op_yaml` | `less`, `mul`, `eq` | multiple `op_yaml` entries with the same `py_method` |
| **2. Input types + `kwonlyargs`** | with or without keyword-only params maps to different `op_yaml` | `div`, `sub`, `add` | some entries declare `kwonlyargs` |
| **3. Old / new interface compatibility** | legacy MS interface differs from new mint / ext interface and needs deprecated compatibility | `flatten`, `pow`, `sub` | contains `deprecated/*.yaml` entries |
| **4. Symbolic alias** | two API names share the same implementation | `__mul__` -> `mul`, `__truediv__` -> `div` | one line `alias: xxx` |

### 25.3 Deprecated YAML Mechanism

**When needed**:
the old MS Tensor interface is incompatible with the new mint / ext interface,
for example due to parameter count, names, or keyword passing behavior.
In that case, a deprecated YAML path plus Python callback preserves old
behavior.

**Files involved**:

| File | Role |
| --- | --- |
| `ops/op_def/deprecated/{op_name}_method.yaml` | defines the old interface signature; parameters must match the `py_method` in `tensor_method.py` |
| `ops/tensor_method.py` | implements `tensor_{op_name}` / `deprecated_tensor_{op_name}` callbacks |
| `_extends/parse/deprecated/deprecated_tensor_method.py` | registers deprecated-interface mapping in static graph mode |

**Matching priority**:
- **dynamic graph**: checks `op_yaml` entries in `api_def` order
- **KBK static graph**: prioritizes the `deprecated` branch

**Example: `flatten.yaml` for old / new compatibility**

```yaml
flatten:
  - op_yaml: flatten_ext_op.yaml       # new interface: flatten(start_dim, end_dim)
    py_method: tensor_flatten
    Ascend: pyboost
    CPU: py_method
    GPU: py_method
    interface: tensor

  - op_yaml: deprecated/flatten_method.yaml  # old interface: flatten(order, *, start_dim, end_dim)
    py_method: deprecated_tensor_flatten
    Ascend: py_method
    CPU: py_method
    GPU: py_method
    interface: tensor
```

Matching deprecated YAML in `ops/op_def/deprecated/flatten_method.yaml`:

```yaml
# deprecated flatten
flatten:
  args:
    input:
      dtype: tensor
    order:
      dtype: str
      default: "'C'"
    start_dim:
      dtype: int
      default: 0
    end_dim:
      dtype: int
      default: -1
  returns:
    output:
      dtype: tensor
```

### 25.4 Notes for `functional` Overloads

`functional` interfaces such as mint have one key difference from Tensor
overloads:

- **there is no Python-callback fallback in static graph mode**:
  `py_method` and deprecated YAML are **not effective** for functional
  overloads
- functional overload dispatch only depends on **input types and argument
  count**

**Integration steps**:
1. in `api_def/{op_name}.yaml`, add `function` to the `interface` field
2. update `mint/__init__.py` so the original source is replaced by the
   auto-generated overload entry from `functional_overload`
3. add the corresponding documentation YAML under `api_def/function_doc/`,
   because missing docs will cause build errors

### 25.5 `alias` Declaration (Symbolic Overloads / API Aliases)

When two API names share the same PyBoost implementation, such as `__mul__` and
`mul`, declare it with one `alias` line:

```yaml
# __mul__.yaml
__mul__:
  alias: mul
```

```yaml
# __truediv__.yaml
__truediv__:
  alias: div
```

The framework will route all alias API calls into the target API’s overload
logic.

### 25.6 Development Notes

1. **Remove the old interface from `tensor.py`**:
   after overload adaptation, move the public interface into
   `tensor_method.py` with the `tensor_` prefix
2. **keep YAML indentation at 2 spaces**
3. **tests**:
   add overload tests under `tests/st/tensor/overload/test_{op_name}.py`,
   use `level0`, and set `jit_level` to `O0`
4. **Ascend-only operators**:
   set CPU / GPU to `py_method` and raise in the callback,
   or use an empty `py_method` implementation if that is the repo convention

<a id="view-operator-development"></a>
## 26. View Operator Development (Zero-Copy Shape / Strides Transform)

> View operators such as `transpose`, `reshape`, `expand_dims`, `slice`,
> `narrow`, and `chunk` do not move data.
> They only change tensor shape / strides / offset, and the output shares the
> same device memory as the input, which makes them zero-copy.

### 26.1 What a View Operator Is

Compared with ordinary ACLNN operators:

| Dimension | Ordinary operator | View operator |
| --- | --- | --- |
| data movement | yes, ACLNN kernel executes compute | no, zero-copy |
| output memory | newly allocated | shares device address with input |
| core implementation | `LAUNCH_ACLNN` / `RunOp` | strides calculation function |
| Infer | handwritten `InferShape` / `InferType` | view-special YAML lets the framework handle infer |
| backward | often requires bprop | often uses `bprop_expander: False` or autodiff from small ops |

**Typical view operators**:
`Transpose`, `Reshape`, `ExpandDims` (`unsqueeze`), `Squeeze`, `BroadcastTo`,
`Narrow`, `Slice`, `Split`, `Chunk`, `Diagonal`.

### 26.2 YAML Markers

View operator YAML uses two key flags:

| Field | Meaning |
| --- | --- |
| `view: True` | enable the PyNative view path based on strides calculation |
| `graph_view: True` | enable the KBK graph-mode view path via host kernel, not ACLNN |
| `labels: side_effect_mem: True` | mark memory side effects for view semantics |

**Two YAML patterns**:

1. **Original operator YAML**, such as `transpose_op.yaml`:
   `view: True` + normal `dispatch`,
   so PyNative uses the View path and KBK uses ACLNN
2. **View-special YAML**, such as `transpose_view_op.yaml`:
   `view: True` + `graph_view: True`,
   so both PyNative and KBK use the View path

```yaml
# transpose_view_op.yaml, view-special example
transpose_view:
  args:
    input:
      dtype: tensor
    input_perm:
      dtype: tuple[int]
  returns:
    output:
      dtype: tensor
  view: True
  graph_view: True
  labels:
    side_effect_mem: True
  dispatch:
    enable: True
```

```yaml
# expand_dims_view_op.yaml
expand_dims_view:
  args:
    input:
      dtype: tensor
    dim:
      dtype: int
      type_cast: tensor
  returns:
    output:
      dtype: tensor
  view: True
  graph_view: True
  labels:
    side_effect_mem: True
  dispatch:
    enable: True
```

<a id="view-strides-calculation"></a>
### 26.3 Strides Calculation Implementation (PyNative View Path)

View operators do not need an ACLNN kernel. The core task is implementing the
strides calculation function: compute the output shape / strides / offset from
the input tensor shape / strides plus operator parameters.

**File locations**:
- header:
  `mindspore/ops/include/view/{op_name}_view_strides_calc.h`
  (export the function with `OPS_API`)
- implementation:
  `mindspore/ops/view/{op_name}_view_strides_calc.cc`

**Function naming convention**:
`{OpName}ViewBasicTypeCalc`, with input arguments as `TensorPtr` plus normal
C++ primitive types.

**Simplified example**, `transpose_view_strides_calc`:

```cpp
// transpose_view_strides_calc.h
OPS_API TensorStorageInfoPtrList TransposeViewBasicTypeCalc(
    const mindspore::tensor::TensorPtr &input_tensor,
    const std::vector<int64_t> &dims);

// transpose_view_strides_calc.cc
#include "view/transpose_view_strides_calc.h"
#include "view/transpose_strides_calc.h"

namespace mindspore::ops {
TensorStorageInfoPtrList TransposeViewBasicTypeCalc(
    const mindspore::tensor::TensorPtr &input_tensor,
    const std::vector<int64_t> &dims) {
  return TransposeBasicTypeCalc(input_tensor, dims);
}
}  // namespace mindspore::ops
```

> **Note**:
> View-special YAML strides-calc functions (`*ViewBasicTypeCalc`) typically
> delegate directly to the original operator’s strides-calc function
> (`*BasicTypeCalc` or `*StridesCalc`), because the actual math is the same and
> only the entry differs.

The original operator’s strides calculation is registered with
`REG_VIEW_STRIDES_CALC_FUN`, usually in
`ops/view/{op_name}_strides_calc.cc`:

```cpp
REG_VIEW_STRIDES_CALC_FUN(ExpandDims, ExpandDimsCalc);
REG_VIEW_STRIDES_CALC_FUN(BroadcastTo, BroadcastToCalc);
REG_VIEW_STRIDES_CALC_FUN(Transpose, TransposeCalc);
```

**Three essentials of strides calculation**:
1. **shape**:
   compute the output shape from the input tensor and parameters
2. **strides**:
   the per-dimension data step, for example shape `(3,4,5)` with strides
   `(20,5,1)`
3. **offset**:
   the starting device-memory offset of the output tensor

<a id="view-kbk-host-kernel"></a>
### 26.4 KBK Host Kernel (Graph-Mode View Path)

When YAML sets `graph_view: True`, KBK does not use an ACLNN kernel.
Instead it uses a host kernel that updates strides directly.

**File location**:
`mindspore/ops/kernel/host/view/kernel_mod_impl/{op_name}_view.cc/.h`

**Key pattern**:
- inherit from `HostKernelMod`
- implement `GetWorkSpaceInfo`
  and call `UpdateOutputTensorInfo` to update output `tensor_storage_info`
- register with `MS_HOST_REG_KERNEL`

**Simplified example**, `transpose_view.cc`:

```cpp
// transpose_view.h
class TransposeView : public HostKernelMod {
 public:
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                        const std::vector<KernelTensor *> &outputs) override;
  void UpdateOutputTensorInfo(const std::vector<KernelTensor *> &inputs,
                              const std::vector<KernelTensor *> &outputs);
};

// transpose_view.cc
void TransposeView::UpdateOutputTensorInfo(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &outputs) {
  const auto &dims = inputs[kIndex1]->GetValueWithCheck<std::vector<int64_t>>();
  const auto &input = inputs[kIndex0];
  auto infos = ops::TransposeStridesCalc(
      input->GetShapeVector(), GetTensorStride(input),
      input->tensor_storage_info(), dims);
  outputs[kIndex0]->set_tensor_storage_info(infos[0]);
}

void TransposeView::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  UpdateOutputTensorInfo(inputs, outputs);
}

MS_HOST_REG_KERNEL(TransposeView, TransposeView);
```

### 26.5 View Feature Toggle and Fallback

In KBK graph mode, the framework uses the `kViewForGraphMode` switch to decide
whether the View path is enabled:
- **enabled** by default:
  operators marked with `graph_view: True` use the host-kernel path
- **disabled or unsupported**:
  they automatically fall back to the ACLNN kernel path through the original
  operator YAML dispatch

**Typical fallback scenarios**:
- the backend lacks a matching View kernel capability
- the input is dynamic shape and strides cannot be determined statically
- special control-flow scenarios

### 26.6 View Operator Development Checklist

| Step | Artifact | Meaning |
| --- | --- | --- |
| 1 | `op_def/yaml/{op_name}_view_op.yaml` | `view: True` + `graph_view: True` + `labels` |
| 2 | `ops/view/{op_name}_view_strides_calc.cc/.h` | strides calculation function (`BasicTypeCalc`) |
| 3 | `ops/kernel/host/view/kernel_mod_impl/{op_name}_view.cc/.h` | KBK host kernel (`MS_HOST_REG_KERNEL`) |
| 4 | add `view: True` to the original operator YAML | for example `transpose_op.yaml` |
| 5 | `ops/view/{op_name}_strides_calc.cc` + `REG_VIEW_STRIDES_CALC_FUN` | register the original operator’s strides calculation |

> **Torch reference**:
> for view-operator stride calculation logic, PyTorch source
> `aten/src/ATen/native/TensorShape.cpp` is a useful reference.
