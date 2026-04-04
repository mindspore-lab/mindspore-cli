# Workflow 9: Documentation

## Goal

Complete the English `function_doc` (YAML) and the **Chinese RST documentation**, keeping the two strictly aligned.

> **Common omission**: the English `function_doc` is usually already created in the Step 1 YAML,
> so the agent may wrongly think "the documentation step is already done" and **skip the Chinese RST**.
> **English doc YAML does not mean the documentation step is complete**. Chinese RST is a separate deliverable and must be confirmed independently.

**What EN/CN consistency means**: parameter names, default values, required/optional status, semantics, and examples must match.
Each language should still follow its own documentation conventions; literal sentence-by-sentence identity is not required.

## Inputs

- **YAML definition**: the `function_doc` section created in Step 1
- **Operator interface implementation**: parameters, defaults, and examples

## Outputs (Two Documentation Types + Interface Lists, Confirmed One By One)

| Type | File Location | Requirement | Status |
| --- | --- | --- | --- |
| **English `function_doc`** | `ops/op_def/yaml/doc/{op}_doc.yaml` | `[MUST]` | ✅ created in Step 1 / needs refinement |
| **Chinese RST** | `docs/api/api_python/ops/*.rst` (or the matching `mint` / `nn` directory) | `[MUST]` required for public APIs | ✅ written / ✅ already exists / ❌ not written |
| **Interface list** | matching `mindspore.xxx.rst` index file | `[MUST]` | both English and Chinese lists must be updated in alphabetical order |

---

## Steps

### Step 1: Refine The English `function_doc`

Make sure the YAML `function_doc` created in Step 1 is complete:
- `desc`: short description of the operator; for public APIs, include principles, formulas, paper references, or other necessary background when appropriate
- `args`: description for each parameter
- `returns`: return-value description
- `examples`: a complete runnable example including imports

### Step 2: Chinese RST (Required For Public APIs)

> ⚠️ **This is the step most likely to be missed.** Search the repository first to see whether a matching Chinese RST already exists.

Follow the rules in `reference.md#documentation-reference`:
- file location: under `docs/api/api_python/ops/` or the corresponding `mint` / `nn` directory
- **first inspect existing Chinese RST files for similar operators** to confirm the format and directory structure
- **filename, in-file title, and interface definition must match exactly** (for functional interfaces, usually only the filename has the extra `func_` prefix)
- the underline of `=` below the title must be at least as long as the title itself
- update interface index files in alphabetical order

**If an older Chinese RST already exists** (for example `acos` exists but `acos_ext` does not), confirm:
- whether the old document needs to be updated to point to the new interface
- whether the new interface, such as `mint.acos`, needs its own standalone Chinese RST

### Step 3: Consistency Check (`reference.md#documentation-general-principles`)

| Check Item | English | Chinese |
| --- | --- | --- |
| Parameter names | ✅ consistent | ✅ consistent |
| Default values | ✅ consistent | ✅ consistent |
| Required/optional status | ✅ consistent | ✅ consistent |
| Examples | ✅ runnable | ✅ runnable |

### Step 4: Confirm The Target Location (`reference.md#documentation-output-mapping`)

| Interface Type | English Location | Chinese Location |
| --- | --- | --- |
| functional | implementation `.py` | `docs/api/.../ops/func_*.rst` |
| mint | mint interface implementation / list | `docs/api/.../mint/*.rst` |
| nn | `nn/*.py` | `docs/api/.../nn/*.rst` |
| Tensor method | `tensor.py` | `docs/api/.../Tensor/` |
| `ops` Primitive | Primitive implementation / list | `docs/api/.../ops/mindspore.ops.*.rst` |

---

## 🔒 Mandatory Check Before Marking Step 9 Complete

```text
Documentation deliverable checklist:

English function_doc (YAML):
  - File path: ops/op_def/yaml/doc/{op}_doc.yaml
  - Status: ✅ created in Step 1 and complete / needs refinement (missing fields: ___)

Chinese RST:
  - File path: docs/api/api_python/ops/mindspore.ops.func_{op}.rst (or the matching mint/Tensor directory)
  - Status: ✅ newly created / ✅ already exists and covers the new interface / ❌ not written (reason: ___)
  - If skipped: is this an internal operator (not a public API)? yes / no

Interface lists (English + Chinese):
  - Added to the matching mindspore.xxx.rst file in alphabetical order? yes / no

EN/CN consistency:
  - Parameter names consistent: yes / no
  - Default values consistent: yes / no
  - Examples consistent and runnable: yes / no
```

> **Public APIs (functional / mint / nn / Tensor) must have Chinese RST.**
> Only **internal operators** that are not exported in `__all__` and do not need public docs may skip it.
> When skipping, you must state the reason explicitly. Silent skipping is not allowed.

## Success Criteria

- [ ] The English `function_doc` is complete (`desc` / `args` / `returns` / `examples`)
- [ ] **The Chinese RST file has been created** for public APIs, or the operator is explicitly marked as internal and skippable
- [ ] Parameter names, default values, and examples are strictly consistent between English and Chinese
- [ ] Examples are runnable and include complete imports
- [ ] The interface lists have been updated in alphabetical order
- [ ] Filename, title, and interface definition all match

---
