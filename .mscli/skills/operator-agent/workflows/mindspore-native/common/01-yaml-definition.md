# Workflow 1: YAML Definition

## Goal

Define the required YAML for the operator (`op_def` + `api_def` + `function_doc`).

## Inputs

- **Feature document**: integration type, parameter list, input/output definitions
- **PTA source review results**: parameter names, types, defaults, and return structure

## Outputs

- **YAML file**: `mindspore/ops/op_def/yaml/{op_name}_op.yaml`
- **Documentation YAML**: `mindspore/ops/op_def/yaml/doc`

---

## Steps

### Step 1: Determine The YAML Structure

YAML definitions are documented in `mindspore/ops/op_def/yaml/README.md`.

Core fields:
- `op_name`: Primitive name (usually PascalCase, optionally with a Customize suffix)
- `args`: each parameter's `name` / `type` / `default` / `desc`
- `outputs`: each output's `name` / `type` / `desc`
- `dispatch`: backend integration mode. `enable: True` alone means auto-generated; adding `Ascend: "XxxAscend"` points to a Customize kernel
- `api`: Python exposure fields such as `py_method` / `module`

### Step 2: Configure `dispatch` According To The Integration Path (For aclnn task)

**This is where the path decision lands in YAML** (`reference.md#dispatch-path-selection`):

**Path 1 (auto-generated)** - direct argument passthrough, no Customize needed:
```yaml
dispatch:
  enable: True
  # omit the Ascend field -> build will auto-generate PyBoost/KBK code
```

**Path 2 (Customize)** - arguments require preprocessing:
```yaml
dispatch:
  enable: True
  Ascend: OpNameAscend    # explicitly names the Customize class
```

Decision rules:
- Parameter count, order, and types match ACLNN exactly -> Path 1
- `tuple -> vector`, `None` handling, `str -> enum`, scalar extraction, argument reordering, or manual output allocation is required -> Path 2
- If uncertain, start with Path 2 and simplify to Path 1 later if possible

### Step 3: Use A Code Skeleton Reference

The minimum YAML skeleton is documented in `reference.md#yaml-skeleton`.

---

## Success Criteria

- [ ] YAML files have been created (for every operator involved)
- [ ] Parameter names, types, and defaults are consistent with the PTA source review conclusions
---
