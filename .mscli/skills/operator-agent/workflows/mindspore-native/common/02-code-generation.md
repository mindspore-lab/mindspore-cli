# Workflow 2: Code Generation

## Goal

Run `mindspore/python/mindspore/ops_generate/gen_ops.py` to generate operator code from YAML. **`gen_ops.py` plays different roles on the two integration paths:**
- **Path 1 (auto-generated)**: generates the complete PyBoost/KBK call code, registration code, and Python interfaces
- **Path 2 (Customize)**: generates wrapper code that calls the handwritten Customize class, plus the Python interfaces

## Inputs

- **YAML files**: the `op_def` / `api_def` / `function_doc` outputs from Workflow 1
- **Integration path**: auto / customize

## Outputs

- **`gen_ops.py` runs successfully**

**Important**: MindSpore builds and interface calls depend on the generated files. Every time YAML changes, you must rerun `gen_ops.py` so the generated files stay up to date.

---

## Steps

### Step 1: Run `gen_ops.py`

```bash
python mindspore/ops/op_def/gen_ops.py
```

### Step 2: Confirm The Generated Artifacts

After it finishes, you **must verify** that the following artifacts were generated correctly:

| File Type | Path 1 | Path 2 | Notes |
| --- | --- | --- | --- |
| PyBoost call code | **fully generated** | wrapper generated | Path 1 directly calls ACLNN; Path 2 calls the Customize class |
| KBK auto-registration | **fully generated** | not generated | Path 2 requires a handwritten kernel and manual registration |

---

## Success Criteria

- [ ] `gen_ops.py` runs without errors
- [ ] On Path 1, confirm that the PyBoost call code and ACLNN kernelmod registration were generated automatically
  - ACLNN kernelmod: `mindspore/ops/kernel/ascend/aclnn/kernel_mod_impl/aclnn_auto_gen`, `mindspore/ops/kernel/ascend/aclnn/kernel_mod_impl/auto_generate`
  - PyBoost: `mindspore/ops/kernel/ascend/aclnn/pyboost_impl/auto_generate`

---
