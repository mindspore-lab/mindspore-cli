# Workflow 7: Export And Placeholder Behavior

## Goal

Ensure the operator is exported correctly through the `mint` package, that non-Ascend devices receive a clear placeholder error, and that the interface matches PyTorch as required.

## Outputs

- **`mint` package exports**: updates to `__init__.py` / `__all__`
- **Interface files**: functional / nn / Tensor methods as needed

**Interface alignment constraint**: interface name, parameter names, parameter order and defaults, and input dtype/range constraints must match PTA.

---

## Steps

### Step 1: Explicit Export In `mint`

- Add the operator name in both the **corresponding operator category import block** and **`__all__`** inside the relevant `__init__.py` under `mindspore/python/mindspore/mint/`.
- Ensure the new operator appears in the `__all__` list.

### Step 2: Interface Development (`reference.md#api-development`)

| Interface Type | Key Points |
| --- | --- |
| **functional** | Use `_get_cache_prim` internally to obtain the Primitive and avoid repeated `__init__` |
| **nn** | Use a `Cell` subclass; do not raise directly in `construct`, use `@constexpr` instead |
| **Tensor method** | Cover PyNative / KBK / GE modes if required by the project. For **GE mode**, register the mapping in `resource.cc` and implement it in `standard_method.py`; the validation function there must not accept Tensor inputs (see 2. Interface Development 2.4) |

### Step 2.5: Interface Overload Configuration (If Multiple Same-Name Signatures Exist, `reference.md#api-overload-adaptation`)

If the target operator has overloads with the same name, such as Tensor-Scalar vs Tensor-Tensor, or signatures with and without keyword-only parameters, follow this process:

1. **Analyze the overload scenario**: determine which of the `reference.md#api-overload-scenarios` cases applies (different input types / keyword-only args / old-new compatibility / symbolic alias)
2. **Write `api_def` YAML**: define multiple `op_yaml` entries in `ops/api_def/{op_name}.yaml`, one per signature
3. **If the old interface is incompatible** -> add `ops/op_def/deprecated/{op_name}_method.yaml` and register the old-interface mapping in `deprecated_tensor_method.py`
4. **If there is a symbolic alias** -> add alias YAML such as `__mul__.yaml: alias: mul`
5. **If functional overload is involved** -> add `function` to the `interface` field and update the import source in `mint/__init__.py` to `functional_overload`

## Limitation

Interfaces named `xxxExt` or `xxx_ext` are internal only. When exporting them publicly, always use `import xxx_ext as xxx` and remove the `ext` suffix. Never expose an `ext` interface directly as the public API.

---

## Success Criteria

- [ ] The operator can be imported normally from `mint`
- [ ] functional / nn / Tensor interfaces work as required
- [ ] `_get_cache_prim` is used correctly in the functional interface
- [ ] For overload cases, the multi-entry `api_def` config is correct and deprecated YAML parameters match `py_method` (if applicable)

---
