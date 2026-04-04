# MindSpore NPU (ACLNN) Dispatch Reference

This reference collects static backend dispatch facts for one active branch that
has already been resolved at the API identity layer.

Use this only after the API has already been resolved to the correct `op_yaml`
and Primitive/OperatorName.

## Scope

Questions on:

- Whether ACLNN dispatch is visible from static source layout.
- Whether the branch is `auto_generate`, `customize`, or unsupported.
- Whether `kbk` evidence is present.
- Whether `pyboost` evidence is present.

### Source-of-Truth Paths

| Purpose | Path |
| --- | --- |
| Operator YAML | `mindspore/ops/op_def/yaml/*_op.yaml` |
| Auto-generated ACLNN mapping | `mindspore/python/mindspore/ops_generate/pyboost/aclnn_config.yaml` |
| Customize PyBoost implementations | `mindspore/ccsrc/plugin/device/ascend/kernel/pyboost_impl/customize/` |
| Customize KBK implementations | `mindspore/ccsrc/plugin/device/ascend/kernel/opapi/aclnn_kernel/mod_impl/customize/` |
| Generated KBK registration evidence | `aclnn_kernel_register_auto.cc` |

### Workflow

For a correct op name, check its op definition `op_yaml` in `mindspore/ops/op_def/yaml/`.

1. check if yaml has `dispatch` settings and  read `dispatch.enable` value
2. check whether an explicit `Ascend: XxxAscend` entry is present
3. infer if aclnn is supported for the operator, by `auto_generate` or `customize` implementaion, and kbk / pyboost supporting evidence.


#### Case 1: `dispatch.enable: True` and no `Ascend: XxxAscend`

This means the operator goes through the auto-generated ACLNN path, both kbk / pyboost are supported.
It's primitive and aclnn mapping (like `AcosExt: 'aclnnAcos'`) usually presence in `aclnn_config.yaml`.

#### Case 2: `dispatch.enable: True` and `Ascend: XxxAscend`

This usually means the branch uses a handwritten ACLNN customize path.

Common customize evidence locations:

- `.../kernel/pyboost_impl/customize/xxx.h` and `.cc` (pyboost aclnn supported)
- `.../kernel/kernel_mod_impl/customize/xxx_aclnn_kernel.h` and `.cc`  (kbk aclnn supported)

#### Case 3: no `dispatch.enable`

This usually means no static ACLNN path is visible from the branch YAML.
You can treat this operator as not integrated with ACLNN.

### Local Correctness Facts

- ACLNN evidence is branch-local, not public-API-global.
- `dispatch.enable` is the first gate.
- `customize` requires explicit source evidence.
- This lens reports static inventory only.


## Output Shape

Answer whether this operator is integrated with ACLNN and what the integration
mode is.

Keep the answer short and branch-local:

- whether the operator has ACLNN integration
- if yes, whether the integration mode is `auto_generate` or `customize` and evidence when it is visible from source.
- if no, report aclnn implementaion not supported for this operator


### Worked Examples

#### Example 1

We have the correct operator name `AcosExt` / `acos_ext`. It's `op_yaml` is `mindspore/ops/op_def/yaml/acos_ext_op.yaml`:

```yaml
#operator acos_ext
acos_ext:
    ...
    dispatch:
        enable: True
        GPU: None
```

the `dispatch.enable: True` -> auto-generated ACLNN path.
and `AcosExt: 'aclnnAcos'` in `aclnn_config.yaml` means the aclnn is `aclnnAcos`


#### Example 2

We have the correct operator name `AddScalar` / `add_scalar`. It's `op_yaml` is `mindspore/ops/op_def/yaml/add_scalar_op.yaml`.

```yaml
dispatch:
  enable: True
  Ascend: AddScalarAscend
```
gives:
`dispatch.enable: True` + `Ascend: XxxAscend` -> customize ACLNN path

- customize kernel and customize PyBoost evidence should be checked explicitly
  - file `/aclnn/pyboost_impl/customize/add_scalar.cc` -> pyboost aclnn supported
  - file `/kernel_mod_impl/customize/add_scalar_aclnn_kernel.cc` detected -> kbk aclnn supported


## Reference Files

- `./api-to-operator.md` - Common mint.* operator name mappings
- `./operator-to-backend.md` - how an operator dispatch to the npu backend