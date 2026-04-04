# MindSpore Operator To Backend Dispatch

This is an optional knowledge reference for branch-local backend dispatch
inspection. Use it only after the operator identity is already resolved.

## Source-Of-Truth Paths

| Purpose | Path |
| --- | --- |
| operator YAML | `mindspore/ops/op_def/yaml/*_op.yaml` |
| auto-generated ACLNN mapping | `mindspore/python/mindspore/ops_generate/pyboost/aclnn_config.yaml` |
| customize PyBoost implementations | `mindspore/ccsrc/plugin/device/ascend/kernel/pyboost_impl/customize/` |
| customize Aclnn KernenlMod implementations | `mindspore/ccsrc/plugin/device/ascend/kernel/opapi/aclnn_kernel/mod_impl/customize/` |

## Workflow

1. Check whether the target YAML branch has `dispatch.enable`.
2. Check whether an explicit `Ascend: XxxAscend` customize entry is present.
3. If the route looks auto-generated, confirm the generated ACLNN mapping when
   source evidence exists.
4. If the route looks customized, confirm the matching PyBoost and Aclnn KernelMod source
   evidence.

## Output Shape

Keep the answer branch-local:

- whether ACLNN integration is visible
- whether the mode is `auto-generated` or `customize`
- what source evidence supports that conclusion
