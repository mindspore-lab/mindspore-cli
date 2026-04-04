# MindSpore API To Operator Identity

This is an optional reference for Stage 1 analysis. Use it only when
the public API name is not enough to identify the real primitive or operator
branch.

## Source-Of-Truth Paths

| Purpose | Path |
| --- | --- |
| `mindspore.mint.*` public exports | `mindspore/python/mindspore/mint/__init__.py` |
| `mindspore.Tensor.*` methods | `mindspore/python/mindspore/common/tensor/tensor.py` |
| function wrappers | `mindspore/python/mindspore/ops/function/*.py` |
| overload entry definitions | `mindspore/ops/api_def/*.yaml` |
| operator definitions | `mindspore/ops/op_def/yaml/*_op.yaml` |
| gradient definitions | `mindspore/ccsrc/frontend/expander/grad/` |

## Resolution Workflow

1. Start from the public API export and identify the internal symbol.
2. Follow wrapper logic when exported from `ops.function`.
3. Resolve the active `api_def` branches when overloads are involved.
4. Confirm the final primitive family from `op_def` YAML.
5. Treat inplace methods and `_ext` aliases as distinct identities until the
   source proves otherwise.

## Output Shape

Produce the smallest correct identity set needed for routing:

- public API name
- internal symbol name
- primitive or operator name
- active YAML branch or branches

Do not collapse branches prematurely when overloads exist.
