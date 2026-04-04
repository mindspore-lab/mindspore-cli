# MindSpore API/Operator Identity Reference

This guide helps you locate and identify API/Operator names in the MindSpore codebase

### Common Identity Facts

mint entry (public API name and internal symbol) → api definition → active operator definition (op name/Primitive) → backwards definitions and operators

### Source-of-Truth Paths

| Purpose | Path |
| --- | --- |
| `mindspore.mint.*` public exports | `mindspore/python/mindspore/mint/__init__.py` |
| `mindspore.Tensor.*` methods | `mindspore/python/mindspore/common/tensor/tensor.py` |
| Function wrappers | `mindspore/python/mindspore/ops/function/*.py` |
| Overload entry definitions | `mindspore/ops/api_def/*.yaml` |
| Operator definitions | `mindspore/ops/op_def/yaml/*_op.yaml` |
| Gradient definitions | `mindspore/ccsrc/frontend/expander/grad/` |
| Auto-generated Primitive exports | `mindspore/python/mindspore/ops/auto_generate/gen_ops_prim.py` |

### Step 1. Identify Forward Operator Name

 - mint api import -> `mindspore/python/mindspore/mint/__init__.py`
 - API definition -> `mindspore/ops/api_def/`
 - OP definition -> ``mindspore/ops/op_def/yaml/`

```python
# Example: mint.linspace
from mindspore.ops.function.math_func import linspace_ext as linspace
```
This tells you:
- **Module path**: `mindspore.ops.function.math_func`
- **Internal function name**: `linspace_ext`
- **Public API name**: `linspace`
- **Operator name**: `LinSpaceExt` (CamelCase of `linspace_ext`)

#### Case 1: `functional_overload` export

exported from `mindspore.ops.functional_overload` → it has multiple implementations, do not infer a single primitive directly → check `ops/api_def/<api>.yaml` first → continue with each non-deprecated `op_yaml` branch

e.g.

```python
from mindspore.ops.functional_overload import max
```

the public API `max` is exported through `mindspore.ops.functional_overload` → check `mindspore/ops/api_def/max.yaml` first → continue with each non-deprecated `op_yaml` branch independently

e.g. a branch list from `mindspore/ops/api_def/max.yaml` may look like:

```yaml
max:
  - op_yaml: max_op.yaml
    py_method: tensor_max
    Ascend: pyboost
    CPU: pyboost
    GPU: pyboost
    interface: tensor, function

  - op_yaml: max_dim_op.yaml
    py_method: tensor_maxdim
    Ascend: pyboost
    CPU: pyboost
    GPU: pyboost
    disable_scalar_tensor: dim
    interface: tensor, function

  - op_yaml: maximum_op.yaml
    py_method: tensor_maximum
    Ascend: pyboost
    CPU: pyboost
    GPU: pyboost
    interface: function
```

This further tells you:

- `max_op.yaml`, `max_dim_op.yaml`, and `maximum_op.yaml` are different active candidates
- one public API may map to multiple branches
- branch reality matters more than the public name alone

#### Case 2: alias export

exported `xxx_ext as xxx` → primitive is `XxxExt` not `Xxx` → `xxx_ext_op.yaml`

e.g.

```python
from mindspore.ops.function.math_func import linspace_ext as linspace
```

- the public API name is `linspace`
- the internal function name is `linspace_ext`
- the mapping should keep the `_ext` suffix
- the forward YAML is `linspace_ext_op.yaml`
- the corresponding Primitive family is `LinSpaceExt`

Do not collapse `linspace_ext` back to plain `linspace`.

#### Case 3: `ops.auto_generate` direct export

When a symbol is exported directly from `mindspore.ops.auto_generate`, it usually points to one Primitive family directly.

- start from `<symbol>_op.yaml`
- still confirm the final Primitive from the YAML top-level operator name

#### Case 4: `ops.function` wrapper export

exported from `mindspore.ops.function` means the wrapper body itself is part of the identity evidence. 
You should inspect the wrapper body first, the wrapper may continue through `api_def`, or directly through one `op_yaml`

Example 1:

`from mindspore.ops.function.math_func import divide` → navigate to the definition of `divide` → `return div(input, other, rounding_mode=rounding_mode)` → check `ops/api_def/div.yaml` → continue with its non-deprecated branches `divs_op.yaml`, `div_op.yaml`, `divmods_op.yaml`, `divmod_op.yaml`

Example 2

`from mindspore.ops.function.array_func import full_ext as full` → navigate to the definition of `full_ext` → `return fill_scalar_(size, fill_value, dtype)` → `ops/op_def/yaml/fill_scalar_op.yaml`

Example 3:

`from mindspore.ops.function.nn_func import softmax_ext` → navigate to the definition of `softmax_ext` → the function first normalizes `dim` and `dtype`, then `return softmax_impl(input, dim)` → directly check `ops/op_def/yaml/softmax_op.yaml`

#### Case 5: Tensor inplace method

`Tensor.xxx_` should be treated as a distinct public entry.

e.g. `Tensor.sub_`

- the trailing `_` is part of the identity
- the method is an inplace variant, not the plain `sub` family
- the delegated internal symbol should be resolved before deciding the final
  `op_yaml`


### Step 2: Identify Backward Operator Name
1. Search `mindspore/ccsrc/frontend/expander/grad/` for the **CORRECT operator name** from Step 1
2. Look for `REG_BPROP_BUILDER("<CorrectOpName>")` registration (e.g., "AcosExt" not "ACos")
3. **EXTRACT AND LIST each individual operator call from the gradient computation**:
   - Look for `ib->OperatorName(...)` patterns in the BODYFUNC
   - List each operator name (e.g., Mul, Muls, Exp, PowTensorScalar, Add, Sub, etc.)
   - Show the line of code for each operator call

##### Case 1: Dedicated Grad Operator:
using `Emit("XXXGrad", ...)` means a dedicated grad operator.

##### Case 2: Multiple Primitive Operators
`AcosExt` backward body `dx = ib->Neg(dout) * ib->Rsqrt(ib->Sub(ib->Tensor(1, ib->GetDtype(x)), ib->Square(x)))` gives multiple primitive operators `Neg, Mul, Rsqrt, Sub, Square`


### Examples

**Step 0: Verify operator name**
```bash
# Read mint/__init__.py line 308:
from mindspore.ops.function.math_func import acos_ext as acos
```
→ `mint.acos` uses `acos_ext` → Operator name is `AcosExt` (NOT `ACos`!)

**Step 1: Find backward with CORRECT name**
```bash
# Search for AcosExt backward:
grep -A 10 "REG_BPROP_BUILDER(\"AcosExt\")" mindspore/ccsrc/frontend/expander/grad/grad_math_ops.cc
```

**Result**:
```cpp
REG_BPROP_BUILDER("AcosExt").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  dx = ib->Neg(dout) * ib->Rsqrt(ib->Sub(ib->Tensor(1, ib->GetDtype(x)), ib->Square(x)));
  return {dx};
});
```

**Answer**: `mint.acos` (which uses `AcosExt`) has NO dedicated backward operator. It computes gradient inline using:
1. `Neg` - negation
2. `Mul` - multiplication
3. `Rsqrt` - reciprocal square root
4. `Sub` - subtraction
5. `Square` - squaring


### Primitive Naming Conventions

Primitive naming facts:

- The YAML top-level operator name is the source of truth for Primitive naming.
- The name is typically converted to UpperCamelCase.
- Meaningful suffixes such as `Ext`, `Scalar`, `Inplace`, and `Grad` are kept. `fill_scalar` → `FillScalar`, `sub_ext` → `SubExt`, `add_scalar` → `AddScalar`, `softmax_backward` → `SoftmaxBackward` etc.


### Local Correctness Facts

- Overloaded APIs remain branch-based.
- Deprecated branches should not be treated as active branches.
- Alias targets should not be collapsed back to the public name.
- Backward operators should come from visible registered-body evidence.

## Reference Files

- `./api-to-operator.md` - Common mint.* operator name mappings
- `./operator-to-backend.md` - how an operator dispatch to the npu backend