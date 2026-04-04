# ACLNN Call-Chain Inventory - {OpName}

> **Purpose**: record the ACLNN call chain of the PTA interface and the coverage status on the MindSpore side.
> **Document status**: local file, do not commit to Git.

---

## 1. Target Interface

| Attribute | Value |
| ---- | -- |
| **PTA interface** | `torch.{op_name}` |
| **MindSpore target interface** | `mindspore.mint.{op_name}` |

---

## 2. Forward Call Chain

```text
Forward path of torch.{op_name}(args...):
  1. aclnn{Sub1}(input1, input2) -> intermediate_1    # description
  2. aclnn{Sub2}(intermediate_1, input3) -> output    # description
```

---

## 3. Backward Call Chain

```text
Backward path of {op_name}:
  1. aclnn{SubGrad1}(dout, ...) -> (d_input1, d_input2)   # description
```

---

## 4. MindSpore Coverage Inventory

| # | aclnnXxx | Purpose | YAML | Infer | PyBoost | KBK | UT | Status | Notes |
| - | -------- | ---- | ---- | ----- | ------- | --- | -- | ---- | ---- |
| 1 | aclnn{Sub1} | Forward | ✅/❌ | ✅/❌ | ✅/❌ | ✅/❌ | ✅/❌ | ✅ integrated / ⚠️ partial / ❌ missing | |
| 2 | aclnn{Sub2} | Forward | | | | | | | |
| 3 | aclnn{SubGrad1} | Backward | | | | | | | |

---

## 5. Implementation Plan

### Priority Order (Leaf Node First, Composite Later; Forward First, Backward Later)

| No. | Operator | Dependency | Estimated Work | Status |
| ---- | ---- | ---- | ---------- | ---- |
| 1 | aclnn{Sub1} (missing) | no dependency | YAML + Infer + PyBoost + KBK + UT | ⬜ Not started |
| 2 | aclnn{Sub2} (missing PyBoost) | depends on #1 | PyBoost + KBK | ⬜ Not started |
| 3 | Composite implementation of {OpName} | depends on #1, #2 | customize + ST | ⬜ Not started |

### Notes

- Implement missing sub-operators step by step following SKILL.md Steps 1-8.
- Sub-operators usually do not need documentation and exported interface, only YAML + Infer + PyBoost + KBK + UT.
- Implement the composite operator only after all required sub-operators are ready.

---