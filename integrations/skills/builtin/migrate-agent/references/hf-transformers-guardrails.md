# HF Transformers Guardrails

## Scope and precedence
- This file contains route-specific rules for the `hf-transformers` route.
- If any rule conflicts with `SKILL.md`, follow this file.

## Guardrails
- Avoid custom compatibility wrappers unless required.
- Use diff-based insertion when updating auto maps.
- Keep changes minimal and aligned with existing MindOne patterns.
- `register_buffer` is supported in MindSpore; do not remove it as part of device-handling cleanup.
- Preserve the route's `mindspore.mint` and `mindspore.mint.nn` conversions by
  default after auto-convert.
- Do not batch-convert auto-generated `mint.*` and `mint.nn.*` usages back to
  `mindspore.nn.*`, `ops.*`, or legacy MindSpore APIs as a generic cleanup
  step.
- If a specific operator must move away from `mint`, document the concrete
  reason such as missing support, target-repo convention, or verified semantic
  mismatch.
- Model coding standards:
  - Import MindSpore as `import mindspore` (avoid `import mindspore as ms`).
  - Use `from mindspore import nn` and define modules as `nn.Cell`.
  - `nn.Cell` guidance applies to module base classes and structure; it is not
    a blanket instruction to rewrite layer implementations away from `mint.nn`.

## Response expectations
- List reference files consulted.
- Summarize edits and note any risks or TODOs.
- Suggest next tests when appropriate.
