# Comparison Scenarios

Use this file only when the comparison setup itself is unclear. The point is to
choose the right starting move before running detailed diagnostics.

These are common comparison scenarios, not an exhaustive taxonomy. If a case
does not fit cleanly, reduce it to the closest comparable setup and continue
from alignment and first-divergence analysis.

## Scenario 1: MindSpore vs MindSpore

Typical case:

- previous good run vs current run
- previous version vs current version
- baseline model vs local modification

Start here:

1. Confirm config, weights, data order, and determinism.
2. Compare fixed golden input outputs.
3. If needed, compare module outputs with hooks or saved tensors.

Why this path:

- Framework semantics already match.
- You usually do not need cross-framework mapping first.
- The fastest wins usually come from config, weights, preprocessing, or local
  code changes.

## Scenario 2: PyTorch on Ascend vs MindSpore on Ascend

Typical case:

- migration validation
- backend stays the same, framework changes

Start here:

1. Run configuration and determinism checks first.
2. Confirm the baseline is really PyTorch on Ascend, not plain CPU `torch`.
   Check `torch`, `torch_npu`, and actual device placement before trusting the
   comparison.
3. Compare step1 loss on the same input and weights.
4. If step1 differs, narrow the first tensor mismatch from coarse modules down.
5. If step1 matches but later diverges, switch quickly to gradients and
   one-step updates.

Why this path:

- Hardware stays constant, so framework differences are easier to isolate.
- This is a strong candidate for data capture and structured compare tools.

## Scenario 3: PyTorch on GPU vs MindSpore on Ascend

Typical case:

- framework and hardware both changed

Start here:

1. Do not treat this as one undifferentiated mismatch.
2. Separate framework differences from backend differences where possible.
3. Use small deterministic cases before large training runs.
4. Prefer stage-by-stage evidence over direct full-network judgment.

Why this path:

- Numerical behavior may differ for good reasons.
- Kernel path, precision mode, accumulation path, and framework semantics may
  all vary at once.

## Quick Decision Table

| Scenario | First comparison | Best early evidence | Common trap |
| --- | --- | --- | --- |
| MindSpore vs MindSpore | golden input and step1 loss | config diff, hooks, saved tensors | blaming backend too early |
| PyTorch Ascend vs MindSpore Ascend | step1 loss and first mismatching module | config check, capture and compare | skipping config alignment |
| PyTorch GPU vs MindSpore Ascend | small deterministic stage checks | golden inputs, module outputs, precision notes | treating all mismatch as a bug |
