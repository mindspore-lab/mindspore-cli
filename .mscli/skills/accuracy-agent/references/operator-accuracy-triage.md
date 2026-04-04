# Operator Accuracy Triage

Use this file only after the first stable mismatch has already been narrowed to
one operator or to a very small operator cluster. Do not start here from a
full-model symptom.

## Goal

Avoid false operator bug reports. Cross-framework accuracy gaps often come from
callsite mismatch rather than from the operator implementation itself.

## Step 1: Recheck the Operator Callsite Before Blaming the Operator

Compare how both model scripts call the operator, not just the operator name.

Check for:

- explicit arguments
- implicit default arguments
- positional versus keyword argument mapping
- dtype, cast, layout, and shape expectations
- reduction, eps, axis, mask, keepdim, align_corners, and similar semantic
  knobs

Why this matters:

- operators with similar names across frameworks may not share the same API
  schema
- default values may differ even when explicit arguments look aligned
- a callsite mismatch is usually cheaper to fix than an operator rewrite

Do not claim an operator bug until this callsite check is clean.

## Step 2: Build a Minimal Single-Operator Repro

If the callsite is aligned but the mismatch remains, construct a script that
only exercises the suspected operator.

Keep the repro focused:

- use the same input values, shapes, dtype, and attributes that exposed the
  mismatch
- remove unrelated model logic, optimizer logic, and dataloader noise
- keep determinism controls consistent when randomness is involved

Interpretation:

- if the isolated repro still shows the mismatch, the operator path is a strong
  root-cause candidate
- if the isolated repro does not reproduce the gap, return to the surrounding
  module context instead of escalating an operator claim

## Step 3: Try Replacement or Reimplementation

Once the single-operator repro confirms the issue, try a safer equivalent path
before proposing large changes.

Prefer this order:

1. swap to a semantically equivalent operator that is known to align better for
   the task
2. prefer non-legacy MindSpore operator variants over legacy ones when
   available
3. check whether a `mindspore.mint` operator can replace a legacy
   `mindspore.nn` or `mindspore.ops` path
4. reimplement the operator from smaller primitives only if replacement is not
   viable

Before picking the replacement, check the official PyTorch-to-MindSpore API
mapping table:

- https://www.mindspore.cn/docs/zh-CN/master/note/api_mapping/pytorch_api_mapping.html

Use it to confirm:

- whether the intended MindSpore API is marked as aligned
- whether the mapping points to `mindspore.mint`, `mindspore.ops`, or
  `mindspore.nn`
- whether the mapped API has documented parameter-name, default-value, or other
  semantic differences

MindSpore note:

- some `mindspore.nn` and `mindspore.ops` entries are legacy paths and may show
  accuracy issues in migration work
- `mindspore.mint` operators are often the closest API-schema and precision
  match to the corresponding torch operators

## Step 4: Validate at the Right Scope

After a replacement or reimplementation attempt:

1. rerun the single-operator repro
2. rerun the lowest validation-ladder rung that originally failed
3. return to the full module or model compare only after the smaller scope is
   clean

Do not report the operator fix as confirmed until the user validates the change
at the task-relevant scope.
