# Debug Script Hygiene

Use this file when you are about to write or review a reduced repro, hook
script, or tensor-compare script.

## Goal

Keep the debug script trustworthy. A fast script that compares the wrong
framework path, wrong device path, or nondeterministic execution is worse than
no script.

## 1. Confirm the Compared Stacks First

Print or collect the exact framework versions that matter for the comparison:

- primary framework package
- any backend-specific extension package
- any runtime package that changes execution behavior

Why this matters:

- the framework package and the backend extension may be separate libraries
- missing or mismatched backend extensions can invalidate the comparison
- a version mismatch is often cheaper to rule out than deep tensor capture

Use `scripts/collect_accuracy_context.py` when you need a quick snapshot.

## 2. Verify the Script Really Uses the Intended Device Path

For the chosen baseline and target:

- import the packages required for the intended backend path
- move the model to the intended device when the framework requires it
- move input tensors to the intended device when the framework requires it
- print the device of suspicious tensors instead of assuming it

Do not trust a compare result from an unintended CPU or fallback path when the
intended comparison is on an accelerator backend.

For mixed-device code, check whether an upstream `device` argument is quietly
`None`, `"cpu"`, or another unexpected value.

## 3. Enable Determinism Before Chasing Tiny Deltas

For reduced repros and step-level compare scripts:

- fix seeds
- disable unnecessary randomness such as shuffle or dropout
- enable deterministic execution controls when the stack supports them
- rerun the trusted baseline once to understand its natural variance

Do not explain away an unstable diff until you have checked whether the script
itself is nondeterministic.

## 4. Verify Inputs Before Guessing Operators

If a module or code block mismatches:

- verify its inputs first
- verify the corresponding model parameters
- verify `register_buffer` values, including non-persistent buffers when they
  affect the path under test
- verify dtype and casts
- verify API arguments and defaults
- verify actual device placement

Only after these checks are clean should you narrow to a specific operator
inside the module.

When the first stable mismatch is still unknown, prefer module-level or
stage-level captures first. A trustworthy script that finds the first
divergence point is more useful than an early operator-specific probe aimed at
the wrong scope.

## 5. Keep Scope Tight

Keep the reduced repro focused on one confirmed issue at a time.

If one issue has already been fixed or isolated but a residual gap remains:

- report the residual explicitly
- decide with the user whether the current gap is acceptable
- if the user wants to continue, restart the workflow for the remaining issue

Before the next round, consider neutralizing the known issue so it does not
mask later problems. Typical options:

- replace the path with identity or no-op behavior on both sides
- feed the trusted baseline tensor into the target path
- replace the path with a smaller deterministic equivalent

## Examples

Example 1:

- intended compare: accelerator baseline versus accelerator target
- actual script behavior: model or tensors silently stay on CPU
- result: the compare is not trustworthy until device placement is corrected

Example 2:

- intended compare: `torch_npu` baseline versus `mindspore` target on Ascend
- actual script behavior: the script imports only `torch`, never imports
  `torch_npu`, and never moves the model or tensors to NPU
- result: this is still a CPU baseline, not an Ascend baseline
