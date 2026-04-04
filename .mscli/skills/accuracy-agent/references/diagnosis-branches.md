# Diagnosis Branches

Read only the branch that matches the first divergence stage. If none fits
cleanly, reduce scope and go back to alignment plus first-divergence analysis
instead of forcing a branch.

## Branch A: Step1 Loss Mismatch

Use this branch when the first meaningful mismatch appears before or at step1
loss.

Primary suspects:

- config mismatch
- wrong or partially converted weights
- preprocessing, tokenizer, padding, mask, or label differences
- dtype, AMP, cast path, or operator semantic mismatch
- backend precision mode differences

Recommended sequence:

1. Reconfirm aligned weights and inputs.
2. Check preprocessing outputs before entering the model.
3. Compare model outputs at coarse module boundaries until the first stable
   mismatch is visible.
4. If a module mismatches, verify that its inputs are already aligned.
5. If the module inputs mismatch, walk upstream to the producer instead of
   blaming an operator inside the current module.
6. If the module inputs align, verify model parameters, `register_buffer`
   state, dtype, API parameters, and actual device placement.
7. Narrow to the first mismatching node or operator.
8. If the mismatch is stable at one operator, switch to operator-level triage
   before attributing the bug to the implementation.
9. Only then talk about a concrete fix.

Do not:

- start with optimizer tuning
- treat a later metric drop as the primary symptom
- compare tensors from incompatible precision contexts without explanation

## Branch B: Step1 Matches, Later Divergence

Use this branch when step1 loss is aligned but later steps diverge.

Primary suspects:

- backward mismatch
- optimizer update mismatch
- loss scale or grad clipping differences
- distributed communication or reduction differences
- hidden randomness that survived setup alignment

Recommended sequence:

1. Compare local norm or gradients as early as possible.
2. Compare one-step weight updates.
3. Run an `lr=0` or no-update experiment to isolate backward from update.
4. If needed, inject trusted gradients and compare the resulting update.
   Save trusted step1 gradients from the baseline side, replace the current
   side's gradients for a single optimizer step, then compare the updated
   weights or step2 loss.
5. Inspect optimizer hyperparameters, weight decay scope, and parallel config.

Do not:

- restart from broad forward-only tensor dumps unless new evidence points back
- jump to long-run retraining before a one-step update check

## Branch C: Non-Fatal NaN or Inf

Use this branch only when the job still produces comparable outputs.

Primary suspects:

- overflow or unstable AMP path
- invalid labels or data values
- divide-by-zero, log-of-negative, or extreme activation range
- loss scale mismatch

Recommended sequence:

1. Find the first step or module with invalid values.
2. Check overflow detection if available.
3. Inspect module-level statistics around the failure point.
4. Test smaller input ranges or reduced precision complexity.

Do not:

- keep this branch if NaN or Inf caused a crash or hard stop
- suggest generic learning-rate changes before finding the first invalid value

## Branch D: Cross-Platform Mismatch or Eval-Only Regression

Use this branch when training or inference finishes, but final outputs or
metrics differ across platforms or versions.

Primary suspects:

- metric implementation mismatch
- postprocessing mismatch
- dtype or kernel path differences
- preprocessing or label construction differences
- backend-specific precision behavior

Recommended sequence:

1. Use a fixed golden input and compare the final output first.
2. Check metric and postprocessing definitions next.
3. If needed, walk backward to the earliest internal mismatch that matters.
4. Prefer module-level captures first. If one module mismatches, verify its
   inputs before checking operators inside it.
5. If the module inputs mismatch, keep walking upstream until you reach the
   first stable divergence point.
6. If the module inputs align, verify model parameters, `register_buffer`
   state, dtype, API parameters, and actual device placement before
   operator-level probes.
7. Keep the comparison deterministic and scope-limited.
8. Do not treat a small residual delta as acceptable until you have evidence
   for its source.

Do not:

- compare only the headline metric without checking what feeds it
- assume every cross-platform mismatch is a correctness bug

## Branch E: No Trusted Baseline

Use this branch when the user has no trusted baseline at all.

Primary goals:

- reduce scope
- build a minimal confidence signal
- avoid false certainty

Recommended sequence:

1. Choose a minimal module or golden case.
2. Compare behavior across backend or precision modes when meaningful.
3. Prefer small manual or historical checks over full-network guesses.
4. Focus on convergence pattern, stability, and monotonic sanity.

Do not:

- present a root cause with high confidence
- use full-network no-baseline compare on a large model without narrowing scope
