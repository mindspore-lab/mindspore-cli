# Validation Ladder

Use this file after you have a ranked hypothesis and need a disciplined
verification plan.

## Principle

Validate from cheap to expensive. Each rung should either confirm progress or
reveal a new earlier mismatch.

## Default Ladder

### 1. Golden Input Output Match

Use when:

- the case is inference-heavy
- final output mismatch is the primary symptom

Pass condition:

- output is aligned within the task-appropriate tolerance

### 2. Step1 Loss Alignment

Use when:

- training is involved
- the first suspected mismatch is in forward behavior

Pass condition:

- step1 loss is aligned within the task-appropriate tolerance

### 3. Local Norm or Gradient Alignment

Use when:

- step1 matches but later divergence appears

Pass condition:

- gradient or norm behavior no longer shows the earlier mismatch

### 4. One-Step Weight Update Alignment

Use when:

- you need to separate backward from optimizer update behavior

Pass condition:

- one-step parameter update is aligned well enough for the task

### 5. Short Training Run

Use when:

- lower rungs passed
- the change may still drift after a few steps

Pass condition:

- the short run no longer reproduces the original gap

### 6. Long-Run Training or Evaluation

Use when:

- the user cares about real training or evaluation outcome

Pass condition:

- final metric, loss curve, or evaluation result is back within acceptable
  bounds

### 7. Restore Full Scale

Use when:

- lower-rung validation passed under simplified settings

Pass condition:

- the fix holds under the intended multi-card, full-data, or production setup

## Exit and Rollback Rules

- If a later rung reveals a new mismatch, go back and update the first
  divergence stage.
- If the task only needs inference output alignment, rungs 1 to 2 may be
  enough.
- If the task needs training alignment, do not stop before a short run unless
  the user explicitly accepts limited evidence.
- Do not restore full scale before lower-rung checks are clean.

## Residual Gap Handling

If one issue has already been fixed or isolated but a residual gap remains
elsewhere, make the residual visible before starting a new diagnosis round.

Before continuing, consider neutralizing the already known issue so multiple
error sources do not stack together. Common options:

- replace the confirmed operator with an identity or no-op on both sides
- replay a trusted baseline tensor into the target path
- stub the already-fixed path with a reduced deterministic equivalent
