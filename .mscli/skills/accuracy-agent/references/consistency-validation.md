# Consistency Validation

After building an accuracy profile, validate the most likely consistency causes
against real evidence.

Typical groups:

- data consistency
- config consistency
- model consistency
- checkpoint consistency
- dtype and precision consistency
- framework or platform consistency
- metric and evaluation consistency

Every root-cause claim should carry:

- confidence
- supporting evidence
- a validation check
- a small next experiment

Use this file to turn a vague suspicion into an evidence-backed candidate list.
If a claim cannot point to evidence and a falsifiable check, it is not ready to
rank as a root cause.

## First-Divergence Discipline

If the first stable mismatch is still unknown, do not jump to operator blame.
Reduce scope and capture earlier aligned comparisons until you can point to the
first module, tensor, or stage that diverges in a stable way.

## Module-Then-Operator Escalation

When a module output mismatches, use this escalation order:

1. Verify the module inputs.
2. If the inputs already mismatch, stop. Walk upstream to the producer of that
   input and keep narrowing there.
3. If the inputs align, verify model parameters, `register_buffer` state,
   dtype, API parameters, and actual device placement for the module.
4. Only after those checks are clean may you narrow to an operator inside the
   module.

## Operator-Level Note

If framework or platform consistency checks narrow the first stable mismatch to
one operator, load `references/operator-accuracy-triage.md` and finish that
operator-specific validation before claiming the operator is the root cause.

Before escalating from a mismatching module to an operator claim, verify that
the module already has aligned:

- inputs
- model parameters
- `register_buffer` state, including non-persistent buffers when they affect the
  path under test
- dtype
- API parameters
- actual device placement

Use evidence to justify the claim rather than intuition. Intuition may help you
choose the next check, but it is not evidence on its own.
