# ACLNN Path Selection

## Goal

Choose whether the ACLNN backend lane can stay auto-generated or must switch to
customized Ascend implementation.

## Decision Rule

Stay on the auto-generated path only when MindSpore can pass the operator
arguments through to the ACLNN interface without semantic reshaping.

Switch to customization when any of the following is required:

- argument reordering
- scalar extraction or normalization
- tuple or list conversion
- optional `None` handling
- manual output allocation
- view-specific or composite behavior that generation does not model correctly

## Output

Record:

- selected backend lane mode: `auto-generated` or `customize`
- reason
- evidence
- downstream impact on PyBoost and KBK work

## Success Criteria

- the route decision is made before handwritten Ascend code starts
- later backend work does not rely on guessed ACLNN signatures
