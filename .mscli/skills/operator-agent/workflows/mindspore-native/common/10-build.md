# MindSpore Native Build

## Goal

Build the modified MindSpore tree and validate that the native operator route is
packagable.

## Inputs

- completed native implementation
- selected backend expectations

## Outputs

- successful build or a narrowed blocking error report
- build artifact locations when produced

## Responsibilities

1. Run the relevant MindSpore build command for the target backend.
2. Fix compile or registration errors introduced by the operator work.
3. Report the final artifact paths, including wheel outputs when produced.

## Backend Notes

- ACLNN-specific expectations such as Ascend registration evidence belong to the
  backend lane, but final framework build ownership stays here.

## Success Criteria

- build succeeds, or the remaining blocker is isolated and evidenced
- artifact paths are recorded
- the operator is ready for verification or handoff
