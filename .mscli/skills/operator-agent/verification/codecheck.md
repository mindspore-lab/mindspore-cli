# Codecheck Verification

Use this module when formatting or lint cleanup is required for the files
affected by the operator work.

## Workflow

1. Run `scripts/ms_codecheck.py`.
2. Fix reported formatting or lint issues in the affected files.
3. Run the script again and confirm the relevant issues are gone.

## Constraints

- prefer fixing the code over adding broad suppressions
- only add special filters after the user explicitly agrees when the rule is
  genuinely contradictory to repository formatting

## Success Criteria

- relevant format and lint issues are either fixed or explicitly blocked
- any remaining suppression need is recorded and justified
