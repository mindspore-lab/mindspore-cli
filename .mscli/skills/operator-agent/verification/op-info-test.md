# Op Info Verification

Use this module when public MindSpore APIs changed or when the user explicitly
asks for `op_info`, ST, or remote validation evidence.

## Workflow

Run the op_info workflow in this order:

1. `verification/op-info-workflow/op_info_generation.md`
2. `verification/op-info-workflow/patch_out_old_tests.md` when isolation is
   required
3. `verification/op-info-workflow/remote_deploy_and_test.md` when remote
   validation is required

## Constraints

- treat op_info as generic verification, not ACLNN-specific logic
- do not change unrelated tests
- keep temporary isolation patches out of final retained history
- require the final summary to say what is covered, not covered, and blocked

## Local Assets

- `scripts/remote_runner_client.py`

## Success Criteria

- required op_info coverage exists
- functional and stability rounds pass when requested
- the final report includes explicit remote evidence and remaining gaps
