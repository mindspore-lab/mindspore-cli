# Readiness-Agent Product Contract

`readiness-agent` has one user-visible responsibility:

- certify whether the intended local single-machine task is runnable now

That certification must stay scoped to the selected workspace. Evidence from
other repos, home-directory projects, or bundled skill examples cannot replace
missing workspace-local scripts, assets, or framework signals unless the user
explicitly points to those paths.

The final user-visible result must contain:

- `status`
- `can_run`
- `target`
- `summary`
- `blockers`
- `warnings`
- `next_action`

## Status Rules

`READY`

- no hard blocker remains
- `runtime_smoke` passed
- enough evidence exists to treat the target as runnable now

`WARN`

- no hard blocker is proven
- some readiness evidence remains incomplete or uncertain
- `runtime_smoke` may still have passed, but confidence is reduced

`BLOCKED`

- one or more hard blockers remain
- explicit task smoke failed
- `runtime_smoke` failed

## Internal Result

The internal verdict should continue to preserve:

- `execution_target`
- `evidence_level`
- `task_smoke_state`
- `dependency_closure`
- `checks`
- `blockers_detailed`
- `warnings_detailed`
- `fix_applied`
- `revalidated`

## Interaction Rule

After `READY` or `WARN`, the skill must explicitly ask:

- `Do you want me to run the real model script now?`

That prompt belongs in the final user-facing result. Running the real model
script is a separate action after readiness, not part of the readiness
certification threshold.
