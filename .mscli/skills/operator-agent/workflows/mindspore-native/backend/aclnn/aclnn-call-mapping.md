# ACLNN Call Mapping

## Goal

Resolve the target ACLNN call set for the selected MindSpore operator branch.

## When To Use

- the task is an NPU or Ascend native MindSpore operator adaptation
- the ACLNN interface is not already obvious from local source evidence
- composite operator rollout needs a sub-operator call inventory

## Responsibilities

1. Identify the exact ACLNN API or API family for the operator branch.
2. Distinguish between direct single-call cases and composite call chains.
3. For composite cases, inventory the sub-operator coverage and rollout order.
4. Record evidence instead of guessing from names alone.

## Reusable Artifacts

- `templates/pta-analysis-report.md`
- `templates/aclnn-callchain-analysis.md`

## Success Criteria

- the final ACLNN mapping is branch-specific and evidenced
- composite cases have an explicit rollout plan
- later PyBoost and KBK work can consume the mapping directly
