# API Resolution Validation Checklist

Use this checklist when the task depends on mapping a public MindSpore API to
the correct internal operator branch.

## Before Answering Or Routing

- read the actual export site first
- verify the internal symbol name
- verify the final primitive or YAML branch
- do not assume `mint.*` names map directly to plain primitives
- do not merge overload branches unless the source proves they are equivalent

## Before Backend Dispatch Claims

- confirm the operator identity first
- treat dispatch as branch-local, not public-API-global
- check `dispatch.enable` before claiming ACLNN support
- look for explicit customize evidence when the YAML points to a customize path

## Common Failure Modes

- using the public API name as the primitive name without checking exports
- answering backend support for the wrong overload branch
- inferring ACLNN support from naming rather than source evidence
