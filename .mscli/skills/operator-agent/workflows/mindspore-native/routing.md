# MindSpore Native Routing

This is the central routing and execution contract for the
`native-framework` + `mindspore` path.

Path convention: unless stated otherwise, `reference.md` means
`../reference.md`. `aclnn_doc` means the ACLNN docs available in the current
workspace, installed SDK, or user-provided materials.

**Important: Don't blindly follow existing op's code. If there is a principal defined in this document, follow it with high priority instead of imiatating existing op's code.**

## How To Use This Routing File

When `operator-agent` selects `native-framework` for `mindspore`, **create a
TODOLIST** and execute the following workflow in strict order.

**Steps marked `🔒 must not be skipped` are mandatory in every scenario.**
**Places marked `⛔ HARD GATE` must be completed before you continue, otherwise
stop and wait for user confirmation.**

This file governs ordered execution. The lower-level workflow docs provide the
implementation details for each phase.

## Strict Execution Checklist

- [ ] **[Pre](common/00-pre-checks.md)** `🔒 must not be skipped`: pre-checks
  and route shaping
  - Required outputs: repository inventory, PTA source review report when
    applicable, initialized Feature document when the task is substantial
  - Required route decision: determine whether backend implementation is needed
    after `common/03-general-infer.md`
  - **⛔ HARD GATE**: before entering Step 1, confirm the required pre-check
    outputs exist in the workspace and have been delivered to the user
  - After each later step, backfill the corresponding section of the Feature
    document
- [ ] **[Step 1](common/01-yaml-definition.md)**: YAML definition
  - Backfill Feature: YAML and API-definition sections
- [ ] **[Step 2](common/02-code-generation.md)**: code generation
- [ ] **[Step 3](common/03-general-infer.md)**: GeneralInfer
  - Backfill Feature: dynamic-shape and validation/error sections
- [ ] **[Step 4](backend/routing.md)** `🔒 conditional`: backend implementation
  - Execute only when the selected native route needs backend-specific
    implementation work after infer
  - Follow `backend/routing.md` to select the backend lane and execute its
    internal sub-order
- [ ] **[Step 5](common/06-bprop.md)**: BPROP
  - Backfill Feature: bprop section
- [ ] **[Step 6](common/07-export.md)**: export
- [ ] **[Step 7](common/08-unit-tests.md)**: unit testing
  - Backfill Feature: test-plan section
- [ ] **[Step 8](common/09-docs.md)**: documentation
  - Important: English doc YAML does not mean the documentation step is
    complete. Chinese RST is a separate deliverable.
- [ ] **[Step 9](Feature document finalization)** `🔒 must not be skipped`
  - Complete the Feature document code-change summary and acceptance-report
    sections
  - Update the task list to reflect what was completed, skipped, or deferred
  - Even if intermediate steps were skipped conditionally, the Feature document
    must still be completed and delivered to the user
- [ ] **[Step 10](common/10-build.md)**: build and packaging validation

## Validation Loop (Evidence Required At Every Step) `🔒 must not be skipped`

After every completed step, an execution report **must** be presented to the
user using the template below. It may not be omitted, merged away, or deferred.
This is a mandatory user-facing deliverable, not an internal note.

```text
━━━ Step X Execution Report ━━━

Execution basis (which routing requirement I followed):
- workflow file: workflows/mindspore-native/...
- corresponding routing requirement: (quote the relevant checklist item from routing.md)
- success criteria for this step: (copied from the workflow success criteria)

What I did (deliverables):
- ...

Key evidence (code snippets / file paths / search results):
- ...
- Which existing operator implementation I compared against: ...

Validation result:
- ...

Feature backfill update:
- updated section(s): ...
- file path: ...

Item-by-item success criteria check:
- [ ] Criterion 1: ✅/❌
- [ ] Criterion 2: ✅/❌
- ...

Open issues / risks / next step:
- ...
```

## Backfill Rules

- The Feature document is initialized during `common/00-pre-checks.md` when the
  task is substantial enough to require review and handoff documentation.
- After every later completed step, backfill the corresponding Feature section
  before moving on.
- If a conditional step is skipped because the route does not require it, record
  the skip reason in the Feature task list and in the step execution report.
- The Feature document must be finalized before build handoff is considered
  complete.
