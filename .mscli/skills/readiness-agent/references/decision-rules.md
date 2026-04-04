# Readiness-Agent Decision Rules

## Target and Framework

- prefer explicit `target` and `framework_hint`
- otherwise infer from high-confidence workspace evidence inside the selected
  workspace only
- if evidence is weak or conflicting, downgrade confidence instead of forcing a
  strong conclusion
- do not search sibling repos, home-directory projects, or bundled examples for
  missing workspace evidence
- if workspace evidence points to PTA, stay on PTA checks and do not probe
  MindSpore
- if workspace evidence points to MindSpore, stay on MindSpore checks and do
  not probe PTA
- use `mixed` only when the current workspace contains evidence for both

## Workspace Boundary

- treat the current workspace as the certification boundary
- only inspect workspace-local entry scripts, configs, assets, and virtual
  environments unless the user explicitly points to another path
- external runtime directories may be resolved from environment variables when
  they represent CANN or Hugging Face state

## Runtime Threshold

- `runtime_smoke` is the minimum threshold for `READY`
- if `runtime_smoke` fails, do not emit `READY`
- explicit `task_smoke_cmd` is stronger evidence when present

## Asset Rules

- local assets satisfy the requirement immediately
- explicit Hugging Face repo IDs may satisfy model or dataset requirements when
  the endpoint is reachable and the workflow allows network-backed resolution
- missing entry scripts are only auto-repairable when a known bundled example
  recipe applies

## Final Question

After `READY` or `WARN`, ask whether to run the real model script now.
