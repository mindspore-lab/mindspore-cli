# Validation Playbook

Read this file after choosing a bottleneck and one targeted optimization.

## Principle

Validate the change against the same bottleneck evidence that motivated it.
Do not declare success from a recommendation alone.

## Default Before/After Comparison

Always keep the run comparable:

- same model path
- same config and command
- same batch size unless the optimization explicitly changes it
- same hardware scale

## Compare by Bottleneck Class

### Communication

Compare:

- step time
- communication time share
- collective count
- exposed step tail

### Host Launch or Idle Gap

Compare:

- host-side idle gap
- kernel launch density
- device utilization trend

### Input Pipeline

Compare:

- pre-compute idle time
- input stage time
- end-to-end throughput

### Graph Build or Recompilation

Compare:

- compile time
- compile count
- steady-state latency after warmup

### Compute Hotspot

Compare:

- dominant operator time share
- end-to-end latency or step time

### Memory Pressure

Compare:

- peak memory
- top memory-consuming stage
- whether batch size headroom improved

### Optimizer or Update

Compare:

- update-time share
- collective time near update
- total step tail

## Stop Rules

- If the dominant bottleneck did not move, do not pile on more changes.
- If a new bottleneck becomes dominant, say so explicitly.
- If the evidence is still weak after rerun, ask for the missing trace view
  instead of guessing.
