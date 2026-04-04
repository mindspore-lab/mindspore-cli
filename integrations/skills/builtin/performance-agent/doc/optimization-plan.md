# Performance Agent Productization Plan

This document records the current product direction for
`skills/performance-agent` on `refactor-arch-3.0`.

The active design is a single-path, profiler-led performance diagnosis
workflow. Older feature-trial ideas are deferred and are not part of the
runtime control surface.

## Current Product State

The skill now has a deterministic diagnosis pipeline built around reusable
structured artifacts:

1. locate the best profiler export root
2. build multi-dimensional summaries:
   - step breakdown
   - communication
   - memory pressure
   - input pipeline
   - trace gaps
   - operator hotspots
3. build a machine-readable `PerformanceProfile`
4. classify ranked bottleneck candidates
5. compare before and after metrics when provided
6. emit:
   - `out/report.json`
   - `out/report.md`
   - `out/meta/performance-profile.json`
   - `out/meta/bottlenecks.json`
   - `out/meta/performance-verdict.json`
   - `out/meta/validation-comparison.json`
   - `out/artifacts/perf.lock.json`

This is the product baseline. The skill should not drift back to free-form
diagnosis when structured summaries are available.

## Product Rules

- Keep the workflow single-path and profiler-led.
- Treat structured helper outputs as the primary diagnosis substrate.
- Preserve one dominant bottleneck at a time.
- Prefer machine-readable outputs over narrative-only explanations.
- Require validation evidence before claiming that an optimization worked.
- Ask for the smallest missing high-signal trace artifact when evidence is
  incomplete.

## Remaining Gaps

### 1. Real Sample Coverage

The current tests use synthetic but layout-realistic fixtures. The next step is
to validate the pipeline against 3 to 5 real Ascend profiler exports:

- communication-dominant case
- host launch or idle-gap case
- graph compile or recompilation case
- memory-heavy case
- dataloader stall case

### 2. Parser Robustness

The helper scripts currently support common layouts and field names. They
should be hardened against profiler version drift by collecting more real
samples and expanding column-name compatibility only when needed.

### 3. Adapter Contract

The skill still needs an explicit adapter contract for `ms-cli` integration:

- input mapping from host runtime into skill inputs
- output surfacing rules for `report.json`, `report.md`, and verdict payloads
- fallback rules when trace discovery is weak or missing

### 4. Validation Normalization

The deterministic validation comparator supports the current core metrics, but
real workloads may expose additional names or units. Normalizing these inputs
without widening the contract too much is the next product hardening task.

## Deferred Items

The old two-path design and pre-profiler feature-trial path are deferred. They
may be revisited later, but they are not active runtime behavior for the
current product line.

## Notes

- Keep comments, script output, and generated summaries in English.
- Keep scope limited to Ascend/NPU, MindSpore, and `torch_npu`.
