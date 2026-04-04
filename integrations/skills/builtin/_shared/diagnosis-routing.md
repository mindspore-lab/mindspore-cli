# Diagnosis Routing

This shared note defines the symptom-routing contract for top-level
`/diagnose` and `/fix` commands used directly in Claude Code, Codex, and other
 command-driven environments.

## Problem Categories

Route each post-run problem into exactly one of these categories:

- failure
- accuracy
- performance

## Deterministic Mapping

Use user wording and directly visible evidence. Do not use an LLM-only guess
when a simple keyword and symptom check is enough.

| symptom category | typical keywords | specialist skill |
|------------------|------------------|------------------|
| failure | crash, error, failed, operator, RuntimeError, CANN, ACLNN, hang, timeout, OOM, not implemented | `failure-agent` |
| accuracy | accuracy, drift, regression, eval, wrong result, mismatch, loss diverge, NaN, Inf | `accuracy-agent` |
| performance | throughput, latency, slow, memory, bottleneck, utilization, profiler, tok/s | `performance-agent` |

## Ambiguity Handling

If classification is ambiguous, ask the user to choose one:

1. training crashed or failed to start
2. accuracy or evaluation results are wrong
3. performance is worse than expected

Do not silently pick a specialist skill when the symptom is genuinely unclear.

## Mode Contract

The same specialist skill supports two top-level modes:

- `diagnose`
  - evidence
  - diagnosis
  - report
  - no edits
- `fix`
  - evidence
  - diagnosis
  - fix proposal
  - explicit confirmation
  - apply
  - verify

## Redirect Cases

Do not route to diagnosis specialists when the real user need is:

- pre-run validation or environment compatibility check
  - route to `readiness-agent`
- model migration or initial porting
  - route to `migrate-agent`
- algorithm feature adaptation into an existing model
  - route to `algorithm-agent`
