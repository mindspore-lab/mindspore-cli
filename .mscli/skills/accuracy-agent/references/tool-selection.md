# Tool Selection

Use this file when deciding which method fits the current evidence. Tools are
conditional helpers, not prerequisites for the workflow.

## Start With Methods, Not Tool Names

Pick the method that matches the question:

- Need alignment of setup and randomness:
  - configuration checks, deterministic settings, repeated baseline run
- Need the first internal mismatch:
  - tensor capture, hooks, manual `.npy` comparison, compare tools
- Need late-stage drift:
  - local norm, gradients, one-step weight updates, training-state monitoring
- Need non-fatal invalid values:
  - overflow detection, module statistics, narrowed module replay
- Need no-baseline confidence:
  - small golden case, backend or precision self-compare

## Common Tool and Method Roles

### Configuration Checks

Best for:

- migration validation
- cross-framework comparison
- early elimination of obvious setup drift

Typical targets:

- framework and runtime version
- precision and AMP settings
- optimizer hyperparameters
- dataset version and sample order
- random controls

### Precision Precheck

Best for:

- quick screening before deep capture
- models dominated by supported high-level APIs

Do not use it as proof that the network is correct.

### Data Capture and Compare

Best for:

- finding the first mismatching module or operator
- structured cross-framework comparison

Use it when:

- step1 loss mismatches
- final output mismatches but you need earlier evidence

### Hierarchical Visualization

Best for:

- large models where raw tables are too noisy
- narrowing from network to layer to node

Use it after data capture, not before.

### Training-State Monitoring

Best for:

- later divergence
- long training where the bad step is not obvious
- models that are too large for broad capture

Typical signals:

- loss trend
- gradient trend
- optimizer or communication anomalies

### Checkpoint Comparison

Best for:

- proving that two training runs have already drifted at the parameter level
- later-stage divergence

### No-Benchmark Compare

Best for:

- no trusted baseline
- narrow, local investigation

Do not run it across a whole large model without narrowing scope first.

### Dump, msprobe, and TroubleShooter

Treat these as optional helpers:

- `Dump`
  - good for tensor export and overflow-related evidence
- `msprobe`
  - good for capture, compare, and overflow-oriented workflows on supported
    environments
- `TroubleShooter`
  - good for `.npy` directory comparison, weight conversion validation, and
    automated difference screening in cross-framework workflows

If none of them are available, fall back to:

- hooks
- saved tensors
- manual `.npy` comparison
- module-level statistics
- one-step experiments

## Debug Script Hygiene

When you are about to write or review a reduced repro or debug script:

- confirm the intended framework stack and relevant package versions first
- confirm the script really executes on the intended device path
- enable determinism before comparing tiny deltas
- print dtype and device for suspicious tensors instead of inferring them
