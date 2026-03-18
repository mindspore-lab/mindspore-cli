# Workstream C: Incubating Factory

## Goal
Create ms-factory as `incubating/factory/` in ms-cli. Define schemas, seed initial cards from demo.go data, establish pack format.

## C1: Directory Structure

```
ms-cli/incubating/factory/
├── README.md
├── schemas/
│   ├── operator.schema.yaml
│   ├── failure.schema.yaml
│   ├── perf_feature.schema.yaml
│   ├── algo_feature.schema.yaml
│   ├── model.schema.yaml
│   └── report.schema.yaml
├── cards/
│   ├── operators/
│   ├── failures/
│   ├── perf-features/
│   ├── algo-features/
│   ├── models/
│   └── reports/
├── packs/
│   └── stable/
│       └── v0.1/
├── manifests/
│   ├── index.yaml
│   └── pack.yaml
└── docs/
    ├── overview.md
    ├── cards.md
    └── pack-format.md
```

## C2: Define Schemas

**`schemas/operator.schema.yaml`**
```yaml
required: [id, name, category, level, platforms]
properties:
  id: string (kebab-case)
  name: string
  category: enum [attention, optimizer, activation, norm, comm, memory]
  level: enum [L0, L1, L2, L3]
  vendor: string (optional)
  platforms:
    type: map
    keys: "<framework>-<version> + <device>"
    values:
      status: enum [supported, not_supported, experimental]
      profiled: boolean
      latency_ms: number (optional)
      constraints: array of string (optional)
  fallback: string (optional, op id)
  optimized_variant: string (optional, op id)
  notes: string (optional)
```

**`schemas/failure.schema.yaml`**
```yaml
required: [id, severity, tags, detection, description]
properties:
  id: string
  severity: enum [low, medium, high, critical]
  tags: array of string
  affects_ops: array of string (optional, op ids)
  affects_models: array of string (optional)
  affects_platforms: array of string (optional)
  detection:
    pattern: string (regex or keyword)
    metric: string (optional)
    threshold: number (optional)
  description: string (markdown, root cause + symptom)
  fix_id: string (optional, links to fix section in card or separate card)
  fix_diff: string (optional, inline diff)
  status: enum [open, resolved]
```

**`schemas/perf_feature.schema.yaml`**
```yaml
required: [id, name, level, category]
properties:
  id: string
  name: string
  level: enum [L0, L1, L2]
  category: enum [compute, memory, communication, compilation]
  description: string
  expected_gain: string (e.g., "+10% throughput")
  platforms: array of string
  config_diff: string (optional)
  code_diff: string (optional)
  dependencies: array of string (optional)
```

**`schemas/algo_feature.schema.yaml`**
```yaml
required: [id, name, level, category]
properties:
  id: string
  name: string
  level: enum [L0, L1, L2]
  category: enum [loss, attention, optimizer, regularization, scaling]
  description: string
  expected_gain: string (e.g., "+1.5 pts accuracy")
  compatible_methods: array of string (e.g., ["lora", "full"])
  config_diff: string (optional)
  code_diff: string (optional)
```

**`schemas/model.schema.yaml`**
```yaml
required: [id, model, method, platform, version]
properties:
  id: string
  model: string
  method: string
  platform: string
  framework: string
  version: enum [v1, v2, v3]
  config: map (training config snapshot)
  expected:
    final_loss: number
    throughput: string (range)
    eval_acc: map (benchmark → range)
    training_time: string
  baselines: array of {benchmark, pretrain_acc, posttrain_acc, tolerance}
  known_issues: array of string (failure card ids)
  verified_optimizations: map (optimization id → metrics)
```

**`schemas/report.schema.yaml`**
```yaml
required: [id, submitted_by, submitted_at, status, context, observation]
properties:
  id: string
  submitted_by: string
  submitted_at: datetime
  status: enum [raw, processing, promoted, archived]
  context: {model, method, platform, framework, device_count}
  observation:
    type: enum [failure, accuracy, performance, success]
    summary: string
    error_log: string (optional)
  metrics: {steps_completed, final_loss, throughput, eval_acc} (all optional)
  config_snapshot: map (optional)
  user_note: string (optional)
```

## C3: Seed Initial Cards from demo.go

Extract hardcoded data from `workflow/train/demo.go` (1815 lines) into factory cards:

**Operators** (from demo crash/fix scenarios):
- `cards/operators/dsa.yaml` — DSA op, not_supported on torch 2.7 + ascend
- `cards/operators/adam.yaml` — Adam, 400ms latency on ascend, optimized_variant: fused-adam
- `cards/operators/fused-adam.yaml` — Fused Adam, ~250ms on ascend
- `cards/operators/softmax.yaml` — Softmax, fp16 drift issue on ascend
- `cards/operators/flash-attention.yaml` — FA, requires CANN >= 8.0.RC3
- `cards/operators/sdpa.yaml` — SDPA, fallback for flash-attention

**Failures** (from demo diagnosis flows):
- `cards/failures/dsa-torch27-ascend.md` — DSA not implemented in torch 2.7
- `cards/failures/fp16-softmax-drift.md` — fp16 softmax causes 16.8 pts accuracy drop
- `cards/failures/cann-flash-attn-version.md` — FlashAttention needs CANN >= 8.0.RC3

**Perf-features** (from RunSingleLanePerfFeature):
- `cards/perf-features/fused-adam.md`
- `cards/perf-features/flash-attn-v2.md`
- `cards/perf-features/gradient-ckpt.md`
- `cards/perf-features/bf16-mixed.md`
- `cards/perf-features/graph-mode.md`
- `cards/perf-features/comm-overlap.md`
- `cards/perf-features/zero-offload.md`
- `cards/perf-features/sequence-parallel.md`
- `cards/perf-features/selective-recompute.md`

**Algo-features** (from RunSingleLaneAlgoFeature):
- `cards/algo-features/mhc.md`
- `cards/algo-features/lora-plus.md`
- `cards/algo-features/galore.md`
- `cards/algo-features/dpo.md`
- `cards/algo-features/rope-scaling.md`
- `cards/algo-features/moe-routing.md`
- `cards/algo-features/flash-attn.md`
- `cards/algo-features/sparse-attn.md`
- `cards/algo-features/ddpm-noise.md`

**Models** (from demo training data):
- `cards/models/qwen3-7b/lora-ascend-910b.yaml`
  - Expected: loss 0.831, throughput 518 tok/s, ceval-valid 72.1%
  - With fused-adam: throughput 571 tok/s
  - With MHC: eval_acc 73.6%

## C4: Pack Format

**`manifests/index.yaml`**
```yaml
channels:
  stable:
    latest: "v0.1"
    url: "packs/stable/v0.1/"
  nightly:
    latest: "20260317"
    url: "packs/nightly/20260317/"
```

**`manifests/pack.yaml`** (per pack)
```yaml
version: "v0.1"
channel: stable
created_at: "2026-03-17"
card_count:
  operators: 6
  failures: 3
  perf-features: 9
  algo-features: 9
  models: 1
cards:
  - id: dsa
    type: operator
    path: operators/dsa.yaml
  - id: fp16-softmax-drift
    type: failure
    path: failures/fp16-softmax-drift.md
  # ... all cards listed
```

A pack is a directory with all cards copied at a point in time + pack.yaml manifest.

## C5: Validation

```
# Validate cards against schemas
python -c "import yaml, jsonschema; ..."  # or dedicated validate.py

# Verify all cards listed in pack manifest exist
# Verify no orphan cards (exist on disk but not in manifest)
```

---

## Implementation Order

Suggested sequence (can parallelize across workstreams):

1. **C1 + C2**: Create factory structure + schemas (foundation, no code deps)
2. **C3**: Seed cards from demo.go (provides test data for everything else)
3. **B1**: Shared tools (factory_query.py, ssh_exec.py, etc.)
4. **B2-B7**: Agent skills (can start with one, e.g., failure-agent)
5. **A1**: Remove workflow mode from ms-cli
6. **A2**: Add skill loader
7. **A3**: Add factory client
8. **A4**: Evolve agent loop for skills
9. **B8-B10**: Update AGENTS.md, commands, consistency check
