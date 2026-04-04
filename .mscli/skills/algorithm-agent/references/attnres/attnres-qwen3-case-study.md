# Qwen3 Attention Residuals Case Study

Use this reference when the selected `algorithm-agent` route is
`attnres` and the target model resembles a Hugging Face decoder stack such as
Qwen.

This case study is based on a validated Qwen3 retrofit that added
Attention Residuals to the Hugging Face `transformers` implementation and
checked the resulting code with forward tests and small finetune runs.

## Source artifacts

- Config patch:
  `transformers/src/transformers/models/qwen3/configuration_qwen3.py`
- Model patch:
  `transformers/src/transformers/models/qwen3/modeling_qwen3.py`
- Modular source:
  `transformers/src/transformers/models/qwen3/modular_qwen3.py`
- Tests:
  `transformers/tests/models/qwen3/test_modeling_qwen3.py`

## What the working Qwen3 port adds

### 1. Config surface

`configuration_qwen3.py` adds:

- `attn_residual_variant`
- `attn_residual_block_size`
- `attn_residual_eps`

Expected values:

- `attn_residual_variant`: `"none" | "full" | "block"`
- `attn_residual_block_size`: logical residual-site count
- `attn_residual_eps`: key-side RMSNorm epsilon

The public load API should allow:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_residual_variant="block",
    attn_residual_block_size=8,
    attn_residual_eps=1e-6,
    dtype="auto",
)
```

### 2. Registered mixer modules

The validated port keeps the mixer as a registered module on the model:

- `Qwen3AttentionResidualMixer`
- `Qwen3FullAttentionResidual`
- `Qwen3BlockAttentionResidual`
- `Qwen3BlockAttentionResidualState`

This is important because:

- AttnRes parameters appear in `state_dict`
- `from_pretrained` reports AttnRes-specific missing weights
- the optimizer sees the new parameters
- the mixer is not silently recreated every forward

### 3. Decoder-layer wiring

Each `Qwen3DecoderLayer` keeps the baseline path unchanged and adds:

- `attn_output(...)`
- `mlp_output(...)`

The port does not rewrite token attention itself. It only exposes branch-level
helpers so the model loop can feed AttnRes-computed inputs into the same
attention and MLP modules.

### 4. Logical-site accounting

For Qwen3:

- one decoder block contributes two logical sites
- attention site index: `2 * layer_idx`
- MLP site index: `2 * layer_idx + 1`

Block size must be interpreted in these logical-site units, not in decoder
layer units.

### 5. Model-boundary wiring

`Qwen3Model` decides the route:

- `"none"`: original baseline
- `"full"`: use `embedding + prior logical outputs`
- `"block"`: use `completed_blocks + partial_block`

At the end of the loop, the model runs one final AttnRes mix before the normal
final norm or head path.

### 6. Tests that mattered

The validated Qwen3 port added two minimum tests:

- block variant forward smoke test still returns `[batch, seq, hidden]`
- `block_size=1` matches `full`

Those two tests caught bookkeeping and indexing mistakes early.

## Engineering cautions learned from the case

### Dynamic mixer creation is a real bug

Do not instantiate the depth mixer inside `forward`, for example:

```python
def _forward_block_attnres(...):
    mixer = AttnResDepthMixer(...)
```

That pattern is wrong because:

- mixer parameters are not registered on the model
- checkpoint load will not report AttnRes-specific missing weights
- the optimizer may never own the mixer parameters
- each forward may start from fresh parameters
- first-step loss can jump far above the baseline for non-algorithm reasons

### Load-time logs are a useful invariant

When the port is wired correctly, `from_pretrained` should report newly
initialized AttnRes-specific parameters such as:

- `...attn_residual_mixer.mixer.key_norms.*.weight`
- `...attn_residual_mixer.mixer.queries`

If those warnings disappear after you added a new module, inspect whether the
module is actually registered.

### Dtype alignment matters

The mixer path often normalizes and computes logits in `float32`, but weighted
aggregation must cast the weights back to the source-value dtype before
`einsum`, especially under `bf16` training.

## Qwen3-specific caution

`transformers/src/transformers/models/qwen3/modeling_qwen3.py` may be
generated from `modular_qwen3.py`.

For a durable Hugging Face patch:

- patch the modular source
- regenerate the generated file according to repo workflow
- do not leave the two files semantically out of sync
