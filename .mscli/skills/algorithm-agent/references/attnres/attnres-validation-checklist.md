# Attention Residuals Validation Checklist

Use this checklist when the selected `algorithm-agent` route is
`attnres`.

## Minimum validation

1. Config validation rejects invalid AttnRes values when the target repo has
   config validation logic.
2. `attn_residual_variant="none"` preserves the original forward path.
3. `attn_residual_variant="block"` still returns `[batch, seq, hidden]`.
4. `block_size=1` matches `full` exactly or within numerical tolerance.
5. `from_pretrained(..., attn_residual_variant="block")` or the equivalent
   public load API works.
6. AttnRes-specific parameters appear in `model.named_parameters()` and in the
   optimizer parameter groups.

## Strongly recommended validation

1. Run a tiny train or finetune smoke test and confirm loss decreases or
   remains finite.
2. Compare first-step loss against baseline. A small increase can be normal,
   but a dramatic jump suggests wiring or initialization problems.
3. Run the model in the dtypes you care about, especially `bf16`.
4. If the model supports caching or generation, test that path with
   `attn_residual_variant="none"` and one AttnRes variant.
5. If the model supports gradient checkpointing, confirm the AttnRes path
   works with it enabled.
6. If the model has attention backend variants such as eager, flash, or sdpa,
   confirm the AttnRes path does not break backend selection.

## Example test ideas

### Forward shape test

```python
config.attn_residual_variant = "block"
config.attn_residual_block_size = 2
model = Model(config)
result = model(**inputs)
assert result.last_hidden_state.shape == (batch, seq, hidden)
```

### Block-one-equals-full test

```python
full_config.attn_residual_variant = "full"
block_config.attn_residual_variant = "block"
block_config.attn_residual_block_size = 1

full_model = Model(full_config)
block_model = Model(block_config)
block_model.load_state_dict(full_model.state_dict(), strict=False)

torch.testing.assert_close(
    full_model(**inputs).last_hidden_state,
    block_model(**inputs).last_hidden_state,
    atol=1e-6,
    rtol=0.0,
)
```

### Registered-parameter test

```python
names = dict(model.named_parameters())
assert any("attn_residual" in name or "queries" in name for name in names)
```

### Public API smoke test

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_residual_variant="block",
    attn_residual_block_size=8,
    dtype="auto",
)
```

## Symptom-to-cause hints

- `block_size=1` does not match `full`: logical-site indexing or block-state
  bookkeeping is wrong.
- No AttnRes-specific missing-weight warnings at checkpoint load: the mixer may
  not be registered on the model.
- Very high first-step loss compared with baseline: initialization may be too
  aggressive, or the mixer is being created dynamically instead of loaded and
  optimized normally.
- `bf16` training crashes in the mixer path: depth weights and source values
  have different dtypes.
- Baseline path changes when AttnRes is off: variant gating leaked into the
  original forward path.
- Generation fails only when AttnRes is on: final mix placement, cache flow,
  or model-loop orchestration is wrong.
