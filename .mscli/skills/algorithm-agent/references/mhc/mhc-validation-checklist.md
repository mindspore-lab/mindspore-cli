# mHC Validation Checklist

Use this checklist when the selected `algorithm-agent` route is
`mhc`.

## Minimum validation

1. Config validation rejects invalid values.
2. `use_mhc=False` preserves the original forward path.
3. `use_mhc=True` still returns `[batch, seq, hidden]`.
4. Residual mappings are approximately doubly stochastic:
   - row sums close to `1`
   - column sums close to `1`
5. `from_pretrained(..., use_mhc=True)` or the equivalent public load API
   works.

## Strongly recommended validation

1. Run a forward pass with caching or generation enabled.
2. Run a tiny train or finetune smoke test and confirm loss decreases or
   remains finite.
3. Run the model in the dtypes you care about, especially `bf16`.
4. Exercise any model-specific attention variants such as sliding-window
   attention, grouped-query attention, or custom backends.
5. If the model supports gradient checkpointing, confirm the mHC path works
   with it enabled.

## Example test ideas

### Shape test

```python
config.use_mhc = True
model = Model(config)
result = model(**inputs)
assert result.last_hidden_state.shape == (batch, seq, hidden)
```

### Doubly-stochastic test

```python
hidden_states = model._expand_mhc_streams(model.embed_tokens(input_ids))
_, _, residual_mapping = model.layers[0].attn_mhc.compute_mappings(hidden_states)
torch.testing.assert_close(residual_mapping.sum(dim=-1), torch.ones_like(...), atol=1e-3, rtol=1e-3)
torch.testing.assert_close(residual_mapping.sum(dim=-2), torch.ones_like(...), atol=1e-3, rtol=1e-3)
```

### Public API smoke test

```python
model = AutoModelForCausalLM.from_pretrained(model_name, use_mhc=True, dtype="auto")
```

## Symptom-to-cause hints

- Attention receives 4-D input: stream reduction was skipped before attention
  or MLP.
- LM head receives 4-D input: stream reduction was skipped before final norm or
  task head.
- Rows sum to `1` but columns do not: Sinkhorn implementation is incomplete or
  not iterated enough.
- Loss diverges early: initialization is not near identity, output reduction
  mismatches checkpoint scale, or mapping math runs in low precision.
- Generation fails only with cache: cache or position metadata was coupled to
  the expanded residual tensor instead of base token positions.
