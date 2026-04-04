# Attention Residuals Implementation Pattern

Use this reference when the selected `algorithm-agent` route is
`attnres`.

Treat Attention Residuals as a replacement for fixed residual addition around
attention and MLP sites, not as a new token-attention kernel.

## Core idea

Replace fixed residual accumulation with learned retrieval over depth:

```text
h_i = sum_j softmax(q_i · norm(v_j)) * v_j
```

- `q_i`: learned pseudo-query for one logical residual site
- `norm(v_j)`: key-side normalization, usually RMSNorm
- `v_j`: embedding or prior residual outputs
- softmax runs over depth, not sequence or head dimensions

## Route choices

Two variants are standard:

- `full`
  every logical residual site attends over `embedding + all prior logical outputs`
- `block`
  every logical residual site attends over `completed_blocks + partial_block`

Practical guidance:

- prefer `block` for real training runs
- use `full` as a reference implementation
- verify `block_size=1` against `full`

## Shapes

For decoder-only transformers, values usually stay in the base hidden shape:

- hidden state: `[B, S, D]`
- stacked sources: `[L, B, S, D]`
- depth logits: `[L, B, S]`
- mixed output: `[B, S, D]`

No extra stream dimension is introduced. Attention and MLP still receive
ordinary `[B, S, D]` tensors.

## Generic module sketch

```python
class AttentionResidualMixer(nn.Module):
    def __init__(self, num_queries, hidden_size, eps):
        self.key_norms = nn.ModuleList([RMSNorm(hidden_size, eps=eps) for _ in range(num_queries)])
        self.queries = nn.Parameter(torch.zeros(num_queries, hidden_size))

    def mix(self, query_index, source_values):
        sources = torch.stack(list(source_values), dim=0)              # [L, B, S, D]
        keys = self.key_norms[query_index](sources)
        query = self.queries[query_index].to(dtype=sources.dtype)
        logits = torch.einsum("d,l...d->l...", query, keys)
        weights = F.softmax(logits, dim=0).to(dtype=sources.dtype)
        return torch.einsum("l...,l...d->...d", weights, sources)
```

Keep the mixer as a registered model module. Do not create it inside
`forward`.

## Full AttnRes pattern

```python
class FullAttentionResidual(nn.Module):
    def __init__(self, num_logical_layers, hidden_size, eps):
        self.mixer = AttentionResidualMixer(num_logical_layers + 1, hidden_size, eps)

    def layer_input(self, embedding, prior_outputs, layer_index):
        return self.mixer.mix(layer_index, [embedding, *prior_outputs])

    def final_output(self, embedding, all_outputs):
        return self.mixer.mix(num_logical_layers, [embedding, *all_outputs])
```

## Block AttnRes state pattern

```python
@dataclass
class BlockState:
    completed_blocks: list[Tensor]
    partial_block: Tensor | None
    completed_layers: int
    block_size: int

    def current_sources(self):
        return self.completed_blocks if self.partial_block is None else [*self.completed_blocks, self.partial_block]

    def append_layer_output(self, layer_output):
        self.partial_block = layer_output if self.partial_block is None else self.partial_block + layer_output
        self.completed_layers += 1
        if self.completed_layers % self.block_size == 0:
            self.completed_blocks.append(self.partial_block)
            self.partial_block = None
```

Important:

- count logical residual sites, not transformer blocks
- in many decoder stacks, one block contributes two logical sites:
  attention and MLP

## Decoder block integration

Keep the baseline path intact:

```python
if attn_residual_variant == "none":
    residual = hidden_states
    hidden_states = input_norm(hidden_states)
    hidden_states = self_attn(hidden_states, ...)
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = post_attn_norm(hidden_states)
    hidden_states = mlp(hidden_states)
    return residual + hidden_states
```

Add site-level helpers for the AttnRes path:

```python
def attn_output(self, hidden_states, ...):
    return self.self_attn(self.input_layernorm(hidden_states), ...)

def mlp_output(self, hidden_states):
    return self.mlp(self.post_attention_layernorm(hidden_states))
```

Drive the AttnRes orchestration from the model-level loop, not from the
attention kernel.

## Model-level integration

For `full`:

```python
prior_outputs = []
for layer_idx, layer in enumerate(self.layers):
    attn_input = attnres.layer_input(embedding, prior_outputs, 2 * layer_idx)
    attn_output = layer.attn_output(attn_input, ...)
    prior_outputs.append(attn_output)

    mlp_input = attnres.layer_input(embedding, prior_outputs, 2 * layer_idx + 1)
    mlp_output = layer.mlp_output(mlp_input)
    prior_outputs.append(mlp_output)

hidden_states = attnres.final_output(embedding, prior_outputs)
```

For `block`, replace `prior_outputs` with block state:

```python
state = attnres.init_state(embedding)
attn_input = attnres.layer_input(state, logical_layer_index)
attn_output = layer.attn_output(attn_input, ...)
attnres.append_layer_output(state, attn_output)
```

## Initialization rules

- zero-initialize pseudo-queries so the initial mixer is close to uniform
  averaging
- initialize key-side norm weights identity-like
- load baseline checkpoint weights as usual
- expect only AttnRes-specific parameters to be missing at checkpoint load

## Porting rules

- keep the baseline path behind config gating
- do not change public hidden size
- keep cache, RoPE, masks, and attention backends operating on the normal
  `[B, S, D]` hidden state
- if the model has a generated and a source modular file, patch the source
  file and regenerate according to repository workflow
- create mixer modules in model construction code so they are visible to
  `from_pretrained`, `state_dict`, and the optimizer

## Common failure modes

- counting transformer blocks instead of logical sites
- mixing over the wrong sources in block mode
- creating the mixer inside `forward` so it is recreated every call
- seeing no AttnRes-specific missing weights at load time because the mixer
  was never registered
- dtype mismatch in the mixer path, especially `softmax(float32)` weights with
  `bf16` values inside `einsum`
- interpreting a very high first-step loss as benign initialization noise when
  the real problem is an unregistered or dynamically created mixer
