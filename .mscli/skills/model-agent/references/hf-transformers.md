# HF Transformers Route

Use this route when the source clearly belongs to Hugging Face transformers.

This route migrates Hugging Face transformers models into
`mindone.transformers` with a standard workflow, tool-assisted conversion,
manual MindSpore adaptation, and registration updates.

Top-level `model-agent` should keep the route boundary explicit here instead of
forcing the user to choose it up front.

## When to Use

- Porting Hugging Face transformers models such as LLaMA, BERT, GPT, or Qwen
  to MindSpore-oriented targets.
- Migrating a transformers-family source repo into `mindone.transformers` with
  repo-specific rules.
- Adding a new transformer architecture to a MindOne-style transformers tree.

## Repository Assumptions

Source repository:

- Hugging Face `transformers`
- Core path: `transformers/src/transformers`
- Model tests: `transformers/tests/models`

Target repository:

- `mindone`
- Core path: `mindone/mindone/transformers`
- Model tests: `mindone/tests/transformers_tests`

## Route Workflow

### 1. Intake and target confirmation

Collect these inputs before editing files:

1. Identify the source and target repository paths and check that the source is
   genuinely transformers-family.
2. Identify the exact `{model_name}` from the prompt or workspace evidence.
3. Map the default source and target model directories:
   - source: `transformers/src/transformers/models/{model_name}/`
   - target: `mindone/mindone/transformers/models/{model_name}/`

Prioritize `model_name` precision. If the name is ambiguous or incomplete,
pause and confirm the exact model family or version before proceeding.

### 2. Modeling files migration with the auto-convert tool

Copy only the intended files from source to target:

- `modeling_*.py`
- `processing_*.py`
- `image_processing_*.py`
- `video_processing_*.py`

Only migrate processing files when they actually manipulate torch tensors.
Otherwise, prefer reusing the Hugging Face implementation directly.

Run the route-specific single-file in-place conversion before any manual edits:

```bash
python skills/model-agent/scripts/hf_transformers_auto_convert.py \
  --src_file path/to/file.py --inplace
```

Install the tool dependency first when needed:

```bash
pip install -r skills/model-agent/scripts/hf_transformers_auto_convert.requirements.txt
```

You must run the auto-convert script before manual edits on migrated modeling
files.

### 3. Manual fix checklist

#### 3.1 Structural and API changes

- `torch.nn.Module` -> `mindspore.nn.Cell`
- `forward` -> `construct`
- `torch.nn.Parameter` -> `mindspore.Parameter`
- Replace `torch` and `torch.nn.functional` usages with `mindspore.mint` or
  `mindspore.ops`
- Prefer `mindspore.mint`, then `mindspore.ops`, then `mindspore.nn`
- Treat the auto-converted `mindspore.mint` and `mindspore.mint.nn` forms as
  the default steady-state implementation for this route
- Do not bulk-rewrite auto-converted `mint.*` or `mint.nn.*` calls back into
  `mindspore.nn.*`, `ops.*`, or older MindSpore-style APIs just for style
  consistency
- Only replace a `mint` form when there is a concrete repo-local reason such as
  an existing model-family convention, a missing API, or a verified behavioral
  incompatibility
- Drop unsupported `inplace=True` arguments

#### 3.2 Device handling cleanup

- Remove `.to(device)`, `.cuda()`, `torch.device`, and `mps` branches
- Check function signatures and remove device-only parameters when they are no
  longer needed
- Do not remove `register_buffer` just because the original code had device
  handling nearby; `register_buffer` remains valid in MindSpore

Example pattern:

Before:

```python
def _dynamic_frequency_update(self, position_ids, device):
    seq_len = mint.max(position_ids) + 1
    if seq_len > self.max_seq_len_cached:
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = seq_len

    if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:
        self.original_inv_freq = self.original_inv_freq.to(device)
        self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
        self.max_seq_len_cached = self.original_max_seq_len
```

After:

```python
def _dynamic_frequency_update(self, position_ids):
    seq_len = mint.max(position_ids) + 1
    if seq_len > self.max_seq_len_cached:
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, seq_len=seq_len)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = seq_len

    if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:
        self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
        self.max_seq_len_cached = self.original_max_seq_len
```

#### 3.3 Imports and decorators

- Keep config and tokenizer imports from Hugging Face `transformers`
- Use `mindone.transformers.modeling_utils` for modeling utilities
- Remove unused or PyTorch-only imports that are not migrated
- Remove decorators only when they are not defined in `mindone.transformers`
- If a decorator is removed, remove both its import and usage

Typical decorators that may need removal:

- `@deprecate_kwarg`
- `@auto_docstring`
- `@torch.jit.script`
- `@use_kernel_func_from_hub`

#### 3.4 Tensors and shapes

- Use `mindspore.Tensor` in docstrings
- Wrap shape arguments in tuples such as `.view((b, s, h))`

### 4. Registration and exports

Update the target repo registration chain:

- `mindone/mindone/transformers/models/auto/configuration_auto.py`
- `mindone/mindone/transformers/models/auto/modeling_auto.py`
- `mindone/mindone/transformers/models/auto/processing_auto.py` when processor
  files are migrated
- `mindone/mindone/transformers/models/auto/image_processing_auto.py` when
  image processing files are migrated
- `mindone/mindone/transformers/models/auto/video_processing_auto.py` when
  video processing files are migrated
- `mindone/mindone/transformers/models/{model_name}/__init__.py`
- `mindone/mindone/transformers/models/__init__.py`
- `mindone/mindone/transformers/__init__.py`

For `mindone/mindone/transformers/models/{model_name}/__init__.py`:

- Preserve the file header comment exactly
- After the header, keep only `from .<module> import *` lines
- Remove all other non-header lines
- Verify that each referenced module exists locally
- Drop import lines that point to missing modules

Use Hugging Face auto files as a reference for insertion order.

### 5. Done criteria

- The minimal import validation succeeds via a `from transformers import xxx`
  style import path for the migrated target, and the model imports cleanly in
  MindOne
- Auto mappings and exports are updated
- Verification artifacts or next test commands are recorded

Do not mark the migration complete before the `from transformers import xxx`
minimal import validation has passed for the migrated target.

## Route Guardrails

- Do not migrate `configuration_*.py`, `tokenization_*.py`, or
  `*moduler_*.py`
- Reuse Hugging Face configuration and tokenization implementations directly
  unless the target repo has a compelling local requirement
- Keep changes minimal and aligned with existing MindOne patterns
- Avoid adding custom compatibility wrappers unless they are required
- Use diff-based insertion when updating auto maps

Load the route-specific companion references for environment notes and repo
guardrails:

- `references/hf-transformers-guardrails.md`
- `references/hf-transformers-env.md`

## Route Output

Report at least:

- files changed and why
- tests run, tests generated, or tests recommended
- remaining TODOs and risks
