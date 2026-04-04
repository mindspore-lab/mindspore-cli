# Profiler Injection Templates

Read this file when the user needs fresh profiler collection and you must show
or automate how profiler code should be inserted into a copied `*-perf.py`
script.

## Goal

Keep profiler injection deterministic.

Do not ask the model to "figure out somewhere reasonable" to inject profiler
code. Match the copied script against a known template first. If there is no
clean match, stop and ask for manual confirmation of the insertion point.

## General Rules

- Always copy the original script to `<stem>-perf.py` before making profiler
  edits.
- Do not modify the original script in place unless the user explicitly asks
  for that.
- Preserve the original CLI, argument parsing, imports, and training logic.
- Add only the minimum profiler imports and statements required for collection.
- Prefer AST-based or structure-aware edits over raw string replacement.
- If the script already contains profiler hooks, do not inject a second set.
- Remove or disable conflicting accuracy-collection hooks in the copied script
  before enabling profiler collection.

## Template A: MindSpore Explicit Training Loop

Use this template when the copied script has an explicit loop such as:

```python
for data, label in train_dataset:
    train_step(data, label)
```

Injection rules:

1. Add profiler imports:

```python
from mindspore.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler
```

2. Wrap the training loop with:

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
    schedule=schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
    on_trace_ready=tensorboard_trace_handler("./profiling_data"),
) as prof:
```

3. Keep the original training statements inside the `with` block.

4. Insert:

```python
prof.step()
```

after the per-iteration training step and before the next iteration begins.

Expected result shape:

```python
with profile(...) as prof:
    for data, label in train_dataset:
        train_step(data, label)
        prof.step()
```

## Template B: `torch_npu` Explicit Training Loop

Use this template when the copied script has a PyTorch-style loop such as:

```python
for batch in loader:
    ...
    loss.backward()
    optimizer.step()
```

Injection rules:

1. Add profiler imports:

```python
from torch_npu.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler
```

2. Wrap the training loop with:

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
    schedule=schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=1),
    on_trace_ready=tensorboard_trace_handler("./profiling_data"),
) as prof:
```

3. Keep the original loop body intact.

4. Insert:

```python
prof.step()
```

after `optimizer.step()` or the iteration's last training update.

Expected result shape:

```python
with profile(...) as prof:
    for batch in loader:
        ...
        optimizer.step()
        prof.step()
```

## Template C: Generic `__main__` Entrypoint Wrapper

Use this template when a safe explicit loop cannot be identified but the copied
script has a clear:

```python
if __name__ == "__main__":
    main()
```

or comparable entrypoint call.

Injection rules:

1. Add the stack-matching profiler imports.
2. Wrap the `__main__` body in:

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
    on_trace_ready=tensorboard_trace_handler("./profiling_data"),
) as prof:
    main()
```

3. Do not add `prof.step()` in this fallback mode.

Use this for inference or launcher-style scripts when whole-entry execution is
safer than guessing the real iteration loop.

## Unsupported for Automatic Injection: MindSpore `model.train(...)` Entry

This shape is less deterministic than an explicit loop because the safe
insertion point depends on the framework API used by the project.

Treat this shape as unsupported for automatic injection in the current design.

Rules:

- Do not guess a callback or wrapper API.
- Do not rewrite `model.train(...)` into a custom manual loop unless the user
  explicitly approves changing execution style.
- Guide the user to modify the copied `*-perf.py` script manually for profiler
  collection, or ask the user to expose a loop-level training entry first.

## Stop Conditions

Stop and avoid automatic injection when any of these is true:

- no clear explicit training loop is found
- multiple nested loops could own the real training step
- the script already contains profiling logic
- the script is generated code or heavily macro-driven
- the training entry is hidden behind another launcher and the true Python
  entry is still unknown

## Review Checklist

Before accepting an injected `*-perf.py`, confirm:

- original script still exists unchanged
- copied script name ends with `-perf.py`
- profiler import matches the detected stack
- profiler schedule matches the documented stack default
- `prof.step()` is present exactly once per training iteration
- output path uses `tensorboard_trace_handler("./profiling_data")`
- no duplicate or conflicting profiling hooks remain
