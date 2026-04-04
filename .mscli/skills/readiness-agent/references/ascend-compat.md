# Ascend Compatibility Reference

This is the local compatibility lookup used by `readiness-agent` when it needs
to validate or repair MindSpore or PTA packages against a detected CANN
version.

Use this file after the system layer is healthy. Do not guess package versions
when the selected CANN / Python tuple is unresolved.

## MindSpore on Ascend

| CANN | MindSpore | Python | Typical Use |
|------|-----------|--------|-------------|
| 8.5.0 | 2.8.0 | 3.9-3.12 | Latest published MindSpore row from the official versions page |
| 8.5.0 | 2.7.2 | 3.9-3.12 | Stable 8.5.0 line from the official versions page |
| 8.3.RC1 | 2.8.0 | 3.9-3.12 | Current RC line from the official versions page |
| 8.3.RC1 | 2.7.1 | 3.9-3.11 | Current RC line from the official versions page |
| 8.2.RC1 | 2.7.0 | 3.9-3.11 | Current RC line from the official versions page |
| 8.2.RC1 | 2.7.0-rc1 | 3.9-3.11 | Current RC line from the official versions page |

Policy:
- resolve MindSpore from the exact detected CANN row and selected Python
- prefer the first compatible row in this local table
- if the local table cannot classify the tuple, do not silently auto-install

## PyTorch + torch_npu on Ascend

Normalize versions before lookup:
- strip a leading `v` if present
- compare `torch` on `major.minor.patch`
- compare `torch_npu` on the full version including `.postN`
- treat `torch==2.6.0+cpu` as `2.6.0` for compatibility lookup

| CANN | torch | torch_npu | Python | Github Branch | Typical Use |
|------|-------|-----------|--------|---------------|-------------|
| 8.5.0 | 2.9.0 | 2.9.0 | 3.9-3.11 | v2.9.0-7.3.0 | Latest published PTA line in the upstream compatibility table |
| 8.5.0 | 2.8.0 | 2.8.0.post2 | 3.9-3.11 | v2.8.0-7.3.0 | Newer Ascend stacks |
| 8.5.0 | 2.7.1 | 2.7.1.post2 | 3.9-3.11 | v2.7.1-7.3.0 | Newer Ascend stacks |
| 8.5.0 | 2.6.0 | 2.6.0.post5 | 3.9-3.11 | v2.6.0-7.3.0 | Newer Ascend stacks |
| 8.3.RC1 | 2.8.0 | 2.8.0 | 3.9-3.11 | v2.8.0-7.2.0 | Current RC line |
| 8.3.RC1 | 2.7.1 | 2.7.1 | 3.9-3.11 | v2.7.1-7.2.0 | Current RC line |
| 8.3.RC1 | 2.6.0 | 2.6.0.post3 | 3.9-3.11 | v2.6.0-7.2.0 | Current RC line |
| 8.3.RC1 | 2.1.0 | 2.1.0.post17 | 3.8-3.11 | v2.1.0-7.2.0 | Legacy compatibility |
| 8.2.RC1 | 2.6.0 | 2.6.0 | 3.9-3.11 | v2.6.0-7.1.0 | Transitional line |
| 8.2.RC1 | 2.5.1 | 2.5.1.post1 | 3.9-3.11 | v2.5.1-7.1.0 | Transitional line |
| 8.2.RC1 | 2.1.0 | 2.1.0.post13 | 3.8-3.11 | v2.1.0-7.1.0 | Legacy compatibility |
| 8.1.RC1 | 2.5.1 | 2.5.1 | 3.9-3.11 | v2.5.1-7.0.0 | Common production baseline |
| 8.1.RC1 | 2.4.0 | 2.4.0.post4 | 3.8-3.11 | v2.4.0-7.0.0 | Common production baseline |
| 8.1.RC1 | 2.3.1 | 2.3.1.post6 | 3.8-3.11 | v2.3.1-7.0.0 | Common production baseline |
| 8.1.RC1 | 2.1.0 | 2.1.0.post12 | 3.8-3.11 | v2.1.0-7.0.0 | Older but still common |

Policy:
- resolve PTA from the exact detected CANN row and selected Python
- prefer the first compatible row in this local table
- when the tuple is unresolved, stop auto-remediation instead of guessing
