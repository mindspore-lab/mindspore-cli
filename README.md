# MindSpore CLI

**AI Agent for Training, Debugging, and Development**

An open-source terminal agent that helps ML engineers train models, diagnose failures, migrate code, and optimize performance — powered by LLMs with built-in domain skills for the MindSpore and AI infrastructure ecosystem.

<!-- TODO: Add demo GIF/screenshot here -->

## Why MindSpore CLI?

- **Domain-aware** — Built-in skills for accuracy diagnosis, model migration, operator development, performance analysis, and training readiness
- **Works with any LLM** — OpenAI, Anthropic, DeepSeek, Kimi, OpenRouter, or local models
- **Zero config start** — Free model included, or bring your own API key
- **Full tool access** — Bash, file read/write/edit, grep, glob — all from the chat
- **Permission control** — Safe commands auto-allowed, dangerous ones require approval
- **Session persistence** — Resume where you left off with `mscli --resume`

## Install

```bash
curl -fsSL http://47.115.175.134/mscli/install.sh | bash
```

Or build from source (Go 1.24.2+):

```bash
git clone https://github.com/mindspore-lab/mindspore-cli.git
cd mindspore-cli && go build -o mscli ./cmd/mscli && ./mscli
```

## Quick Start

**Use the free built-in model (zero config):**
```bash
mscli
# Choose "mscli-provided" → "kimi-k2.5 [free]" on first run
```

**Bring your own API key:**
```bash
export MSCLI_API_KEY=sk-...
export MSCLI_MODEL=deepseek-chat
mscli
```

**Use with OpenAI / Anthropic / OpenRouter:**
```bash
# OpenAI
export MSCLI_PROVIDER=openai-completion MSCLI_API_KEY=sk-... MSCLI_MODEL=gpt-4o

# Anthropic
export MSCLI_PROVIDER=anthropic MSCLI_API_KEY=sk-ant-... MSCLI_MODEL=claude-sonnet-4-20250514

# OpenRouter (100+ models)
export MSCLI_PROVIDER=openai-completion MSCLI_API_KEY=sk-or-... MSCLI_BASE_URL=https://openrouter.ai/api/v1

mscli
```

## Features

| | Feature | Description |
|---|---|---|
| | Streaming TUI | Live agent output, tool results, and thinking indicators |
| | `@file` references | `explain @go.mod` — inline workspace files in chat |
| | Tool output viewer | Ctrl+O to expand collapsed results in a scrollable view |
| | Smart permissions | `ls`, `cat`, `git` auto-allowed; `rm -rf` always asks |
| | Context management | `/compact` to free space, auto-compaction when needed |
| | Session resume | `mscli --resume` to continue the last conversation |
| | Multi-line input | Ctrl+J or `\`+Enter for newlines without submitting |

## Built-in Skills

| Skill | What it does |
|---|---|
| accuracy-agent | Diagnose numerical drift, precision mismatches, cross-platform accuracy regressions |
| algorithm-agent | Adapt algorithms from papers or reference implementations into existing models |
| api-helper | Answer MindSpore API questions (tensor ops, forward/backward, CPU/GPU) |
| failure-agent | Diagnose training and runtime failures across MindSpore and PyTorch |
| model-agent | Migrate model implementations into the MindSpore ecosystem |
| operator-agent | Build custom operators through framework-native or integration approaches |
| performance-agent | Diagnose latency, throughput, memory, dataloader, and communication bottlenecks |
| readiness-agent | Check whether a workspace is ready to train or run inference |

## Commands

| Command | Description |
|---|---|
| `/model` | Switch model or provider |
| `/compact` | Free up context space |
| `/clear` | Fresh conversation |
| `/permissions` | Manage tool access |
| `/diagnose` | Investigate issues |
| `/fix` | Apply fixes |
| `/bugs` | Manage bugs |
| `/issues` | Manage issues |
| `/report` | Submit feedback |
| `/help` | Show all commands |

## Documentation

- [Architecture](docs/arch.md)
- [Contributor Guide](docs/agent-contributor-guide.md)
- [Feature Backlog](docs/features-backlog.md)

## Contributing

See the [Contributor Guide](docs/agent-contributor-guide.md) for code style, dependency rules, and testing conventions.

## License

Apache-2.0 — see [LICENSE](LICENSE).
