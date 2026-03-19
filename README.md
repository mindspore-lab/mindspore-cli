# ms-cli

AI infrastructure agent

## Documentation Map

Current documentation in [`docs/`](docs/) is split into:

- shared repository policy: [`docs/agent-contributor-guide.md`](docs/agent-contributor-guide.md)
- current architecture references:
  - [`docs/arch.md`](docs/arch.md)
- active refactor and workstream plans:
  - [`docs/impl-guide/ms-cli-refactor-3.md`](docs/impl-guide/ms-cli-refactor-3.md)
  - [`docs/impl-guide/ms-skills-whole-update-plan.md`](docs/impl-guide/ms-skills-whole-update-plan.md)
  - [`docs/impl-guide/ms-factory-struct-v0.1.md`](docs/impl-guide/ms-factory-struct-v0.1.md)
  - [`docs/features-backlog.md`](docs/features-backlog.md)
  - [`docs/how-to-provide-plan-proposal.md`](docs/how-to-provide-plan-proposal.md)

Important:

- architecture docs describe the current checkout
- refactor/workstream docs describe planned target states
- if they conflict, treat the current code as authoritative

## Prerequisites

- Go 1.24.2+ (see `go.mod`)

## Quick Start

Build:

```bash
go build -o ms-cli ./cmd/ms-cli
```

Run real mode:

```bash
go run ./cmd/ms-cli
# or
./ms-cli
```

### Command-Line Options

```bash
# Select URL and model
./ms-cli --url https://api.openai.com/v1 --model gpt-4o

# Set API key directly
./ms-cli --api-key sk-xxx
```

## LLM API Configuration

`ms-cli` supports three provider modes:

- `openai`: native OpenAI API protocol
- `openai-compatible`: OpenAI-compatible protocol (default)
- `anthropic`: Anthropic Messages API protocol

Provider routing is fully configuration-driven (no runtime protocol probing).

### Config files

Layered merge (low -> high):

1. built-in defaults
2. user config: `~/.mscli/config.yaml`
3. project config: `./.mscli/config.yaml`
4. environment variables: `MSCLI_*`
5. session overrides (`/model` in current process only, not persisted)

Each higher layer overrides only the fields it sets.

```yaml
model:
  provider: openai-compatible
  url: https://api.openai.com/v1
  model: gpt-4o-mini
  key: ""
```

### Environment variables

Use unified `MSCLI_*` names:

- `MSCLI_PROVIDER`
- `MSCLI_MODEL`
- `MSCLI_API_KEY`
- `MSCLI_BASE_URL`
- `MSCLI_TEMPERATURE`
- `MSCLI_MAX_TOKENS`
- `MSCLI_TIMEOUT`

CLI flags `--api-key`, `--url`, `--model` are startup overrides for the current run.

### Use OpenAI API

```bash
export MSCLI_PROVIDER=openai
export MSCLI_API_KEY=sk-...
export MSCLI_MODEL=gpt-4o-mini
./ms-cli
```

### Use Anthropic API

```bash
export MSCLI_PROVIDER=anthropic
export MSCLI_API_KEY=sk-ant-...
export MSCLI_MODEL=claude-3-5-sonnet
./ms-cli
```

### Use OpenRouter (OpenAI-compatible third-party routing)

OpenRouter uses an OpenAI-compatible interface, so set provider to `openai-compatible`:

```bash
export MSCLI_PROVIDER=openai-compatible
export MSCLI_API_KEY=sk-or-...
export MSCLI_BASE_URL=https://openrouter.ai/api/v1
export MSCLI_MODEL=anthropic/claude-3.5-sonnet
./ms-cli
```

You can also set custom headers in `model.headers` in config when required by a gateway.

### In-session model/provider switch

Inside CLI:

- `/model gpt-4o-mini` (switch model, keep current provider)
- `/model openai:gpt-4o`
- `/model openai-compatible:gpt-4o-mini`
- `/model anthropic:claude-3-5-sonnet`

## Repository Structure

See [`docs/arch.md`](docs/arch.md) for the current architecture and package map.

The repository is under active refactor, so this README intentionally does not
duplicate a full package tree. Use the linked architecture docs above as the
source of truth for either:

- the current checkout layout, or
- explicitly labeled target-state planning docs under [`docs/`](docs/)

## Known Limitations

- The real-mode engine flow is still minimal/stub-oriented.
- Running Bubble Tea in non-interactive shells may fail with `/dev/tty` errors.

## Planning Workstreams

The repository currently tracks three related planning streams:

- Workstream A: `ms-cli` refactor into a thinner agent runtime
- Workstream B: `ms-skills` update for prompt-oriented domain skills
- Workstream C: incubating Factory schemas, cards, and pack format

These plans live under [`docs/`](docs/) and are intended to guide staged
implementation across `ms-cli`, `ms-skills`, and the future Factory split.

## Architecture Rule

UI listens to events; agent loop emits events; tool execution does not depend on UI.
