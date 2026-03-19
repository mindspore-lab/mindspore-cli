# ms-cli

AI infrastructure agent

## Documentation Map

Current documentation in [`docs/`](docs/) is split into:

- shared repository policy: [`docs/ai/contributor-guide.md`](docs/ai/contributor-guide.md)
- current architecture references:
  - [`docs/arch.md`](docs/arch.md)
  - [`docs/ms-cli-arch.md`](docs/ms-cli-arch.md)
- active refactor and workstream plans:
  - [`docs/ms-cli-refactor.md`](docs/ms-cli-refactor.md)
  - [`docs/ms-skills-update-plan.md`](docs/ms-skills-update-plan.md)
  - [`docs/incubating-factory-plan.md`](docs/incubating-factory-plan.md)
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

Run demo mode:

```bash
go run ./cmd/ms-cli --demo
# or
./ms-cli --demo
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

# Use custom config file
./ms-cli --config /path/to/config.yaml

# Set API key directly
./ms-cli --api-key sk-xxx
```

## LLM API Configuration

`ms-cli` supports three provider modes:

- `openai`: native OpenAI API protocol
- `openai-compatible`: OpenAI-compatible protocol (default)
- `anthropic`: Anthropic Messages API protocol

Provider routing is fully configuration-driven (no runtime protocol probing).

### Config file (`mscli.yaml`)

```yaml
model:
  provider: openai-compatible
  url: https://api.openai.com/v1
  model: gpt-4o-mini
  key: ""
```

### Environment variable precedence

- Provider: `MSCLI_PROVIDER` > `model.provider` > default `openai-compatible`
- API key:
  - `openai` / `openai-compatible`: `MSCLI_API_KEY` > `OPENAI_API_KEY` > `model.key`
  - `anthropic`: `ANTHROPIC_AUTH_TOKEN` > `ANTHROPIC_API_KEY` > `model.key`
- Base URL:
  - all providers: `MSCLI_BASE_URL` (highest)
  - `openai` / `openai-compatible`: then `OPENAI_BASE_URL`
  - `anthropic`: then `ANTHROPIC_BASE_URL`
  - then `model.url`
  - then provider default:
    - OpenAI/OpenAI-compatible: `https://api.openai.com/v1`
    - Anthropic: `https://api.anthropic.com/v1/messages`

CLI flags `--api-key` and `--url` are explicit runtime overrides for the current run.

### Use OpenAI API

```bash
export MSCLI_PROVIDER=openai
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini
./ms-cli
```

### Use Anthropic API

```bash
export MSCLI_PROVIDER=anthropic
export ANTHROPIC_AUTH_TOKEN=sk-ant-...
export MSCLI_MODEL=claude-3-5-sonnet
./ms-cli
```

### Use OpenRouter (OpenAI-compatible third-party routing)

OpenRouter uses an OpenAI-compatible interface, so set provider to `openai-compatible`:

```bash
export MSCLI_PROVIDER=openai-compatible
export OPENAI_API_KEY=sk-or-...
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
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

See [`docs/arch.md`](docs/arch.md) and [`docs/ms-cli-arch.md`](docs/ms-cli-arch.md)
for the current architecture and package map.

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
