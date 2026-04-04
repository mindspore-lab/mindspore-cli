# mscli

AI Infra Agent

## Install

### One-liner (public mirror)

```bash
curl -fsSL http://47.115.175.134/mscli/install.sh | bash
```

Optional overrides:

```bash
# Force one source instead of auto-probing.
MSCLI_INSTALL_SOURCE=github curl -fsSL http://47.115.175.134/mscli/install.sh | bash
MSCLI_INSTALL_SOURCE=mirror curl -fsSL http://47.115.175.134/mscli/install.sh | bash

# Override the mirror base URL if you host your own Caddy/Nginx mirror.
MSCLI_MIRROR_BASE_URL=http://47.115.175.134/mscli/releases curl -fsSL http://47.115.175.134/mscli/install.sh | bash
```

### One-liner (GitHub raw, internal use)

```bash
curl -fsSL https://raw.githubusercontent.com/vigo999/mindspore-code/main/scripts/install.sh | bash
```

### Build from source

Requires Go 1.24.2+.

```bash
git clone https://github.com/vigo999/mindspore-code.git
cd mindspore-code
go build -o mscli ./cmd/mscli
./mscli
```

## Quick Start

```bash
# Set your LLM API key
export MSCLI_API_KEY=sk-...

# Run
mscli
```

### Slash Commands

| Command | Description |
|---------|-------------|
| `/project` | Show project status (overview, milestones, tasks, support) |
| `/project add tasks "title" --owner X --due 03-25` | Add a task (admin) |
| `/project add milestones "title" --progress 50` | Add a milestone (admin) |
| `/project add support "feature name"` | Add a support entry (admin) |
| `/project update <id> --progress 80` | Update a task by ID (admin) |
| `/project update "title" --progress 80` | Update a milestone by title (admin) |
| `/project rm <id\|title>` | Remove an item (admin) |
| `/project overview --phase X --owner Y` | Edit project overview (admin) |
| `/bugs` | List bugs |
| `/report <title>` | Report a bug |
| `/claim <id>` | Claim a bug |
| `/dock` | Show bug dashboard |
| `/login <token>` | Login to the bug/project server |
| `/model <provider:model>` | Switch LLM model |
| `/clear` | Clear chat |

### `@file` Input Expansion

`ms-cli` supports inline file expansion for plain chat input and these text-tail commands:

- `/report`
- `/diagnose`
- `/fix`
- `/skill <name> ...`
- direct skill aliases such as `/pdf ...`

Use a standalone workspace-relative token like `@docs/bug.md` to inline a text file into the prompt.
Use `@@name` to keep a literal `@name` token.

v1 limits:

- only standalone whitespace-delimited `@relative/path` tokens are expanded
- paths with spaces are not supported
- files must stay inside the current workspace
- files must be UTF-8 text without NUL bytes
- files larger than `64 KiB` are rejected
- any invalid `@file` reference fails the whole input

### Server Setup

The bug and project server runs separately:

```bash
# Build and deploy
go build -o /opt/mscli/ms-cli-server ./cmd/ms-cli-server

# Start with config
cd /opt/mscli && ./ms-cli-server -config ./server.yaml
```

See `configs/server.yaml` for server configuration (auth tokens, database, listen address).

## Documentation

- Architecture: [`docs/arch.md`](docs/arch.md)
- Contributor guide: [`docs/agent-contributor-guide.md`](docs/agent-contributor-guide.md)
- Implementation plans: [`docs/impl-guide/`](docs/impl-guide/)
- Feature backlog: [`docs/features-backlog.md`](docs/features-backlog.md)

### Command-Line Options

```bash
# Select URL and model
./ms-cli --url https://api.openai.com/v1 --model gpt-4o

# Set API key directly
./ms-cli --api-key sk-xxx
```

## LLM API Configuration

`ms-cli` supports three provider modes:

- `openai-completion`: OpenAI Chat Completions API and compatible gateways (default)
- `openai-responses`: OpenAI Responses API
- `anthropic`: Anthropic Messages API protocol

Provider routing is fully configuration-driven (no runtime protocol probing).

### Configuration sources

Runtime config now comes from:

1. built-in defaults
2. environment variables: `MSCLI_*`
3. session overrides (`/model` in the current process only, not persisted)

`~/.ms-cli/config.yaml` and `./.ms-cli/config.yaml` are no longer read.
The shared skills repo URL and branch are built into the binary.

### Environment variables

Use unified `MSCLI_*` names:

- `MSCLI_PROVIDER`
- `MSCLI_MODEL`
- `MSCLI_API_KEY`
- `MSCLI_BASE_URL`
- `MSCLI_TEMPERATURE`
- `MSCLI_MAX_TOKENS`
- `MSCLI_TIMEOUT`
- `MSCLI_CONTEXT_WINDOW`

CLI flags `--api-key`, `--url`, `--model` are startup overrides for the current run.



### Model token defaults (auto + override)

When `model.model` matches known families (`gpt-5` ~ `gpt-5.4`, `claude-4.5` ~ `claude-4.6`, `glm-4.7*`, `glm-5*`, `kimi-k2*`, `kimi-k2.5*`, `minimax-m2.5*`, `minimax-m2.7*`, `deepseek*`, `qwen3*`, `qwen3.5*`), ms-cli auto-fills:

- `model.max_tokens`
- `context.window`

Precedence is:

1. `MSCLI_MAX_TOKENS` / `MSCLI_CONTEXT_WINDOW`
2. auto defaults from model name
3. built-in defaults
### Use OpenAI API

```bash
export MSCLI_PROVIDER=openai-completion
export MSCLI_API_KEY=sk-...
export MSCLI_MODEL=gpt-4o-mini
./mscli
```

If you specifically want the Responses API path, use `openai-responses`.

### Use Anthropic API

```bash
export MSCLI_PROVIDER=anthropic
export MSCLI_API_KEY=sk-ant-...
export MSCLI_MODEL=claude-3-5-sonnet
./mscli
```

### Use OpenRouter (OpenAI-compatible third-party routing)

OpenRouter uses an OpenAI-compatible interface, so set provider to `openai-completion`:

```bash
export MSCLI_PROVIDER=openai-completion
export MSCLI_API_KEY=sk-or-...
export MSCLI_BASE_URL=https://openrouter.ai/api/v1
export MSCLI_MODEL=anthropic/claude-3.5-sonnet
./mscli
```
