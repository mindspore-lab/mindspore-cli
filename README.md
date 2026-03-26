# mindspore-cli

AI Infra Agent

## Install

### One-liner (recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/vigo999/ms-cli/main/scripts/install.sh | bash
```

### Build from source

Requires Go 1.24.2+.

```bash
git clone https://github.com/vigo999/ms-cli.git
cd ms-cli
go build -o mscli ./cmd/ms-cli
./mscli
```

## Quick Start

```bash
# Set your LLM API key
export MSCLI_API_KEY=sk-...

# Run
mscli
```

### Use OpenAI API

```bash
export MSCLI_PROVIDER=openai-completion
export MSCLI_API_KEY=sk-...
export MSCLI_MODEL=gpt-4o-mini
./ms-cli
```

If you specifically want the Responses API path, use `openai-responses`.

### Use Anthropic API

```bash
export MSCLI_PROVIDER=anthropic
export MSCLI_API_KEY=sk-ant-...
export MSCLI_MODEL=claude-3-5-sonnet
./ms-cli
```

### Use OpenRouter (OpenAI-compatible third-party routing)

OpenRouter uses an OpenAI-compatible interface, so set provider to `openai-completion`:

```bash
export MSCLI_PROVIDER=openai-completion
export MSCLI_API_KEY=sk-or-...
export MSCLI_BASE_URL=https://openrouter.ai/api/v1
export MSCLI_MODEL=anthropic/claude-3.5-sonnet
./ms-cli
```
