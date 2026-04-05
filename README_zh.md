[English](README.md) | 中文

# MindSpore Model Agent

MindSpore Model Agent 是一个面向 MindSpore 生态的、聚焦模型训练场景的 AI agent solution。它面向模型训练周边高频工程工作而设计，适用于那些仅靠通用代码生成还不够、还需要训练领域专项能力支持的场景。

它由两个紧密相关的部分组成：

- [`mindspore-skills`](https://github.com/mindspore-lab/mindspore-skills)：面向模型训练与调试任务的领域能力层，提供可复用的技能，包括 readiness 检查、failure diagnosis、accuracy analysis、performance analysis、model migration、algorithm adaptation 和 operator implementation。这些 skills 不仅可用于 MindSpore Model Agent，也可以与 Claude Code、OpenCode、Codex 等其他 agentic CLI 环境配合使用。
- `mindspore-cli`：MindSpore Model Agent 的官方 CLI。它与相关 skills 有更好的集成，并针对模型训练场景进行了优化，提供更统一的端到端训练任务交互体验。

## v0.1.0 新增内容

- MindSpore Model Agent 首个公开版本发布
- 引入 `mindspore-skills` 作为面向模型训练与调试任务的可复用能力层
- 引入 `mindspore-cli` 作为面向模型训练工作流优化的官方 CLI

## MindSpore CLI

MindSpore CLI 是 MindSpore Model Agent 的官方端到端交互入口。它面向训练任务工作流提供统一的 CLI 体验，并与方案背后的相关 skills 做更紧密的集成。

## Installation

### 脚本安装

```bash
curl -fsSL https://raw.githubusercontent.com/mindspore-lab/mindspore-cli/main/scripts/install.sh | bash
```

### 从源码构建

需要 Go 1.24.2+：

```bash
git clone https://github.com/mindspore-lab/mindspore-cli.git
cd mindspore-cli
go build -o mscli ./cmd/mscli
./mscli
```

## Quick Start

### 使用免费内置模型

```bash
mscli
# 首次运行时选择 "mscli-provided" → "kimi-k2.5 [free]"
```

### 使用自己的 API Key

```bash
export MSCLI_API_KEY=sk-...
export MSCLI_MODEL=deepseek-chat
mscli
```

### 使用 OpenAI / Anthropic / OpenRouter

```bash
# OpenAI
export MSCLI_PROVIDER=openai-completion
export MSCLI_API_KEY=sk-...
export MSCLI_MODEL=gpt-4o

# Anthropic
export MSCLI_PROVIDER=anthropic
export MSCLI_API_KEY=sk-ant-...
export MSCLI_MODEL=claude-sonnet-4-20250514

# OpenRouter
export MSCLI_PROVIDER=openai-completion
export MSCLI_API_KEY=sk-or-...
export MSCLI_BASE_URL=https://openrouter.ai/api/v1

mscli
```

## Documentation

- [架构](docs/arch.md)
- [贡献者指南](docs/agent-contributor-guide.md)

## Contributing

请参阅 [贡献者指南](docs/agent-contributor-guide.md) 了解代码风格、依赖规则和测试规范。
