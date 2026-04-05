[English](README.md) | 中文

# MindSpore CLI

面向 **AI 基础设施与模型训练工作流** 的 Agent CLI。

MindSpore CLI 帮助机器学习工程师和 AI 基础设施开发者 **启动训练任务、诊断故障、对齐精度、迁移模型代码、优化性能**。

与通用编程 CLI 不同，MindSpore CLI 围绕 **训练任务导向的工作流** 设计，而非宽泛的代码生成。

MindSpore CLI 是 MindSpore 模型训练 Agent 能力的 **官方端到端入口**，将可复用的领域技能集成在统一的 CLI 体验中。

---

## 1. 为什么选择 MindSpore CLI

模型训练不只是写代码。

真实的训练工作流常被环境问题、运行时故障、精度不一致、性能分析噪声、框架行为差异、算子缺失和迁移工作所阻塞。这些问题高频出现、跨层交织、高度依赖经验。

真正拖慢团队的，往往不是想法本身，而是围绕它的重复性工程工作：

- 检查工作空间是否就绪
- 阅读日志并缩小根因
- 将结果与基线对比
- 识别性能瓶颈
- 路由迁移和适配工作
- 迭代直到训练稳定高效运行

MindSpore CLI 正是为推动这些训练任务而构建的。

简而言之：

- 通用编程 Agent 关注的是 **怎么写代码**
- MindSpore CLI 关注的是 **怎么让训练跑起来、跑对、跑快**

---

## 2. MindSpore CLI 的适用场景

MindSpore CLI 面向以下任务：

- 检查本地训练工作空间是否就绪
- 诊断训练和运行时故障
- 分析精度不一致与回退
- 识别常见性能瓶颈
- 路由模型迁移和算子适配工作
- 辅助将论文算法特性适配到现有模型代码中

这些任务通常涉及：

- 框架和运行时行为
- 模型代码和训练脚本
- 数据预处理和结果验证
- 算子支持和后端差异
- 性能分析信号和瓶颈定位

---

## 3. 它不是什么

MindSpore CLI **不是** 通用编程 CLI。

它不以以下内容为中心：

- 通用代码生成
- 宽泛的仓库级软件工程工作流
- 无关的开发效率工具

它是面向 **AI 基础设施和模型训练工作流** 的 CLI。

---

## 4. 当前聚焦

MindSpore CLI 当前聚焦于 **模型训练工作流** 中最常见、最高价值的任务，尤其是在单机或受控环境下：

- 运行前就绪检查
- 训练/运行时故障诊断
- 精度调试
- 性能分析
- 模型迁移和算子相关工作

当前优先级是让核心训练工作流更深入、更可靠、更易用。

---

## 5. 内置能力领域

MindSpore CLI 集成了以下领域能力：

### 就绪检查
检查工作空间是否具备训练或推理的运行条件。

### 故障诊断
诊断训练和运行时故障，缩小到具体的层或组件。

### 精度分析
排查执行成功后出现的精度不一致、漂移、回退或错误结果。

### 性能分析
检查吞吐量、延迟、显存、数据加载、利用率、Host/Device 行为和瓶颈。

### 迁移与适配
路由模型迁移、算子实现和算法特性适配工作。

---

## 6. 内置技能

| 技能 | 功能描述 |
|---|---|
| `readiness-agent` | 检查工作空间是否具备训练或推理条件 |
| `failure-agent` | 诊断 MindSpore 和 PyTorch (torch_npu) 的训练和运行时故障 |
| `accuracy-agent` | 诊断数值漂移、精度不一致和跨平台精度回退 |
| `performance-agent` | 诊断延迟、吞吐量、显存、数据加载和通信瓶颈 |
| `migrate-agent` | 将模型实现迁移到 MindSpore 生态 |
| `operator-agent` | 通过框架原生或集成方式构建自定义算子 |
| `algorithm-agent` | 将论文或参考实现中的算法适配到现有模型代码 |

---

## 7. 命令

| 命令 | 描述 |
|---|---|
| `/model` | 切换模型或提供商 |
| `/compact` | 释放上下文空间 |
| `/clear` | 开启新对话 |
| `/permissions` | 管理工具访问权限 |
| `/diagnose` | 排查故障、精度问题和性能问题 |
| `/fix` | 诊断并在确认后应用修复 |
| `/help` | 显示所有命令 |

---

## 8. 快速开始

### 8.1 使用免费内置模型

```bash
mscli
# 首次运行时选择 "mscli-provided" → "kimi-k2.5 [free]"
```

### 8.2 使用自己的 API Key

```bash
export MSCLI_API_KEY=sk-...
export MSCLI_MODEL=deepseek-chat
mscli
```

### 8.3 使用 OpenAI / Anthropic / OpenRouter

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

---

## 9. 安装

### 9.1 脚本安装

```bash
# GitCode（推荐）
curl -fsSL https://gitcode.com/mindspore-lab/mindspore-cli/raw/main/scripts/install.sh | bash

# GitHub
curl -fsSL https://raw.githubusercontent.com/mindspore-lab/mindspore-cli/main/scripts/install.sh | bash
```

### 9.2 从源码构建

需要 Go 1.24.2+：

```bash
git clone https://github.com/mindspore-lab/mindspore-cli.git
cd mindspore-cli
go build -o mscli ./cmd/mscli
./mscli
```

---

## 10. 与 MindSpore Skills 的关系

MindSpore CLI 和 MindSpore Skills 承担不同角色：

- **MindSpore CLI** 是官方的端到端交互入口
- **MindSpore Skills** 是面向 AI 基础设施和模型训练工作流的可复用能力层

这种分离使得相同的训练导向能力既可以集成到官方 CLI 中，也可以作为领域技能被复用。

---

## 11. 典型使用场景

MindSpore CLI 尤其适用于：

- 首次运行前检查就绪状态
- 训练启动后诊断故障
- 执行成功后排查精度不一致
- 识别性能瓶颈
- 路由迁移和适配工作


---

## 12. 文档

- [架构](docs/arch.md)
- [贡献者指南](docs/agent-contributor-guide.md)

---

## 13. 贡献

请参阅 [贡献者指南](docs/agent-contributor-guide.md) 了解代码风格、依赖规则和测试规范。

贡献时请保持仓库定位一致：

- 面向 AI 基础设施和模型训练工作流
- 训练任务导向的 CLI 行为
- CLI 入口与可复用技能层之间的清晰边界

---

## 14. 许可证

Apache-2.0 — 详见 [LICENSE](LICENSE)。
