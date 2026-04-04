# MindSpore CLI

Agent CLI for **AI infra and model training workflows**.

MindSpore CLI helps ML engineers and AI infra developers **get training jobs running, diagnose failures, align results, migrate model code, and improve performance**.

Unlike general-purpose coding CLIs, MindSpore CLI is designed around **training-task-oriented workflows** rather than broad repository-wide code generation.

MindSpore CLI is the **official end-to-end entrypoint** for interacting with MindSpore's model-training-oriented agent capabilities. It integrates reusable domain skills behind a single CLI experience.

---

## 1. Why MindSpore CLI

Model training is not just a coding problem.

Real-world training workflows are often blocked by a mix of environment issues, runtime failures, accuracy mismatch, profiling noise, framework behavior, operator gaps, and migration work. These problems are high-frequency, cross-layer, and highly experience-driven.

What slows teams down is often not the idea itself, but the repeated engineering-heavy work around it:

- checking whether the workspace is ready
- reading logs and narrowing root causes
- comparing results against a baseline
- identifying performance bottlenecks
- routing migration and adaptation work
- iterating toward a stable and efficient run

MindSpore CLI is built to help move those training tasks forward.

In short:

- general coding agents focus on **how to write the code**
- MindSpore CLI focuses on **how to get the training task running, running correctly, and running efficiently**

---

## 2. What MindSpore CLI is for

MindSpore CLI is built for tasks such as:

- checking whether a local training workspace is ready
- diagnosing training and runtime failures
- analyzing accuracy mismatch and regression
- identifying common performance bottlenecks
- routing model migration and operator adaptation work
- assisting algorithm feature adaptation into an existing model codebase

These tasks commonly cross:

- framework and runtime behavior
- model code and training scripts
- data preprocessing and result validation
- operator support and backend differences
- profiling signals and performance bottlenecks

---

## 3. What it is not

MindSpore CLI is **not** a general-purpose coding CLI.

It is not centered on:

- generic code generation
- broad repository-wide software engineering workflows
- unrelated developer productivity tasks

It is a CLI designed for **AI infra and model training workflows**.

---

## 4. Current focus

MindSpore CLI currently focuses on the most common and highest-value tasks in the **model training workflow**, especially in single-machine or otherwise controlled environments:

- preflight readiness checks
- training/runtime failure diagnosis
- accuracy debugging
- performance analysis
- model migration and operator-related work

The current priority is to make core training workflows deeper, more reliable, and easier to use.

---

## 5. Built-in capability areas

MindSpore CLI integrates built-in domain capabilities for:

### Readiness
Check whether a workspace is ready to train or run inference.

### Failure
Diagnose training and runtime failures and narrow the responsible layer or component.

### Accuracy
Investigate mismatch, drift, regression, or wrong results after execution succeeds.

### Performance
Inspect throughput, latency, memory, dataloader, utilization, host/device behavior, and bottlenecks.

### Migration and adaptation
Route model migration, operator implementation, and algorithm feature adaptation work.

---

## 6. Built-in skills

| Skill | What it does |
|---|---|
| `readiness-agent` | Check whether a workspace is ready to train or run inference |
| `failure-agent` | Diagnose training and runtime failures across MindSpore and PyTorch (torch_npu) |
| `accuracy-agent` | Diagnose numerical drift, precision mismatches, and cross-platform accuracy regressions |
| `performance-agent` | Diagnose latency, throughput, memory, dataloader, and communication bottlenecks |
| `migrate-agent` | Migrate model implementations into the MindSpore ecosystem |
| `operator-agent` | Build custom operators through framework-native or integration approaches |
| `algorithm-agent` | Adapt algorithms from papers or reference implementations into existing models |

---

## 7. Commands

| Command | Description |
|---|---|
| `/model` | Switch model or provider |
| `/compact` | Free up context space |
| `/clear` | Start a fresh conversation |
| `/permissions` | Manage tool access |
| `/diagnose` | Investigate failures, accuracy problems, and performance issues |
| `/fix` | Diagnose and apply fixes with confirmation |
| `/help` | Show all commands |

---

## 8. Quick start

### 8.1 Use the free built-in model

```bash
mscli
# Choose "mscli-provided" → "kimi-k2.5 [free]" on first run
```

### 8.2 Bring your own API key

```bash
export MSCLI_API_KEY=sk-...
export MSCLI_MODEL=deepseek-chat
mscli
```

### 8.3 Use OpenAI / Anthropic / OpenRouter

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

## 9. Install

### 9.1 Install from script

```bash
curl -fsSL http://47.115.175.134/mscli/install.sh | bash
```

### 9.2 Build from source

Go 1.24.2+:

```bash
git clone https://github.com/mindspore-lab/mindspore-cli.git
cd mindspore-cli
go build -o mscli ./cmd/mscli
./mscli
```

---

## 10. Relationship to MindSpore Skills

MindSpore CLI and MindSpore Skills play different roles:

- **MindSpore CLI** is the official end-to-end interaction surface
- **MindSpore Skills** is the reusable capability layer for AI infra and model training workflows

This separation allows the same training-oriented capabilities to be integrated into the official CLI while remaining reusable as domain skills.

---

## 11. Typical use cases

MindSpore CLI is especially useful for:

- checking readiness before the first run
- diagnosing failures after training starts
- investigating accuracy mismatch after execution succeeds
- identifying performance bottlenecks
- routing migration and adaptation work

See [docs/use-cases.md](docs/use-cases.md) for a lifecycle-oriented view.

---

## 12. Documentation

- [Architecture](docs/arch.md)
- [Contributor Guide](docs/agent-contributor-guide.md)
- [Feature Backlog](docs/features-backlog.md)
- [Use Cases](docs/use-cases.md)
- [Roadmap](docs/roadmap.md)

---

## 13. Contributing

See the [Contributor Guide](docs/agent-contributor-guide.md) for code style, dependency rules, and testing conventions.

When contributing, please keep the repo positioning aligned with:

- AI infra and model training workflows
- training-task-oriented CLI behavior
- clear boundaries between CLI entrypoint and reusable skill layer

---

## 14. License

Apache-2.0 — see [LICENSE](LICENSE).
