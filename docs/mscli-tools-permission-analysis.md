# ms-cli 工具系统与权限管理系统设计模式分析

## 背景与范围

本文基于当前仓库代码，对 `ms-cli` 的工具系统（`tools/*`, `runtime/shell/*`）和权限管理系统（`permission/*`）进行结构化总结，并分析潜在缺陷。

- 主要运行链路：`internal/app -> agent/loop -> tools -> runtime/shell`
- 关键参考：`docs/arch.md`、`tools/*`、`permission/*`、`internal/app/wire.go`

---

## 一、工具系统设计模式

### 1. 分层架构与职责隔离

工具调用在架构上采用分层：

- `agent/loop` 负责模型循环和工具调度
- `tools` 负责定义 LLM 可调用的工具接口
- `runtime/shell` 负责 shell 的状态化执行

这种模式降低了耦合度，使工具定义与执行基础设施分离。

代码参考：

- `docs/arch.md:57`
- `docs/arch.md:144`

### 2. 插件化工具接口（统一契约）

`tools.Tool` 定义了统一四元接口：

- `Name()`
- `Description()`
- `Schema()`
- `Execute()`

这本质是“接口驱动 + schema 驱动”模式，便于扩展新工具。

代码参考：

- `tools/types.go:12`

### 3. 注册中心模式（Registry）

`tools.Registry` 负责：

- 去重注册
- 维持注册顺序
- 转换为 LLM 可消费的 tools 描述

属于典型的“集中注册 + 分发”模式。

代码参考：

- `tools/registry.go:10`
- `tools/registry.go:92`

### 4. 双层安全边界

- 第一层：`agent/loop` 在调用工具前先走权限检查
- 第二层：具体工具内部进行参数校验（如路径限制）

例如文件系统工具通过 `resolveSafePath` 限制路径不逃逸工作目录。

代码参考：

- `agent/loop/engine.go:266`
- `tools/fs/pathutil.go:10`

### 5. shell 执行器封装

`shell` 工具把真实执行委托给 `runtime/shell.Runner`，由 Runner 统一处理：

- 工作目录
- 超时
- 输出截断
- allow/block 规则

代码参考：

- `runtime/shell/runner.go:53`
- `runtime/shell/runner.go:194`

---

## 二、权限管理系统设计模式

### 1. 权限等级格（Permission Lattice）

系统定义了严格有序的权限等级：

`deny < ask < allow_once < allow_session < allow_always`

通过数值序和 `min/maxPermission` 做“更严格优先”的合并策略。

代码参考：

- `permission/types.go:11`
- `permission/service.go:384`

### 2. 多作用域策略叠加

权限在三个维度上判断并取最严格值：

- 工具级（tool）
- 命令级（command，仅 shell）
- 路径级（path）

代码参考：

- `permission/service.go:131`

### 3. 可替换策略与存储抽象

- `PermissionService` 抽象决策逻辑
- `PermissionStore` 抽象持久化
- Store 支持 `file` 与 `memory`

代码参考：

- `permission/service.go:18`
- `permission/store.go:220`

### 4. 危险命令规则库

通过正则规则集合识别危险命令，并给出类别与建议等级。

代码参考：

- `permission/dangerous.go:16`

---

## 三、可能存在的缺陷与风险

以下按风险高到低排序。

### 1. 高风险：`ask` 在无 UI 时会默认放行

在 `PermissionAsk` 分支中，如果没有注入 `PermissionUI`，逻辑直接 `return true`。

这意味着“应询问”的操作在无 UI 场景下会被自动批准。

代码参考：

- `permission/service.go:175`
- `permission/service.go:211`

此外，当前应用接线仅创建并注入 `DefaultPermissionService`，未看到 `SetUI/SetStore` 的实际调用点。

代码参考：

- `internal/app/wire.go:173`
- `permission/service.go:102`

### 2. 高风险：配置字段语义混用（工具名单 vs 命令名单）

`PermissionsConfig.AllowedTools/BlockedTools` 字段名表达的是“工具”，但在 `initTools` 中被传给 `runtime/shell` 作为命令 allow/block 规则。

这会导致：

- 配置语义模糊
- 误配置概率高
- 安全策略不可预期

代码参考：

- `configs/types.go:54`
- `internal/app/wire.go:268`

### 3. 高风险：危险命令 `deny` 级别未在主判定中生效

危险命令库中有 `PermissionDeny` 规则（如 `rm -rf /`），但 `CheckCommand` 仅调用 `IsDangerousCommand`（布尔）后返回 `min(default, ask)`。

这会丢失规则中更强的 `deny` 语义。

代码参考：

- `permission/dangerous.go:20`
- `permission/service.go:260`

### 4. 中风险：命令策略粒度过粗

命令权限策略按“首个 token”匹配（`extractCommandName`），如 `git`/`python` 下不同子命令无法细分控制。

代码参考：

- `permission/service.go:398`

### 5. 中风险：`allow_session` 被持久化，语义与名字不一致

当用户“记住”权限时写入的是 `PermissionAllowSession`，但该决策会持久化到文件并在后续进程加载，相当于跨 session。

代码参考：

- `permission/service.go:187`
- `permission/service.go:197`

### 6. 中风险：路径策略匹配可能不稳定

`CheckPath` 使用 `filepath.Match` 对传入字符串直接匹配，未统一做归一化/绝对化，可能导致策略在不同路径表示下效果不一致。

代码参考：

- `permission/service.go:269`

### 7. 低到中风险：Runner 内部安全函数未接入主执行路径

`runtime/shell.Runner` 中有 `IsDangerous`、`RequiresConfirm`，但 `Run` 逻辑未使用它们进行阻断或确认。

代码参考：

- `runtime/shell/runner.go:172`
- `runtime/shell/runner.go:221`

### 8. 工程风险：缺少测试覆盖

对关键包执行测试：

- `go test ./permission ./tools/... ./runtime/shell`

结果均为 `[no test files]`。权限与执行安全链路缺少回归保护。

---

## 四、结论

当前实现在“架构分层、接口统一、可扩展性”方面模式清晰；但在权限系统上存在几个实质性安全缺口，尤其是：

1. `ask` 在无 UI 时自动放行
2. 危险命令 deny 语义在主路径被削弱
3. 配置字段语义混用导致策略可理解性与可控性下降

如果作为生产级 agent runtime，这三点应优先修复。
