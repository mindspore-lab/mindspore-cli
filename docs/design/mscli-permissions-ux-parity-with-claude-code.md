# mscli 权限系统与 Claude Code 体验一致化设计文档

## 1. 目标

将 `mscli` 的权限管理系统升级为与 Claude Code 权限体验一致（功能、交互、默认行为、规则语义、管理入口一致），重点覆盖：

- 工具调用时的权限询问交互
- `/permissions` 管理界面与规则可视化
- 权限规则语法（`Tool` / `Tool(specifier)`）
- 权限模式（`default` / `acceptEdits` / `plan` / `dontAsk` / `bypassPermissions`）
- 规则优先级与持久化行为

> 说明：本文中的“完全一致”指**用户可感知行为与信息架构一致**。内部实现可以与 Claude Code 不同，但外部表现必须对齐。

---

## 2. 设计依据（官方参考）

主要依据 Claude Code 官方文档：

- Configure permissions: https://code.claude.com/docs/en/permissions

文档中关键要求（摘取语义）：

- 读工具默认无需审批；Bash 与文件修改需审批
- `/permissions` 可查看规则与来源
- 规则优先级为 `deny -> ask -> allow`，首个匹配生效
- 支持权限模式：`default`、`acceptEdits`、`plan`、`dontAsk`、`bypassPermissions`
- 规则语法为 `Tool` 或 `Tool(specifier)`，Bash 支持 `*` 通配
- “Yes, don’t ask again” 对 Bash 的持久化粒度为“按项目目录 + 命令规则”；文件编辑为“会话级”

---

## 3. 现状与差距

### 3.1 mscli 当前能力（简述）

- 已有工具级/命令级/路径级权限框架
- 已有 `PermissionPrompt` 专门事件
- 已有 `PermissionStore`（文件存储）
- 已支持 `/permission` 与 `/yolo`

### 3.2 与目标体验的主要差距

1. 管理入口与信息结构不一致：当前是 `/permission`（单条设置），缺少 Claude 风格 `/permissions` 规则列表+来源展示。
2. 规则语法不一致：当前内部结构不是 `Tool(specifier)` 统一语法。
3. 规则优先级实现未完全按 “deny -> ask -> allow + first-match wins” 通用模型组织。
4. 询问选项语义不完全对齐：缺少 “Yes, don’t ask again” 在不同工具上的差异化落地策略。
5. 模式体系不一致：缺少完整 `default/acceptEdits/plan/dontAsk/bypassPermissions`。
6. Bash 规则能力不足：缺少稳定的通配规则、复合命令拆分存储策略与规则来源管理。

---

## 4. 目标体验规范（对齐 Claude Code）

## 4.1 命令与入口

- 新增并主推：`/permissions`
- 保留 `/permission` 作为兼容别名（后续可提示迁移）

`/permissions` 页面必须展示：

- 当前生效规则列表（按优先级顺序）
- 每条规则的来源（例如：managed / user / project / session）
- 当前权限模式（default/acceptEdits/plan/dontAsk/bypassPermissions）

## 4.2 权限询问交互（Prompt UX）

当工具调用触发 ask 时，统一弹出权限卡片（通过 `PermissionPrompt` 事件）：

- 显示：工具名、规范化 specifier、规则匹配解释、建议动作
- 对 `Edit/Write` 的提示文案必须对齐 Claude 示例交互：
  - `Do you want to make this edit to <path>?`
  - `1. Yes`
  - `2. Yes, allow all edits during this session`
  - `3. No`
  - `Esc to cancel`
- 输入语义：
  - `1` / `y` / `yes` -> 本次允许
  - `2` / `a` -> 会话内允许全部编辑（`Edit + Write`）
  - `3` / `n` / `no` / `esc` -> 拒绝
  - 在 TUI 中，默认通过 `↑/↓` 选择 + `Enter` 确认，不要求用户手动输入答案

差异化“记忆”策略：

- Bash：持久化为项目级命令规则（可跨会话）
- 文件编辑（Edit/Write）：仅会话内生效（会话结束失效），作用域为“全部编辑类工具（Edit+Write）”，不是单一路径
- 读工具通常不触发询问（除非规则显式 ask）

## 4.3 规则语法（统一 DSL）

统一使用字符串规则：

- `Tool`：匹配该工具全部调用
- `Tool(specifier)`：匹配该工具特定调用

首批支持：

- `Bash(...)`
- `Read(...)`
- `Edit(...)`
- `Write(...)`
- `WebFetch(...)`（为后续扩展预留）
- `Agent(...)`（子代理控制预留）

Bash specifier：

- 支持 `*` 通配
- 具备 shell operator 感知（避免 `safe-cmd *` 间接放行 `safe-cmd && dangerous-cmd`）
- 复合命令在 “don’t ask again” 时拆分为最多 N 条子规则（建议 N=5，对齐官方描述）

## 4.4 规则优先级与匹配

统一匹配模型：

- 规则顺序：`deny -> ask -> allow`
- 同类规则按配置顺序匹配
- first-match wins

决策输出：

- `deny`: 直接拒绝
- `ask`: 触发 Prompt
- `allow`: 直接执行

## 4.5 权限模式

新增 `permissions.defaultMode`：

1. `default`
- 标准模式：按工具首次使用触发询问

2. `acceptEdits`
- 自动接受本会话文件编辑权限
- Bash 仍按规则询问

3. `plan`
- 仅允许分析，不允许执行命令与写文件

4. `dontAsk`
- 未预先 allow 的工具默认拒绝（不弹询问）

5. `bypassPermissions`
- 跳过常规询问
- 但对受保护目录写入仍强制确认（见 4.6）

## 4.6 受保护目录策略（bypass 下强制确认）

在 `bypassPermissions` 下，写入以下目录仍需强制确认：

- `.git`
- `.claude`
- `.vscode`
- `.idea`

并保留白名单例外目录机制（例如未来若 mscli 有自身 agent/skills 目录，可按产品策略配置）。

---

## 5. 配置模型设计

```yaml
permissions:
  defaultMode: default
  allow:
    - "Bash(npm test *)"
    - "Read"
  ask:
    - "Bash(git push *)"
  deny:
    - "Bash(rm -rf /)"
  protectedWrites:
    - ".git/**"
    - ".claude/**"
    - ".vscode/**"
    - ".idea/**"
```

说明：

- 用 `allow/ask/deny` 三数组替代旧的多结构并行策略，降低歧义
- 内部可编译为高性能 matcher，但外部配置保持 DSL 一致

---

## 6. 持久化与来源体系

## 6.1 存储分层

与现有配置层次对齐，新增“规则来源”概念：

- managed（组织策略，不可覆盖）
- user（`~/.ms-cli/config.yaml`）
- project（`./.ms-cli/config.yaml`）
- session（运行期动态）

优先级建议：`managed > session > project > user > default`

> 如需强对齐 Claude 的 precedence 文档，可在实现阶段逐项对齐并在文档附录给出精确顺序表。

## 6.2 “don’t ask again” 持久化规则

- Bash：写入 project 级 permission state 文件（建议 `./.ms-cli/permissions.state.json`）
- Edit/Write：写入 session 内存态，不落盘
- `/permissions` 页面必须标注来源（state/user/project/managed）

---

## 7. TUI 交互规范

## 7.1 事件模型

保留并扩展 `PermissionPrompt` 专门事件：

- `Type: PermissionPrompt`
- `Payload` 增加结构化字段（建议）：
  - `Tool`
  - `Specifier`
  - `Mode`
  - `MatchedRule`
  - `Source`
  - `Options`

## 7.2 视觉与操作

权限卡片应具备：

- 独立视觉层（边框/颜色/标题）
- 清晰热键：
  - `1` / `y` -> Yes
  - `2` / `a` -> Yes, allow all edits during this session（Edit/Write 场景）
  - `3` / `n` / `Esc` -> No
- 焦点锁定：有 pending prompt 时，普通输入先用于权限决策

---

## 8. 实现方案（分阶段）

### Phase 1：规则引擎重构

- 引入 `Tool(specifier)` 解析器与 matcher
- 实现 deny/ask/allow first-match 模型
- 保持旧配置兼容读取（自动映射）

### Phase 2：模式系统

- 增加 `defaultMode`
- 实现五种模式行为
- 增加 protected write 强制确认

### Phase 3：交互与管理界面

- 新增 `/permissions` UI
- 展示规则、来源、模式
- 支持在 UI 中增删改规则

### Phase 4：持久化与来源可观测

- 增加 project state 文件
- 接入 Bash don’t-ask-again 持久化
- UI 中显示来源与生效顺序

### Phase 5：兼容与迁移

- `/permission` 输出 deprecation 提示
- 提供配置迁移器（旧字段 -> 新 DSL）

---

## 9. 验收标准

1. 交互一致性
- 用户在工具调用时看到与 Claude 语义一致的三选项权限询问。

2. 模式一致性
- 五种模式行为可验证且符合定义。

3. 规则一致性
- `Tool(specifier)`、通配匹配、优先级与 first-match 行为正确。

4. 持久化一致性
- Bash don’t-ask-again 跨会话生效；Edit/Write don’t-ask-again 会话结束失效。

5. 管理可观测
- `/permissions` 能看到规则、来源、模式与最终生效顺序。

---

## 10. 风险与注意事项

- “完全一致”需要持续对照官方文档演进，建议建立版本对齐清单。
- Bash 参数级安全约束天然脆弱，需配合 sandbox / hook 形成纵深防御。
- 受保护目录策略与本项目工作流（例如技能/配置目录）可能存在冲突，需在实现前确认例外清单。

---

## 11. 附录：对 mscli 当前代码的影响面

主要涉及包：

- `permission/`（核心重构）
- `internal/app/commands.go`（新增 `/permissions` 入口）
- `internal/app/run.go`（pending prompt 输入优先级已具备，需结构化升级）
- `ui/model/` 与 `ui/app.go`（权限卡片展示）
- `configs/`（新配置 schema 与迁移）
