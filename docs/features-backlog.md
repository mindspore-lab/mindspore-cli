# Features Backlog

Planned features not yet scheduled. Each item may become its own feature plan when prioritized.

**Command interface note:** Interactive UI uses slash commands (`/factory status`).
CLI subcommands (`mscli factory query ...`) are for tooling, scripts, and
Python tool bridge. Both route to the same Go implementation.

---

## Factory Query API & Tool Bridge

**Context:** A3 (`mindspore-code-refactor-3.md`) builds the factory store and updater.
This feature adds a query CLI that Python tools (and other consumers) can
call for search/get/list operations.

**Scope:**
- `mscli factory query` CLI subcommand (JSON output for tool consumption)
  - `mscli factory query search --type operator --tags ascend,softmax`
  - `mscli factory query get --id fp16-softmax-drift`
  - `mscli factory query list --type perf-features`
- `MS_FACTORY_PATH` env var for Python tools to locate cache dir
- Integration with `factory_query.py` shared tool in ms-skills
- Single query implementation in Go, usable by both Go code and Python tools via CLI

**Not in scope:** Semantic resolution (ResolveFailure, CheckTrick, constraint
resolution). That is a separate item below.

**Depends on:** A3 (factory store + updater)

---

## Factory Resolver for Runtime Consumers

**Context:** Demo stories (failure diagnosis, perf optimization, accuracy
recovery) need semantic answers, not just search/get/list. Agent skills
need higher-level queries like "what's wrong and how to fix it."

**Scope:**
- `ResolveFailure(errorLog, platform) → []FailureCard` — match error patterns
  to known failure cards, return ranked candidates with fixes
- `CheckTrick(model, method, category) → []FeatureCard` — find applicable
  perf/algo features given model context
- Operator constraint resolution — given op + platform + framework version,
  resolve support status, fallbacks, optimized variants
- Lives in `internal/factory/` as Go API (consumed by agent tools, not exposed as CLI)

**Depends on:** Factory Query API, C2 (schemas), C3 (seeded cards)

---

## Factory Report Submission

**Context:** Deferred from A3. Collect session data and submit report cards.

**Scope:**
- `/factory report` slash command (interactive UI)
- Auto-collect: error logs, config snapshot, metrics
- Package as report card (follows report.schema.yaml)
- Submit to factory (local or remote)

**Depends on:** A3 (factory store), C2 (report schema)
