# Refactor Arch 4 — Issue Command System (Phase 1: Bug)

## Overview

A shared bug tracking backend (server + client domain layer + slash commands).
Phase 1 covers bug-only; Phase 2 extends to full issue system.

## Architecture

```
TUI client → internal/issues (domain) → remote_store (HTTP) → server → SQLite
```

## New Files (17)

### Server Side (9 files)

| File | Purpose |
|---|---|
| `cmd/mscli-server/main.go` | Read config, open SQLite, init store, register routes, start HTTP |
| `configs/server_config.go` | ServerConfig struct + YAML loader |
| external `server.yaml` | Server config file (addr, dsn, auth tokens) |
| `internal/server/store.go` | SQLite schema init + CRUD (7 methods) |
| `internal/server/routes.go` | Register all routes on net/http.ServeMux |
| `internal/server/middleware.go` | Bearer token extraction → context user/role |
| `internal/server/auth_handler.go` | GET /me |
| `internal/server/bug_handler.go` | POST /bugs, GET /bugs, GET /bugs/{id} |
| `internal/server/note_handler.go` | POST /bugs/{id}/notes |
| `internal/server/claim_handler.go` | POST /bugs/{id}/claim |
| `internal/server/dock_handler.go` | GET /dock |
| `internal/server/activity_handler.go` | GET /bugs/{id}/activity |

### Client Domain Layer (4 files)

| File | Purpose |
|---|---|
| `internal/issues/model.go` | Bug, Note, Activity, DockData structs |
| `internal/issues/store.go` | Store interface (7 methods) |
| `internal/issues/service.go` | Service facade wrapping Store |
| `internal/issues/remote_store.go` | HTTP client impl of Store |

### Client App Layer (2 new files)

| File | Purpose |
|---|---|
| `internal/app/auth.go` | /login handler, save token to ~/.mscli/credentials |
| `internal/app/bugs.go` | /report, /bugs, /bug, /claim, /dock handlers + rendering |

## Edited Files (4)

| File | Change |
|---|---|
| `internal/app/commands.go` | Add 6 new cases to switch |
| `internal/app/wire.go` | Init issues.Service + remote_store |
| `configs/types.go` | Add ServerURL + TokenPath fields |
| `ui/slash/commands.go` | Register 6 new commands |

## API Surface

| Method | Path | Auth | Handler |
|---|---|---|---|
| GET | /healthz | no | routes.go inline |
| GET | /me | yes | auth_handler.go |
| POST | /bugs | yes | bug_handler.go |
| GET | /bugs | yes | bug_handler.go |
| GET | /bugs/{id} | yes | bug_handler.go |
| POST | /bugs/{id}/notes | yes | note_handler.go |
| POST | /bugs/{id}/claim | yes | claim_handler.go |
| GET | /bugs/{id}/activity | yes | activity_handler.go |
| GET | /dock | yes | dock_handler.go |

## Slash Commands

| Command | Handler file | Rendering |
|---|---|---|
| /login | auth.go | one-liner |
| /report | bugs.go | one-liner confirmation |
| /bugs | bugs.go | styled table |
| /bug <id> | bugs.go | styled box (notes + activity) |
| /claim <id> | bugs.go | one-liner confirmation |
| /dock | bugs.go | styled summary box |

## SQLite Tables

```sql
bugs:       id, title, status, lead, reporter, created_at, updated_at
notes:      id, bug_id, author, content, created_at
activities: id, bug_id, actor, type, text, created_at
```

## Dependencies

No new deps. Uses: net/http (stdlib), mattn/go-sqlite3 (existing), lipgloss (existing).

## Implementation Order

1. internal/issues/model.go
2. configs/server_config.go + external server.yaml
3. internal/server/ (store → middleware → handlers → routes)
4. cmd/mscli-server/main.go
5. internal/issues/ (store interface → service → remote_store)
6. internal/app/ (auth.go → bugs.go → edit commands.go + wire.go)
7. ui/slash/commands.go edits
