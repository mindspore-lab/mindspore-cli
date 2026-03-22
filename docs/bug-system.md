# Bug System Usage Guide

A shared bug tracking system for ms-cli teams. Covers installation, server setup, and all slash commands.

## 1. Install ms-cli

### Option A: Install script (recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/vigo999/ms-cli/refactor-arch-4.2-issue-cmd/scripts/install.sh | bash
```

This downloads the latest release to `~/.ms-cli/bin/mscli`. Add to PATH:

```bash
export PATH="$HOME/.ms-cli/bin:$PATH"
```

### Option B: Build from source

```bash
git clone https://github.com/vigo999/ms-cli.git
cd ms-cli
go build -o ms-cli ./cmd/ms-cli
go build -o ms-cli-server ./cmd/ms-cli-server
```

## 2. Server Setup

The bug server stores bugs in SQLite and serves them over HTTP with token auth.

### 2.1 Config

Create or edit `configs/server.yaml`:

```yaml
server:
  addr: ":8080"

storage:
  driver: sqlite
  dsn: /opt/mscli/issues.db

auth:
  tokens:
    - token: mscli_token_alice
      user: alice
      role: member
    - token: mscli_token_bob
      user: bob
      role: member
    - token: mscli_token_travis
      user: travis
      role: admin
```

- `addr` — listen address (default `:8080`)
- `dsn` — SQLite database path (created automatically)
- `tokens` — each entry maps a Bearer token to a user and role

### 2.2 Start the server

```bash
./ms-cli-server --config configs/server.yaml
```

The server creates the SQLite tables on first start. Verify with:

```bash
curl http://localhost:8080/healthz
# ok
```

## 3. Login

Before using bug commands in the TUI, log in once:

```
/login http://localhost:8080 mscli_token_alice
```

Output:

```
logged in as alice (member)
```

Credentials are saved to `~/.ms-cli/credentials.json`. Subsequent sessions reuse them automatically.

## 4. Bug Commands

### /report — Create a bug

```
/report training loss diverges after epoch 3
```

Output:

```
created BUG-1: training loss diverges after epoch 3
```

### /bugs — List bugs

```
/bugs
```

Shows a styled table:

```
╭────────────────────────────────────────────────────────╮
│ bug list                                               │
│   ID      TITLE                    STATUS  LEAD  ...   │
│   BUG-1   training loss diverges   open    -     alice  │
│   BUG-2   OOM on 8xA100           doing   bob   alice  │
╰────────────────────────────────────────────────────────╯
```

Filter by status:

```
/bugs open
/bugs doing
```

### /claim — Claim a bug

```
/claim 1
```

Output:

```
you claimed BUG-1
```

Sets you as the lead and changes status to `doing`.

### /dock — Dashboard

```
/dock
```

Shows a summary box:

```
╭──────────────────────────────────────╮
│ dock                                 │
│                                      │
│   open bugs  3                       │
│                                      │
│   ready (unassigned)                 │
│     BUG-3  checkpoint save fails     │
│                                      │
│   recent activity                    │
│     03-22 10:15  alice  reported ... │
│     03-22 10:18  bob    claimed ...  │
╰──────────────────────────────────────╯
```

## 5. Server API Reference

All endpoints except `/healthz` require `Authorization: Bearer <token>`.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/healthz` | Liveness check (no auth) |
| GET | `/me` | Current user info |
| POST | `/bugs` | Create bug (`{"title":"..."}`) |
| GET | `/bugs` | List bugs (optional `?status=open`) |
| GET | `/bugs/{id}` | Get single bug |
| POST | `/bugs/{id}/claim` | Claim bug as lead |
| POST | `/bugs/{id}/notes` | Add note (`{"content":"..."}`) |
| GET | `/bugs/{id}/activity` | List bug activity |
| GET | `/dock` | Dashboard summary |

### curl examples

```bash
# create bug
curl -X POST -H "Authorization: Bearer mscli_token_alice" \
  -H "Content-Type: application/json" \
  -d '{"title":"gradient explosion on large batch"}' \
  http://localhost:8080/bugs

# list all bugs
curl -H "Authorization: Bearer mscli_token_alice" http://localhost:8080/bugs

# claim bug 1
curl -X POST -H "Authorization: Bearer mscli_token_bob" \
  http://localhost:8080/bugs/1/claim

# add note
curl -X POST -H "Authorization: Bearer mscli_token_bob" \
  -H "Content-Type: application/json" \
  -d '{"content":"reproduced with batch_size=256"}' \
  http://localhost:8080/bugs/1/notes

# view activity
curl -H "Authorization: Bearer mscli_token_alice" \
  http://localhost:8080/bugs/1/activity

# dock summary
curl -H "Authorization: Bearer mscli_token_alice" http://localhost:8080/dock
```

## 6. Data Model

### SQLite Tables

```
bugs:       id, title, status, lead, reporter, created_at, updated_at
notes:      id, bug_id, author, content, created_at
activities: id, bug_id, actor, type, text, created_at
```

- **status**: `open` (default) or `doing` (after claim)
- **activity types**: `report`, `claim`, `note`, `status`

## 7. Architecture

```
TUI (/login, /report, /bugs, /claim, /dock)
  → internal/app/bugs.go (command dispatch)
    → internal/issues/service.go (domain facade)
      → internal/issues/remote_store.go (HTTP client)
        → ms-cli-server (HTTP API)
          → internal/server/store.go (SQLite)
```

Rendering lives in `ui/render/` — shared styles and box layout used by all boxy commands.
