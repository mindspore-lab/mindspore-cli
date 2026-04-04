# Add Tip Line Below Thinking Spinner in Live Status Area

## Context

Display a "Tip: ..." line below the thinking spinner in the live status area, matching Claude Code's design:

```
⣻ Thinking... 01:02
  ⎿  Tip: Use /report to submit feedback!
```

The tip appears while the model is thinking (`WaitModel`), rendered with the `⎿` connector and dim styling.

## Files to Modify

1. **`ui/components/tips.go`** (new) — tip list and random selection
2. **`ui/components/thinking.go`** (modify) — add `ViewWithTip()` method
3. **`ui/components/styles.go`** (modify) — add tip styles
4. **`ui/inline.go`** (modify) — use `ViewWithTip()` in `activePreview()`

## Implementation Steps

### Step 1: Create `ui/components/tips.go`

```go
package components

import "math/rand"

var tips = []string{
    "Use /report to submit feedback",
    "Press ctrl+o to view full tool output",
    "Use /model to switch models",
    "Use ↑/↓ to recall input history",
    "Use /compact to free up context space",
    "Use /clear to start a fresh conversation",
    "Use /permissions to manage tool access",
    "Use /train to launch training sessions",
    "Use /diagnose to investigate issues",
    "Use /help to see all commands",
}

func RandomTip() string {
    return tips[rand.Intn(len(tips))]
}
```

Tips reference only commands that exist: `/report`, `/model`, `/compact`, `/clear`, `/permissions`, `/train`, `/diagnose`, `/help`, plus keybindings `ctrl+o` and `↑/↓`.

### Step 2: Modify `ui/components/thinking.go`

- Add `tipText string` field to `ThinkingSpinner` struct
- Set `tipText: RandomTip()` in `NewThinkingSpinner()` and `NewThinkingSpinnerWithText()`
- Re-roll tip in `Reset()` via `RandomTip()`
- Add `ViewWithTip() string` method:

```go
func (t ThinkingSpinner) ViewWithTip() string {
    base := t.View()
    tip := fmt.Sprintf("  %s  %s",
        tipPrefixStyle.Render("⎿"),
        tipStyle.Render("Tip: "+t.tipText))
    return base + "\n" + tip
}
```

### Step 3: Modify `ui/components/styles.go`

Add two styles in `InitStyles()`:
```go
tipPrefixStyle = lipgloss.NewStyle().Foreground(t.TextSecondary)
tipStyle = lipgloss.NewStyle().Foreground(t.TextMuted).Italic(true)
```

And declare package-level vars:
```go
var tipPrefixStyle lipgloss.Style
var tipStyle lipgloss.Style
```

### Step 4: Modify `ui/inline.go` — `activePreview()`

- `WaitModel` case: `a.thinking.View()` → `a.thinking.ViewWithTip()`
- Delta buffering case: same change → `a.thinking.ViewWithTip()`

## Why This Works

- `renderMainView()` uses `lipgloss.JoinVertical` with variable-height parts and `trimViewHeight` clips to terminal height — two lines is fine
- The tip is stored in the struct field, so it stays stable across 80ms tick updates
- `ThinkingSpinner` is a value type — the `tipText` field copies correctly through Bubble Tea's update cycle
- `Reset()` re-rolls the tip so users see variety across thinking sessions

## Verification

1. `go build ./...` — confirms compilation
2. `go test ./ui/...` — run existing UI tests
3. Manual: run `./mscli` and send a prompt — verify the tip line appears below spinner during thinking
