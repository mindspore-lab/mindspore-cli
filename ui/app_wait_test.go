package ui

import (
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/vigo999/mindspore-cli/ui/model"
)

func TestEnterStartsThinkingWaitImmediately(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 20})
	app = next.(App)
	app.input.Model.SetValue("hello")

	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = next.(App)

	if !app.state.IsThinking {
		t.Fatal("expected enter to start model wait immediately")
	}
	if got, want := app.state.WaitKind, model.WaitModel; got != want {
		t.Fatalf("wait kind = %v, want %v", got, want)
	}
	if !app.state.IsThinking {
		t.Fatal("expected thinking state to be set after enter")
	}
	if app.state.WaitStartedAt.IsZero() {
		t.Fatal("expected wait timer to be started")
	}
}

func TestToolCallStartShowsPendingToolWait(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 20})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type:     model.ToolCallStart,
		ToolName: "shell",
		Message:  "go test ./ui",
	})
	app = next.(App)

	if app.state.IsThinking {
		t.Fatal("expected tool wait to stop thinking state")
	}
	if got, want := app.state.WaitKind, model.WaitTool; got != want {
		t.Fatalf("wait kind = %v, want %v", got, want)
	}
	view := app.View()
	if !strings.Contains(view, "Bash($ go test ./ui)") {
		t.Fatalf("expected pending tool call line, got:\n%s", view)
	}
	if !strings.Contains(view, "running") {
		t.Fatalf("expected tool wait timer in view, got:\n%s", view)
	}
}

func TestToolWarningClearsWaitState(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false
	app.state = app.startWait(model.WaitTool)

	next, _ := app.handleEvent(model.Event{
		Type:     model.ToolWarning,
		ToolName: "Engine",
		Message:  "request timeout",
	})
	app = next.(App)

	if app.state.IsThinking {
		t.Fatal("expected warning to clear thinking state")
	}
	if got, want := app.state.WaitKind, model.WaitNone; got != want {
		t.Fatalf("wait kind = %v, want %v", got, want)
	}
	if got, want := app.state.Messages[len(app.state.Messages)-1].Display, model.DisplayWarning; got != want {
		t.Fatalf("last message display = %v, want %v", got, want)
	}
}

func TestReplayWaitFastForwardsElapsedTime(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false
	app.state.WaitKind = model.WaitTool
	app.state.WaitStartedAt = time.Now().Add(-2500 * time.Millisecond)
	app.replayWait = &model.ReplayWaitData{
		OriginalDuration:  12 * time.Second,
		SimulatedDuration: 5 * time.Second,
	}

	got := app.currentWaitElapsed()
	if got < 5*time.Second || got > 7*time.Second {
		t.Fatalf("fast-forwarded wait = %v, want about 6s", got)
	}
}

func TestToolReplayResolvesHistoricalPendingShellByCallID(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.handleEvent(model.Event{
		Type:       model.ToolCallStart,
		ToolName:   "shell",
		ToolCallID: "call_shell_1",
		Message:    "sleep 10",
	})
	app = next.(App)

	if !app.state.Messages[0].Pending {
		t.Fatal("expected pending shell call after replay tool start")
	}

	next, _ = app.handleEvent(model.Event{
		Type:       model.ToolReplay,
		ToolName:   "shell",
		ToolCallID: "call_shell_1",
		Message:    "done",
	})
	app = next.(App)

	if len(app.state.Messages) != 1 {
		t.Fatalf("message count = %d, want 1", len(app.state.Messages))
	}
	if app.state.Messages[0].Pending {
		t.Fatal("expected replay tool result to resolve historical pending shell")
	}
	if got, want := app.state.Messages[0].Content, "done"; got != want {
		t.Fatalf("tool content = %q, want %q", got, want)
	}
	if got, want := app.state.Messages[0].ToolArgs, "$ sleep 10"; got != want {
		t.Fatalf("tool args = %q, want %q", got, want)
	}
}
