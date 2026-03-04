package ui

import (
	"strings"
	"testing"

	"github.com/vigo999/ms-cli/ui/model"
)

func applyEvent(t *testing.T, a App, ev model.Event) App {
	t.Helper()
	m, _ := a.handleEvent(ev)
	updated, ok := m.(App)
	if !ok {
		t.Fatalf("expected App model")
	}
	return updated
}

func newProgressTestApp() App {
	ch := make(chan model.Event)
	a := New(ch, nil, "v0.0.0", ".", "", "openrouter", "deepseek", 24000)
	a.width = 120
	a.height = 40
	a.resizeViewport()
	return a
}

func TestHandleEvent_ReplyCompletesThinkingStepWithSummary(t *testing.T) {
	a := newProgressTestApp()

	a = applyEvent(t, a, model.Event{Type: model.AgentThinking})
	a = applyEvent(t, a, model.Event{Type: model.CmdStarted, Message: "go test ./..."})
	a = applyEvent(t, a, model.Event{Type: model.ToolRead, Message: "ui/app.go"})
	a = applyEvent(t, a, model.Event{Type: model.ToolEdit, Message: "ui/app.go\n\n- old\n+ new"})
	a = applyEvent(t, a, model.Event{Type: model.AgentReply, Message: "all done"})

	thinkingCount := 0
	doneCount := 0
	for _, msg := range a.state.Messages {
		if msg.Kind == model.MsgThinking {
			thinkingCount++
		}
		if msg.Kind == model.MsgAgent && strings.HasPrefix(msg.Content, "Done  -- ") {
			doneCount++
			if msg.Content != "Done  -- commands: 1, files viewed: 1, files modified: 1" {
				t.Fatalf("unexpected done summary: %q", msg.Content)
			}
		}
	}
	if thinkingCount != 0 {
		t.Fatalf("expected no thinking message after reply, got %d", thinkingCount)
	}
	if doneCount != 1 {
		t.Fatalf("expected one done summary, got %d", doneCount)
	}

	last := a.state.Messages[len(a.state.Messages)-1]
	if last.Kind != model.MsgAgent || last.Content != "all done" {
		t.Fatalf("expected final agent reply, got kind=%v content=%q", last.Kind, last.Content)
	}
}

func TestHandleEvent_NextThinkingFinalizesPreviousStepAndKeepsSingleThinking(t *testing.T) {
	a := newProgressTestApp()

	a = applyEvent(t, a, model.Event{Type: model.AgentThinking})
	a = applyEvent(t, a, model.Event{Type: model.CmdStarted, Message: "ls -la"})
	a = applyEvent(t, a, model.Event{Type: model.AgentThinking})

	thinkingCount := 0
	doneFound := false
	for _, msg := range a.state.Messages {
		if msg.Kind == model.MsgThinking {
			thinkingCount++
		}
		if msg.Kind == model.MsgAgent && msg.Content == "Done  -- commands: 1" {
			doneFound = true
		}
	}
	if thinkingCount != 1 {
		t.Fatalf("expected exactly one thinking message, got %d", thinkingCount)
	}
	if !doneFound {
		t.Fatalf("expected done summary for previous step")
	}
}

func TestHandleEvent_NoOpsStepDoesNotShowDone(t *testing.T) {
	a := newProgressTestApp()

	a = applyEvent(t, a, model.Event{Type: model.AgentThinking})
	a = applyEvent(t, a, model.Event{Type: model.AgentThinking})

	for _, msg := range a.state.Messages {
		if msg.Kind == model.MsgAgent && strings.HasPrefix(msg.Content, "Done  -- ") {
			t.Fatalf("did not expect done summary for zero-op step, got %q", msg.Content)
		}
	}
}
