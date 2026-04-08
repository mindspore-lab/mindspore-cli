package ui

import (
	"testing"
	"time"

	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

func TestClearScreenClearsVisibleChatState(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false
	app.state = app.state.WithMessage(model.Message{Kind: model.MsgUser, Content: "hello"})
	app.state = app.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: "hi"})
	app.state = app.state.WithThinking(true)
	app.state = app.state.WithWait(model.WaitModel, time.Now())
	app.followBottom = false
	app.unreadCount = 3
	*app.deltaStarted = true
	app.deltaBuf.WriteString("partial")
	*app.cmdOutputStarted = true
	*app.cmdOutputLines = 4

	next, _ := app.handleEvent(model.Event{Type: model.ClearScreen, Message: "Chat history cleared."})
	app = next.(App)

	if len(app.state.Messages) != 0 {
		t.Fatalf("message count after clear = %d, want 0", len(app.state.Messages))
	}
	if app.state.IsThinking {
		t.Fatal("expected clear to reset thinking state")
	}
	if got := app.state.WaitKind; got != model.WaitNone {
		t.Fatalf("wait kind after clear = %v, want %v", got, model.WaitNone)
	}
	if !app.followBottom {
		t.Fatal("expected followBottom=true after clear")
	}
	if got := app.unreadCount; got != 0 {
		t.Fatalf("unreadCount after clear = %d, want 0", got)
	}
	if got := app.deltaBuf.Len(); got != 0 {
		t.Fatalf("delta buffer len after clear = %d, want 0", got)
	}
	if *app.deltaStarted {
		t.Fatal("expected deltaStarted=false after clear")
	}
	if *app.cmdOutputStarted {
		t.Fatal("expected cmdOutputStarted=false after clear")
	}
	if got := *app.cmdOutputLines; got != 0 {
		t.Fatalf("cmdOutputLines after clear = %d, want 0", got)
	}
}
