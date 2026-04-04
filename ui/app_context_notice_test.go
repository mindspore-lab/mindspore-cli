package ui

import (
	"testing"

	"github.com/vigo999/mindspore-cli/ui/model"
)

func TestContextNoticeDoesNotInterruptStreamingAgentMessage(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.handleEvent(model.Event{Type: model.AgentThinking})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{Type: model.AgentReplyDelta, Message: "partial"})
	app = next.(App)

	app.state = app.state.WithThinking(true)

	next, _ = app.handleEvent(model.Event{
		Type:    model.ContextNotice,
		Message: "Context compacted automatically: 80 -> 40 tokens.",
	})
	app = next.(App)

	if !app.state.IsThinking {
		t.Fatal("context notice should not clear thinking state")
	}
	if got, want := len(app.state.Messages), 2; got != want {
		t.Fatalf("message count after context notice = %d, want %d", got, want)
	}
	if !app.state.Messages[0].Streaming {
		t.Fatal("streaming agent message should remain streaming after context notice")
	}

	next, _ = app.handleEvent(model.Event{Type: model.AgentReply, Message: "done"})
	app = next.(App)

	if got, want := len(app.state.Messages), 2; got != want {
		t.Fatalf("message count after final agent reply = %d, want %d", got, want)
	}
	if got, want := app.state.Messages[0].Content, "done"; got != want {
		t.Fatalf("finalized agent message = %q, want %q", got, want)
	}
	if app.state.Messages[0].Streaming {
		t.Fatal("finalized agent message should not be streaming")
	}
	if got, want := app.state.Messages[1].Content, "Context compacted automatically: 80 -> 40 tokens."; got != want {
		t.Fatalf("context notice message = %q, want %q", got, want)
	}
}

func TestAgentReplyPreservesRawANSIFlag(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.handleEvent(model.Event{
		Type:     model.AgentReply,
		Message:  "\x1b[38;5;252m[ OVERVIEW ]\x1b[0m",
		RawANSI:  true,
	})
	app = next.(App)

	if got, want := len(app.state.Messages), 1; got != want {
		t.Fatalf("message count = %d, want %d", got, want)
	}
	if !app.state.Messages[0].RawANSI {
		t.Fatal("expected AgentReply RawANSI to propagate to stored message")
	}
	if got, want := app.state.Messages[0].Content, "\x1b[38;5;252m[ OVERVIEW ]\x1b[0m"; got != want {
		t.Fatalf("stored message = %q, want %q", got, want)
	}
}
