package app

import (
	"testing"

	"github.com/vigo999/ms-cli/agent/loop"
	"github.com/vigo999/ms-cli/ui/model"
)

func TestConvertLoopEvent_TaskStartedIsNotRendered(t *testing.T) {
	ev := loop.Event{
		Type:    loop.EventTaskStarted,
		Message: "Task: repeated user input",
	}

	got := convertLoopEvent(ev)
	if got != nil {
		t.Fatalf("convertLoopEvent(TaskStarted) = %+v, want nil", got)
	}
}

func TestConvertLoopEvent_UnknownWithMessageFallsBackToAgentReply(t *testing.T) {
	ev := loop.Event{
		Type:    "UnknownEvent",
		Message: "some status",
	}

	got := convertLoopEvent(ev)
	if got == nil {
		t.Fatalf("convertLoopEvent(UnknownEvent) = nil, want non-nil")
	}
	if got.Type != model.AgentReply {
		t.Fatalf("convertLoopEvent type = %v, want %v", got.Type, model.AgentReply)
	}
	if got.Message != ev.Message {
		t.Fatalf("convertLoopEvent message = %q, want %q", got.Message, ev.Message)
	}
}
