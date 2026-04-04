package app

import (
	"testing"

	"github.com/vigo999/mindspore-cli/ui/model"
)

func TestCmdExitShowsGoodbyeOnly(t *testing.T) {
	app := &Application{
		EventCh: make(chan model.Event, 16),
	}

	app.cmdExit()

	ev := drainUntilEventType(t, app, model.AgentReply)
	if ev.Message != "Goodbye!" {
		t.Fatalf("expected goodbye message, got %q", ev.Message)
	}
}
