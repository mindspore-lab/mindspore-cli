package app

import (
	"testing"
	"time"

	"github.com/vigo999/ms-cli/ui/model"
)

func TestProcessInputQuitTriggersExit(t *testing.T) {
	app := newTestApp()

	app.processInput("/quit")

	_ = drainUntil(t, app, model.AgentReply, 2*time.Second)
	done := drainUntil(t, app, model.Done, 2*time.Second)
	if done.Type != model.Done {
		t.Fatalf("expected done event, got %s", done.Type)
	}
}
