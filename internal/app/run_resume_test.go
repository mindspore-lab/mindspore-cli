package app

import (
	"strings"
	"testing"

	"github.com/vigo999/ms-cli/agent/session"
)

func TestExitResumeHintRequiresPersistedDialogue(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	runtimeSession, err := session.Create(t.TempDir(), "system prompt")
	if err != nil {
		t.Fatalf("create session: %v", err)
	}
	t.Cleanup(func() {
		_ = runtimeSession.Close()
	})

	app := &Application{session: runtimeSession}
	if got := app.exitResumeHint(); got != "" {
		t.Fatalf("expected no resume hint before persistence, got %q", got)
	}

	if err := runtimeSession.AppendUserInput("hello"); err != nil {
		t.Fatalf("append buffered user input: %v", err)
	}
	if got := app.exitResumeHint(); got != "" {
		t.Fatalf("expected no resume hint before activation, got %q", got)
	}

	if err := runtimeSession.Activate(); err != nil {
		t.Fatalf("activate session: %v", err)
	}

	got := app.exitResumeHint()
	if !strings.Contains(got, "ms-cli resume "+runtimeSession.ID()) {
		t.Fatalf("expected resume hint with session id, got %q", got)
	}
}
