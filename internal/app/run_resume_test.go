package app

import (
	"strings"
	"testing"

	"github.com/vigo999/mindspore-cli/agent/session"
)

func TestExitResumeHintSkippedWithoutLiveLLMActivity(t *testing.T) {
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
		t.Fatalf("expected no resume hint without live llm activity, got %q", got)
	}
}

func TestExitResumeHintEmptyWithoutSession(t *testing.T) {
	app := &Application{}
	if got := app.exitResumeHint(); got != "" {
		t.Fatalf("expected no resume hint without session, got %q", got)
	}
}

func TestExitResumeHintShowsSessionIDAfterLiveLLMActivity(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	runtimeSession, err := session.Create(t.TempDir(), "system prompt")
	if err != nil {
		t.Fatalf("create session: %v", err)
	}
	t.Cleanup(func() {
		_ = runtimeSession.Close()
	})

	app := &Application{session: runtimeSession}
	if err := app.noteLiveLLMActivity(); err != nil {
		t.Fatalf("note live llm activity: %v", err)
	}

	got := app.exitResumeHint()
	if !strings.Contains(got, "mscli resume "+runtimeSession.ID()) {
		t.Fatalf("expected resume hint with session id, got %q", got)
	}
}

func TestExitResumeHintSkippedForReplay(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	runtimeSession, err := session.Create(t.TempDir(), "system prompt")
	if err != nil {
		t.Fatalf("create session: %v", err)
	}
	t.Cleanup(func() {
		_ = runtimeSession.Close()
	})

	app := &Application{session: runtimeSession, replayOnly: true}
	if err := app.noteLiveLLMActivity(); err != nil {
		t.Fatalf("note live llm activity: %v", err)
	}
	if got := app.exitResumeHint(); got != "" {
		t.Fatalf("expected no resume hint for replay, got %q", got)
	}
}
