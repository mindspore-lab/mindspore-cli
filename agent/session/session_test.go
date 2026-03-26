package session

import (
	"os"
	"testing"

	"github.com/vigo999/ms-cli/integrations/llm"
)

func TestCreateDefersDiskWritesUntilActivate(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	workDir := t.TempDir()
	s, err := Create(workDir, "system prompt")
	if err != nil {
		t.Fatalf("create session: %v", err)
	}

	if _, err := os.Stat(s.Path()); !os.IsNotExist(err) {
		t.Fatalf("expected no trajectory before activate, got err=%v", err)
	}
	if _, err := os.Stat(snapshotPath(s.Path())); !os.IsNotExist(err) {
		t.Fatalf("expected no snapshot before activate, got err=%v", err)
	}

	if err := s.AppendSkillActivation("demo-skill"); err != nil {
		t.Fatalf("append skill activation: %v", err)
	}
	if err := s.AppendUserInput("hello"); err != nil {
		t.Fatalf("append user input: %v", err)
	}
	if err := s.AppendAssistant("hi"); err != nil {
		t.Fatalf("append assistant reply: %v", err)
	}
	if err := s.SaveSnapshot("updated prompt", []llm.Message{
		llm.NewUserMessage("hello"),
		llm.NewAssistantMessage("hi"),
	}); err != nil {
		t.Fatalf("save buffered snapshot: %v", err)
	}

	if _, err := os.Stat(s.Path()); !os.IsNotExist(err) {
		t.Fatalf("expected no trajectory before activate after buffering, got err=%v", err)
	}
	if _, err := os.Stat(snapshotPath(s.Path())); !os.IsNotExist(err) {
		t.Fatalf("expected no snapshot before activate after buffering, got err=%v", err)
	}

	if err := s.Activate(); err != nil {
		t.Fatalf("activate session: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("close activated session: %v", err)
	}

	if _, err := os.Stat(s.Path()); err != nil {
		t.Fatalf("expected trajectory after activate: %v", err)
	}
	if _, err := os.Stat(snapshotPath(s.Path())); err != nil {
		t.Fatalf("expected snapshot after activate: %v", err)
	}

	loaded, err := LoadByID(workDir, s.ID())
	if err != nil {
		t.Fatalf("load activated session: %v", err)
	}
	t.Cleanup(func() {
		_ = loaded.Close()
	})

	if !loaded.HasPersistedDialogue() {
		t.Fatal("expected persisted dialogue after activation")
	}
	if got := loaded.Meta().SystemPrompt; got != "updated prompt" {
		t.Fatalf("meta system prompt = %q, want %q", got, "updated prompt")
	}

	systemPrompt, restored := loaded.RestoreContext()
	if systemPrompt != "updated prompt" {
		t.Fatalf("restored system prompt = %q, want %q", systemPrompt, "updated prompt")
	}
	if len(restored) != 2 {
		t.Fatalf("restored message count = %d, want 2", len(restored))
	}

	replay := loaded.ReplayEvents()
	if len(replay) != 3 {
		t.Fatalf("replay event count = %d, want 3", len(replay))
	}
}
