package app

import (
	"strings"
	"testing"
	"time"

	agentctx "github.com/mindspore-lab/mindspore-cli/agent/context"
	"github.com/mindspore-lab/mindspore-cli/agent/session"
	"github.com/mindspore-lab/mindspore-cli/integrations/llm"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

func TestCmdResumeOpensSessionPicker(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	workDir := t.TempDir()
	saved, err := session.Create(workDir, "system prompt")
	if err != nil {
		t.Fatalf("create saved session: %v", err)
	}
	if err := saved.AppendUserInput("fix the replay command"); err != nil {
		t.Fatalf("append user input: %v", err)
	}
	if err := saved.AppendAssistant("done"); err != nil {
		t.Fatalf("append assistant reply: %v", err)
	}
	if err := saved.Activate(); err != nil {
		t.Fatalf("activate saved session: %v", err)
	}
	if err := saved.Close(); err != nil {
		t.Fatalf("close saved session: %v", err)
	}

	app := newModelCommandTestApp()
	app.WorkDir = workDir

	app.cmdResume(nil)

	ev := drainUntilEventType(t, app, model.SessionPickerOpen)
	if ev.SessionPicker == nil {
		t.Fatal("expected session picker payload")
	}
	if got, want := ev.SessionPicker.Mode, model.SessionPickerResume; got != want {
		t.Fatalf("picker mode = %q, want %q", got, want)
	}
	if got, want := len(ev.SessionPicker.Items), 1; got != want {
		t.Fatalf("picker item count = %d, want %d", got, want)
	}
	if got, want := ev.SessionPicker.Items[0].ID, saved.ID(); got != want {
		t.Fatalf("picker item id = %q, want %q", got, want)
	}
	if got, want := ev.SessionPicker.Items[0].FirstUserInput, "fix the replay command"; got != want {
		t.Fatalf("picker first user input = %q, want %q", got, want)
	}
}

func TestCmdResumeSwitchesConversationAndShowsReturnHint(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	workDir := t.TempDir()
	current, err := session.Create(workDir, "system prompt")
	if err != nil {
		t.Fatalf("create current session: %v", err)
	}
	if err := current.AppendUserInput("current conversation"); err != nil {
		t.Fatalf("append current user input: %v", err)
	}
	if err := current.AppendAssistant("current reply"); err != nil {
		t.Fatalf("append current assistant reply: %v", err)
	}
	if err := current.Activate(); err != nil {
		t.Fatalf("activate current session: %v", err)
	}

	target, err := session.Create(workDir, "system prompt")
	if err != nil {
		t.Fatalf("create target session: %v", err)
	}
	if err := target.AppendUserInput("target conversation"); err != nil {
		t.Fatalf("append target user input: %v", err)
	}
	if err := target.AppendAssistant("target reply"); err != nil {
		t.Fatalf("append target assistant reply: %v", err)
	}
	if err := target.Activate(); err != nil {
		t.Fatalf("activate target session: %v", err)
	}
	if err := target.SaveSnapshotWithUsage("system prompt", []llm.Message{
		llm.NewUserMessage("target conversation"),
		llm.NewAssistantMessage("target reply"),
	}, &session.UsageSnapshot{
		Provider:   "anthropic",
		TokenScope: "total",
		Tokens:     1809,
		LocalDelta: 21,
	}); err != nil {
		t.Fatalf("save target snapshot with usage: %v", err)
	}
	if err := target.Close(); err != nil {
		t.Fatalf("close target session: %v", err)
	}

	ctxManager := agentctx.NewManager(agentctx.ManagerConfig{
		ContextWindow: 4096,
		ReserveTokens: 512,
	})
	ctxManager.SetSystemPrompt("system prompt")
	if err := ctxManager.AddMessage(llm.NewUserMessage("current conversation")); err != nil {
		t.Fatalf("add current user context: %v", err)
	}
	if err := ctxManager.AddMessage(llm.NewAssistantMessage("current reply")); err != nil {
		t.Fatalf("add current assistant context: %v", err)
	}

	app := newModelCommandTestApp()
	app.WorkDir = workDir
	app.session = current
	app.ctxManager = ctxManager
	app.sessionLLMActivity.Store(true)

	app.cmdResume([]string{target.ID()})

	clearEv := drainUntilEventType(t, app, model.ClearScreen)
	if !strings.Contains(clearEv.Summary, current.ID()) {
		t.Fatalf("clear summary = %q, want current session hint", clearEv.Summary)
	}
	if got, want := app.session.ID(), target.ID(); got != want {
		t.Fatalf("active session id = %q, want %q", got, want)
	}
	if got, want := app.ctxManager.TokenUsage().Current, 1830; got != want {
		t.Fatalf("ctx current after resume = %d, want %d", got, want)
	}

	timer := time.NewTimer(2 * time.Second)
	defer timer.Stop()

	var replayed []model.EventType
	for len(replayed) < 2 {
		select {
		case ev := <-app.EventCh:
			if ev.Type == model.UserInput || ev.Type == model.AgentReply {
				replayed = append(replayed, ev.Type)
			}
		case <-timer.C:
			t.Fatalf("timed out waiting for replayed session events, got %v", replayed)
		}
	}
	if replayed[0] != model.UserInput || replayed[1] != model.AgentReply {
		t.Fatalf("replayed event order = %v, want [UserInput AgentReply]", replayed)
	}
}

func TestCmdResumeFromEmptyConversationSkipsClearScreen(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	workDir := t.TempDir()
	current, err := session.Create(workDir, "system prompt")
	if err != nil {
		t.Fatalf("create current session: %v", err)
	}
	if err := current.Close(); err != nil {
		t.Fatalf("close current session: %v", err)
	}

	target, err := session.Create(workDir, "system prompt")
	if err != nil {
		t.Fatalf("create target session: %v", err)
	}
	if err := target.AppendUserInput("target conversation"); err != nil {
		t.Fatalf("append target user input: %v", err)
	}
	if err := target.AppendAssistant("target reply"); err != nil {
		t.Fatalf("append target assistant reply: %v", err)
	}
	if err := target.Activate(); err != nil {
		t.Fatalf("activate target session: %v", err)
	}
	if err := target.Close(); err != nil {
		t.Fatalf("close target session: %v", err)
	}

	ctxManager := agentctx.NewManager(agentctx.ManagerConfig{
		ContextWindow: 4096,
		ReserveTokens: 512,
	})
	ctxManager.SetSystemPrompt("system prompt")

	app := newModelCommandTestApp()
	app.WorkDir = workDir
	app.session = current
	app.ctxManager = ctxManager

	app.cmdResume([]string{target.ID()})

	timer := time.NewTimer(2 * time.Second)
	defer timer.Stop()

	var replayed []model.EventType
	for len(replayed) < 2 {
		select {
		case ev := <-app.EventCh:
			if ev.Type == model.ClearScreen {
				t.Fatalf("unexpected clear screen event: %#v", ev)
			}
			if ev.Type == model.ToolError {
				t.Fatalf("unexpected tool error: %#v", ev)
			}
			if ev.Type == model.UserInput || ev.Type == model.AgentReply {
				replayed = append(replayed, ev.Type)
			}
		case <-timer.C:
			t.Fatalf("timed out waiting for replayed session events, got %v", replayed)
		}
	}

	if replayed[0] != model.UserInput || replayed[1] != model.AgentReply {
		t.Fatalf("replayed event order = %v, want [UserInput AgentReply]", replayed)
	}
}
