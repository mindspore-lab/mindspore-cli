package app

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	ctxmanager "github.com/mindspore-lab/mindspore-cli/agent/context"
	"github.com/mindspore-lab/mindspore-cli/integrations/llm"
	"github.com/mindspore-lab/mindspore-cli/integrations/skills"
	issuepkg "github.com/mindspore-lab/mindspore-cli/internal/issues"
	"github.com/mindspore-lab/mindspore-cli/internal/project"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

func TestExpandInputTextExpandsStandaloneTokensAndEscapes(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, root, "a.txt", "alpha")
	writeTestFile(t, root, "b.txt", "beta")

	app := &Application{WorkDir: root}
	got, err := app.expandInputText("read @a.txt and @@literal then @b.txt")
	if err != nil {
		t.Fatalf("expandInputText returned error: %v", err)
	}

	if !strings.Contains(got, `[file path="`+filepath.ToSlash(filepath.Join(root, "a.txt"))+`"]`) {
		t.Fatalf("expected a.txt contents to be expanded, got %q", got)
	}
	if !strings.Contains(got, `[file path="`+filepath.ToSlash(filepath.Join(root, "b.txt"))+`"]`) {
		t.Fatalf("expected b.txt contents to be expanded, got %q", got)
	}
	if strings.Contains(got, "alpha") || strings.Contains(got, "beta") {
		t.Fatalf("expected file contents not to be inlined, got %q", got)
	}
	if !strings.Contains(got, "@literal") {
		t.Fatalf("expected @@ escape to keep literal @, got %q", got)
	}
}

func TestExpandInputTextLeavesUnsupportedAtFormsUnchanged(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, root, "ctx.txt", "context")

	app := &Application{WorkDir: root}
	input := "see @ctx.txt, (@ctx.txt) user@ctx.txt"
	got, err := app.expandInputText(input)
	if err != nil {
		t.Fatalf("expandInputText returned error: %v", err)
	}
	if got != input {
		t.Fatalf("unsupported @ forms should stay unchanged, got %q", got)
	}
}

func TestExpandInputTextRejectsUnsafeFiles(t *testing.T) {
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, "dir"), 0o755); err != nil {
		t.Fatal(err)
	}

	app := &Application{WorkDir: root}
	tests := []struct {
		input string
		want  string
	}{
		{"@missing.txt", "file not found"},
		{"@dir", "path is a directory"},
		{"@../escape.txt", "path escapes working directory"},
	}

	for _, tc := range tests {
		_, err := app.expandInputText(tc.input)
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Fatalf("expandInputText(%q) error = %v, want substring %q", tc.input, err, tc.want)
		}
	}
}

func TestProcessInputExpandsPlainChatBeforeRunTask(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, root, "ctx.txt", "context payload")

	app := &Application{
		WorkDir:  root,
		EventCh:  make(chan model.Event, 8),
		llmReady: false,
		ctxManager: ctxmanager.NewManager(ctxmanager.ManagerConfig{
			ContextWindow: 24000,
			ReserveTokens: 4000,
		}),
	}

	app.processInput("please read @ctx.txt")

	ev := drainUntilEventType(t, app, model.AgentReply)
	if ev.Message != provideAPIKeyFirstMsg {
		t.Fatalf("expected unavailable reply, got %q", ev.Message)
	}

	msgs := app.ctxManager.GetNonSystemMessages()
	if len(msgs) < 1 || msgs[0].Role != "user" {
		t.Fatalf("expected recorded user message, got %#v", msgs)
	}
	if !strings.Contains(msgs[0].Content, `[file path="`+filepath.ToSlash(filepath.Join(root, "ctx.txt"))+`"]`) {
		t.Fatalf("expected expanded plain chat to be recorded, got %q", msgs[0].Content)
	}
	if strings.Contains(msgs[0].Content, "context payload") {
		t.Fatalf("expected file content not to be recorded inline, got %q", msgs[0].Content)
	}
}

func TestProcessInputEmitsExpandedUserInputEvent(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, root, "ctx.txt", "context payload")

	app := &Application{
		WorkDir:  root,
		EventCh:  make(chan model.Event, 8),
		llmReady: false,
		ctxManager: ctxmanager.NewManager(ctxmanager.ManagerConfig{
			ContextWindow: 24000,
			ReserveTokens: 4000,
		}),
	}

	app.processInput("please read @ctx.txt")

	ev := drainUntilEventType(t, app, model.UserInput)
	if !strings.Contains(ev.Message, `[file path="`+filepath.ToSlash(filepath.Join(root, "ctx.txt"))+`"]`) {
		t.Fatalf("expected expanded user input event, got %q", ev.Message)
	}
	if strings.Contains(ev.Message, "context payload") {
		t.Fatalf("expected user input event not to inline file content, got %q", ev.Message)
	}
}

func TestHandleCommandProjectDoesNotExpandExcludedCommand(t *testing.T) {
	store := newMockProjectStore()
	app := &Application{
		WorkDir:        t.TempDir(),
		EventCh:        make(chan model.Event, 8),
		projectService: project.NewService(store),
		issueUser:      "alice",
		issueRole:      "admin",
	}

	app.handleCommand(`/project add tasks "@missing.txt" --owner bob --progress 30`)

	ev := drainUntilEventType(t, app, model.AgentReply)
	if !strings.Contains(ev.Message, "created task #1") {
		t.Fatalf("expected project command to succeed unchanged, got %q", ev.Message)
	}
	if got := store.tasks[0].Title; got != "@missing.txt" {
		t.Fatalf("excluded command should keep literal title, got %q", got)
	}
}

func TestHandleCommandReportExpandsOnlyTitleRemainder(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, root, "ctx.txt", "reported context")

	store := &fakeAppIssueStore{}
	app := &Application{
		WorkDir:      root,
		EventCh:      make(chan model.Event, 8),
		issueService: issuepkg.NewService(store),
		issueUser:    "alice",
	}

	app.handleCommand(`/report accuracy @ctx.txt`)

	drainUntilEventType(t, app, model.AgentReply)
	if got := store.lastCreateKind; got != issuepkg.KindAccuracy {
		t.Fatalf("kind = %q, want %q", got, issuepkg.KindAccuracy)
	}
	if !strings.Contains(store.lastCreateTitle, `[file path="`+filepath.ToSlash(filepath.Join(root, "ctx.txt"))+`"]`) {
		t.Fatalf("expected expanded report title, got %q", store.lastCreateTitle)
	}
	if strings.Contains(store.lastCreateTitle, "reported context") {
		t.Fatalf("expected report title not to inline file content, got %q", store.lastCreateTitle)
	}
}

func TestHandleCommandReportBadReferenceFailsWholeInput(t *testing.T) {
	store := &fakeAppIssueStore{}
	app := &Application{
		WorkDir:      t.TempDir(),
		EventCh:      make(chan model.Event, 8),
		issueService: issuepkg.NewService(store),
		issueUser:    "alice",
	}

	app.handleCommand(`/report accuracy @missing.txt`)

	ev := drainUntilEventType(t, app, model.ToolError)
	if !strings.Contains(ev.Message, "Failed to expand @file input") {
		t.Fatalf("expected input expansion error, got %q", ev.Message)
	}
	if store.lastCreateTitle != "" {
		t.Fatalf("report command should not execute on bad @file, got title %q", store.lastCreateTitle)
	}
}

func TestHandleCommandFixPreservesIssueModeAndExpandsPromptRemainder(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, root, "ctx.txt", "fix context")
	store := &fakeAppIssueStore{
		issue: &issuepkg.Issue{ID: 42, Key: "ISSUE-42", Title: "demo issue", Kind: issuepkg.KindFailure},
	}

	app := &Application{
		WorkDir:  root,
		EventCh:  make(chan model.Event, 8),
		llmReady: false,
		ctxManager: ctxmanager.NewManager(ctxmanager.ManagerConfig{
			ContextWindow: 24000,
			ReserveTokens: 4000,
		}),
		issueService: issuepkg.NewService(store),
	}

	app.handleCommand(`/fix ISSUE-42 @ctx.txt`)

	ev := drainUntilEventType(t, app, model.AgentReply)
	if ev.Message != provideAPIKeyFirstMsg {
		t.Fatalf("expected unavailable reply, got %q", ev.Message)
	}
	msgs := app.ctxManager.GetNonSystemMessages()
	if !containsUserMessage(msgs, "ISSUE-42") {
		t.Fatalf("expected issue-target mode to be preserved, got %#v", msgs)
	}
	if !containsUserMessage(msgs, `[file path="`+filepath.ToSlash(filepath.Join(root, "ctx.txt"))+`"]`) {
		t.Fatalf("expected expanded prompt remainder, got %#v", msgs)
	}
	if containsUserMessage(msgs, "fix context") {
		t.Fatalf("expected fix prompt not to inline file content, got %#v", msgs)
	}
}

func TestHandleCommandFixFileFirstStaysFreeTextMode(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, root, "ctx.txt", "fix context")

	app := &Application{
		WorkDir:  root,
		EventCh:  make(chan model.Event, 8),
		llmReady: false,
		ctxManager: ctxmanager.NewManager(ctxmanager.ManagerConfig{
			ContextWindow: 24000,
			ReserveTokens: 4000,
		}),
	}

	app.handleCommand(`/fix @ctx.txt ISSUE-42`)

	ev := drainUntilEventType(t, app, model.AgentReply)
	if ev.Message != provideAPIKeyFirstMsg {
		t.Fatalf("expected unavailable reply, got %q", ev.Message)
	}
	msgs := app.ctxManager.GetNonSystemMessages()
	if containsUserMessage(msgs, "Issue: ISSUE-42") {
		t.Fatalf("file-first input should not switch to issue mode, got %#v", msgs)
	}
	if !containsUserMessage(msgs, filepath.ToSlash(filepath.Join(root, "ctx.txt"))) || !containsUserMessage(msgs, "ISSUE-42") {
		t.Fatalf("expected free-text mode with expanded content, got %#v", msgs)
	}
	if containsUserMessage(msgs, "fix context") {
		t.Fatalf("expected free-text mode not to inline file content, got %#v", msgs)
	}
}

func TestHandleCommandSkillAndAliasExpandOnlyRequestRemainder(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, root, "req.txt", "skill request")
	skillDir := filepath.Join(root, "skills")
	createTestSkill(t, skillDir, "demo")

	app := &Application{
		WorkDir:     root,
		EventCh:     make(chan model.Event, 16),
		llmReady:    false,
		ctxManager:  ctxmanager.NewManager(ctxmanager.ManagerConfig{ContextWindow: 24000, ReserveTokens: 4000}),
		skillLoader: skills.NewLoader(skillDir),
	}

	app.handleCommand(`/skill demo @req.txt`)
	drainUntilEventType(t, app, model.ToolSkill)
	drainUntilEventType(t, app, model.AgentReply)

	msgs := app.ctxManager.GetNonSystemMessages()
	if !containsUserMessage(msgs, `[file path="`+filepath.ToSlash(filepath.Join(root, "req.txt"))+`"]`) {
		t.Fatalf("expected /skill request to be expanded, got %#v", msgs)
	}

	app.ctxManager.Clear()
	app.handleCommand(`/demo @req.txt`)
	drainUntilEventType(t, app, model.ToolSkill)
	drainUntilEventType(t, app, model.AgentReply)

	msgs = app.ctxManager.GetNonSystemMessages()
	if !containsUserMessage(msgs, `[file path="`+filepath.ToSlash(filepath.Join(root, "req.txt"))+`"]`) {
		t.Fatalf("expected skill alias request to be expanded, got %#v", msgs)
	}
}

func containsUserMessage(msgs []llm.Message, needle string) bool {
	for _, msg := range msgs {
		if msg.Role == "user" && strings.Contains(msg.Content, needle) {
			return true
		}
	}
	return false
}

func writeTestFile(t *testing.T, root, relativePath, content string) {
	t.Helper()
	path := filepath.Join(root, relativePath)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func createTestSkill(t *testing.T, skillsRoot, name string) {
	t.Helper()
	skillPath := filepath.Join(skillsRoot, name)
	if err := os.MkdirAll(skillPath, 0o755); err != nil {
		t.Fatal(err)
	}
	content := "---\nname: " + name + "\ndescription: demo skill\n---\n\nbody"
	if err := os.WriteFile(filepath.Join(skillPath, "SKILL.md"), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}
