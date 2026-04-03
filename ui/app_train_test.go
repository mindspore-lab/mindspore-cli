package ui

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/vigo999/mindspore-code/ui/model"
)

const largePastedBlock = "line 01\nline 02\nline 03\nline 04\nline 05\nline 06\nline 07\nline 08\n"

func TestTrainFixActionClearsStaleButtonAndKeepsCompletionMessage(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.handleEvent(model.Event{
		Type: model.TrainModeOpen,
		Train: &model.TrainEventData{
			RunID:    "primary",
			RawInput: "qwen3 lora",
			Model:    "qwen3",
			Method:   "lora",
		},
	})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type:    model.TrainAnalysisReady,
		Message: "Analysis complete. Ready to apply fix.",
		Train: &model.TrainEventData{
			RunID:       "primary",
			IssueType:   "failure",
			ActionID:    "fix-dsa-op",
			ActionKind:  "apply_patch",
			ActionLabel: "apply fix",
		},
	})
	app = next.(App)

	if got := len(app.trainView.GlobalActions.Items); got == 0 || app.trainView.GlobalActions.Items[0].ID != "fix-dsa-op" {
		t.Fatalf("expected fix action before execution, got %#v", app.trainView.GlobalActions.Items)
	}

	next, _ = app.handleEvent(model.Event{
		Type:    model.TrainActionApplied,
		Message: "op-agent: implementing DSA operator and compiling custom torch-npu...",
		Train: &model.TrainEventData{
			RunID:     "primary",
			IssueType: "failure",
			ActionID:  "fix-dsa-op",
		},
	})
	app = next.(App)

	run := app.trainView.RunByID("primary")
	if run == nil {
		t.Fatal("expected primary run")
	}
	if run.Phase != model.TrainPhaseFixing {
		t.Fatalf("expected fixing phase, got %s", run.Phase)
	}
	if len(run.AgentActions) != 0 {
		t.Fatalf("expected stale agent actions to be cleared, got %#v", run.AgentActions)
	}
	if got := len(app.trainView.GlobalActions.Items); got != 1 || app.trainView.GlobalActions.Items[0].ID != "stop" {
		t.Fatalf("expected only stop action while fixing, got %#v", app.trainView.GlobalActions.Items)
	}

	next, _ = app.handleEvent(model.Event{
		Type:    model.TrainFixApplied,
		Message: "op-agent: DSA operator finished. New torch wheel is ready. Please rerun experiment.",
		Train: &model.TrainEventData{
			RunID:      "primary",
			FixSummary: "DSA operator implemented and torch-npu recompiled",
		},
	})
	app = next.(App)

	run = app.trainView.RunByID("primary")
	if run == nil {
		t.Fatal("expected primary run after fix")
	}
	if run.Phase != model.TrainPhaseReady {
		t.Fatalf("expected ready phase after fix, got %s", run.Phase)
	}
	if !run.FixApplied {
		t.Fatal("expected run to be marked as fix-applied")
	}
	if got := len(app.trainView.GlobalActions.Items); got != 1 || app.trainView.GlobalActions.Items[0].Label != "rerun" {
		t.Fatalf("expected rerun action after fix, got %#v", app.trainView.GlobalActions.Items)
	}
	if run.StatusMessage != "op-agent: DSA operator finished. New torch wheel is ready. Please rerun experiment." {
		t.Fatalf("expected status message to keep fix completion text, got %q", run.StatusMessage)
	}
	last := app.state.Messages[len(app.state.Messages)-1]
	if !strings.Contains(last.Content, "DSA operator finished") {
		t.Fatalf("expected final agent message to include fix completion text, got %#v", last)
	}
}

func TestTrainViewUsesSharedChatSurface(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 80, Height: 24})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type: model.TrainModeOpen,
		Train: &model.TrainEventData{
			RunID:    "primary",
			RawInput: "qwen3 lora",
			Model:    "qwen3",
			Method:   "lora",
		},
	})
	app = next.(App)

	view := app.View()
	if !strings.Contains(view, "train job") {
		t.Fatalf("expected train HUD in view, got:\n%s", view)
	}
	if strings.Contains(view, "setup env") {
		t.Fatalf("expected old train panel layout to be gone, got:\n%s", view)
	}
	if !strings.Contains(view, "❯ ") {
		t.Fatalf("expected global composer to stay visible, got:\n%s", view)
	}
}

func TestTrainHUDShowsSingleVIPLine(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 140, Height: 32})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type: model.TrainModeOpen,
		Train: &model.TrainEventData{
			RunID:    "primary",
			RawInput: "qwen3 lora",
			Model:    "qwen3",
			Method:   "lora",
		},
	})
	app = next.(App)

	app.trainView.Request.Dataset = "alpaca_gpt4_zh"
	app.trainView.SetupContext.BaseModelRef = "models/qwen3-7b"
	app.trainView.Request.TargetName = "torch-npu-910b-0"
	if run := app.trainView.RunByID("primary"); run != nil {
		run.TargetName = "torch-npu-910b-0"
		run.Device = "Ascend"
	}
	app.trainView.UpsertCheck("primary", model.ChecklistItem{
		Group:   model.TrainCheckGroupLocal,
		Name:    "local_repo",
		Status:  model.TrainCheckPass,
		Summary: "repo detected",
	})
	app.trainView.UpsertCheck("primary", model.ChecklistItem{
		Group:   model.TrainCheckGroupTarget,
		Name:    "ssh",
		Status:  model.TrainCheckRunning,
		Summary: "checking target ssh",
	})

	view := app.View()
	if !strings.Contains(view, "run_id") || !strings.Contains(view, "machine") || !strings.Contains(view, "model") || !strings.Contains(view, "ckpt") || !strings.Contains(view, "dataset") {
		t.Fatalf("expected train HUD VIP fields, got:\n%s", view)
	}
	for _, want := range []string{"primary", "torch-npu-910b-0 npu", "qwen3", "qwen3-7b", "alpaca_gpt4_zh"} {
		if !strings.Contains(view, want) {
			t.Fatalf("expected train HUD to contain %q, got:\n%s", want, view)
		}
	}
	if strings.Contains(view, "local checks") || strings.Contains(view, "target checks") {
		t.Fatalf("expected checklist details to stay out of train HUD, got:\n%s", view)
	}
}

func TestTrainSetupStreamsProgressAndSummaryToChat(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 140, Height: 32})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type: model.TrainModeOpen,
		Train: &model.TrainEventData{
			RunID:    "primary",
			RawInput: "qwen3 lora alpaca_gpt4_zh",
			Model:    "qwen3",
			Method:   "lora",
		},
	})
	app = next.(App)

	events := []model.Event{
		{
			Type: model.TrainConnect,
			Train: &model.TrainEventData{
				RunID:   "primary",
				Host:    "torch-npu-910b-0",
				Address: "8.9.72.194:22",
				Status:  "connecting",
			},
		},
		{
			Type: model.TrainConnect,
			Train: &model.TrainEventData{
				RunID:   "primary",
				Host:    "torch-npu-910b-0",
				Address: "8.9.72.194:22",
				Status:  "connected",
			},
		},
		{
			Type: model.TrainSetup,
			Train: &model.TrainEventData{
				RunID:  "primary",
				Check:  "local_repo",
				Status: "checking",
				Scope:  "local",
			},
		},
		{
			Type: model.TrainSetup,
			Train: &model.TrainEventData{
				RunID:  "primary",
				Check:  "local_repo",
				Status: "passed",
				Detail: "repo detected",
				Scope:  "local",
			},
		},
		{
			Type: model.TrainSetup,
			Train: &model.TrainEventData{
				RunID:  "primary",
				Check:  "ssh",
				Status: "checking",
				Scope:  "target",
			},
		},
		{
			Type: model.TrainSetup,
			Train: &model.TrainEventData{
				RunID:  "primary",
				Check:  "ssh",
				Status: "passed",
				Detail: "weizheng@8.9.72.194:22",
				Scope:  "target",
			},
		},
		{
			Type:    model.TrainReady,
			Message: "all preflight checks passed. ready to start training.",
			Train: &model.TrainEventData{
				RunID:        "primary",
				ActionSource: "setup-helper",
			},
		},
	}

	for _, ev := range events {
		next, _ = app.handleEvent(ev)
		app = next.(App)
	}

	var contents []string
	for _, msg := range app.state.Messages {
		contents = append(contents, msg.Content)
	}
	all := strings.Join(contents, "\n")

	for _, want := range []string{
		"setup-agent",
		"connecting to torch-npu-910b-0 (8.9.72.194:22)...",
		"connected to torch-npu-910b-0 (8.9.72.194:22)",
		"checking repo...",
		"repo ok: repo detected",
		"checking ssh...",
		"ssh ok: weizheng@8.9.72.194:22",
		"setup summary",
		"local checks",
		"target checks",
		"[x] repo: repo detected",
		"[x] ssh: weizheng@8.9.72.194:22",
		"all preflight checks passed. ready to start training.",
		"╭",
		"╰",
	} {
		if !strings.Contains(all, want) {
			t.Fatalf("expected streamed setup content %q, got:\n%s", want, all)
		}
	}
}

func TestUpDownRecallInputHistoryInsteadOfScrollingViewport(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 20})
	app = next.(App)

	app.input.Model.SetValue("first message")
	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = next.(App)

	app.input.Model.SetValue("second message")
	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = next.(App)

	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyUp})
	app = next.(App)
	if got := app.input.Value(); got != "second message" {
		t.Fatalf("expected up to recall latest history entry, got %q", got)
	}

	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyUp})
	app = next.(App)
	if got := app.input.Value(); got != "first message" {
		t.Fatalf("expected second up to recall earlier history entry, got %q", got)
	}

	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyDown})
	app = next.(App)
	if got := app.input.Value(); got != "second message" {
		t.Fatalf("expected down to move forward in history, got %q", got)
	}
}

func TestSubmittedChatInputPersistsThroughAppender(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	var persisted []string
	app = app.WithInputHistoryAppender(func(value string) {
		persisted = append(persisted, value)
	})
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 20})
	app = next.(App)

	app.input.Model.SetValue("hello history")
	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = next.(App)

	select {
	case msg := <-userCh:
		if msg != "hello history" {
			t.Fatalf("expected submit to reach backend, got %q", msg)
		}
	default:
		t.Fatal("expected submit to reach backend")
	}
	if len(persisted) != 1 || persisted[0] != "hello history" {
		t.Fatalf("expected submit to persist once, got %#v", persisted)
	}
}

func TestSubmittedChatInputDoesNotPersistConsecutiveDuplicateTwice(t *testing.T) {
	userCh := make(chan string, 2)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	var persisted []string
	app = app.WithInputHistoryAppender(func(value string) {
		persisted = append(persisted, value)
	})
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 20})
	app = next.(App)

	app.input.Model.SetValue("same input")
	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = next.(App)
	app.input.Model.SetValue("same input")
	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = next.(App)

	if len(persisted) != 1 || persisted[0] != "same input" {
		t.Fatalf("expected consecutive duplicate submit to persist once, got %#v", persisted)
	}
}

func TestCtrlJInsertsComposerNewlineWithoutSubmitting(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 20})
	app = next.(App)

	app.input.Model.SetValue("line one")
	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyCtrlJ})
	app = next.(App)

	if got := app.input.Value(); got != "line one\n" {
		t.Fatalf("expected ctrl+j to insert newline, got %q", got)
	}
	select {
	case msg := <-userCh:
		t.Fatalf("expected ctrl+j not to submit input, got %q", msg)
	default:
	}

	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("line two")})
	app = next.(App)
	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = next.(App)

	select {
	case msg := <-userCh:
		if msg != "line one\nline two" {
			t.Fatalf("expected multiline submit, got %q", msg)
		}
	default:
		t.Fatal("expected enter to submit multiline input")
	}
}

func TestSoftWrappedComposerLinesStayVisibleInAppView(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 15, Height: 12})
	app = next.(App)

	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("alpha beta")})
	app = next.(App)

	view := app.View()
	if !strings.Contains(view, "❯ alpha ") {
		t.Fatalf("expected first wrapped composer row in app view, got:\n%s", view)
	}
	if !strings.Contains(view, "  beta") {
		t.Fatalf("expected second wrapped composer row in app view, got:\n%s", view)
	}
}

func TestPastedComposerLinesStayVisibleInAppView(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 20, Height: 12})
	app = next.(App)

	next, _ = app.handleKey(tea.KeyMsg{
		Type:  tea.KeyRunes,
		Runes: []rune("line 01\nline 02\nline 03\nline 04"),
		Paste: true,
	})
	app = next.(App)

	view := app.View()
	for _, want := range []string{"line 01", "line 02", "line 03", "line 04"} {
		if !strings.Contains(view, want) {
			t.Fatalf("expected pasted line %q in app view, got:\n%s", want, view)
		}
	}
}

func TestBackslashEnterInsertsComposerNewlineWithoutSubmitting(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 20})
	app = next.(App)

	app.input.Model.SetValue("line one\\")
	app.input.Model.SetCursor(len([]rune("line one\\")))
	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = next.(App)

	if got := app.input.Value(); got != "line one\n" {
		t.Fatalf("expected backslash+enter to insert newline, got %q", got)
	}
	select {
	case msg := <-userCh:
		t.Fatalf("expected backslash+enter not to submit input, got %q", msg)
	default:
	}

	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("line two")})
	app = next.(App)
	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = next.(App)

	select {
	case msg := <-userCh:
		if msg != "line one\nline two" {
			t.Fatalf("expected multiline submit after backslash+enter, got %q", msg)
		}
	default:
		t.Fatal("expected enter to submit multiline input")
	}
}

func TestEnterAcceptsSlashSuggestion(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 20})
	app = next.(App)

	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'/'}})
	app = next.(App)
	if !app.input.IsSlashMode() {
		t.Fatal("expected slash suggestions to be visible")
	}

	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = next.(App)

	if got := app.input.Value(); got == "/" || got == "" {
		t.Fatalf("expected enter to accept a slash suggestion, got %q", got)
	}
	select {
	case msg := <-userCh:
		t.Fatalf("expected suggestion selection not to submit input, got %q", msg)
	default:
	}
}

func TestBackslashEnterAcceptsSlashSuggestionWhenCandidatesExist(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 20})
	app = next.(App)

	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'/'}})
	app = next.(App)
	if !app.input.HasSuggestions() {
		t.Fatal("expected slash suggestions to be visible")
	}

	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = next.(App)

	if got := app.input.Value(); got == "/" || got == "" {
		t.Fatalf("expected enter to accept a slash suggestion, got %q", got)
	}
	select {
	case msg := <-userCh:
		t.Fatalf("expected suggestion selection not to submit input, got %q", msg)
	default:
	}
}

func TestBackslashEnterInsertsNewlineWhenSlashModeHasNoSuggestions(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 20})
	app = next.(App)

	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("/nope")})
	app = next.(App)
	if app.input.HasSuggestions() {
		t.Fatal("expected no slash suggestions for unmatched command")
	}

	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'\\'}})
	app = next.(App)
	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = next.(App)

	if got := app.input.Value(); got != "/nope\n" {
		t.Fatalf("expected backslash+enter to insert newline when no suggestions exist, got %q", got)
	}
	select {
	case msg := <-userCh:
		t.Fatalf("expected backslash+enter not to submit input, got %q", msg)
	default:
	}
}

func TestLargePastedUserMessageRendersFullContent(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 20})
	app = next.(App)

	next, _ = app.handleKey(tea.KeyMsg{
		Type:  tea.KeyRunes,
		Runes: []rune(largePastedBlock),
		Paste: true,
	})
	app = next.(App)

	if got := app.input.Value(); got != largePastedBlock {
		t.Fatalf("expected pasted content to remain verbatim in composer, got %q", got)
	}
	if view := app.View(); !strings.Contains(view, "line 01") || !strings.Contains(view, "line 07") {
		t.Fatalf("expected composer view to show full pasted content, got:\n%s", view)
	}

	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = next.(App)

	select {
	case msg := <-userCh:
		if msg != strings.TrimSpace(largePastedBlock) {
			t.Fatalf("expected submitted value to preserve raw pasted content, got %q", msg)
		}
	default:
		t.Fatal("expected pasted input submit to reach backend")
	}

	found := false
	for _, msg := range app.state.Messages {
		if msg.Kind == model.MsgUser && strings.Contains(msg.Content, "line 01") && strings.Contains(msg.Content, "line 07") {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("expected pasted content to be recorded in state messages")
	}
}

func TestUpDoesNotRecallHistoryWhileInsideMultilineComposer(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 20})
	app = next.(App)

	app.input.Model.SetValue("first message")
	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = next.(App)

	app.input.Model.SetValue("second message")
	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = next.(App)

	app.input.Model.SetValue("draft line 1\ndraft line 2")
	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyUp})
	app = next.(App)

	if got := app.input.Value(); got != "draft line 1\ndraft line 2" {
		t.Fatalf("expected up to stay inside multiline draft instead of recalling history, got %q", got)
	}
}

func TestUpDownContinueHistoryAcrossSlashCommandsWithoutEnteringSuggestionNavigation(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 20})
	app = next.(App)

	app.input.Model.SetValue("/project")
	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = next.(App)

	app.input.Model.SetValue("hello")
	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = next.(App)

	app.input.Model.SetValue("draft")
	app.input.Model.SetCursor(len("draft"))

	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyUp})
	app = next.(App)
	if got := app.input.Value(); got != "hello" {
		t.Fatalf("expected first up to recall latest history entry, got %q", got)
	}
	if app.input.IsSlashMode() {
		t.Fatal("expected no slash suggestions while browsing history")
	}

	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyUp})
	app = next.(App)
	if got := app.input.Value(); got != "/project" {
		t.Fatalf("expected second up to recall slash history entry, got %q", got)
	}
	if app.input.IsSlashMode() {
		t.Fatal("expected slash history recall not to enter suggestion navigation")
	}

	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyDown})
	app = next.(App)
	if got := app.input.Value(); got != "hello" {
		t.Fatalf("expected down to keep moving forward in history, got %q", got)
	}

	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyDown})
	app = next.(App)
	if got := app.input.Value(); got != "draft" {
		t.Fatalf("expected second down to restore draft, got %q", got)
	}
	if app.input.IsSlashMode() {
		t.Fatal("expected restored draft to stay out of slash suggestion mode")
	}
}

func TestEscInterruptTokenSentForQueuedTrain(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false
	app.trainView.Active = true
	app.queuedInputs = []string{"/train qwen3 lora"}
	app.trainView.Runs = []model.TrainRunState{{
		ID:    "primary",
		Phase: model.TrainPhaseSetup,
	}}
	app.trainView.ActiveRunID = "primary"

	next, _ := app.handleKey(tea.KeyMsg{Type: tea.KeyEsc})
	app = next.(App)
	_ = app

	select {
	case msg := <-userCh:
		if msg != "/train exit" {
			t.Fatalf("expected esc to send /train exit for queued interrupt, got %q", msg)
		}
	default:
		t.Fatal("expected esc to send /train exit for queued interrupt")
	}
}

func TestCtrlCSendsInterruptTokenForActiveTask(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false
	app.state = app.state.WithThinking(true)

	next, _ := app.handleKey(tea.KeyMsg{Type: tea.KeyCtrlC})
	app = next.(App)

	select {
	case msg := <-userCh:
		if msg != interruptActiveTaskToken {
			t.Fatalf("expected ctrl+c to send interrupt token, got %q", msg)
		}
	default:
		t.Fatal("expected ctrl+c to send interrupt token")
	}

	last := app.state.Messages[len(app.state.Messages)-1]
	if !strings.Contains(last.Content, "Interrupt requested.") {
		t.Fatalf("expected interrupt hint message, got %#v", last)
	}
	if strings.Contains(last.Content, "mscode resume") {
		t.Fatalf("expected interrupt hint without resume text, got %#v", last)
	}
	if app.state.IsThinking {
		t.Fatal("expected ctrl+c to clear thinking state")
	}
	if view := app.View(); strings.Contains(view, "Working...") {
		t.Fatalf("expected ctrl+c to remove thinking indicator, got:\n%s", view)
	}
}

func TestInlineModeCtrlCPrintsInterruptHint(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false
	app.state = app.state.WithThinking(true)

	next, cmd := app.handleKey(tea.KeyMsg{Type: tea.KeyCtrlC})
	app = next.(App)

	select {
	case msg := <-userCh:
		if msg != interruptActiveTaskToken {
			t.Fatalf("expected ctrl+c to send interrupt token, got %q", msg)
		}
	default:
		t.Fatal("expected ctrl+c to send interrupt token")
	}

	if cmd == nil {
		t.Fatal("expected inline mode ctrl+c to print interrupt hint into normal buffer")
	}
	last := app.state.Messages[len(app.state.Messages)-1]
	if !strings.Contains(last.Content, "Interrupt requested.") {
		t.Fatalf("expected interrupt hint message, got %#v", last)
	}
	if app.state.IsThinking {
		t.Fatal("expected ctrl+c to clear thinking state")
	}
	if view := app.View(); strings.Contains(view, "Working...") {
		t.Fatalf("expected ctrl+c to remove inline thinking indicator, got:\n%s", view)
	}
}

func TestToolErrorClearsThinkingIndicator(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 20})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{Type: model.AgentThinking})
	app = next.(App)
	if view := app.View(); !strings.Contains(view, "Working...") {
		t.Fatalf("expected thinking indicator to render from state, got:\n%s", view)
	}

	next, _ = app.handleEvent(model.Event{
		Type:     model.ToolError,
		ToolName: "Engine",
		Message:  "LLM error: boom",
	})
	app = next.(App)

	if app.state.IsThinking {
		t.Fatal("expected tool error to clear thinking state")
	}
	view := app.View()
	if strings.Contains(view, "Working...") {
		t.Fatalf("expected no stale thinking indicator after tool error, got:\n%s", view)
	}
}

func TestBusyTrainQueuesInputInBannerInsteadOfChatStream(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	var persisted []string
	app = app.WithInputHistoryAppender(func(value string) {
		persisted = append(persisted, value)
	})
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 120, Height: 24})
	app = next.(App)
	app.trainView.Active = true
	app.trainView.Runs = []model.TrainRunState{{
		ID:    "primary",
		Phase: model.TrainPhaseSetup,
	}}
	app.trainView.ActiveRunID = "primary"

	app.input.Model.SetValue("/train qwen3 lora")
	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = next.(App)

	if got := len(app.queuedInputs); got != 1 {
		t.Fatalf("expected one queued input, got %d", got)
	}
	if len(app.state.Messages) != 0 {
		t.Fatalf("expected queued input to stay out of chat stream, got %#v", app.state.Messages)
	}
	select {
	case msg := <-userCh:
		t.Fatalf("expected no immediate backend submit while busy, got %q", msg)
	default:
	}

	view := app.View()
	if !strings.Contains(view, "messages queued (press esc to interrupt)") {
		t.Fatalf("expected queued-input banner in view, got:\n%s", view)
	}
	if strings.Contains(view, "> /train qwen3 lora") {
		t.Fatalf("expected queued command to stay out of chat stream, got:\n%s", view)
	}
	if len(persisted) != 1 || persisted[0] != "/train qwen3 lora" {
		t.Fatalf("expected queued input to persist once, got %#v", persisted)
	}
}
