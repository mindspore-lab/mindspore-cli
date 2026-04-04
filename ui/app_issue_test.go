package ui

import (
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	issuepkg "github.com/mindspore-lab/mindspore-cli/internal/issues"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

func TestIssueViewUsesDedicatedSurfaceAndShowsComposerInDetail(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 28})
	app = next.(App)

	now := time.Now()
	next, _ = app.handleEvent(model.Event{
		Type: model.IssueIndexOpen,
		IssueView: &model.IssueEventData{
			Filter: "all",
			Items: []issuepkg.Issue{
				{ID: 42, Key: "ISSUE-42", Title: "acc failure in migrate", Kind: issuepkg.KindAccuracy, Status: "ready", Reporter: "alice", UpdatedAt: now},
			},
		},
	})
	app = next.(App)

	view := app.View()
	if !strings.Contains(view, "ISSUES") {
		t.Fatalf("expected issue view header, got:\n%s", view)
	}

	next, _ = app.handleEvent(model.Event{
		Type: model.IssueDetailOpen,
		IssueView: &model.IssueEventData{
			ID:        42,
			Issue:     &issuepkg.Issue{ID: 42, Key: "ISSUE-42", Title: "acc failure in migrate", Kind: issuepkg.KindAccuracy, Status: "doing", Lead: "bob", Reporter: "alice", Summary: "baseline vs candidate mismatch", UpdatedAt: now},
			Notes:     []issuepkg.Note{{IssueID: 42, Author: "alice", Content: "maybe dtype mismatch", CreatedAt: now}},
			Activity:  []issuepkg.Activity{{IssueID: 42, Actor: "bob", Text: "bob took lead", CreatedAt: now}},
			FromIndex: true,
		},
	})
	app = next.(App)

	detail := app.View()
	if !strings.Contains(detail, "SUMMARY") || !strings.Contains(detail, "NOTES") {
		t.Fatalf("expected issue detail sections, got:\n%s", detail)
	}
	if !strings.Contains(detail, "Add note to ISSUE-42") {
		t.Fatalf("expected issue note composer, got:\n%s", detail)
	}
}

func TestIssueDetailComposerSubmitsMultilineNote(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	var persisted []string
	app = app.WithInputHistoryAppender(func(value string) {
		persisted = append(persisted, value)
	})
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 28})
	app = next.(App)

	now := time.Now()
	next, _ = app.handleEvent(model.Event{
		Type: model.IssueDetailOpen,
		IssueView: &model.IssueEventData{
			ID:    42,
			Issue: &issuepkg.Issue{ID: 42, Key: "ISSUE-42", Title: "acc failure in migrate", Kind: issuepkg.KindAccuracy, Status: "doing", Lead: "bob", Reporter: "alice", UpdatedAt: now},
		},
	})
	app = next.(App)

	app.input.Model.SetValue("line one")
	next, _ = app.handleIssueDetailKey(tea.KeyMsg{Type: tea.KeyCtrlJ})
	app = next.(App)
	next, _ = app.handleIssueDetailKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("line two")})
	app = next.(App)
	next, _ = app.handleIssueDetailKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = next.(App)

	select {
	case msg := <-userCh:
		if msg != "/__issue_note ISSUE-42 line one\nline two" {
			t.Fatalf("expected multiline issue note submit, got %q", msg)
		}
	default:
		t.Fatal("expected multiline issue note submit")
	}

	if got := app.input.Value(); got != "" {
		t.Fatalf("expected composer reset after submit, got %q", got)
	}
	if len(persisted) != 1 || persisted[0] != "line one\nline two" {
		t.Fatalf("expected issue detail submit to persist original input, got %#v", persisted)
	}
}

func TestIssueDetailComposerBackslashEnterInsertsNewlineWithoutSubmitting(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 28})
	app = next.(App)

	now := time.Now()
	next, _ = app.handleEvent(model.Event{
		Type: model.IssueDetailOpen,
		IssueView: &model.IssueEventData{
			ID:    42,
			Issue: &issuepkg.Issue{ID: 42, Key: "ISSUE-42", Title: "acc failure in migrate", Kind: issuepkg.KindAccuracy, Status: "doing", Lead: "bob", Reporter: "alice", UpdatedAt: now},
		},
	})
	app = next.(App)

	app.input.Model.SetValue("line one\\")
	app.input.Model.SetCursor(len([]rune("line one\\")))
	next, _ = app.handleIssueDetailKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = next.(App)

	if got := app.input.Value(); got != "line one\n" {
		t.Fatalf("expected backslash+enter to insert newline, got %q", got)
	}
	select {
	case msg := <-userCh:
		t.Fatalf("expected backslash+enter not to submit issue note, got %q", msg)
	default:
	}

	next, _ = app.handleIssueDetailKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("line two")})
	app = next.(App)
	next, _ = app.handleIssueDetailKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = next.(App)

	select {
	case msg := <-userCh:
		if msg != "/__issue_note ISSUE-42 line one\nline two" {
			t.Fatalf("expected multiline issue note submit, got %q", msg)
		}
	default:
		t.Fatal("expected multiline issue note submit")
	}
}

func TestIssueDetailBackslashEnterAcceptsSlashSuggestionWhenCandidatesExist(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 28})
	app = next.(App)

	now := time.Now()
	next, _ = app.handleEvent(model.Event{
		Type: model.IssueDetailOpen,
		IssueView: &model.IssueEventData{
			ID:    42,
			Issue: &issuepkg.Issue{ID: 42, Key: "ISSUE-42", Title: "acc failure in migrate", Kind: issuepkg.KindAccuracy, Status: "doing", Lead: "bob", Reporter: "alice", UpdatedAt: now},
		},
	})
	app = next.(App)

	next, _ = app.handleIssueDetailKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'/'}})
	app = next.(App)
	if !app.input.HasSuggestions() {
		t.Fatal("expected slash suggestions to be visible")
	}

	next, _ = app.handleIssueDetailKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = next.(App)

	if got := app.input.Value(); got == "/" || got == "" {
		t.Fatalf("expected enter to accept a slash suggestion, got %q", got)
	}
	select {
	case msg := <-userCh:
		t.Fatalf("expected suggestion selection not to submit issue input, got %q", msg)
	default:
	}
}
