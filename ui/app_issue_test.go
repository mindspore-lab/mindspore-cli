package ui

import (
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	issuepkg "github.com/vigo999/ms-cli/internal/issues"
	"github.com/vigo999/ms-cli/ui/model"
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
