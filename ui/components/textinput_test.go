package components

import (
	"fmt"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/vigo999/ms-cli/ui/slash"
)

func TestTextInputHistoryRecall(t *testing.T) {
	input := NewTextInput()
	input = input.PushHistory("first prompt")
	input = input.PushHistory("second prompt")
	input.Model.SetValue("draft")
	input.Model.SetCursor(len("draft"))

	input = input.PrevHistory()
	if got := input.Value(); got != "second prompt" {
		t.Fatalf("expected latest history entry, got %q", got)
	}

	input = input.PrevHistory()
	if got := input.Value(); got != "first prompt" {
		t.Fatalf("expected previous history entry, got %q", got)
	}

	input = input.NextHistory()
	if got := input.Value(); got != "second prompt" {
		t.Fatalf("expected forward history entry, got %q", got)
	}

	input = input.NextHistory()
	if got := input.Value(); got != "draft" {
		t.Fatalf("expected draft restoration after leaving history, got %q", got)
	}
}

func TestTextInputHistoryDoesNotBreakSlashSuggestions(t *testing.T) {
	input := NewTextInput()
	input = input.PushHistory("/project")
	var cmd tea.Cmd
	input, cmd = input.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'/'}})
	_ = cmd
	if !input.IsSlashMode() {
		t.Fatal("expected slash suggestions after typing slash")
	}
}

func TestTextInputHistoryWorksEvenWhenRecalledValueStartsWithSlash(t *testing.T) {
	input := NewTextInput()
	input = input.PushHistory("/project")
	input = input.PushHistory("hello")

	input = input.PrevHistory()
	if got := input.Value(); got != "hello" {
		t.Fatalf("expected latest history entry, got %q", got)
	}

	input = input.PrevHistory()
	if got := input.Value(); got != "/project" {
		t.Fatalf("expected slash command from history, got %q", got)
	}
	if !input.IsSlashMode() {
		t.Fatal("expected slash suggestions to appear for recalled slash command")
	}

	input = input.NextHistory()
	if got := input.Value(); got != "hello" {
		t.Fatalf("expected down to continue history recall even in slash mode, got %q", got)
	}
}

func TestTextInputSuggestionsScrollDownToKeepSelectionVisible(t *testing.T) {
	input := newSlashSuggestionInput(10)
	input.selectedIdx = 7
	input.suggestionOffset = 0

	input, _ = input.Update(tea.KeyMsg{Type: tea.KeyDown})

	if input.selectedIdx != 8 {
		t.Fatalf("expected selected index 8, got %d", input.selectedIdx)
	}
	if input.suggestionOffset != 1 {
		t.Fatalf("expected suggestion offset 1, got %d", input.suggestionOffset)
	}

	view := input.View()
	if !strings.Contains(view, "/cmd08") {
		t.Fatalf("expected view to include newly selected command, got %q", view)
	}
	if strings.Contains(view, "/cmd00") {
		t.Fatalf("expected first command to scroll out of view, got %q", view)
	}
}

func TestTextInputSuggestionsScrollUpToKeepSelectionVisible(t *testing.T) {
	input := newSlashSuggestionInput(10)
	input.selectedIdx = 2
	input.suggestionOffset = 2

	input, _ = input.Update(tea.KeyMsg{Type: tea.KeyUp})

	if input.selectedIdx != 1 {
		t.Fatalf("expected selected index 1, got %d", input.selectedIdx)
	}
	if input.suggestionOffset != 1 {
		t.Fatalf("expected suggestion offset 1, got %d", input.suggestionOffset)
	}

	view := input.View()
	if !strings.Contains(view, "/cmd01") {
		t.Fatalf("expected view to include newly selected command, got %q", view)
	}
	if strings.Contains(view, "/cmd09") {
		t.Fatalf("expected last command to scroll out of view, got %q", view)
	}
}

func TestTextInputSuggestionsWrapDownToTopPage(t *testing.T) {
	input := newSlashSuggestionInput(10)
	input.selectedIdx = 9
	input.suggestionOffset = 2

	input, _ = input.Update(tea.KeyMsg{Type: tea.KeyDown})

	if input.selectedIdx != 0 {
		t.Fatalf("expected selected index 0 after wrap, got %d", input.selectedIdx)
	}
	if input.suggestionOffset != 0 {
		t.Fatalf("expected suggestion offset 0 after wrap, got %d", input.suggestionOffset)
	}

	view := input.View()
	if !strings.Contains(view, "/cmd00") {
		t.Fatalf("expected wrapped view to include first command, got %q", view)
	}
	if strings.Contains(view, "/cmd09") {
		t.Fatalf("expected last command to leave view after wrapping to top, got %q", view)
	}
}

func TestTextInputSuggestionsWrapUpToLastPage(t *testing.T) {
	input := newSlashSuggestionInput(10)
	input.selectedIdx = 0
	input.suggestionOffset = 0

	input, _ = input.Update(tea.KeyMsg{Type: tea.KeyUp})

	if input.selectedIdx != 9 {
		t.Fatalf("expected selected index 9 after wrap, got %d", input.selectedIdx)
	}
	if input.suggestionOffset != 2 {
		t.Fatalf("expected suggestion offset 2 after wrap, got %d", input.suggestionOffset)
	}

	view := input.View()
	if !strings.Contains(view, "/cmd09") {
		t.Fatalf("expected wrapped view to include last command, got %q", view)
	}
	if strings.Contains(view, "/cmd00") {
		t.Fatalf("expected first command to leave view after wrapping to bottom, got %q", view)
	}
}

func newSlashSuggestionInput(count int) TextInput {
	input := NewTextInput()
	registry := slash.NewRegistry()
	input.slashRegistry = registry
	input.showSuggestions = true
	input.slashMode = true
	input.suggestions = make([]string, 0, count)

	for i := 0; i < count; i++ {
		name := fmt.Sprintf("/cmd%02d", i)
		registry.Register(slash.Command{
			Name:        name,
			Description: fmt.Sprintf("Command %02d", i),
			Usage:       name,
		})
		input.suggestions = append(input.suggestions, name)
	}

	return input
}
