package ui

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

func TestAskUserQuestionPrompt_SingleSelectEnterSubmits(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.handleEvent(model.Event{
		Type: model.AskUserQuestionPrompt,
		AskUserQuestion: &model.AskUserQuestionPromptData{
			Title:        "Answer Questions",
			SubmitPrefix: "ask:",
			Questions: []model.AskUserQuestionView{{
				Header:   "Scope",
				Question: "Which scope should we optimize first?",
				Options: []model.AskUserQuestionOption{
					{Label: "backend", Description: "Optimize backend first"},
					{Label: "frontend", Description: "Optimize frontend first"},
				},
			}},
		},
	})
	app = next.(App)

	if app.askUserQuestionPrompt == nil {
		t.Fatal("askUserQuestionPrompt should be set after prompt event")
	}
	if view := app.renderMainView(); !strings.Contains(view, "Which scope should we optimize first?") {
		t.Fatalf("rendered view missing question text:\n%s", view)
	}

	nextModel, _ := app.handleKey(tea.KeyMsg{Type: tea.KeyDown})
	app = nextModel.(App)
	nextModel, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = nextModel.(App)

	token := readAskUserQuestionToken(t, userCh)
	if !strings.HasPrefix(token, "ask:") {
		t.Fatalf("submitted token = %q, want prefix %q", token, "ask:")
	}

	var payload struct {
		Answers []struct {
			Question string `json:"question"`
			Answer   string `json:"answer"`
		} `json:"answers"`
		Declined bool `json:"declined"`
	}
	if err := json.Unmarshal([]byte(strings.TrimPrefix(token, "ask:")), &payload); err != nil {
		t.Fatalf("json.Unmarshal() err = %v", err)
	}
	if payload.Declined {
		t.Fatal("payload.Declined = true, want false")
	}
	if got := payload.Answers[0].Answer; got != "frontend" {
		t.Fatalf("selected answer = %q, want %q", got, "frontend")
	}
	if app.askUserQuestionPrompt != nil {
		t.Fatal("askUserQuestionPrompt should be cleared after submit")
	}
}

func TestAskUserQuestionPrompt_EscDeclines(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.handleEvent(model.Event{
		Type: model.AskUserQuestionPrompt,
		AskUserQuestion: &model.AskUserQuestionPromptData{
			Title:        "Answer Questions",
			SubmitPrefix: "ask:",
			Questions: []model.AskUserQuestionView{{
				Header:   "Scope",
				Question: "Which scope should we optimize first?",
				Options: []model.AskUserQuestionOption{
					{Label: "backend", Description: "Optimize backend first"},
					{Label: "frontend", Description: "Optimize frontend first"},
				},
			}},
		},
	})
	app = next.(App)

	nextModel, _ := app.handleKey(tea.KeyMsg{Type: tea.KeyEsc})
	app = nextModel.(App)

	token := readAskUserQuestionToken(t, userCh)
	if !strings.Contains(token, `"declined":true`) {
		t.Fatalf("submitted token = %q, want declined payload", token)
	}
	if app.askUserQuestionPrompt != nil {
		t.Fatal("askUserQuestionPrompt should be cleared after esc")
	}
}

func TestAskUserQuestionPrompt_MultiSelectSpaceThenEnterSubmits(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.handleEvent(model.Event{
		Type: model.AskUserQuestionPrompt,
		AskUserQuestion: &model.AskUserQuestionPromptData{
			Title:        "Answer Questions",
			SubmitPrefix: "ask:",
			Questions: []model.AskUserQuestionView{{
				Header:      "Tests",
				Question:    "Which tests should we add?",
				MultiSelect: true,
				Options: []model.AskUserQuestionOption{
					{Label: "unit", Description: "Add unit tests"},
					{Label: "integration", Description: "Add integration tests"},
				},
			}},
		},
	})
	app = next.(App)

	nextModel, _ := app.handleKey(tea.KeyMsg{Type: tea.KeySpace})
	app = nextModel.(App)
	nextModel, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyDown})
	app = nextModel.(App)
	nextModel, _ = app.handleKey(tea.KeyMsg{Type: tea.KeySpace})
	app = nextModel.(App)
	nextModel, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = nextModel.(App)

	token := readAskUserQuestionToken(t, userCh)
	if !strings.Contains(token, `"answer":"unit, integration"`) {
		t.Fatalf("submitted token = %q, want joined multi-select answer", token)
	}
	if app.askUserQuestionPrompt != nil {
		t.Fatal("askUserQuestionPrompt should be cleared after multi-select submit")
	}
}

func TestAskUserQuestionPrompt_OtherAnswerSubmitsCustomText(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.handleEvent(model.Event{
		Type: model.AskUserQuestionPrompt,
		AskUserQuestion: &model.AskUserQuestionPromptData{
			Title:        "Answer Questions",
			SubmitPrefix: "ask:",
			Questions: []model.AskUserQuestionView{{
				Header:   "Scope",
				Question: "Which scope should we optimize first?",
				Options: []model.AskUserQuestionOption{
					{Label: "backend", Description: "Optimize backend first"},
					{Label: "frontend", Description: "Optimize frontend first"},
				},
			}},
		},
	})
	app = next.(App)

	nextModel, _ := app.handleKey(tea.KeyMsg{Type: tea.KeyDown})
	app = nextModel.(App)
	nextModel, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyDown})
	app = nextModel.(App)
	nextModel, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = nextModel.(App)
	for _, r := range []rune("custom scope") {
		keyType := tea.KeyRunes
		if r == ' ' {
			keyType = tea.KeySpace
		}
		nextModel, _ = app.handleKey(tea.KeyMsg{Type: keyType, Runes: []rune{r}})
		app = nextModel.(App)
	}
	nextModel, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = nextModel.(App)

	token := readAskUserQuestionToken(t, userCh)
	if !strings.Contains(token, `"answer":"custom scope"`) {
		t.Fatalf("submitted token = %q, want custom answer", token)
	}
}

func TestAskUserQuestionToolEvent_ResolvesPendingToolMessage(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.handleEvent(model.Event{
		Type:       model.ToolCallStart,
		ToolName:   "AskUserQuestion",
		ToolCallID: "tool-1",
		Message:    "Scope",
	})
	app = next.(App)

	if got := len(app.state.Messages); got != 1 {
		t.Fatalf("message count = %d, want 1", got)
	}
	if !app.state.Messages[0].Pending {
		t.Fatal("pending tool message should be pending after ToolCallStart")
	}
	if got := app.state.Messages[0].Summary; got != "waiting for answers..." {
		t.Fatalf("pending summary = %q, want %q", got, "waiting for answers...")
	}

	next, _ = app.handleEvent(model.Event{
		Type: model.AskUserQuestionPrompt,
		AskUserQuestion: &model.AskUserQuestionPromptData{
			Title:        "Answer Questions",
			SubmitPrefix: "ask:",
			Questions: []model.AskUserQuestionView{{
				Header:   "Scope",
				Question: "Which scope should we optimize first?",
				Options: []model.AskUserQuestionOption{
					{Label: "backend", Description: "Optimize backend first"},
					{Label: "frontend", Description: "Optimize frontend first"},
				},
			}},
		},
	})
	app = next.(App)
	if app.askUserQuestionPrompt == nil {
		t.Fatal("askUserQuestionPrompt should be active after prompt event")
	}

	next, _ = app.handleEvent(model.Event{
		Type:       model.ToolAskUserQuestion,
		ToolName:   "AskUserQuestion",
		ToolCallID: "tool-1",
		Message:    `User has answered your questions: - "Which scope should we optimize first?" = "frontend"`,
		Summary:    "1 answer collected",
	})
	app = next.(App)

	if app.askUserQuestionPrompt != nil {
		t.Fatal("askUserQuestionPrompt should be cleared after resolved tool event")
	}
	if got := len(app.state.Messages); got != 1 {
		t.Fatalf("message count after resolve = %d, want 1", got)
	}
	last := app.state.Messages[0]
	if last.Pending {
		t.Fatal("resolved tool message should not be pending")
	}
	if got := last.Summary; got != "1 answer collected" {
		t.Fatalf("resolved summary = %q, want %q", got, "1 answer collected")
	}
	if !strings.Contains(last.Content, `"frontend"`) {
		t.Fatalf("resolved content = %q, want collected answer", last.Content)
	}
}

func readAskUserQuestionToken(t *testing.T, userCh <-chan string) string {
	t.Helper()

	select {
	case token := <-userCh:
		return token
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for ask-user-question submission")
		return ""
	}
}
