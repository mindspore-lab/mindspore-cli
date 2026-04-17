package app

import (
	"context"
	"testing"
	"time"

	askuserquestion "github.com/mindspore-lab/mindspore-cli/tools/ask_user_question"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

func TestAskUserQuestionPromptUI_AskAndHandleInput(t *testing.T) {
	eventCh := make(chan model.Event, 2)
	ui := NewAskUserQuestionPromptUI(eventCh)

	req := askuserquestion.PromptRequest{
		Questions: []askuserquestion.Question{{
			Header:   "Scope",
			Question: "Which scope should we optimize first?",
			Options: []askuserquestion.QuestionOption{
				{Label: "backend", Description: "Optimize backend first"},
				{Label: "frontend", Description: "Optimize frontend first"},
			},
		}},
	}

	resultCh := make(chan struct {
		resp askuserquestion.PromptResponse
		err  error
	}, 1)
	go func() {
		resp, err := ui.Ask(context.Background(), req)
		resultCh <- struct {
			resp askuserquestion.PromptResponse
			err  error
		}{resp: resp, err: err}
	}()

	select {
	case ev := <-eventCh:
		if ev.Type != model.AskUserQuestionPrompt {
			t.Fatalf("event type = %s, want %s", ev.Type, model.AskUserQuestionPrompt)
		}
		if ev.AskUserQuestion == nil {
			t.Fatal("event.AskUserQuestion = nil, want prompt payload")
		}
		if got := ev.AskUserQuestion.Title; got != "Answer Questions" {
			t.Fatalf("prompt title = %q, want %q", got, "Answer Questions")
		}
		if got := len(ev.AskUserQuestion.Questions); got != 1 {
			t.Fatalf("question count = %d, want 1", got)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for ask-user-question prompt event")
	}

	if handled := ui.HandleInput(askUserQuestionInputPrefix + `{"answers":[{"question":"Which scope should we optimize first?","answer":"frontend"}]}`); !handled {
		t.Fatal("HandleInput() = false, want true for serialized prompt response")
	}

	select {
	case out := <-resultCh:
		if out.err != nil {
			t.Fatalf("Ask() err = %v", out.err)
		}
		if out.resp.Declined {
			t.Fatal("Ask() declined = true, want false")
		}
		if got := len(out.resp.Answers); got != 1 {
			t.Fatalf("answer count = %d, want 1", got)
		}
		if got := out.resp.Answers[0].Answer; got != "frontend" {
			t.Fatalf("answer = %q, want %q", got, "frontend")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for Ask() result")
	}
}

func TestAskUserQuestionPromptUI_HandleMalformedPayloadDeclines(t *testing.T) {
	eventCh := make(chan model.Event, 1)
	ui := NewAskUserQuestionPromptUI(eventCh)

	resultCh := make(chan askuserquestion.PromptResponse, 1)
	go func() {
		resp, _ := ui.Ask(context.Background(), askuserquestion.PromptRequest{
			Questions: []askuserquestion.Question{{
				Header:   "Tests",
				Question: "Which tests should we add?",
				Options: []askuserquestion.QuestionOption{
					{Label: "unit", Description: "Add unit tests"},
					{Label: "integration", Description: "Add integration tests"},
				},
			}},
		})
		resultCh <- resp
	}()

	<-eventCh
	if handled := ui.HandleInput(askUserQuestionInputPrefix + "{bad json"); !handled {
		t.Fatal("HandleInput() = false, want true for malformed prefixed payload")
	}

	select {
	case resp := <-resultCh:
		if !resp.Declined {
			t.Fatal("resp.Declined = false, want true")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for declined response")
	}
}

func TestApplicationProcessInput_AskUserQuestionReplyPriority(t *testing.T) {
	app := &Application{
		EventCh: make(chan model.Event, 2),
	}
	app.questionUI = NewAskUserQuestionPromptUI(app.EventCh)

	resultCh := make(chan askuserquestion.PromptResponse, 1)
	go func() {
		resp, _ := app.questionUI.Ask(context.Background(), askuserquestion.PromptRequest{
			Questions: []askuserquestion.Question{{
				Header:   "Scope",
				Question: "Which scope should we optimize first?",
				Options: []askuserquestion.QuestionOption{
					{Label: "backend", Description: "Optimize backend first"},
					{Label: "frontend", Description: "Optimize frontend first"},
				},
			}},
		})
		resultCh <- resp
	}()

	select {
	case <-app.EventCh:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for question prompt")
	}

	app.processInput(askUserQuestionInputPrefix + `{"declined":true}`)

	select {
	case resp := <-resultCh:
		if !resp.Declined {
			t.Fatal("resp.Declined = false, want true")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for processInput to resolve question prompt")
	}
}
