package app

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	askuserquestion "github.com/mindspore-lab/mindspore-cli/tools/ask_user_question"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

const askUserQuestionInputPrefix = "\x00ask_user_question:"

// AskUserQuestionPromptUI bridges interactive question prompts into the TUI flow.
type AskUserQuestionPromptUI struct {
	mu      sync.Mutex
	pending *pendingAskUserQuestionRequest
	eventCh chan<- model.Event
}

type pendingAskUserQuestionRequest struct {
	wait chan askuserquestion.PromptResponse
}

func NewAskUserQuestionPromptUI(eventCh chan<- model.Event) *AskUserQuestionPromptUI {
	return &AskUserQuestionPromptUI{eventCh: eventCh}
}

func (p *AskUserQuestionPromptUI) Ask(ctx context.Context, req askuserquestion.PromptRequest) (askuserquestion.PromptResponse, error) {
	p.mu.Lock()
	if p.pending != nil {
		p.mu.Unlock()
		return askuserquestion.PromptResponse{}, fmt.Errorf("question prompt already pending")
	}
	pending := &pendingAskUserQuestionRequest{
		wait: make(chan askuserquestion.PromptResponse, 1),
	}
	p.pending = pending
	p.mu.Unlock()

	p.eventCh <- model.Event{
		Type:            model.AskUserQuestionPrompt,
		AskUserQuestion: buildAskUserQuestionPromptData(req),
	}

	select {
	case resp := <-pending.wait:
		return resp, nil
	case <-ctx.Done():
		p.clearPending(pending)
		p.eventCh <- model.Event{Type: model.AskUserQuestionClose}
		return askuserquestion.PromptResponse{}, ctx.Err()
	}
}

// HandleInput resolves the active prompt when the UI submits a serialized response.
func (p *AskUserQuestionPromptUI) HandleInput(input string) bool {
	if !strings.HasPrefix(input, askUserQuestionInputPrefix) {
		return false
	}

	p.mu.Lock()
	pending := p.pending
	if pending != nil {
		p.pending = nil
	}
	p.mu.Unlock()
	if pending == nil {
		return true
	}

	var resp askuserquestion.PromptResponse
	raw := strings.TrimPrefix(input, askUserQuestionInputPrefix)
	if err := json.Unmarshal([]byte(raw), &resp); err != nil {
		resp = askuserquestion.PromptResponse{Declined: true}
	}
	pending.wait <- resp
	return true
}

func (p *AskUserQuestionPromptUI) clearPending(target *pendingAskUserQuestionRequest) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.pending == target {
		p.pending = nil
	}
}

func buildAskUserQuestionPromptData(req askuserquestion.PromptRequest) *model.AskUserQuestionPromptData {
	questions := make([]model.AskUserQuestionView, 0, len(req.Questions))
	for _, question := range req.Questions {
		options := make([]model.AskUserQuestionOption, 0, len(question.Options))
		for _, option := range question.Options {
			options = append(options, model.AskUserQuestionOption{
				Label:       option.Label,
				Description: option.Description,
			})
		}
		questions = append(questions, model.AskUserQuestionView{
			Header:      question.Header,
			Question:    question.Question,
			Options:     options,
			MultiSelect: question.MultiSelect,
		})
	}

	return &model.AskUserQuestionPromptData{
		Title:        "Answer Questions",
		SubmitPrefix: askUserQuestionInputPrefix,
		Questions:    questions,
	}
}
