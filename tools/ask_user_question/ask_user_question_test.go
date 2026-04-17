package askuserquestion

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

type stubPromptUI struct {
	resp PromptResponse
	err  error
	req  PromptRequest
}

func (s *stubPromptUI) Ask(_ context.Context, req PromptRequest) (PromptResponse, error) {
	s.req = req
	return s.resp, s.err
}

func TestToolSchema_ContainsNestedQuestionShape(t *testing.T) {
	tool := NewTool(nil)
	schema := tool.Schema()

	questions, ok := schema.Properties["questions"]
	if !ok {
		t.Fatal("questions property missing")
	}
	if questions.Type != "array" {
		t.Fatalf("questions.Type = %q, want array", questions.Type)
	}
	if questions.Items == nil {
		t.Fatal("questions.Items = nil, want nested question schema")
	}
	if got := questions.Items.Properties["options"].Items; got == nil {
		t.Fatal("options.Items = nil, want nested option schema")
	}
}

func TestToolExecute_ReturnsCollectedAnswers(t *testing.T) {
	ui := &stubPromptUI{
		resp: PromptResponse{
			Answers: []PromptAnswer{
				{Question: "Which scope should we optimize first?", Answer: "backend"},
				{Question: "Which tests do you want?", Answer: "unit, integration"},
			},
		},
	}
	tool := NewTool(ui)
	params := mustJSON(t, PromptRequest{
		Questions: []Question{
			{
				Header:   "Scope",
				Question: "Which scope should we optimize first?",
				Options: []QuestionOption{
					{Label: "backend", Description: "Optimize backend first"},
					{Label: "frontend", Description: "Optimize frontend first"},
				},
			},
			{
				Header:      "Tests",
				Question:    "Which tests do you want?",
				MultiSelect: true,
				Options: []QuestionOption{
					{Label: "unit", Description: "Add unit tests"},
					{Label: "integration", Description: "Add integration tests"},
				},
			},
		},
	})

	result, err := tool.Execute(context.Background(), params)
	if err != nil {
		t.Fatalf("Execute() err = %v", err)
	}
	if result.Error != nil {
		t.Fatalf("Execute() result.Error = %v", result.Error)
	}
	if got, want := result.Summary, "2 answers collected"; got != want {
		t.Fatalf("result.Summary = %q, want %q", got, want)
	}
	if !strings.Contains(result.Content, `"Which scope should we optimize first?" = "backend"`) {
		t.Fatalf("result.Content missing first answer:\n%s", result.Content)
	}
	if len(ui.req.Questions) != 2 {
		t.Fatalf("prompt ui saw %d questions, want 2", len(ui.req.Questions))
	}
}

func TestToolExecute_Declined(t *testing.T) {
	tool := NewTool(&stubPromptUI{resp: PromptResponse{Declined: true}})
	params := mustJSON(t, PromptRequest{
		Questions: []Question{{
			Header:   "Scope",
			Question: "Which scope should we optimize first?",
			Options: []QuestionOption{
				{Label: "backend", Description: "Optimize backend first"},
				{Label: "frontend", Description: "Optimize frontend first"},
			},
		}},
	})

	result, err := tool.Execute(context.Background(), params)
	if err != nil {
		t.Fatalf("Execute() err = %v", err)
	}
	if result.Error != nil {
		t.Fatalf("Execute() result.Error = %v", result.Error)
	}
	if got, want := result.Summary, "declined"; got != want {
		t.Fatalf("result.Summary = %q, want %q", got, want)
	}
}

func TestToolExecute_ValidatesRequest(t *testing.T) {
	tool := NewTool(&stubPromptUI{})
	params := mustJSON(t, PromptRequest{
		Questions: []Question{{
			Header:   "",
			Question: "",
			Options: []QuestionOption{
				{Label: "backend", Description: "Optimize backend first"},
			},
		}},
	})

	result, err := tool.Execute(context.Background(), params)
	if err != nil {
		t.Fatalf("Execute() err = %v", err)
	}
	if result.Error == nil {
		t.Fatal("result.Error = nil, want validation error")
	}
}

func TestToolExecute_StripsExplicitOtherOptionAndKeepsCustomInputPath(t *testing.T) {
	ui := &stubPromptUI{
		resp: PromptResponse{
			Answers: []PromptAnswer{
				{Question: "Which CANN path should we use?", Answer: "/home/cann_custom_path/8.5.0/ascend-toolkit/set_env.sh"},
			},
		},
	}
	tool := NewTool(ui)
	params := mustJSON(t, PromptRequest{
		Questions: []Question{{
			Header:   "CANN Path",
			Question: "Which CANN path should we use?",
			Options: []QuestionOption{
				{Label: "/usr/local/Ascend/ascend-toolkit/latest", Description: "Typical CANN toolkit installation path."},
				{Label: "Other", Description: "I will type a custom path."},
			},
		}},
	})

	result, err := tool.Execute(context.Background(), params)
	if err != nil {
		t.Fatalf("Execute() err = %v", err)
	}
	if result.Error != nil {
		t.Fatalf("Execute() result.Error = %v", result.Error)
	}
	if got := len(ui.req.Questions[0].Options); got != 1 {
		t.Fatalf("normalized option count = %d, want 1 concrete option after stripping explicit Other", got)
	}
	if got := ui.req.Questions[0].Options[0].Label; got != "/usr/local/Ascend/ascend-toolkit/latest" {
		t.Fatalf("remaining option label = %q, want toolkit path", got)
	}
	if !strings.Contains(result.Content, `"/home/cann_custom_path/8.5.0/ascend-toolkit/set_env.sh"`) {
		t.Fatalf("result.Content missing custom path answer:\n%s", result.Content)
	}
}

func mustJSON(t *testing.T, v any) json.RawMessage {
	t.Helper()
	data, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("json.Marshal() err = %v", err)
	}
	return data
}
