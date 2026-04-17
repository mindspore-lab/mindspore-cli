package askuserquestion

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/mindspore-lab/mindspore-cli/integrations/llm"
	"github.com/mindspore-lab/mindspore-cli/tools"
)

// PromptUI collects answers from the user through the application UI.
type PromptUI interface {
	Ask(ctx context.Context, req PromptRequest) (PromptResponse, error)
}

// QuestionOption is a single option shown to the user.
type QuestionOption struct {
	Label       string `json:"label"`
	Description string `json:"description"`
}

// Question describes one question to ask the user.
type Question struct {
	Header      string           `json:"header"`
	Question    string           `json:"question"`
	Options     []QuestionOption `json:"options"`
	MultiSelect bool             `json:"multiSelect"`
}

// PromptRequest is the tool input schema.
type PromptRequest struct {
	Questions []Question `json:"questions"`
}

// PromptAnswer is one final answer returned by the UI.
type PromptAnswer struct {
	Question string `json:"question"`
	Answer   string `json:"answer"`
}

// PromptResponse is the final question result returned by the UI.
type PromptResponse struct {
	Answers  []PromptAnswer `json:"answers"`
	Declined bool           `json:"declined"`
}

// Tool asks the user clarifying multiple-choice questions during execution.
type Tool struct {
	promptUI PromptUI
}

// NewTool creates a new AskUserQuestion tool.
func NewTool(promptUI PromptUI) *Tool {
	return &Tool{promptUI: promptUI}
}

// Name returns the tool name.
func (t *Tool) Name() string {
	return "AskUserQuestion"
}

// Description returns the tool description for the model.
func (t *Tool) Description() string {
	return "Ask the user one to four multiple-choice questions to clarify requirements, gather preferences, or choose between implementation options. Provide one to four concrete options per question and never add an explicit Other/manual-input option because the UI always provides Other for custom text."
}

// Schema returns the nested JSON schema used for tool calling.
func (t *Tool) Schema() llm.ToolSchema {
	optionSchema := llm.Property{
		Type: "object",
		Properties: map[string]llm.Property{
			"label": {
				Type:        "string",
				Description: "Short label shown for this option. Keep it concise and distinct from the other options.",
			},
			"description": {
				Type:        "string",
				Description: "One sentence explaining what the option means or what tradeoff it implies.",
			},
		},
		Required: []string{"label", "description"},
	}

	questionSchema := llm.Property{
		Type: "object",
		Properties: map[string]llm.Property{
			"header": {
				Type:        "string",
				Description: "Very short label for this question, such as 'Scope' or 'Tests'.",
			},
			"question": {
				Type:        "string",
				Description: "The full question to ask the user.",
			},
			"options": {
				Type:        "array",
				Description: "One to four concrete options for the user to choose from. Do not include an explicit Other or manual-input option because the UI adds Other automatically.",
				Items:       &optionSchema,
			},
			"multiSelect": {
				Type:        "boolean",
				Description: "Set true when the user may choose more than one option.",
			},
		},
		Required: []string{"header", "question", "options"},
	}

	return llm.ToolSchema{
		Type: "object",
		Properties: map[string]llm.Property{
			"questions": {
				Type:        "array",
				Description: "One to four questions to ask the user.",
				Items:       &questionSchema,
			},
		},
		Required: []string{"questions"},
	}
}

// Execute asks the user the requested questions and returns the collected answers.
func (t *Tool) Execute(ctx context.Context, params json.RawMessage) (*tools.Result, error) {
	var req PromptRequest
	if err := tools.ParseParams(params, &req); err != nil {
		return tools.ErrorResult(err), nil
	}
	req = normalizeRequest(req)
	if err := validateRequest(req); err != nil {
		return tools.ErrorResult(err), nil
	}
	if t.promptUI == nil {
		return tools.ErrorResultf("ask user question ui is not configured"), nil
	}

	resp, err := t.promptUI.Ask(ctx, req)
	if err != nil {
		if errors.Is(err, context.Canceled) {
			return nil, err
		}
		return tools.ErrorResultf("ask user: %w", err), nil
	}

	if resp.Declined {
		return tools.StringResultWithSummary(
			"User declined to answer the questions. Continue by making reasonable assumptions based on the current conversation and code context.",
			"declined",
		), nil
	}

	if len(resp.Answers) == 0 {
		return tools.ErrorResultf("ask user returned no answers"), nil
	}

	lines := make([]string, 0, len(resp.Answers)+2)
	lines = append(lines, "User has answered your questions:")
	for _, answer := range resp.Answers {
		lines = append(lines, fmt.Sprintf("- %q = %q", answer.Question, answer.Answer))
	}
	lines = append(lines, "You can now continue with the user's answers in mind.")

	summary := fmt.Sprintf("%d answers collected", len(resp.Answers))
	if len(resp.Answers) == 1 {
		summary = "1 answer collected"
	}
	return tools.StringResultWithSummary(strings.Join(lines, "\n"), summary), nil
}

func normalizeRequest(req PromptRequest) PromptRequest {
	normalized := PromptRequest{
		Questions: make([]Question, 0, len(req.Questions)),
	}
	for _, question := range req.Questions {
		normalized = appendNormalizedQuestion(normalized, question)
	}
	return normalized
}

func appendNormalizedQuestion(req PromptRequest, question Question) PromptRequest {
	req.Questions = append(req.Questions, Question{
		Header:      strings.TrimSpace(question.Header),
		Question:    strings.TrimSpace(question.Question),
		Options:     normalizeQuestionOptions(question.Options),
		MultiSelect: question.MultiSelect,
	})
	return req
}

func normalizeQuestionOptions(options []QuestionOption) []QuestionOption {
	normalized := make([]QuestionOption, 0, len(options))
	for _, option := range options {
		label := strings.TrimSpace(option.Label)
		description := strings.TrimSpace(option.Description)
		if isBuiltInOtherOption(label, description) {
			continue
		}
		normalized = append(normalized, QuestionOption{
			Label:       label,
			Description: description,
		})
	}
	return normalized
}

func isBuiltInOtherOption(label, description string) bool {
	normalizedLabel := normalizeOptionToken(label)
	switch normalizedLabel {
	case "other", "use manual input", "manual input", "manual entry", "enter a custom value manually", "custom input":
		return true
	}

	normalizedDescription := normalizeOptionToken(description)
	return strings.HasPrefix(normalizedLabel, "other") && (strings.Contains(normalizedDescription, "custom") || strings.Contains(normalizedDescription, "manual"))
}

func normalizeOptionToken(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	replacer := strings.NewReplacer("-", " ", "_", " ", "/", " ", "(", " ", ")", " ", ",", " ", ".", " ", ":", " ")
	value = replacer.Replace(value)
	return strings.Join(strings.Fields(value), " ")
}

func validateRequest(req PromptRequest) error {
	if len(req.Questions) < 1 || len(req.Questions) > 4 {
		return fmt.Errorf("questions must contain 1 to 4 entries")
	}

	seenQuestions := make(map[string]struct{}, len(req.Questions))
	for i, question := range req.Questions {
		if strings.TrimSpace(question.Header) == "" {
			return fmt.Errorf("questions[%d].header is required", i)
		}
		text := strings.TrimSpace(question.Question)
		if text == "" {
			return fmt.Errorf("questions[%d].question is required", i)
		}
		if _, exists := seenQuestions[text]; exists {
			return fmt.Errorf("question text must be unique: %q", text)
		}
		seenQuestions[text] = struct{}{}

		if len(question.Options) < 1 || len(question.Options) > 4 {
			return fmt.Errorf("questions[%d].options must contain 1 to 4 concrete entries", i)
		}

		seenLabels := make(map[string]struct{}, len(question.Options))
		for j, option := range question.Options {
			label := strings.TrimSpace(option.Label)
			if label == "" {
				return fmt.Errorf("questions[%d].options[%d].label is required", i, j)
			}
			if strings.TrimSpace(option.Description) == "" {
				return fmt.Errorf("questions[%d].options[%d].description is required", i, j)
			}
			if _, exists := seenLabels[label]; exists {
				return fmt.Errorf("option labels must be unique within question %q", text)
			}
			seenLabels[label] = struct{}{}
		}
	}

	return nil
}
