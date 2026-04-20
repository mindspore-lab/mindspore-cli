package ui

import (
	"encoding/json"
	"strconv"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
	"github.com/mindspore-lab/mindspore-cli/ui/theme"
)

type askUserQuestionPromptState struct {
	title          string
	submitPrefix   string
	questions      []model.AskUserQuestionView
	current        int
	selectedOption int
	answers        []askUserQuestionAnswerState
}

type askUserQuestionAnswerState struct {
	selected map[int]bool
	other    string
}

type askUserQuestionResponsePayload struct {
	Answers  []askUserQuestionAnswerPayload `json:"answers"`
	Declined bool                           `json:"declined"`
}

type askUserQuestionAnswerPayload struct {
	Question string `json:"question"`
	Answer   string `json:"answer"`
}

const (
	askUserQuestionChatLabel       = "Type something here"
	askUserQuestionChatDescription = ""
)

func toAskUserQuestionPromptState(ev model.Event) *askUserQuestionPromptState {
	data := ev.AskUserQuestion
	if data == nil {
		return nil
	}

	questions := make([]model.AskUserQuestionView, 0, len(data.Questions))
	answers := make([]askUserQuestionAnswerState, 0, len(data.Questions))
	for _, question := range data.Questions {
		options := append([]model.AskUserQuestionOption(nil), question.Options...)
		questions = append(questions, model.AskUserQuestionView{
			Header:      question.Header,
			Question:    question.Question,
			Options:     options,
			MultiSelect: question.MultiSelect,
		})
		answers = append(answers, askUserQuestionAnswerState{
			selected: make(map[int]bool),
		})
	}

	return &askUserQuestionPromptState{
		title:          valueOrString(strings.TrimSpace(data.Title), "Answer Questions"),
		submitPrefix:   strings.TrimSpace(data.SubmitPrefix),
		questions:      questions,
		answers:        answers,
		selectedOption: 0,
	}
}

func (a App) handleAskUserQuestionKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	p := a.askUserQuestionPrompt
	if p == nil {
		return a, nil
	}
	if p.optionCount() == 0 {
		return a, nil
	}

	if p.handleInputOptionEdit(msg) {
		return a, nil
	}

	switch msg.String() {
	case "up", "left":
		p.selectedOption--
		if p.selectedOption < 0 {
			p.selectedOption = p.optionCount() - 1
		}
		return a, nil
	case "down", "right", "tab":
		p.selectedOption = (p.selectedOption + 1) % p.optionCount()
		return a, nil
	case "space":
		if p.isInputFocused() {
			return a, nil
		}
		if !p.currentQuestion().MultiSelect {
			return a, nil
		}
		answer := &p.answers[p.current]
		if answer.selected[p.selectedOption] {
			delete(answer.selected, p.selectedOption)
		} else {
			answer.selected[p.selectedOption] = true
		}
		return a, nil
	case "enter":
		if p.isInputFocused() {
			if !p.canSubmitInputOption() {
				return a, nil
			}
			if !p.currentQuestion().MultiSelect {
				p.currentAnswer().selected = map[int]bool{}
			}
			return a.finishCurrentAskUserQuestion()
		}
		answer := &p.answers[p.current]
		if p.currentQuestion().MultiSelect {
			if !p.hasAnswerForCurrentQuestion() {
				return a, nil
			}
			return a.finishCurrentAskUserQuestion()
		}
		answer.selected = map[int]bool{p.selectedOption: true}
		answer.other = ""
		return a.finishCurrentAskUserQuestion()
	case "esc":
		a.submitAskUserQuestionResponse(askUserQuestionResponsePayload{Declined: true})
		a.askUserQuestionPrompt = nil
		return a, nil
	default:
		return a, nil
	}
}

func (a App) finishCurrentAskUserQuestion() (tea.Model, tea.Cmd) {
	p := a.askUserQuestionPrompt
	if p == nil {
		return a, nil
	}

	if !p.hasAnswerForCurrentQuestion() {
		return a, nil
	}

	if p.current < len(p.questions)-1 {
		p.current++
		p.selectedOption = 0
		return a, nil
	}

	a.submitAskUserQuestionResponse(p.responsePayload())
	a.askUserQuestionPrompt = nil
	return a, nil
}

func (a App) submitAskUserQuestionResponse(payload askUserQuestionResponsePayload) {
	if a.askUserQuestionPrompt == nil || a.userCh == nil {
		return
	}

	data, err := json.Marshal(payload)
	if err != nil {
		data = []byte(`{"declined":true}`)
	}
	token := a.askUserQuestionPrompt.submitPrefix + string(data)
	select {
	case a.userCh <- token:
	default:
	}
}

func (p *askUserQuestionPromptState) currentQuestion() model.AskUserQuestionView {
	if p == nil || p.current < 0 || p.current >= len(p.questions) {
		return model.AskUserQuestionView{}
	}
	return p.questions[p.current]
}

func (p *askUserQuestionPromptState) optionCount() int {
	return len(p.currentQuestion().Options) + 1
}

func (p *askUserQuestionPromptState) isInputFocused() bool {
	return p != nil && p.selectedOption == len(p.currentQuestion().Options)
}

func (p *askUserQuestionPromptState) currentAnswer() *askUserQuestionAnswerState {
	if p == nil || p.current < 0 || p.current >= len(p.answers) {
		return nil
	}
	return &p.answers[p.current]
}

func (p *askUserQuestionPromptState) customAnswer() string {
	answer := p.currentAnswer()
	if answer == nil {
		return ""
	}
	return answer.other
}

func (p *askUserQuestionPromptState) canSubmitInputOption() bool {
	answer := p.currentAnswer()
	if answer == nil {
		return false
	}
	if strings.TrimSpace(answer.other) != "" {
		return true
	}
	return p.currentQuestion().MultiSelect && len(answer.selected) > 0
}

func (p *askUserQuestionPromptState) handleInputOptionEdit(msg tea.KeyMsg) bool {
	if p == nil || !p.isInputFocused() {
		return false
	}

	answer := p.currentAnswer()
	if answer == nil {
		return false
	}

	switch {
	case msg.Type == tea.KeyRunes && len(msg.Runes) > 0:
		answer.other += string(msg.Runes)
	case msg.Type == tea.KeySpace:
		answer.other += " "
	case msg.String() == "backspace":
		runes := []rune(answer.other)
		if len(runes) == 0 {
			return true
		}
		answer.other = string(runes[:len(runes)-1])
	default:
		return false
	}

	if !p.currentQuestion().MultiSelect {
		answer.selected = map[int]bool{}
	}
	return true
}

func (p *askUserQuestionPromptState) hasAnswerForCurrentQuestion() bool {
	if p == nil || p.current < 0 || p.current >= len(p.answers) {
		return false
	}
	answer := p.answers[p.current]
	return len(answer.selected) > 0 || strings.TrimSpace(answer.other) != ""
}

func (p *askUserQuestionPromptState) responsePayload() askUserQuestionResponsePayload {
	payload := askUserQuestionResponsePayload{
		Answers: make([]askUserQuestionAnswerPayload, 0, len(p.questions)),
	}
	for i, question := range p.questions {
		answerText := p.answerTextForQuestion(i)
		if strings.TrimSpace(answerText) == "" {
			continue
		}
		payload.Answers = append(payload.Answers, askUserQuestionAnswerPayload{
			Question: question.Question,
			Answer:   answerText,
		})
	}
	return payload
}

func (p *askUserQuestionPromptState) answerTextForQuestion(index int) string {
	if p == nil || index < 0 || index >= len(p.questions) || index >= len(p.answers) {
		return ""
	}
	question := p.questions[index]
	answer := p.answers[index]

	if !question.MultiSelect {
		if strings.TrimSpace(answer.other) != "" {
			return strings.TrimSpace(answer.other)
		}
		for optionIndex := range question.Options {
			if answer.selected[optionIndex] {
				return question.Options[optionIndex].Label
			}
		}
		return ""
	}

	parts := make([]string, 0, len(answer.selected)+1)
	for optionIndex, option := range question.Options {
		if answer.selected[optionIndex] {
			parts = append(parts, option.Label)
		}
	}
	if strings.TrimSpace(answer.other) != "" {
		parts = append(parts, strings.TrimSpace(answer.other))
	}
	return strings.Join(parts, ", ")
}

func renderAskUserQuestionPromptPopup(p *askUserQuestionPromptState) string {
	if p == nil || len(p.questions) == 0 {
		return ""
	}

	t := theme.Current
	titleStyle := lipgloss.NewStyle().Foreground(t.Accent).Bold(true)
	subtitleStyle := lipgloss.NewStyle().Foreground(t.TextMuted)
	questionStyle := lipgloss.NewStyle().Foreground(t.TextPrimary).Bold(true)
	selectedStyle := lipgloss.NewStyle().Foreground(t.Accent).Bold(true)
	normalStyle := lipgloss.NewStyle().Foreground(t.TextPrimary)
	descStyle := lipgloss.NewStyle().Foreground(t.TextSecondary)
	inlineInputStyle := lipgloss.NewStyle().Foreground(t.TextPrimary)
	inlineInputInactiveStyle := lipgloss.NewStyle().Foreground(t.TextMuted)
	hintStyle := lipgloss.NewStyle().Foreground(t.TextMuted).Italic(true)

	question := p.currentQuestion()
	progress := ""
	if len(p.questions) > 1 {
		progress = lipgloss.NewStyle().Foreground(t.TextMuted).Render(
			strings.TrimSpace(
				lipgloss.JoinHorizontal(lipgloss.Left,
					"Question ",
					itoa(p.current+1),
					"/",
					itoa(len(p.questions)),
				),
			),
		)
	}

	lines := []string{titleStyle.Render(p.title)}
	if progress != "" {
		lines = append(lines, progress)
	}
	if header := strings.TrimSpace(question.Header); header != "" {
		lines = append(lines, subtitleStyle.Render("["+header+"]"))
	}
	lines = append(lines, questionStyle.Render(question.Question), "")

	for i, option := range question.Options {
		lines = append(lines, renderAskUserQuestionOptionLine(question.MultiSelect, p.selectedOption == i, p.answers[p.current].selected[i], option.Label, option.Description, selectedStyle, normalStyle, descStyle)...)
	}
	lines = append(lines, renderAskUserQuestionOptionLine(question.MultiSelect, p.isInputFocused(), strings.TrimSpace(p.answers[p.current].other) != "", askUserQuestionChatLabel, askUserQuestionChatDescription, selectedStyle, normalStyle, descStyle)...)
	lines = append(lines, renderAskUserQuestionInlineInputLine(
		p.customAnswer(),
		p.isInputFocused(),
		inlineInputStyle,
		inlineInputInactiveStyle,
	))

	lines = append(lines, "")
	if question.MultiSelect {
		lines = append(lines, hintStyle.Render("up/down move | space toggle | enter continue | esc cancel"))
	} else {
		lines = append(lines, hintStyle.Render("up/down move | enter confirm | esc cancel"))
	}

	content := strings.Join(lines, "\n")
	return lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(t.Accent).
		Padding(0, 1).
		Render(content)
}

func renderAskUserQuestionOptionLine(multiSelect, isCursor, isSelected bool, label, description string, selectedStyle, normalStyle, descStyle lipgloss.Style) []string {
	cursor := "  "
	style := normalStyle
	if isCursor {
		cursor = "> "
		style = selectedStyle
	}

	choice := ""
	if multiSelect {
		choice = "[ ]"
		if isSelected {
			choice = "[x]"
		}
	}

	prefix := ""
	if choice != "" {
		prefix = choice + " "
	}

	lines := []string{cursor + style.Render(prefix+label)}
	if strings.TrimSpace(description) != "" {
		lines = append(lines, "   "+descStyle.Render(description))
	}
	return lines
}

func renderAskUserQuestionInlineInputLine(value string, active bool, activeStyle, inactiveStyle lipgloss.Style) string {
	rendered := renderAskUserQuestionInputValue(value, active)
	if active {
		return "   " + activeStyle.Render(rendered)
	}
	return "   " + inactiveStyle.Render(rendered)
}

func renderAskUserQuestionInputValue(value string, active bool) string {
	if strings.TrimSpace(value) == "" {
		if active {
			return "|"
		}
		return "start typing"
	}
	if active {
		return value + "|"
	}
	return value
}

func itoa(v int) string {
	return strconv.Itoa(v)
}
