package ui

import (
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/vigo999/ms-cli/ui/model"
	"github.com/vigo999/ms-cli/ui/panels"

	tea "github.com/charmbracelet/bubbletea"
)

func (a *App) openIssueIndex(data *model.IssueEventData) {
	filter := "all"
	if data != nil && data.Filter != "" {
		filter = data.Filter
	}
	index := model.IssueIndexState{Filter: filter}
	if data != nil {
		index.Items = data.Items
		if data.Err != nil {
			index.Err = data.Err.Error()
		}
	}
	a.issueView.Mode = model.IssueModeIndex
	a.issueView.Index = index
	a.input = a.input.Blur()
	a.input.Model.Placeholder = ""
	a.resizeActiveLayout()
}

func (a *App) openIssueDetail(data *model.IssueEventData) {
	if data == nil {
		return
	}
	a.issueView.Mode = model.IssueModeDetail
	a.issueView.Detail = model.IssueDetailState{
		ID:        data.ID,
		Issue:     data.Issue,
		Notes:     data.Notes,
		Activity:  data.Activity,
		FromIndex: data.FromIndex,
	}
	if data.Err != nil {
		a.issueView.Detail.Err = data.Err.Error()
	}
	a.input = a.input.Reset()
	if data.Issue != nil {
		a.input.Model.Placeholder = "Add note to " + data.Issue.Key + "..."
	}
	a.input, _ = a.input.Focus()
	a.resizeActiveLayout()
}

func (a *App) closeIssueView() {
	a.issueView = model.IssueViewState{}
	a.input.Model.Placeholder = ""
	a.input, _ = a.input.Focus()
	a.resizeActiveLayout()
}

func (a App) handleIssueKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch a.issueView.Mode {
	case model.IssueModeIndex:
		return a.handleIssueIndexKey(msg)
	case model.IssueModeDetail:
		return a.handleIssueDetailKey(msg)
	default:
		return a, nil
	}
}

func (a App) handleIssueIndexKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "esc":
		a.closeIssueView()
	case "j", "down":
		if a.issueView.Index.Cursor < len(a.issueView.Index.Items)-1 {
			a.issueView.Index.Cursor++
		}
	case "k", "up":
		if a.issueView.Index.Cursor > 0 {
			a.issueView.Index.Cursor--
		}
	case "enter":
		if len(a.issueView.Index.Items) == 0 || a.userCh == nil {
			return a, nil
		}
		id := a.issueView.Index.Items[a.issueView.Index.Cursor].ID
		select {
		case a.userCh <- fmt.Sprintf("/__issue_detail %d", id):
		default:
		}
	}
	return a, nil
}

func (a App) handleIssueDetailKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	if a.input.IsSlashMode() {
		switch msg.String() {
		case "tab", "esc":
			var cmd tea.Cmd
			a.input, cmd = a.input.Update(msg)
			a.resizeActiveLayout()
			return a, cmd
		case "up", "down":
			if a.input.HasSuggestions() {
				var cmd tea.Cmd
				a.input, cmd = a.input.Update(msg)
				return a, cmd
			}
		}
	}

	if strings.TrimSpace(a.input.Value()) == "" {
		switch msg.String() {
		case "esc":
			if a.issueView.Detail.FromIndex {
				a.issueView.Mode = model.IssueModeIndex
				a.input = a.input.Reset().Blur()
				a.input.Model.Placeholder = ""
				return a, nil
			}
			a.closeIssueView()
			return a, nil
		case "d":
			return a.dispatchIssueCommand("/diagnose")
		case "f":
			return a.dispatchIssueCommand("/fix")
		case "l":
			a.applyIssueClaim(a.issueView.Detail.ID)
			return a.dispatchIssueCommand("/claim")
		case "s":
			a.applyIssueNextStatus(a.issueView.Detail.ID)
			return a.dispatchIssueCommand("/status")
		}
	}

	switch msg.String() {
	case "enter":
		if a.input.IsSlashMode() {
			var cmd tea.Cmd
			a.input, cmd = a.input.Update(msg)
			a.resizeActiveLayout()
			return a, cmd
		}
		val := strings.TrimSpace(a.input.Value())
		if val == "" {
			return a, nil
		}
		input := a.rewriteIssueInput(val)
		a.input = a.input.PushHistory(val)
		a.input = a.input.Reset()
		if a.issueView.Detail.Issue != nil {
			a.input.Model.Placeholder = "Add note to " + a.issueView.Detail.Issue.Key + "..."
		}
		a.resizeActiveLayout()
		if a.userCh != nil && input != "" {
			select {
			case a.userCh <- input:
			default:
			}
		}
		return a, nil
	case "up", "down":
		if msg.String() == "up" {
			a.input = a.input.PrevHistory()
		} else {
			a.input = a.input.NextHistory()
		}
		a.resizeActiveLayout()
		return a, nil
	default:
		var cmd tea.Cmd
		a.input, cmd = a.input.Update(msg)
		a.resizeActiveLayout()
		return a, cmd
	}
}

func (a App) dispatchIssueCommand(base string) (tea.Model, tea.Cmd) {
	if a.issueView.Detail.Issue == nil || a.userCh == nil {
		return a, nil
	}
	cmd := a.rewriteIssueInput(base)
	if cmd == "" {
		return a, nil
	}
	select {
	case a.userCh <- cmd:
	default:
	}
	return a, nil
}

func (a App) rewriteIssueInput(input string) string {
	if a.issueView.Detail.Issue == nil {
		return input
	}
	key := a.issueView.Detail.Issue.Key
	trimmed := strings.TrimSpace(input)
	switch {
	case trimmed == "":
		return ""
	case !strings.HasPrefix(trimmed, "/"):
		return fmt.Sprintf("/__issue_note %s %s", key, trimmed)
	case trimmed == "/diagnose":
		return "/diagnose " + key
	case trimmed == "/fix":
		return "/fix " + key
	case trimmed == "/claim":
		return "/__issue_claim " + key
	case trimmed == "/status":
		return "/status " + key + " closed"
	case strings.HasPrefix(trimmed, "/status "):
		return "/status " + key + " " + strings.TrimSpace(strings.TrimPrefix(trimmed, "/status "))
	default:
		return trimmed
	}
}

func (a *App) applyIssueClaim(id int) {
	now := time.Now()
	actor := a.state.IssueUser
	for i := range a.issueView.Index.Items {
		if a.issueView.Index.Items[i].ID == id {
			a.issueView.Index.Items[i].Status = "doing"
			a.issueView.Index.Items[i].Lead = actor
			a.issueView.Index.Items[i].UpdatedAt = now
		}
	}
	if a.issueView.Detail.Issue != nil && a.issueView.Detail.Issue.ID == id {
		a.issueView.Detail.Issue.Status = "doing"
		a.issueView.Detail.Issue.Lead = actor
		a.issueView.Detail.Issue.UpdatedAt = now
	}
}

func (a *App) applyIssueNextStatus(id int) {
	var next string
	if a.issueView.Detail.Issue != nil && a.issueView.Detail.Issue.ID == id {
		next = nextIssueStatus(a.issueView.Detail.Issue.Status)
		a.issueView.Detail.Issue.Status = next
		a.issueView.Detail.Issue.UpdatedAt = time.Now()
	}
	for i := range a.issueView.Index.Items {
		if a.issueView.Index.Items[i].ID == id {
			if next == "" {
				next = nextIssueStatus(a.issueView.Index.Items[i].Status)
			}
			a.issueView.Index.Items[i].Status = next
			a.issueView.Index.Items[i].UpdatedAt = time.Now()
		}
	}
}

func nextIssueStatus(current string) string {
	switch strings.ToLower(strings.TrimSpace(current)) {
	case "ready":
		return "doing"
	case "doing":
		return "closed"
	default:
		return "ready"
	}
}

func (a App) renderIssueView() string {
	topBar := panels.RenderTopBar(a.state, a.width)
	hintBar := panels.RenderIssueHintBar(a.width, a.issueView.Mode)
	input := ""
	if a.issueView.Mode == model.IssueModeDetail {
		input = "  " + a.input.View()
	}
	bodyHeight := a.height - lipgloss.Height(topBar) - lipgloss.Height(hintBar) - lipgloss.Height(input) - bottomSafePadding
	if bodyHeight < 1 {
		bodyHeight = 1
	}
	bodyWidth := a.width
	if bodyWidth < 1 {
		bodyWidth = 1
	}

	body := ""
	switch a.issueView.Mode {
	case model.IssueModeIndex:
		body = panels.RenderIssueIndex(bodyWidth, bodyHeight, a.issueView.Index)
	case model.IssueModeDetail:
		body = panels.RenderIssueDetail(bodyWidth, bodyHeight, a.issueView.Detail)
	}

	parts := []string{topBar, body}
	if input != "" {
		parts = append(parts, input)
	}
	parts = append(parts, hintBar, "", "")
	return trimViewHeight(lipgloss.JoinVertical(lipgloss.Left, parts...), a.height)
}
