package ui

import (
	"fmt"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/vigo999/ms-cli/internal/bugs"
	"github.com/vigo999/ms-cli/ui/model"
	"github.com/vigo999/ms-cli/ui/panels"

	tea "github.com/charmbracelet/bubbletea"
)

func (a *App) openBugIndex(data *model.BugEventData) {
	filter := "all"
	if data != nil && data.Filter != "" {
		filter = data.Filter
	}
	index := model.BugIndexState{Filter: filter}
	if data != nil {
		index.Items = data.Items
		if data.Err != nil {
			index.Err = data.Err.Error()
		}
	}
	a.bugView.Mode = model.BugModeIndex
	a.bugView.Index = index
	a.input = a.input.Blur()
	a.resizeActiveLayout()
}

func (a *App) openBugDetail(data *model.BugEventData) {
	if data == nil {
		return
	}
	a.bugView.Mode = model.BugModeDetail
	a.bugView.Detail = model.BugDetailState{
		ID:        data.ID,
		Bug:       data.Bug,
		Activity:  data.Activity,
		FromIndex: data.FromIndex,
	}
	if data.Err != nil {
		a.bugView.Detail.Err = data.Err.Error()
	}
	a.input = a.input.Blur()
	a.resizeActiveLayout()
}

func (a *App) closeBugView() {
	a.bugView = model.BugViewState{}
	a.input, _ = a.input.Focus()
	a.resizeActiveLayout()
}

func (a App) handleBugKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch a.bugView.Mode {
	case model.BugModeIndex:
		return a.handleBugIndexKey(msg)
	case model.BugModeDetail:
		return a.handleBugDetailKey(msg)
	default:
		return a, nil
	}
}

func (a App) handleBugIndexKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "esc":
		a.closeBugView()
	case "j", "down":
		if a.bugView.Index.Cursor < len(a.bugView.Index.Items)-1 {
			a.bugView.Index.Cursor++
		}
	case "k", "up":
		if a.bugView.Index.Cursor > 0 {
			a.bugView.Index.Cursor--
		}
	case "enter":
		if len(a.bugView.Index.Items) == 0 || a.userCh == nil {
			return a, nil
		}
		id := a.bugView.Index.Items[a.bugView.Index.Cursor].ID
		select {
		case a.userCh <- fmt.Sprintf("/__bug_detail %d", id):
		default:
		}
	case "c":
		if len(a.bugView.Index.Items) == 0 || a.userCh == nil {
			return a, nil
		}
		id := a.bugView.Index.Items[a.bugView.Index.Cursor].ID
		a.applyBugClaim(id)
		select {
		case a.userCh <- fmt.Sprintf("/claim %d", id):
		default:
		}
	}
	return a, nil
}

func (a App) handleBugDetailKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "esc":
		if a.bugView.Detail.FromIndex {
			a.bugView.Mode = model.BugModeIndex
			return a, nil
		}
		a.closeBugView()
	case "c":
		if a.bugView.Detail.ID == 0 || a.userCh == nil {
			return a, nil
		}
		a.applyBugClaim(a.bugView.Detail.ID)
		select {
		case a.userCh <- fmt.Sprintf("/claim %d", a.bugView.Detail.ID):
		default:
		}
	case "C":
		if a.bugView.Detail.ID == 0 || a.userCh == nil {
			return a, nil
		}
		a.applyBugClose(a.bugView.Detail.ID)
		select {
		case a.userCh <- fmt.Sprintf("/close %d", a.bugView.Detail.ID):
		default:
		}
	}
	return a, nil
}

func (a *App) applyBugClaim(id int) {
	now := time.Now()
	actor := a.state.IssueUser
	for i := range a.bugView.Index.Items {
		if a.bugView.Index.Items[i].ID == id {
			a.bugView.Index.Items[i].Status = "doing"
			if actor != "" {
				a.bugView.Index.Items[i].Lead = actor
			}
			a.bugView.Index.Items[i].UpdatedAt = now
			break
		}
	}
	if a.bugView.Detail.Bug != nil && a.bugView.Detail.Bug.ID == id {
		a.bugView.Detail.Bug.Status = "doing"
		if actor != "" {
			a.bugView.Detail.Bug.Lead = actor
		}
		a.bugView.Detail.Bug.UpdatedAt = now
		if actor != "" {
			a.bugView.Detail.Activity = append(a.bugView.Detail.Activity, bugs.Activity{
				BugID:     id,
				Actor:     actor,
				Type:      "claim",
				Text:      fmt.Sprintf("%s claimed bug", actor),
				CreatedAt: now,
			})
		}
	}
}

func (a *App) applyBugClose(id int) {
	now := time.Now()
	actor := a.state.IssueUser
	for i := range a.bugView.Index.Items {
		if a.bugView.Index.Items[i].ID == id {
			a.bugView.Index.Items[i].Status = "closed"
			a.bugView.Index.Items[i].UpdatedAt = now
			break
		}
	}
	if a.bugView.Detail.Bug != nil && a.bugView.Detail.Bug.ID == id {
		a.bugView.Detail.Bug.Status = "closed"
		a.bugView.Detail.Bug.UpdatedAt = now
		if actor != "" {
			a.bugView.Detail.Activity = append(a.bugView.Detail.Activity, bugs.Activity{
				BugID:     id,
				Actor:     actor,
				Type:      "close",
				Text:      fmt.Sprintf("%s closed bug", actor),
				CreatedAt: now,
			})
		}
	}
}

func (a App) renderBugView() string {
	topBar := panels.RenderTopBar(a.state, a.width)
	hintBar := panels.RenderBugHintBar(a.width, a.bugView.Mode)
	bodyHeight := a.height - lipgloss.Height(topBar) - lipgloss.Height(hintBar) - bottomSafePadding
	if bodyHeight < 1 {
		bodyHeight = 1
	}
	bodyWidth := a.width
	if bodyWidth < 1 {
		bodyWidth = 1
	}

	body := ""
	switch a.bugView.Mode {
	case model.BugModeIndex:
		body = panels.RenderBugIndex(bodyWidth, bodyHeight, a.bugView.Index)
	case model.BugModeDetail:
		body = panels.RenderBugDetail(bodyWidth, bodyHeight, a.bugView.Detail)
	}

	return trimViewHeight(lipgloss.JoinVertical(lipgloss.Left, topBar, body, hintBar, "", ""), a.height)
}
