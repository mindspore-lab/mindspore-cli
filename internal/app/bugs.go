package app

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/vigo999/ms-cli/ui/model"
	"github.com/vigo999/ms-cli/ui/render"
)

func (a *Application) cmdReport(args []string) {
	if !a.ensureIssueService() {
		return
	}
	if len(args) == 0 {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: "Usage: /report <bug title>"}
		return
	}
	title := strings.Join(args, " ")
	bug, err := a.issueService.ReportBug(title, a.issueUser)
	if err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: fmt.Sprintf("report failed: %v", err)}
		return
	}
	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: fmt.Sprintf("created BUG-%d: %s", bug.ID, bug.Title),
	}
}

func (a *Application) cmdBugs(args []string) {
	if !a.ensureIssueService() {
		return
	}
	status := ""
	if len(args) > 0 {
		status = args[0]
	}
	bugs, err := a.issueService.ListBugs(status)
	if err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: fmt.Sprintf("list bugs failed: %v", err)}
		return
	}
	if len(bugs) == 0 {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: "no bugs found."}
		return
	}
	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: render.BugList(bugs),
	}
}

func (a *Application) cmdClaim(args []string) {
	if !a.ensureIssueService() {
		return
	}
	if len(args) == 0 {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: "Usage: /claim <bug-id>"}
		return
	}
	id, err := strconv.Atoi(args[0])
	if err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: "invalid bug id"}
		return
	}
	if err := a.issueService.ClaimBug(id, a.issueUser); err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: fmt.Sprintf("claim failed: %v", err)}
		return
	}
	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: fmt.Sprintf("you claimed BUG-%d", id),
	}
}

func (a *Application) cmdDock() {
	if !a.ensureIssueService() {
		return
	}
	data, err := a.issueService.DockSummary()
	if err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: fmt.Sprintf("dock failed: %v", err)}
		return
	}
	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: render.Dock(data),
	}
}
