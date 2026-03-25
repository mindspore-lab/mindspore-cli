package app

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/vigo999/ms-cli/internal/bugs"
	"github.com/vigo999/ms-cli/ui/model"
	"github.com/vigo999/ms-cli/ui/render"
)

func (a *Application) cmdReport(args []string) {
	a.cmdReportInput(strings.Join(args, " "))
}

func (a *Application) cmdReportInput(input string) {
	if !a.ensureBugService() {
		return
	}

	title, tags, err := parseReportInput(input)
	if err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: err.Error()}
		return
	}
	bug, err := a.bugService.ReportBug(title, a.issueUser, tags)
	if err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: fmt.Sprintf("report failed: %v", err)}
		return
	}
	tagSummary := renderBugTagSummary(bug.Tags)
	if tagSummary != "" {
		tagSummary = " " + tagSummary
	}
	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: fmt.Sprintf("created bug #%d%s: %s", bug.ID, tagSummary, bug.Title),
	}
}

func (a *Application) cmdBugs(args []string) {
	if !a.ensureBugService() {
		return
	}
	status := "all"
	if len(args) > 0 {
		status = args[0]
	}
	listStatus := status
	if status == "all" {
		listStatus = ""
	}
	bugs, err := a.bugService.ListBugs(listStatus)
	if err != nil {
		a.EventCh <- model.Event{
			Type: model.BugIndexOpen,
			BugView: &model.BugEventData{
				Filter: status,
				Err:    err,
			},
		}
		return
	}
	a.EventCh <- model.Event{
		Type: model.BugIndexOpen,
		BugView: &model.BugEventData{
			Filter: status,
			Items:  bugs,
		},
	}
}

func (a *Application) cmdBugDetail(args []string) {
	if !a.ensureBugService() {
		return
	}
	if len(args) == 0 {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: "Usage: /__bug_detail <bug-id>"}
		return
	}
	id, err := strconv.Atoi(args[0])
	if err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: "invalid bug id"}
		return
	}
	bug, err := a.bugService.GetBug(id)
	if err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: fmt.Sprintf("get bug failed: %v", err)}
		return
	}
	acts, err := a.bugService.GetActivity(id)
	if err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: fmt.Sprintf("list activity failed: %v", err)}
		return
	}
	a.EventCh <- model.Event{
		Type: model.BugDetailOpen,
		BugView: &model.BugEventData{
			ID:        id,
			Bug:       bug,
			Activity:  acts,
			FromIndex: true,
		},
	}
}

func (a *Application) cmdClaim(args []string) {
	if !a.ensureBugService() {
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
	if err := a.bugService.ClaimBug(id, a.issueUser); err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: fmt.Sprintf("claim failed: %v", err)}
		return
	}
	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: fmt.Sprintf("you claimed bug #%d", id),
	}
}

func (a *Application) cmdClose(args []string) {
	if !a.ensureBugService() {
		return
	}
	if len(args) == 0 {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: "Usage: /close <bug-id>"}
		return
	}
	id, err := strconv.Atoi(args[0])
	if err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: "invalid bug id"}
		return
	}
	if err := a.bugService.CloseBug(id); err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: fmt.Sprintf("close failed: %v", err)}
		return
	}
	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: fmt.Sprintf("closed bug #%d", id),
	}
}

func (a *Application) cmdDock() {
	if !a.ensureBugService() {
		return
	}
	data, err := a.bugService.DockSummary()
	if err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: fmt.Sprintf("dock failed: %v", err)}
		return
	}
	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: render.Dock(data),
	}
}

func parseReportInput(input string) (string, []string, error) {
	input = strings.TrimSpace(input)
	if input == "" {
		return "", nil, fmt.Errorf("Usage: /report [tag1,tag2] <bug title>")
	}
	if !strings.HasPrefix(input, "[") {
		return input, nil, nil
	}

	end := strings.Index(input, "]")
	if end <= 0 {
		return "", nil, fmt.Errorf("Usage: /report [tag1,tag2] <bug title>")
	}

	tags := bugs.NormalizeTags(strings.Split(input[1:end], ","))
	title := strings.TrimSpace(input[end+1:])
	if title == "" {
		return "", nil, fmt.Errorf("Usage: /report [tag1,tag2] <bug title>")
	}
	return title, tags, nil
}

func renderBugTagSummary(tags []string) string {
	tags = bugs.NormalizeTags(tags)
	if len(tags) == 0 {
		return ""
	}
	return "[" + strings.Join(tags, ",") + "]"
}
