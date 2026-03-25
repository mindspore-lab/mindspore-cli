package app

import (
	"fmt"
	"strconv"
	"strings"

	issuepkg "github.com/vigo999/ms-cli/internal/issues"
	"github.com/vigo999/ms-cli/ui/model"
)

func (a *Application) cmdIssueReportInput(input string) {
	if !a.ensureIssueService() {
		return
	}

	kind, title, err := parseIssueReportInput(input)
	if err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: err.Error()}
		return
	}
	issue, err := a.issueService.CreateIssue(title, kind, a.issueUser)
	if err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: fmt.Sprintf("report failed: %v", err)}
		return
	}
	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: fmt.Sprintf("%s created [%s]: %s", issue.Key, issue.Kind, issue.Title),
	}
}

func (a *Application) cmdIssues(args []string) {
	if !a.ensureIssueService() {
		return
	}
	status := "all"
	if len(args) > 0 {
		status = strings.ToLower(strings.TrimSpace(args[0]))
	}
	listStatus := status
	if status == "all" {
		listStatus = ""
	}
	issueList, err := a.issueService.ListIssues(listStatus)
	if err != nil {
		a.EventCh <- model.Event{
			Type: model.IssueIndexOpen,
			IssueView: &model.IssueEventData{
				Filter: status,
				Err:    err,
			},
		}
		return
	}
	a.EventCh <- model.Event{
		Type: model.IssueIndexOpen,
		IssueView: &model.IssueEventData{
			Filter: status,
			Items:  issueList,
		},
	}
}

func (a *Application) cmdIssueDetail(args []string) {
	if !a.ensureIssueService() {
		return
	}
	if len(args) == 0 {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: "Usage: /__issue_detail <issue-id>"}
		return
	}
	id, err := parseIssueRef(args[0])
	if err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: "invalid issue id"}
		return
	}
	a.emitIssueDetail(id, true)
}

func (a *Application) cmdIssueNoteInput(input string) {
	if !a.ensureIssueService() {
		return
	}
	ref, content, err := splitIssueNoteInput(input)
	if err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: err.Error()}
		return
	}
	id, err := parseIssueRef(ref)
	if err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: "invalid issue id"}
		return
	}
	if _, err := a.issueService.AddNote(id, a.issueUser, content); err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: fmt.Sprintf("add note failed: %v", err)}
		return
	}
	a.emitIssueDetail(id, true)
}

func (a *Application) cmdIssueClaim(args []string) {
	if !a.ensureIssueService() {
		return
	}
	if len(args) == 0 {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: "Usage: /__issue_claim <issue-id>"}
		return
	}
	id, err := parseIssueRef(args[0])
	if err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: "invalid issue id"}
		return
	}
	if _, err := a.issueService.ClaimIssue(id, a.issueUser); err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: fmt.Sprintf("claim issue failed: %v", err)}
		return
	}
	a.emitIssueDetail(id, true)
}

func (a *Application) cmdIssueStatus(args []string) {
	if !a.ensureIssueService() {
		return
	}
	if len(args) < 2 {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: "Usage: /status <ISSUE-id> <ready|doing|closed>"}
		return
	}
	id, err := parseIssueRef(args[0])
	if err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: "invalid issue id"}
		return
	}
	status, err := issuepkg.NormalizeStatus(args[1])
	if err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: err.Error()}
		return
	}
	if _, err := a.issueService.UpdateStatus(id, status, a.issueUser); err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: fmt.Sprintf("update issue status failed: %v", err)}
		return
	}
	a.emitIssueDetail(id, true)
}

func (a *Application) cmdDiagnose(input string) {
	target, err := parseIssueCommandTarget(input, "/diagnose")
	if err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: err.Error()}
		return
	}
	if target.HasIssue && !a.ensureIssueService() {
		return
	}
	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: pendingIssueFlowMessage("diagnose", target),
	}
}

func (a *Application) cmdFix(input string) {
	target, err := parseIssueCommandTarget(input, "/fix")
	if err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: err.Error()}
		return
	}
	if target.HasIssue && !a.ensureIssueService() {
		return
	}
	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: pendingIssueFlowMessage("fix", target),
	}
}

func (a *Application) emitIssueDetail(id int, fromIndex bool) {
	issue, err := a.issueService.GetIssue(id)
	if err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: fmt.Sprintf("get issue failed: %v", err)}
		return
	}
	notes, err := a.issueService.ListNotes(id)
	if err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: fmt.Sprintf("list issue notes failed: %v", err)}
		return
	}
	acts, err := a.issueService.GetActivity(id)
	if err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: fmt.Sprintf("list issue activity failed: %v", err)}
		return
	}
	a.EventCh <- model.Event{
		Type: model.IssueDetailOpen,
		IssueView: &model.IssueEventData{
			ID:        id,
			Issue:     issue,
			Notes:     notes,
			Activity:  acts,
			FromIndex: fromIndex,
		},
	}
}

func parseIssueReportInput(input string) (issuepkg.Kind, string, error) {
	input = strings.TrimSpace(input)
	if input == "" {
		return "", "", fmt.Errorf("Usage: /report <failure|accuracy|performance> <title>")
	}
	parts := strings.Fields(input)
	if len(parts) < 2 {
		return "", "", fmt.Errorf("Usage: /report <failure|accuracy|performance> <title>")
	}
	kind, err := issuepkg.NormalizeKind(parts[0])
	if err != nil {
		return "", "", fmt.Errorf("Usage: /report <failure|accuracy|performance> <title>")
	}
	title := strings.TrimSpace(strings.TrimPrefix(input, parts[0]))
	if title == "" {
		return "", "", fmt.Errorf("Usage: /report <failure|accuracy|performance> <title>")
	}
	return kind, title, nil
}

func parseIssueRef(ref string) (int, error) {
	ref = strings.TrimSpace(strings.ToUpper(ref))
	ref = strings.TrimPrefix(ref, "ISSUE-")
	return strconv.Atoi(ref)
}

func splitIssueNoteInput(input string) (string, string, error) {
	input = strings.TrimSpace(input)
	if input == "" {
		return "", "", fmt.Errorf("Usage: /__issue_note <ISSUE-id> <content>")
	}
	parts := strings.Fields(input)
	if len(parts) < 2 {
		return "", "", fmt.Errorf("Usage: /__issue_note <ISSUE-id> <content>")
	}
	ref := parts[0]
	content := strings.TrimSpace(strings.TrimPrefix(input, ref))
	if content == "" {
		return "", "", fmt.Errorf("Usage: /__issue_note <ISSUE-id> <content>")
	}
	return ref, content, nil
}

type issueCommandTarget struct {
	HasIssue bool
	IssueID  int
	Prompt   string
}

func parseIssueCommandTarget(input string, command string) (issueCommandTarget, error) {
	trimmed := strings.TrimSpace(input)
	if trimmed == "" {
		return issueCommandTarget{}, fmt.Errorf("Usage: %s <problem text|ISSUE-id>", command)
	}

	parts := strings.Fields(trimmed)
	first := parts[0]
	if looksLikeIssueKey(first) {
		id, err := parseIssueRef(first)
		if err != nil {
			return issueCommandTarget{}, fmt.Errorf("invalid issue id")
		}
		return issueCommandTarget{
			HasIssue: true,
			IssueID:  id,
			Prompt:   strings.TrimSpace(strings.TrimPrefix(trimmed, first)),
		}, nil
	}

	return issueCommandTarget{Prompt: trimmed}, nil
}

func looksLikeIssueKey(token string) bool {
	return strings.HasPrefix(strings.ToUpper(strings.TrimSpace(token)), "ISSUE-")
}

func pendingIssueFlowMessage(action string, target issueCommandTarget) string {
	if target.HasIssue {
		if target.Prompt != "" {
			return fmt.Sprintf("%s flow for %s is not wired yet (extra context: %s)", action, issuepkg.IssueKey(target.IssueID), target.Prompt)
		}
		return fmt.Sprintf("%s flow for %s is not wired yet", action, issuepkg.IssueKey(target.IssueID))
	}
	return fmt.Sprintf("%s flow for %q is not wired yet", action, target.Prompt)
}
