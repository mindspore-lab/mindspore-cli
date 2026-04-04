package app

import (
	"fmt"
	"strconv"
	"strings"

	issuepkg "github.com/mindspore-lab/mindspore-cli/internal/issues"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
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
	a.runSkillCommand(input, "/diagnose")
}

func (a *Application) cmdFix(input string) {
	a.runSkillCommand(input, "/fix")
}

func (a *Application) cmdMigrate(input string) {
	task := fmt.Sprintf(
		"Load skill migrate-agent.\n\nUser request: %s",
		strings.TrimSpace(input),
	)
	a.EventCh <- model.Event{Type: model.AgentThinking}
	go a.runTask(task)
}

func (a *Application) runSkillCommand(input, command string) {
	mode := strings.TrimPrefix(command, "/")
	target, err := parseIssueCommandTarget(input, command)
	if err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: err.Error()}
		return
	}
	if target.HasIssue && !a.ensureIssueService() {
		return
	}

	task, err := a.buildSkillTask(target, mode)
	if err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: err.Error()}
		return
	}

	a.EventCh <- model.Event{Type: model.AgentThinking}
	go a.runTask(task)
}

// buildSkillTask constructs a task description that instructs the agent to load
// the appropriate diagnosis skill in the given mode (diagnose or fix).
func (a *Application) buildSkillTask(target issueCommandTarget, mode string) (string, error) {
	if target.HasIssue {
		issueCtx, err := a.buildIssueContext(target.IssueID)
		if err != nil {
			return "", fmt.Errorf("fetch issue failed: %w", err)
		}
		task := fmt.Sprintf("Load skill %s-agent in %s mode.\n\n%s", issueCtx.kind, mode, issueCtx.text)
		if target.Prompt != "" {
			task += "\n\nAdditional context: " + target.Prompt
		}
		return task, nil
	}

	return fmt.Sprintf(
		"Load the appropriate diagnosis skill (failure-agent, accuracy-agent, or performance-agent) in %s mode.\n\nUser problem: %s",
		mode, target.Prompt,
	), nil
}

type issueContext struct {
	kind string // failure, accuracy, performance
	text string // formatted issue details
}

func (a *Application) buildIssueContext(id int) (issueContext, error) {
	issue, err := a.issueService.GetIssue(id)
	if err != nil {
		return issueContext{}, err
	}

	var b strings.Builder
	fmt.Fprintf(&b, "Issue: %s — %s\n", issue.Key, issue.Title)
	fmt.Fprintf(&b, "Kind: %s\n", issue.Kind)
	if issue.Summary != "" {
		fmt.Fprintf(&b, "Summary: %s\n", issue.Summary)
	}

	notes, err := a.issueService.ListNotes(id)
	if err == nil && len(notes) > 0 {
		b.WriteString("\nNotes:\n")
		for _, n := range notes {
			fmt.Fprintf(&b, "- [%s] %s\n", n.Author, n.Content)
		}
	}

	return issueContext{kind: string(issue.Kind), text: b.String()}, nil
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
