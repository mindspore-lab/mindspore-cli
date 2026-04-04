package panels

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

// Style vars are populated by InitStyles() in styles.go.
var (
	issueKindStyle    lipgloss.Style
	issueSummaryStyle lipgloss.Style
)

func RenderTrainIssue(tv model.TrainWorkspaceState, width int) []string {
	run := tv.ActiveRun()
	if run == nil {
		return []string{"   " + checkPendingStyle.Render("No active issue")}
	}
	if run.CurrentIssue == nil && run.Issue == nil && len(run.AgentActions) == 0 {
		return []string{"   " + checkPendingStyle.Render("No active issue")}
	}

	lines := []string{}
	if run.CurrentIssue != nil {
		lines = append(lines, fmt.Sprintf("   %s %s %s",
			checkFailedStyle.Render("[!]"),
			issueKindStyle.Render(strings.ToUpper(string(run.CurrentIssue.Kind))),
			issueSummaryStyle.Render(run.CurrentIssue.Summary)))
	}
	if run.Issue != nil && run.Issue.Title != "" {
		title := run.Issue.Title
		if !strings.Contains(title, "[!]") {
			title = "[!] " + title
		}
		lines = append(lines, "   "+metricLabelStyle.Render(title))
		if run.Issue.Detail != "" {
			lines = append(lines, "   "+checkDetailStyle.Render(run.Issue.Detail))
		}
	}
	if len(run.AgentActions) > 0 {
		lines = append(lines, "   "+metricLabelStyle.Render("Suggested actions:"))
		for _, action := range run.AgentActions {
			lines = append(lines, "   "+checkPassedStyle.Render("• "+action.Label))
		}
	}
	return lines
}
