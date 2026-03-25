package render

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
	issuepkg "github.com/vigo999/ms-cli/internal/issues"
)

var issueRowSelectedStyle = lipgloss.NewStyle().Background(lipgloss.Color("237"))

func IssueIndex(items []issuepkg.Issue, cursor, width, rows int) string {
	if rows < 1 {
		rows = 1
	}
	if width < 36 {
		width = 36
	}

	keyW := 8
	kindW := 12
	statusW := 8
	leadW := 10
	markerW := 2
	spaceBudget := markerW + keyW + kindW + statusW + leadW + 4
	titleW := width - spaceBudget
	if titleW < 12 {
		titleW = 12
	}

	header := visPad("", markerW) +
		centerPad("key", keyW) + " " +
		centerPad("title", titleW) + " " +
		visPad("kind", kindW) + " " +
		visPad("status", statusW) + " " +
		visPad("lead", leadW)

	lines := []string{bugHeaderStyle.Render(header)}
	if len(items) == 0 {
		lines = append(lines, ValueStyle.Render("no issues found."))
		return strings.Join(lines, "\n")
	}

	maxRows := rows - len(lines)
	if maxRows < 1 {
		maxRows = 1
	}
	start := bugWindowStart(len(items), cursor, maxRows)
	end := start + maxRows
	if end > len(items) {
		end = len(items)
	}
	for i := start; i < end; i++ {
		it := items[i]
		marker := "  "
		if i == cursor {
			marker = "❯ "
		}
		title := visPad(visTruncate(strings.TrimSpace(it.Title), titleW), titleW)
		if i == cursor {
			title = issueRowSelectedStyle.Render(title)
		}
		line := marker +
			centerPad(it.Key, keyW) + " " +
			title + " " +
			visPad(string(it.Kind), kindW) + " " +
			bugStatusStyle(it.Status).Render(visPad(it.Status, statusW)) + " " +
			visPad(issueLeadValue(it), leadW)
		lines = append(lines, line)
	}

	return strings.Join(lines, "\n")
}

func IssueDetail(it issuepkg.Issue, notes []issuepkg.Note, activity []issuepkg.Activity, width, rows int) string {
	if width < 40 {
		width = 40
	}

	sectionWidth := width - 2
	if sectionWidth < 1 {
		sectionWidth = 1
	}
	lines := []string{
		TitleStyle.Render(fmt.Sprintf("%s  %s", it.Key, strings.TrimSpace(it.Title))),
		"",
		LabelStyle.Render(fmt.Sprintf("kind: %s    status: %s    lead: %s", it.Kind, it.Status, issueLeadValue(it))),
		LabelStyle.Render(fmt.Sprintf("reporter: %s", strings.TrimSpace(it.Reporter))),
		separatorLine(sectionWidth),
		TitleStyle.Render("SUMMARY"),
		renderIssueSummary(it.Summary),
		separatorLine(sectionWidth),
		TitleStyle.Render("NOTES"),
	}

	lines = append(lines, renderIssueNotes(notes)...)
	lines = append(lines,
		separatorLine(sectionWidth),
		TitleStyle.Render("ACTIVITY"),
	)
	lines = append(lines, renderIssueActivity(activity)...)
	lines = append(lines,
		separatorLine(sectionWidth),
		TitleStyle.Render("ACTION"),
		ValueStyle.Render("[d] diagnose   [f] fix   [l] take lead   [s] status"),
	)

	return strings.Join(trimBugLines(lines, rows), "\n")
}

func renderIssueSummary(summary string) string {
	summary = strings.TrimSpace(summary)
	if summary == "" {
		return ValueStyle.Render("-")
	}
	return ValueStyle.Render(summary)
}

func renderIssueNotes(notes []issuepkg.Note) []string {
	if len(notes) == 0 {
		return []string{ValueStyle.Render("no notes yet.")}
	}
	start := 0
	if len(notes) > 6 {
		start = len(notes) - 6
	}
	lines := make([]string, 0, len(notes)-start)
	for _, note := range notes[start:] {
		lines = append(lines, fmt.Sprintf("• %s   %s", note.Author, strings.TrimSpace(note.Content)))
	}
	return lines
}

func renderIssueActivity(activity []issuepkg.Activity) []string {
	if len(activity) == 0 {
		return []string{ValueStyle.Render("no activity yet.")}
	}
	start := 0
	if len(activity) > 6 {
		start = len(activity) - 6
	}
	lines := make([]string, 0, len(activity)-start)
	for _, act := range activity[start:] {
		lines = append(lines, fmt.Sprintf("%s  %s", act.Actor, strings.TrimSpace(act.Text)))
	}
	return lines
}

func issueLeadValue(it issuepkg.Issue) string {
	if strings.TrimSpace(it.Lead) == "" {
		return "no owner"
	}
	return it.Lead
}

func separatorLine(width int) string {
	if width < 1 {
		width = 1
	}
	return strings.Repeat("─", width)
}
