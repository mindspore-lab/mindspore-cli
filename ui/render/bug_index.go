package render

import (
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/vigo999/ms-cli/internal/bugs"
)

var statusClosedStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("244"))
var bugRowSelectedStyle = lipgloss.NewStyle().Background(lipgloss.Color("237"))
var bugHeaderStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("15"))
var bugDetailLabelStyle = lipgloss.NewStyle()
var bugDetailValueStyle = lipgloss.NewStyle()
var bugDetailTitleStyle = lipgloss.NewStyle()

func BugIndex(items []bugs.Bug, cursor, width, rows int) string {
	if rows < 1 {
		rows = 1
	}
	if width < 24 {
		width = 24
	}

	idW := 4
	statusW := 8
	leadW := 10
	updatedW := 11

	markerW := 2
	spaceBudget := markerW + idW + statusW + leadW + updatedW + 4
	titleW := int(float64(width-spaceBudget) * 0.7)
	if titleW < 12 {
		titleW = 12
	}
	header := visPad("", markerW) +
		centerPad("id", idW) + " " +
		centerPad("title", titleW) + " " +
		visPad("status", statusW) + " " +
		visPad("lead", leadW) + " " +
		visPad("updated", updatedW)

	lines := []string{bugHeaderStyle.Render(header)}
	if len(items) == 0 {
		lines = append(lines, ValueStyle.Render("no bugs found."))
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
		b := items[i]
		marker := "  "
		if i == cursor {
			marker = "❯ "
		}
		status := bugStatusLabel(b.Status)
		title := visPad(visTruncate(bugTitleValue(b), titleW), titleW)
		if i == cursor {
			title = bugRowSelectedStyle.Render(title)
		}
		line := marker +
			centerPad(fmt.Sprintf("%d", b.ID), idW) + " " +
			title + " " +
			bugStatusStyle(b.Status).Render(visPad(status, statusW)) + " " +
			visPad(bugLeadValue(b), leadW) + " " +
			visPad(formatBugUpdatedAt(b.UpdatedAt), updatedW)
		lines = append(lines, line)
	}

	return strings.Join(lines, "\n")
}

func BugDetail(st bugs.Bug, activity []bugs.Activity, width, rows int) string {
	if rows < 1 {
		rows = 1
	}
	if width < 24 {
		width = 24
	}
	labelW := 10

	lines := []string{
		TitleStyle.Render(fmt.Sprintf("BUG %d:", st.ID)),
		bugDetailField("title:", st.Title, labelW, width, bugDetailTitleStyle),
		bugDetailField("tags:", bugTagsValue(st.Tags), labelW, width, bugDetailValueStyle),
		bugDetailField("status:", bugStatusLabel(st.Status), labelW, width, bugDetailValueStyle),
		bugDetailField("lead:", bugLeadValue(st), labelW, width, bugDetailValueStyle),
		bugDetailField("reporter:", st.Reporter, labelW, width, bugDetailValueStyle),
		bugDetailField("updated:", st.UpdatedAt.Format("01-02 15:04"), labelW, width, bugDetailValueStyle),
		"",
		TitleStyle.Render("activity"),
	}

	if len(activity) == 0 {
		lines = append(lines, ValueStyle.Render("no activity yet."))
		return strings.Join(trimBugLines(lines, rows), "\n")
	}

	available := rows - len(lines)
	if available < 1 {
		available = 1
	}
	start := 0
	if len(activity) > available {
		start = len(activity) - available
	}
	for _, act := range activity[start:] {
		lines = append(lines, fmt.Sprintf("%s  %s  %s", act.CreatedAt.Format("01-02 15:04"), act.Actor, bugActivityText(act, st.Title)))
	}

	return strings.Join(trimBugLines(lines, rows), "\n")
}

func bugActivityText(act bugs.Activity, title string) string {
	text := strings.TrimSpace(act.Text)
	reportLine := fmt.Sprintf("reported bug: %s", strings.TrimSpace(title))
	if act.Type == "report" && text == reportLine {
		return "reported bug"
	}
	actorPrefix := strings.TrimSpace(act.Actor) + " "
	if strings.TrimSpace(act.Actor) != "" && strings.HasPrefix(text, actorPrefix) {
		return strings.TrimPrefix(text, actorPrefix)
	}
	return text
}

func bugDetailField(label, value string, labelW, width int, valueStyle lipgloss.Style) string {
	if labelW < 1 {
		labelW = 1
	}
	valueW := width - labelW
	if valueW < 1 {
		valueW = 1
	}
	return bugDetailLabelStyle.Render(visPad(label, labelW)) + valueStyle.Width(valueW).Render(value)
}

func bugWindowStart(total, cursor, size int) int {
	if total <= size {
		return 0
	}
	start := cursor - size + 1
	if start < 0 {
		start = 0
	}
	if start+size > total {
		start = total - size
	}
	return start
}

func trimBugLines(lines []string, limit int) []string {
	if limit <= 0 || len(lines) <= limit {
		return lines
	}
	return lines[:limit]
}

func bugStatusStyle(status string) lipgloss.Style {
	switch strings.ToLower(strings.TrimSpace(status)) {
	case "open", "ready":
		return StatusOpenStyle
	case "doing":
		return StatusDoingStyle
	default:
		return statusClosedStyle
	}
}

func bugStatusLabel(status string) string {
	if strings.EqualFold(strings.TrimSpace(status), "doing") {
		return "doing"
	}
	return strings.ToLower(strings.TrimSpace(status))
}

func bugLeadValue(b bugs.Bug) string {
	if strings.TrimSpace(b.Lead) == "" {
		return "no owner"
	}
	return b.Lead
}

func bugTagsValue(tags []string) string {
	if len(tags) == 0 {
		return "-"
	}
	return strings.Join(tags, ", ")
}

func bugTitleValue(b bugs.Bug) string {
	title := strings.TrimSpace(b.Title)
	if len(b.Tags) == 0 {
		return title
	}
	return title + " [" + strings.Join(b.Tags, ",") + "]"
}

func formatBugUpdatedAt(ts time.Time) string {
	if ts.IsZero() {
		return "-"
	}
	return ts.Format("01-02 15:04")
}

func bugFilterLabel(filter string) string {
	filter = strings.TrimSpace(strings.ToLower(filter))
	if filter == "" {
		return "all"
	}
	return filter
}

func centerPad(s string, w int) string {
	visible := lipgloss.Width(s)
	if visible >= w {
		return s
	}
	left := (w - visible) / 2
	right := w - visible - left
	return strings.Repeat(" ", left) + s + strings.Repeat(" ", right)
}
