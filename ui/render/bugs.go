package render

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/vigo999/ms-cli/internal/bugs"
)

func visPad(s string, w int) string {
	visible := lipgloss.Width(s)
	if visible >= w {
		return s
	}
	return s + strings.Repeat(" ", w-visible)
}

func visTruncate(s string, w int) string {
	if lipgloss.Width(s) <= w {
		return s
	}
	// Truncate rune by rune until it fits.
	runes := []rune(s)
	for i := len(runes) - 1; i >= 0; i-- {
		candidate := string(runes[:i]) + "..."
		if lipgloss.Width(candidate) <= w {
			return candidate
		}
	}
	return "..."
}

func BugList(bugs []bugs.Bug) string {
	idW, titleW, statusW, leadW, reporterW := 2, 12, 6, 4, 8
	for _, b := range bugs {
		if l := len(fmt.Sprintf("%d", b.ID)); l > idW {
			idW = l
		}
		if w := lipgloss.Width(b.Title); w > titleW {
			titleW = w
		}
		if w := lipgloss.Width(b.Status); w > statusW {
			statusW = w
		}
		lead := b.Lead
		if lead == "" {
			lead = "-"
		}
		if w := lipgloss.Width(lead); w > leadW {
			leadW = w
		}
		if w := lipgloss.Width(b.Reporter); w > reporterW {
			reporterW = w
		}
	}
	if titleW > 50 {
		titleW = 50
	}

	var lines []string
	header := visPad("  "+visPad("id", idW)+"  "+visPad("title", titleW)+"  "+visPad("status", statusW)+"  "+visPad("lead", leadW)+"  "+visPad("reporter", reporterW), 0)
	lines = append(lines, TitleStyle.Render("BUG LIST"))
	lines = append(lines, TitleStyle.Render(header))

	for _, b := range bugs {
		title := visTruncate(b.Title, titleW)
		lead := b.Lead
		if lead == "" {
			lead = "-"
		}
		statusStyle := StatusOpenStyle
		if b.Status == "doing" {
			statusStyle = StatusDoingStyle
		}

		line := "  " +
			visPad(fmt.Sprintf("%d", b.ID), idW) + "  " +
			visPad(title, titleW) + "  " +
			statusStyle.Render(visPad(b.Status, statusW)) + "  " +
			visPad(lead, leadW) + "  " +
			visPad(b.Reporter, reporterW)
		lines = append(lines, line)
	}
	return strings.Join(lines, "\n")
}

func Dock(data *bugs.DockData) string {
	lines := []string{
		TitleStyle.Render("DOCK"),
		"",
		fmt.Sprintf("  %s %s    %s %s",
			LabelStyle.Render("open bugs"),
			ValueStyle.Render(fmt.Sprintf("%d", data.OpenCount)),
			LabelStyle.Render("online (24h)"),
			ValueStyle.Render(fmt.Sprintf("%d", data.OnlineCount)),
		),
	}

	if len(data.ReadyBugs) > 0 {
		lines = append(lines, "", LabelStyle.Render("  ready (unassigned)"))
		for _, b := range data.ReadyBugs {
			lines = append(lines, fmt.Sprintf("    %d  %s  %s",
				b.ID, b.Title, StatusOpenStyle.Render(b.Status)))
		}
	}

	if len(data.RecentFeed) > 0 {
		lines = append(lines, "", LabelStyle.Render("  recent activity"))
		for _, a := range data.RecentFeed {
			ts := a.CreatedAt.Format("01-02 15:04")
			lines = append(lines, ActivityStyle.Render(fmt.Sprintf("    %s  %s  %s", ts, a.Actor, a.Text)))
		}
	}

	return strings.Join(lines, "\n")
}
