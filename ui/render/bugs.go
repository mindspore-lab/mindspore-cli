package render

import (
	"fmt"

	"github.com/vigo999/ms-cli/internal/issues"
)

func BugList(bugs []issues.Bug) string {
	idW, titleW := 6, 12
	for _, b := range bugs {
		idStr := fmt.Sprintf("BUG-%d", b.ID)
		if len(idStr) > idW {
			idW = len(idStr)
		}
		if len(b.Title) > titleW {
			titleW = len(b.Title)
		}
	}
	if titleW > 50 {
		titleW = 50
	}

	var lines []string
	header := fmt.Sprintf("  %-*s  %-*s  %-8s  %-10s  %s",
		idW, "ID", titleW, "TITLE", "STATUS", "LEAD", "REPORTER")
	lines = append(lines, TitleStyle.Render("bug list"))
	lines = append(lines, LabelStyle.Render(header))

	for _, b := range bugs {
		idStr := fmt.Sprintf("BUG-%d", b.ID)
		title := b.Title
		if len(title) > titleW {
			title = title[:titleW-3] + "..."
		}
		lead := b.Lead
		if lead == "" {
			lead = "-"
		}
		statusStyle := StatusOpenStyle
		if b.Status == "doing" {
			statusStyle = StatusDoingStyle
		}
		line := fmt.Sprintf("  %-*s  %-*s  %s  %-10s  %s",
			idW, idStr, titleW, title,
			statusStyle.Render(fmt.Sprintf("%-8s", b.Status)),
			lead, b.Reporter)
		lines = append(lines, line)
	}
	return Box(lines)
}

func Dock(data *issues.DockData) string {
	lines := []string{
		TitleStyle.Render("dock"),
		"",
		fmt.Sprintf("  %s %s",
			LabelStyle.Render("open bugs"),
			ValueStyle.Render(fmt.Sprintf("%d", data.OpenCount)),
		),
	}

	if len(data.ReadyBugs) > 0 {
		lines = append(lines, "", LabelStyle.Render("  ready (unassigned)"))
		for _, b := range data.ReadyBugs {
			lines = append(lines, fmt.Sprintf("    BUG-%d  %s  %s",
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

	return Box(lines)
}
