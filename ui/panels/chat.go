package panels

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/vigo999/ms-cli/ui/model"
)

var (
	userStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("39")).
			Bold(true)

	agentStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("252"))

	thinkingStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("205")).
			Italic(true)

	// expanded tool block
	toolBorderStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("240"))

	toolHeaderStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("214")).
			Bold(true)

	toolContentStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("250")).
				PaddingLeft(2)

	// collapsed tool (dim, single line)
	collapsedIconStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("240"))

	collapsedNameStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("244"))

	collapsedSummaryStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("240")).
				Italic(true)

	// error tool block
	errorBorderStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("196"))

	errorHeaderStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("196")).
				Bold(true)

	errorContentStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("203")).
				PaddingLeft(2)

	// edit/write diff styles
	diffAddStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("114"))

	diffRemoveStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("203"))

	diffNeutralStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("250")).
				PaddingLeft(2)
)

// RenderMessages converts messages into styled text for the viewport.
// stats is used to show summary when thinking is done.
// compact uses single-line spacing (for the train agent box).
func RenderMessages(state model.State, spinnerView string, width int, compact ...bool) string {
	var parts []string
	messages := state.Messages
	stats := state.Stats
	isThinking := state.IsThinking
	if width < 12 {
		width = 12
	}

	for _, m := range messages {
		switch m.Kind {
		case model.MsgUser:
			parts = append(parts, renderUserMsg(m.Content, width))
		case model.MsgAgent:
			parts = append(parts, renderAgentMsg(m.Content, width))
		case model.MsgThinking:
			if isThinking {
				// Show animated thinking indicator
				parts = append(parts, renderThinking(spinnerView, width))
			} else {
				// Show "Done" with summary
				parts = append(parts, renderDone(stats, width))
			}
		case model.MsgTool:
			parts = append(parts, renderTool(m, width))
		}
	}

	sep := "\n"
	return strings.Join(parts, sep)
}

func renderUserMsg(content string, width int) string {
	return renderPrefixedBlock(userStyle.Render(content), width, "  "+userStyle.Render(">")+" ", "    ")
}

func renderAgentMsg(content string, width int) string {
	return renderPrefixedBlock(agentStyle.Render(content), width, "  ", "  ")
}

func renderThinking(thinkingView string, width int) string {
	// Animated thinking indicator with Braille spinner
	// thinkingView already contains the spinner and text from ThinkingSpinner.View()
	return renderPrefixedBlock(thinkingView, width, "  ", "  ")
}

// renderDone shows completed task summary without animation
func renderDone(stats model.TaskStats, width int) string {
	var parts []string

	// Always show Done
	doneStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("114")) // green
	parts = append(parts, doneStyle.Render("✓ Done"))

	// Show simple execution summary if there's anything to report
	summaryParts := []string{}
	if stats.Commands > 0 {
		summaryParts = append(summaryParts, fmt.Sprintf("%d commands", stats.Commands))
	}
	if stats.FilesRead > 0 {
		summaryParts = append(summaryParts, fmt.Sprintf("%d files read", stats.FilesRead))
	}
	if stats.FilesEdited > 0 {
		summaryParts = append(summaryParts, fmt.Sprintf("%d files modified", stats.FilesEdited))
	}
	if stats.Searches > 0 {
		summaryParts = append(summaryParts, fmt.Sprintf("%d searches", stats.Searches))
	}
	if stats.Errors > 0 {
		summaryParts = append(summaryParts, fmt.Sprintf("%d errors", stats.Errors))
	}

	if len(summaryParts) > 0 {
		summaryStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("244"))
		parts = append(parts, summaryStyle.Render("("+strings.Join(summaryParts, ", ")+")"))
	}

	return renderPrefixedBlock(strings.Join(parts, " "), width, "  ", "  ")
}

func renderTool(m model.Message, width int) string {
	switch m.Display {
	case model.DisplayCollapsed:
		return renderCollapsedTool(m, width)
	case model.DisplayError:
		return renderErrorTool(m, width)
	default:
		return renderExpandedTool(m, width)
	}
}

// --- Collapsed: single dim line ---
// "  ▸ Read model/layer3.go — 42 lines"
// "  ▸ Grep "allocTensor" — 5 matches"
func renderCollapsedTool(m model.Message, width int) string {
	summary := ""
	if m.Summary != "" {
		summary = " — " + collapsedSummaryStyle.Render(m.Summary)
	}
	body := collapsedNameStyle.Render(strings.TrimSpace(m.ToolName + " " + m.Content))
	return renderPrefixedBlock(body+summary, width, "  "+collapsedIconStyle.Render("▸")+" ", "    ")
}

// --- Expanded: full output with header + body ---
func renderExpandedTool(m model.Message, width int) string {
	// edit/write get diff rendering
	if m.ToolName == "Edit" || m.ToolName == "Write" {
		return renderDiffTool(m, width)
	}

	header := renderToolHeader("▸", m.ToolName, toolBorderStyle, toolHeaderStyle, width)

	lines := strings.Split(m.Content, "\n")
	styled := make([]string, len(lines))
	for i, line := range lines {
		styled[i] = toolContentStyle.Width(maxBodyWidth(width)).Render(line)
	}
	body := strings.Join(styled, "\n")

	return header + "\n" + body
}

// --- Diff: edit/write with +/- coloring ---
func renderDiffTool(m model.Message, width int) string {
	header := renderToolHeader("▸", m.ToolName, toolBorderStyle, toolHeaderStyle, width)

	lines := strings.Split(m.Content, "\n")
	styled := make([]string, len(lines))
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		switch {
		case strings.HasPrefix(trimmed, "+"):
			styled[i] = renderPrefixedBlock(diffAddStyle.Render(line), width, "  ", "  ")
		case strings.HasPrefix(trimmed, "-"):
			styled[i] = renderPrefixedBlock(diffRemoveStyle.Render(line), width, "  ", "  ")
		default:
			styled[i] = diffNeutralStyle.Width(maxBodyWidth(width)).Render(line)
		}
	}
	body := strings.Join(styled, "\n")

	return header + "\n" + body
}

// --- Error: red highlighted block ---
func renderErrorTool(m model.Message, width int) string {
	header := renderToolHeader("✗", m.ToolName+" failed", errorBorderStyle, errorHeaderStyle, width)

	lines := strings.Split(m.Content, "\n")
	styled := make([]string, len(lines))
	for i, line := range lines {
		styled[i] = errorContentStyle.Width(maxBodyWidth(width)).Render(line)
	}
	body := strings.Join(styled, "\n")

	return header + "\n" + body
}

func renderPrefixedBlock(content string, width int, firstPrefix, restPrefix string) string {
	prefixWidth := lipgloss.Width(firstPrefix)
	if w := lipgloss.Width(restPrefix); w > prefixWidth {
		prefixWidth = w
	}
	bodyWidth := width - prefixWidth
	if bodyWidth < 1 {
		bodyWidth = 1
	}
	wrapped := lipgloss.NewStyle().Width(bodyWidth).Render(content)
	lines := strings.Split(wrapped, "\n")
	for i := range lines {
		if i == 0 {
			lines[i] = firstPrefix + lines[i]
			continue
		}
		lines[i] = restPrefix + lines[i]
	}
	return strings.Join(lines, "\n")
}

func renderToolHeader(icon, title string, borderStyle, titleStyle lipgloss.Style, width int) string {
	dividerWidth := width - lipgloss.Width(title) - 6
	if dividerWidth < 6 {
		dividerWidth = 6
	}
	return fmt.Sprintf("  %s %s %s",
		borderStyle.Render(icon),
		titleStyle.Render(title),
		borderStyle.Render(strings.Repeat("─", dividerWidth)),
	)
}

func maxBodyWidth(width int) int {
	if width < 1 {
		return 1
	}
	return width
}
