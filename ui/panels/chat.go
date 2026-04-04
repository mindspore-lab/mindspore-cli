package panels

import (
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
	// uirender "github.com/mindspore-lab/mindspore-cli/ui/render"
)

// Style vars are populated by InitStyles() in styles.go.
var (
	userStyle              lipgloss.Style
	userBlockStyle         lipgloss.Style
	agentStyle             lipgloss.Style
	thinkingStyle          lipgloss.Style
	toolBorderStyle        lipgloss.Style
	toolHeaderStyle        lipgloss.Style
	toolContentStyle       lipgloss.Style
	collapsedIconStyle     lipgloss.Style
	collapsedNameStyle     lipgloss.Style
	collapsedTitleStyle    lipgloss.Style
	collapsedSummaryStyle  lipgloss.Style
	errorBorderStyle       lipgloss.Style
	errorHeaderStyle       lipgloss.Style
	errorContentStyle      lipgloss.Style
	diffAddStyle           lipgloss.Style
	diffRemoveStyle        lipgloss.Style
	diffNeutralStyle       lipgloss.Style
	toolPendingDotStyle    lipgloss.Style
	toolSuccessDotStyle    lipgloss.Style
	toolWarningDotStyle    lipgloss.Style
	toolErrorDotStyle      lipgloss.Style
	toolCallLineStyle      lipgloss.Style
	toolPendingStatusStyle lipgloss.Style
	toolResultPrefixStyle  lipgloss.Style
	toolResultSummaryStyle lipgloss.Style
	toolResultDetailStyle  lipgloss.Style
	toolResultWarningStyle lipgloss.Style
	toolResultErrorStyle   lipgloss.Style
)

// RenderMessages converts messages into styled text for the viewport.
// compact uses single-line spacing (for the train agent box).
func RenderMessages(state model.State, spinnerView, spinnerFrame string, width int, compact ...bool) string {
	var parts []string
	messages := state.Messages
	if width < 12 {
		width = 12
	}

	for _, m := range messages {
		switch m.Kind {
		case model.MsgUser:
			parts = append(parts, renderUserMsg(m.Content, width)+"\n")
		case model.MsgAgent:
			parts = append(parts, renderAgentMsg(m, width))
		case model.MsgTool:
			parts = append(parts, renderTool(state, m, spinnerFrame, width))
		}
	}

	if state.IsThinking {
		parts = append(parts, renderThinking(spinnerView, width))
	}

	sep := "\n\n"
	if len(compact) > 0 && compact[0] {
		sep = "\n"
	}
	return strings.Join(parts, sep)
}

func renderUserMsg(content string, width int) string {
	bodyWidth := width - 4
	if bodyWidth < 1 {
		bodyWidth = 1
	}
	// Wrap plain text — no nested ANSI, so background fills the full line.
	wrapped := lipgloss.NewStyle().Width(bodyWidth).Render(content)
	lines := strings.Split(wrapped, "\n")
	firstLineStyle := userBlockStyle.Copy().Bold(true).Width(width)
	restLineStyle := userBlockStyle.Copy().Width(width)
	for i, line := range lines {
		if i == 0 {
			lines[i] = firstLineStyle.Render("  ❯ " + line)
		} else {
			lines[i] = restLineStyle.Render("    " + line)
		}
	}
	return strings.Join(lines, "\n")
}

func renderAgentMsg(msg model.Message, width int) string {
	// Pre-render the prefix as a complete ANSI string so it doesn't
	// interfere with glamour's escape sequences (same approach as crush).
	bulletPrefix := agentStyle.Render("• ")
	blankPrefix := agentStyle.Render("  ")
	if msg.Streaming || msg.Display == model.DisplayNotice {
		bulletPrefix = blankPrefix
	}

	prefixWidth := lipgloss.Width(bulletPrefix)
	bodyWidth := cappedMessageWidth(width) - prefixWidth
	if bodyWidth < 1 {
		bodyWidth = 1
	}
	rendered := msg.Content
	if !msg.RawANSI {
		rendered = RenderMarkdown(msg.Content, bodyWidth)
	}
	lines := strings.Split(rendered, "\n")
	for i, line := range lines {
		if i == 0 {
			lines[i] = bulletPrefix + line
		} else {
			lines[i] = blankPrefix + line
		}
	}
	return strings.Join(lines, "\n")
}

func renderThinking(thinkingView string, width int) string {
	// Animated thinking indicator with Braille spinner
	// thinkingView already contains the spinner and text from ThinkingSpinner.View()
	return renderPrefixedBlock(thinkingView, width, "  ", "  ")
}

func renderTool(state model.State, m model.Message, spinnerFrame string, width int) string {
	call := renderToolCallLine(state, m, spinnerFrame)
	if m.Pending {
		return "  " + call
	}
	summary, details := toolResult(m)
	if summary == "" && len(details) == 0 {
		return "  " + call
	}
	// Tool call is indented under agent message; result lines indent further.
	bodyWidth := width - 7
	if bodyWidth < 1 {
		bodyWidth = 1
	}
	wrapStyle := lipgloss.NewStyle().Width(bodyWidth)
	lines := []string{"  " + call}
	if summary != "" {
		lines = append(lines, "    "+toolResultPrefixStyle.Render("⎿")+"  "+renderToolSummary(m, summary))
	}
	for _, line := range details {
		if strings.TrimSpace(line) == "" {
			continue
		}
		lines = append(lines, "       "+wrapStyle.Render(renderToolDetail(m, line)))
	}
	return strings.Join(lines, "\n")
}

// RenderToolCallHeader renders a tool call header line for inline printing.
func RenderToolCallHeader(toolName, args string) string {
	dot := toolPendingDotStyle.Render("⏺")
	return toolCallLineStyle.Render(dot + " " + strings.TrimSpace(toolName) + "(" + strings.TrimSpace(args) + ")")
}

func renderToolCallLine(state model.State, m model.Message, spinnerFrame string) string {
	dot := toolPendingDotStyle.Render("⏺")
	suffix := ""
	switch {
	case m.Pending || m.Streaming:
		if strings.TrimSpace(spinnerFrame) != "" && state.WaitKind == model.WaitTool {
			dot = spinnerFrame
		} else {
			dot = toolPendingDotStyle.Render("⏺")
		}
		suffix = renderPendingToolStatus(state, m)
	case m.Display == model.DisplayWarning:
		dot = toolWarningDotStyle.Render("⚠")
	case m.Display == model.DisplayError:
		dot = toolErrorDotStyle.Render("✗")
	default:
		dot = toolSuccessDotStyle.Render("✓")
	}
	name := toolCallLineStyle.Copy().Bold(true).Render(strings.TrimSpace(m.ToolName))
	args := toolCallLineStyle.Render("(" + strings.TrimSpace(toolCallArgs(m)) + ")")
	return dot + " " + name + args + suffix
}

func renderPendingToolStatus(state model.State, m model.Message) string {
	status := strings.TrimSpace(m.Summary)
	if status == "" {
		if strings.EqualFold(strings.TrimSpace(m.ToolName), "Shell") {
			status = "running command..."
		} else {
			status = "running..."
		}
	}
	if state.WaitKind == model.WaitTool && !state.WaitStartedAt.IsZero() {
		elapsed := state.WaitElapsed
		if elapsed <= 0 {
			elapsed = time.Since(state.WaitStartedAt)
		}
		status += " ctrl+o to expand " + model.FormatWaitDuration(elapsed)
	}
	return " " + toolPendingStatusStyle.Render(status)
}

func toolCallArgs(m model.Message) string {
	args := strings.TrimSpace(m.ToolArgs)
	if args == "" {
		args = strings.TrimSpace(toolHeadline(m.Content))
	}
	if args == "" {
		args = "none"
	}
	return args
}

func toolResult(m model.Message) (string, []string) {
	lines := nonEmptyLines(m.Content)
	summary := strings.TrimSpace(m.Summary)
	if summary == "" && len(lines) > 0 {
		summary = lines[0]
		lines = lines[1:]
	} else if summary != "" && len(lines) > 0 && strings.TrimSpace(lines[0]) == summary {
		lines = lines[1:]
	}
	return summary, lines
}

func renderToolSummary(m model.Message, line string) string {
	if m.Display == model.DisplayWarning {
		return toolResultWarningStyle.Render(line)
	}
	if m.Display == model.DisplayError {
		return toolResultErrorStyle.Render(line)
	}
	return toolResultSummaryStyle.Render(line)
}

func renderToolDetail(m model.Message, line string) string {
	if m.Display == model.DisplayWarning {
		return toolResultWarningStyle.Render(line)
	}
	if m.Display == model.DisplayError {
		return toolResultErrorStyle.Render(line)
	}
	return toolResultDetailStyle.Render(line)
}

func toolHeadline(content string) string {
	lines := strings.Split(strings.TrimSpace(content), "\n")
	for _, line := range lines {
		headline := strings.TrimSpace(line)
		if headline != "" {
			return headline
		}
	}
	return ""
}

func nonEmptyLines(content string) []string {
	raw := strings.Split(strings.TrimSpace(content), "\n")
	out := make([]string, 0, len(raw))
	for _, line := range raw {
		if strings.TrimSpace(line) == "" {
			continue
		}
		out = append(out, line)
	}
	return out
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
