package panels

import (
	"strings"

	"github.com/charmbracelet/lipgloss"
)

var (
	logTitleStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("252")).
			Background(lipgloss.Color("236")).
			Padding(0, 1)

	logLineStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("250"))

	logMetricLineStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("39"))

	logErrorLineStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("196")).
				Bold(true)

	logHighlightStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("114"))

	logTimestampStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("240"))

	logBorderStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("236"))
)

// RenderLaneLogs renders log lines for a lane or general log output.
func RenderLaneLogs(logs []string, title string, width, height int) string {
	var sections []string

	logCount := len(logs)
	titleText := title
	if titleText == "" {
		titleText = " Output Log"
	}
	if logCount > 0 {
		titleText += lipgloss.NewStyle().
			Foreground(lipgloss.Color("240")).
			Render(" (" + itoa(logCount) + ")")
	}
	sections = append(sections, " "+logTitleStyle.Render(titleText))
	sections = append(sections, " "+logBorderStyle.Render(strings.Repeat("─", width-2)))

	headerLines := len(sections)
	logHeight := height - headerLines
	if logHeight < 1 {
		logHeight = 1
	}

	visible := logs
	if len(visible) > logHeight {
		visible = visible[len(visible)-logHeight:]
	}

	for _, line := range visible {
		styled := styleLogLine(line, width-2)
		sections = append(sections, " "+styled)
	}

	content := strings.Join(sections, "\n")
	lines := strings.Split(content, "\n")
	for len(lines) < height {
		lines = append(lines, "")
	}
	if len(lines) > height {
		lines = lines[:height]
	}

	return strings.Join(lines, "\n")
}

// styleLogLine applies context-sensitive styling to a log line.
func styleLogLine(line string, maxLen int) string {
	display := truncate(line, maxLen)
	lower := strings.ToLower(line)

	switch {
	case strings.Contains(lower, "error") || strings.Contains(lower, "failed") || strings.Contains(lower, "fatal"):
		return logErrorLineStyle.Render(display)
	case strings.Contains(lower, "loss") && strings.Contains(lower, "step"):
		return logMetricLineStyle.Render(display)
	case strings.Contains(lower, "saved") || strings.Contains(lower, "complete") || strings.Contains(lower, "passed"):
		return logHighlightStyle.Render(display)
	case strings.Contains(lower, "loading") || strings.Contains(lower, "checking") || strings.Contains(lower, "rsync"):
		return logTimestampStyle.Render(display)
	default:
		return logLineStyle.Render(display)
	}
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen || maxLen <= 0 {
		return s
	}
	return s[:maxLen-1] + "…"
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	s := ""
	for n > 0 {
		s = string(rune('0'+n%10)) + s
		n /= 10
	}
	return s
}
