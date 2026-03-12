package panels

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/vigo999/ms-cli/ui/model"
)

var (
	hintDividerStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("238"))

	hintTextStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("240")).
			PaddingLeft(1)

	hintKeyStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("250"))

	hintDescStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("240"))

	hintSepStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("238"))
)

type hint struct {
	key  string
	desc string
}

var hints = []hint{
	{"/", "commands"},
	{"↑/↓", "navigate"},
	{"wheel", "scroll"},
	{"pgup/pgdn", "scroll"},
	{"ctrl+c", "quit"},
}

// RenderHintBar renders the bottom keybinding hint bar.
func RenderHintBar(width int) string {
	divider := hintDividerStyle.Render(repeatChar("━", width))

	parts := make([]string, len(hints))
	for i, h := range hints {
		parts[i] = hintKeyStyle.Render(h.key) + " " + hintDescStyle.Render(h.desc)
	}

	sep := hintSepStyle.Render(" • ")
	line := hintTextStyle.Render("")
	for i, p := range parts {
		if i > 0 {
			line += sep
		}
		line += p
	}

	return divider + "\n" + line
}

// RenderTrainHintBar renders the hint bar for the train workspace with focus context.
func RenderTrainHintBar(width int, focused model.TrainPanelID, opts ...bool) string {
	maximized := len(opts) > 0 && opts[0]
	divider := hintDividerStyle.Render(repeatChar("━", width))

	var trainHints []hint
	trainHints = append(trainHints, hint{"Tab", "cycle panels"})
	if maximized {
		trainHints = append(trainHints, hint{"z", "unzoom"})
	} else {
		trainHints = append(trainHints, hint{"c", "collapse"}, hint{"z", "zoom"})
	}

	switch focused {
	case model.TrainPanelRunList:
		trainHints = append(trainHints, hint{"↑/↓", "switch run"})
	case model.TrainPanelActions:
		trainHints = append(trainHints, hint{"←/→", "select"}, hint{"Enter", "activate"})
	case model.TrainPanelMetrics:
		trainHints = append(trainHints, hint{"Esc", "actions"})
	case model.TrainPanelLogs:
		trainHints = append(trainHints, hint{"↑/↓", "scroll"}, hint{"Esc", "actions"})
	case model.TrainPanelAgent:
		trainHints = append(trainHints, hint{"↑/↓", "scroll"}, hint{"Esc", "actions"})
	case model.TrainPanelStatus:
		trainHints = append(trainHints, hint{"Esc", "actions"})
	}
	trainHints = append(trainHints, hint{"ctrl+c", "quit"})

	parts := make([]string, len(trainHints))
	for i, h := range trainHints {
		parts[i] = hintKeyStyle.Render(h.key) + " " + hintDescStyle.Render(h.desc)
	}

	sep := hintSepStyle.Render(" • ")
	line := hintTextStyle.Render("")
	for i, p := range parts {
		if i > 0 {
			line += sep
		}
		line += p
	}

	// Show focused panel indicator
	panelName := "status"
	switch focused {
	case model.TrainPanelRunList:
		panelName = "train job"
	case model.TrainPanelActions:
		panelName = "actions"
	case model.TrainPanelLogs:
		panelName = "logs"
	case model.TrainPanelStatus:
		panelName = "setup env"
	case model.TrainPanelMetrics:
		panelName = "metrics"
	case model.TrainPanelAgent:
		panelName = "agent"
	}
	indicator := hintDescStyle.Render(fmt.Sprintf("  [%s]", panelName))

	return divider + "\n" + line + indicator
}

// RenderTrainMetricsHeader renders the metrics header row for the right panel.
func RenderTrainMetricsHeader(m model.TrainMetricsView, width int, focused bool) string {
	parts := []string{}

	valStyle := lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("39"))
	lblStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("244"))

	if m.TotalSteps > 0 {
		pct := float64(m.Step) / float64(m.TotalSteps) * 100
		parts = append(parts, lblStyle.Render("step ")+valStyle.Render(fmt.Sprintf("%d/%d", m.Step, m.TotalSteps)))
		parts = append(parts, valStyle.Render(fmt.Sprintf("%.0f%%", pct)))
	}
	if m.Loss > 0 {
		parts = append(parts, lblStyle.Render("loss ")+valStyle.Render(fmt.Sprintf("%.4f", m.Loss)))
	}
	if m.LR > 0 {
		parts = append(parts, lblStyle.Render("lr ")+valStyle.Render(fmt.Sprintf("%.1e", m.LR)))
	}
	if m.Throughput > 0 {
		parts = append(parts, lblStyle.Render("tput ")+valStyle.Render(fmt.Sprintf("%.0f tok/s", m.Throughput)))
	}

	line := " " + strings.Join(parts, "  ")

	// Progress bar
	if m.TotalSteps > 0 {
		barWidth := width - 4
		if barWidth > 60 {
			barWidth = 60
		}
		if barWidth > 0 {
			filled := int(float64(m.Step) / float64(m.TotalSteps) * float64(barWidth))
			if filled > barWidth {
				filled = barWidth
			}
			bar := lipgloss.NewStyle().Foreground(lipgloss.Color("39")).Render(strings.Repeat("█", filled))
			empty := lipgloss.NewStyle().Foreground(lipgloss.Color("236")).Render(strings.Repeat("░", barWidth-filled))
			line += "\n " + bar + empty
		}
	}

	return line
}
