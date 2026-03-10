package panels

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/vigo999/ms-cli/ui/model"
)

var (
	laneHeaderStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("252"))

	laneSubStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("244"))

	laneBadgeRunning = lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.Color("16")).
				Background(lipgloss.Color("39")).
				Padding(0, 1)

	laneBadgeCompleted = lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.Color("16")).
				Background(lipgloss.Color("114")).
				Padding(0, 1)

	laneBadgeFailed = lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.Color("255")).
				Background(lipgloss.Color("196")).
				Padding(0, 1)

	laneBadgeRerunning = lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.Color("16")).
				Background(lipgloss.Color("69")).
				Padding(0, 1)

	laneBadgePending = lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.Color("16")).
				Background(lipgloss.Color("240")).
				Padding(0, 1)

	laneMetricLabel = lipgloss.NewStyle().
				Foreground(lipgloss.Color("244"))

	laneMetricValue = lipgloss.NewStyle().
				Bold(true)
)

// RenderLanePanel renders a complete lane panel: header + chart + logs.
func RenderLanePanel(lane model.TrainLaneView, width, height int) string {
	if width < 10 || height < 6 {
		return strings.Repeat("\n", height-1)
	}

	var sections []string

	// ── Header: title + badge ──
	badge := laneBadgeForStatus(lane.Status)
	sections = append(sections, " "+laneHeaderStyle.Render(lane.Title)+"  "+badge)
	sections = append(sections, " "+laneSubStyle.Render(fmt.Sprintf("%s · %s · %s", lane.Host, lane.Device, lane.Framework)))

	// ── Metrics row (when running or completed) ──
	if lane.Metrics.TotalSteps > 0 {
		metricsLine := metricsRowForLane(lane)
		sections = append(sections, " "+metricsLine)
	}

	headerHeight := len(sections)

	// ── Chart + Logs ──
	remaining := height - headerHeight
	chartHeight := remaining * 40 / 100
	if chartHeight < 4 {
		chartHeight = 4
	}
	logsHeight := remaining - chartHeight - 1 // 1 for separator
	if logsHeight < 2 {
		logsHeight = 2
	}

	pointColor, lineColor := laneColors(lane.ID)
	chart := RenderLaneChart(lane.LossSeries, "", pointColor, lineColor, width, chartHeight)

	sepStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("236"))
	sep := sepStyle.Render(strings.Repeat("─", width))

	logs := RenderLaneLogs(lane.Logs, "", width, logsHeight)

	sections = append(sections, chart)
	sections = append(sections, sep)
	sections = append(sections, logs)

	final := strings.Join(sections, "\n")
	lines := strings.Split(final, "\n")
	if len(lines) > height {
		lines = lines[:height]
	}
	for len(lines) < height {
		lines = append(lines, "")
	}

	return strings.Join(lines, "\n")
}

func laneBadgeForStatus(status string) string {
	switch status {
	case "running":
		return laneBadgeRunning.Render(" RUNNING ")
	case "completed":
		return laneBadgeCompleted.Render(" COMPLETED ")
	case "failed":
		return laneBadgeFailed.Render(" FAILED ")
	case "rerunning":
		return laneBadgeRerunning.Render(" RERUNNING ")
	default:
		return laneBadgePending.Render(" PENDING ")
	}
}

func metricsRowForLane(lane model.TrainLaneView) string {
	m := lane.Metrics
	parts := []string{}

	if m.TotalSteps > 0 {
		valStyle := laneMetricValue.Foreground(laneAccent(lane.ID))
		parts = append(parts,
			laneMetricLabel.Render("step ")+valStyle.Render(fmt.Sprintf("%d/%d", m.Step, m.TotalSteps)))
	}
	if m.Loss > 0 {
		valStyle := laneMetricValue.Foreground(laneAccent(lane.ID))
		parts = append(parts,
			laneMetricLabel.Render("loss ")+valStyle.Render(fmt.Sprintf("%.4f", m.Loss)))
	}
	if m.Throughput > 0 {
		valStyle := laneMetricValue.Foreground(laneAccent(lane.ID))
		parts = append(parts,
			laneMetricLabel.Render("tput ")+valStyle.Render(fmt.Sprintf("%.0f", m.Throughput)))
	}

	return strings.Join(parts, "  ")
}

// laneColors returns point and line colors for chart rendering.
func laneColors(id model.TrainLaneID) (pointColor, lineColor string) {
	switch id {
	case model.LaneGPU:
		return "39", "69" // cyan, light blue
	case model.LaneNPU:
		return "114", "78" // green, lighter green
	default:
		return "252", "244"
	}
}

// laneAccent returns the primary accent color for a lane.
func laneAccent(id model.TrainLaneID) lipgloss.Color {
	switch id {
	case model.LaneGPU:
		return lipgloss.Color("39") // cyan
	case model.LaneNPU:
		return lipgloss.Color("114") // green
	default:
		return lipgloss.Color("252")
	}
}
