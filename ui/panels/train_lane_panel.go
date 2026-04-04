package panels

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

// Style vars are populated by InitStyles() in styles.go.
// laneMetricValue is not themed (color set at call sites).
var (
	laneHeaderStyle    lipgloss.Style
	laneSubStyle       lipgloss.Style
	laneBadgeRunning   lipgloss.Style
	laneBadgeCompleted lipgloss.Style
	laneBadgeFailed    lipgloss.Style
	laneBadgePending   lipgloss.Style
	laneMetricLabel    lipgloss.Style
	laneMetricValue    = lipgloss.NewStyle().Bold(true)
)

// RenderLanePanel remains as a reusable per-run renderer for optional compare views.
func RenderLanePanel(run model.TrainRunState, width, height int) string {
	if width < 10 || height < 6 {
		return strings.Repeat("\n", height-1)
	}

	var sections []string
	sections = append(sections, " "+laneHeaderStyle.Render(run.Label)+"  "+laneBadgeForPhase(run.Phase))
	sections = append(sections, " "+laneSubStyle.Render(fmt.Sprintf("%s · %s · %s", run.TargetName, run.Device, run.Framework)))

	if run.CurrentMetrics.TotalSteps > 0 {
		sections = append(sections, " "+metricsRowForRun(run))
	}

	headerHeight := len(sections)
	remaining := height - headerHeight
	chartHeight := remaining * 40 / 100
	if chartHeight < 4 {
		chartHeight = 4
	}
	logsHeight := remaining - chartHeight - 1
	if logsHeight < 2 {
		logsHeight = 2
	}

	pointColor, lineColor := laneColors(run.ID)
	chart := RenderLaneChart(run.LossSeries, "", pointColor, lineColor, width, chartHeight)
	sep := lipgloss.NewStyle().Foreground(lipgloss.Color("236")).Render(strings.Repeat("─", width))
	logs := RenderLaneLogs(run.Logs.Lines, "", width, logsHeight)

	sections = append(sections, chart, sep, logs)
	return trimPanelHeight(strings.Join(sections, "\n"), height)
}

func laneBadgeForPhase(phase model.TrainPhase) string {
	switch phase {
	case model.TrainPhaseRunning, model.TrainPhaseEvaluating:
		return laneBadgeRunning.Render(" RUNNING ")
	case model.TrainPhaseCompleted, model.TrainPhaseReady:
		return laneBadgeCompleted.Render(" READY ")
	case model.TrainPhaseFailed, model.TrainPhaseDriftDetected:
		return laneBadgeFailed.Render(" FAILED ")
	default:
		return laneBadgePending.Render(" PENDING ")
	}
}

func metricsRowForRun(run model.TrainRunState) string {
	m := run.CurrentMetrics
	parts := []string{}

	if m.TotalSteps > 0 {
		valStyle := laneMetricValue.Foreground(laneAccent(run.ID))
		parts = append(parts, laneMetricLabel.Render("step ")+valStyle.Render(fmt.Sprintf("%d/%d", m.Step, m.TotalSteps)))
	}
	if m.Loss > 0 {
		valStyle := laneMetricValue.Foreground(laneAccent(run.ID))
		parts = append(parts, laneMetricLabel.Render("loss ")+valStyle.Render(fmt.Sprintf("%.4f", m.Loss)))
	}
	if m.Throughput > 0 {
		valStyle := laneMetricValue.Foreground(laneAccent(run.ID))
		parts = append(parts, laneMetricLabel.Render("tput ")+valStyle.Render(fmt.Sprintf("%.0f", m.Throughput)))
	}

	return strings.Join(parts, "  ")
}

func laneColors(id string) (pointColor, lineColor string) {
	switch id {
	case "torch_npu":
		return "39", "69"
	case "mindspore_npu":
		return "114", "78"
	default:
		return "252", "244"
	}
}

func laneAccent(id string) lipgloss.Color {
	switch id {
	case "torch_npu":
		return lipgloss.Color("39")
	case "mindspore_npu":
		return lipgloss.Color("114")
	default:
		return lipgloss.Color("252")
	}
}
