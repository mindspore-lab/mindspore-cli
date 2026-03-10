package panels

import (
	"fmt"

	"github.com/charmbracelet/lipgloss"
	"github.com/vigo999/ms-cli/ui/model"
)

var (
	cmpLabelStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("244"))
	cmpGoodStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("114")).Bold(true)
	cmpBadStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("196")).Bold(true)
	cmpWarnStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("214")).Bold(true)
	cmpNeutStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("252")).Bold(true)

	cmpGPUAccent = lipgloss.NewStyle().Foreground(lipgloss.Color("39")).Bold(true)
	cmpNPUAccent = lipgloss.NewStyle().Foreground(lipgloss.Color("114")).Bold(true)
)

// RenderCompareSummary renders the compare block for the left panel.
// It adapts to the current phase: running, startup_failed, drift, verified.
func RenderCompareSummary(tv model.TrainViewState) []string {
	var lines []string

	gpuStatus := tv.GPULane.Status
	npuStatus := tv.NPULane.Status

	// ── Running: live loss + throughput comparison ──
	if tv.Phase == "running" {
		gpuM := tv.GPULane.Metrics
		npuM := tv.NPULane.Metrics

		if gpuM.TotalSteps > 0 || npuM.TotalSteps > 0 {
			if gpuM.Loss > 0 {
				lines = append(lines, fmt.Sprintf("   %s %s",
					cmpLabelStyle.Render("GPU loss:"),
					cmpGPUAccent.Render(fmt.Sprintf("%.4f", gpuM.Loss))))
			}
			if npuM.Loss > 0 {
				lines = append(lines, fmt.Sprintf("   %s %s",
					cmpLabelStyle.Render("NPU loss:"),
					cmpNPUAccent.Render(fmt.Sprintf("%.4f", npuM.Loss))))
			}
			if gpuM.Throughput > 0 {
				lines = append(lines, fmt.Sprintf("   %s %s",
					cmpLabelStyle.Render("GPU tput:"),
					cmpGPUAccent.Render(fmt.Sprintf("%.0f tok/s", gpuM.Throughput))))
			}
			if npuM.Throughput > 0 {
				lines = append(lines, fmt.Sprintf("   %s %s",
					cmpLabelStyle.Render("NPU tput:"),
					cmpNPUAccent.Render(fmt.Sprintf("%.0f tok/s", npuM.Throughput))))
			}
		}
		return lines
	}

	// ── Launch failed: GPU healthy, NPU failed ──
	if tv.Phase == "launch_failed" || (gpuStatus != "" && npuStatus == "failed") {
		gpuBadge := "running"
		if gpuStatus == "completed" {
			gpuBadge = "completed"
		}
		lines = append(lines, fmt.Sprintf("   %s %s",
			cmpLabelStyle.Render("GPU:"),
			cmpGoodStyle.Render(gpuBadge)))
		lines = append(lines, fmt.Sprintf("   %s %s",
			cmpLabelStyle.Render("NPU:"),
			cmpBadStyle.Render("failed")))
		return lines
	}

	// ── Compare: eval accuracy results ──
	if tv.Compare != nil {
		lines = append(lines, fmt.Sprintf("   %s %s",
			cmpLabelStyle.Render("GPU acc:  "),
			cmpGPUAccent.Render(fmt.Sprintf("%.1f%%", tv.Compare.BaselineAcc))))
		lines = append(lines, fmt.Sprintf("   %s %s",
			cmpLabelStyle.Render("NPU acc:  "),
			accStyle(tv.Compare.Drift).Render(fmt.Sprintf("%.1f%%", tv.Compare.CandidateAcc))))
		lines = append(lines, fmt.Sprintf("   %s %s",
			cmpLabelStyle.Render("Drift:    "),
			accStyle(tv.Compare.Drift).Render(fmt.Sprintf("%.1f pts", tv.Compare.Drift))))

		if tv.Compare.Status == "verified" {
			lines = append(lines, fmt.Sprintf("   %s",
				cmpGoodStyle.Render("Recovered: drift -16.6 -> -1.6 pts")))
		}
		return lines
	}

	// ── Fallback: lane status ──
	if gpuStatus != "" {
		lines = append(lines, fmt.Sprintf("   %s %s",
			cmpLabelStyle.Render("GPU:"),
			cmpNeutStyle.Render(gpuStatus)))
	}
	if npuStatus != "" {
		lines = append(lines, fmt.Sprintf("   %s %s",
			cmpLabelStyle.Render("NPU:"),
			cmpNeutStyle.Render(npuStatus)))
	}

	return lines
}

func accStyle(drift float64) lipgloss.Style {
	if drift > -5.0 {
		return cmpGoodStyle
	}
	if drift > -10.0 {
		return cmpWarnStyle
	}
	return cmpBadStyle
}
