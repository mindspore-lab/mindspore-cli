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
)

func RenderCompareSummary(tv model.TrainWorkspaceState) []string {
	if tv.Compare == nil || !tv.Compare.Enabled {
		return nil
	}

	var lines []string
	left := tv.RunByID(tv.Compare.LeftRunID)
	right := tv.RunByID(tv.Compare.RightRunID)
	if left != nil {
		lines = append(lines, fmt.Sprintf("   %s %s",
			cmpLabelStyle.Render(left.Label+":"),
			cmpNeutStyle.Render(string(left.Phase))))
	}
	if right != nil {
		lines = append(lines, fmt.Sprintf("   %s %s",
			cmpLabelStyle.Render(right.Label+":"),
			cmpNeutStyle.Render(string(right.Phase))))
	}

	if tv.Compare.BaselineAcc > 0 || tv.Compare.CandidateAcc > 0 || tv.Compare.Drift != 0 {
		lines = append(lines, fmt.Sprintf("   %s %s",
			cmpLabelStyle.Render("baseline acc:"),
			cmpGoodStyle.Render(fmt.Sprintf("%.1f%%", tv.Compare.BaselineAcc))))
		lines = append(lines, fmt.Sprintf("   %s %s",
			cmpLabelStyle.Render("candidate acc:"),
			accStyle(tv.Compare.Drift).Render(fmt.Sprintf("%.1f%%", tv.Compare.CandidateAcc))))
		lines = append(lines, fmt.Sprintf("   %s %s",
			cmpLabelStyle.Render("drift:"),
			accStyle(tv.Compare.Drift).Render(fmt.Sprintf("%.1f pts", tv.Compare.Drift))))
	}

	if tv.Compare.Summary != "" {
		lines = append(lines, "   "+cmpLabelStyle.Render(tv.Compare.Summary))
	} else if tv.Compare.Status != "" {
		lines = append(lines, "   "+cmpLabelStyle.Render("status: ")+cmpNeutStyle.Render(tv.Compare.Status))
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
