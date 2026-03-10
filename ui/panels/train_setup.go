package panels

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/vigo999/ms-cli/ui/model"
)

var (
	// Title and headers
	trainTitleStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("214"))

	sectionHeaderStyle = lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.Color("252")).
				Background(lipgloss.Color("236")).
				Padding(0, 1)

	// Phase badges
	phaseBadgeSetup = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("16")).
			Background(lipgloss.Color("214")).
			Padding(0, 1)

	phaseBadgeReady = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("16")).
			Background(lipgloss.Color("114")).
			Padding(0, 1)

	phaseBadgeRunning = lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.Color("16")).
				Background(lipgloss.Color("39")).
				Padding(0, 1)

	phaseBadgeCompleted = lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.Color("16")).
				Background(lipgloss.Color("114")).
				Padding(0, 1)

	phaseBadgeFailed = lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.Color("255")).
				Background(lipgloss.Color("196")).
				Padding(0, 1)

	phaseBadgeStopped = lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.Color("16")).
				Background(lipgloss.Color("240")).
				Padding(0, 1)

	phaseBadgeDrift = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("255")).
			Background(lipgloss.Color("196")).
			Padding(0, 1)

	phaseBadgeAnalyzing = lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.Color("16")).
				Background(lipgloss.Color("214")).
				Padding(0, 1)

	phaseBadgeVerified = lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.Color("16")).
				Background(lipgloss.Color("114")).
				Padding(0, 1)

	phaseBadgeRerunning = lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.Color("16")).
				Background(lipgloss.Color("69")).
				Padding(0, 1)

	// Check items
	checkPassedStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("114"))

	checkFailedStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("196"))

	checkRunningStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("214"))

	checkPendingStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("240"))

	checkDetailStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("244"))

	// Hosts
	hostConnectedStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("114"))

	hostConnectingStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("214"))

	hostAddrStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("244"))

	// Hints and borders
	trainHintStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("240")).
			Italic(true)

	trainDividerStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("236"))

	// Metrics
	metricLabelStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("244"))

	// Action buttons
	actionNormalStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("252")).
				Background(lipgloss.Color("238")).
				Padding(0, 2)

	actionFocusedStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("16")).
				Background(lipgloss.Color("39")).
				Bold(true).
				Padding(0, 2)

	actionDangerStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("255")).
				Background(lipgloss.Color("196")).
				Bold(true).
				Padding(0, 2)

	actionDangerFocusedStyle = lipgloss.NewStyle().
					Foreground(lipgloss.Color("16")).
					Background(lipgloss.Color("203")).
					Bold(true).
					Padding(0, 2)

	actionDisabledStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("238")).
				Background(lipgloss.Color("234")).
				Strikethrough(true).
				Padding(0, 2)
)

// RenderTrainSetup renders the left panel: request, phase, targets, compare, issue, actions.
func RenderTrainSetup(tv model.TrainViewState, width, height int) string {
	var sections []string

	// ── Header: model/method + phase badge ──
	title := trainTitleStyle.Render(fmt.Sprintf(" %s / %s", tv.Model, tv.Method))
	badge := phaseBadge(tv.Phase)
	sections = append(sections, title+"  "+badge)
	sections = append(sections, "")

	// ── Preflight Checks (during setup) ──
	if tv.Phase == "setup" || tv.Phase == "ready" {
		sections = append(sections, " "+sectionHeaderStyle.Render("Preflight Checks"))
		for _, c := range tv.Checks {
			sections = append(sections, renderCheck(c))
		}
	}

	// ── Hosts (during setup) ──
	if (tv.Phase == "setup" || tv.Phase == "ready") && len(tv.Hosts) > 0 {
		sections = append(sections, "")
		sections = append(sections, " "+sectionHeaderStyle.Render("Compute Hosts"))
		for _, h := range tv.Hosts {
			sections = append(sections, renderHost(h))
		}
	}

	// ── Targets (after setup) ──
	showTargets := tv.Phase != "setup" && tv.Phase != "ready"
	if showTargets {
		sections = append(sections, " "+sectionHeaderStyle.Render("Targets"))
		sections = append(sections, renderTarget(tv.GPULane))
		sections = append(sections, renderTarget(tv.NPULane))
	}

	// ── Compare summary ──
	cmpLines := RenderCompareSummary(tv)
	if len(cmpLines) > 0 {
		sections = append(sections, "")
		sections = append(sections, " "+sectionHeaderStyle.Render("Compare"))
		sections = append(sections, cmpLines...)
	}

	// ── Issue summary ──
	if tv.Issue != nil {
		sections = append(sections, "")
		issueHeader := "Issue"
		if tv.Issue.Type == "runtime" {
			issueHeader = "Issue (Runtime)"
		} else if tv.Issue.Type == "accuracy" {
			issueHeader = "Issue (Accuracy)"
		}
		sections = append(sections, " "+sectionHeaderStyle.Render(issueHeader))
		sections = append(sections, "   "+metricLabelStyle.Render(tv.Issue.Title))
		if tv.Issue.Detail != "" {
			for _, line := range wrapText(tv.Issue.Detail, width-6) {
				sections = append(sections, "   "+checkDetailStyle.Render(line))
			}
		}
		if tv.Issue.FixSummary != "" {
			sections = append(sections, "   "+checkPassedStyle.Render("Fix: "+tv.Issue.FixSummary))
		}
		if tv.Issue.Confidence != "" {
			confStyle := checkPassedStyle
			if tv.Issue.Confidence != "high" {
				confStyle = checkRunningStyle
			}
			sections = append(sections, "   "+metricLabelStyle.Render("Confidence: ")+confStyle.Render(tv.Issue.Confidence))
		}
		if tv.Issue.DiffText != "" {
			sections = append(sections, "   "+metricLabelStyle.Render("Diff:"))
			for _, line := range renderDiffPreview(tv.Issue.DiffText, width-6, 8) {
				sections = append(sections, "   "+line)
			}
		}
	}

	// ── Action row (pinned to bottom) ──
	content := strings.Join(sections, "\n")
	contentLines := strings.Split(content, "\n")
	actionLines := 3
	gapNeeded := height - len(contentLines) - actionLines
	if gapNeeded > 0 {
		for i := 0; i < gapNeeded; i++ {
			sections = append(sections, "")
		}
	}

	divider := trainDividerStyle.Render(" " + strings.Repeat("─", width-2))
	sections = append(sections, divider)
	sections = append(sections, renderActionRow(tv))

	final := strings.Join(sections, "\n")
	lines := strings.Split(final, "\n")
	if len(lines) > height {
		lines = lines[:height]
	}

	return strings.Join(lines, "\n")
}

func renderTarget(lane model.TrainLaneView) string {
	statusStyle := checkPendingStyle
	icon := "○"
	switch lane.Status {
	case "running", "rerunning":
		statusStyle = checkRunningStyle
		icon = "●"
	case "completed":
		statusStyle = checkPassedStyle
		icon = "✓"
	case "failed":
		statusStyle = checkFailedStyle
		icon = "✗"
	}

	label := fmt.Sprintf("%s %s", lane.Host, lane.Device)
	tag := ""
	if lane.Role == "baseline" {
		tag = " " + checkDetailStyle.Render("[baseline]")
	} else if lane.Role == "candidate" {
		tag = " " + checkDetailStyle.Render("[candidate]")
	}
	return "   " + statusStyle.Render(icon+" "+label) + tag
}

func phaseBadge(phase string) string {
	switch phase {
	case "setup":
		return phaseBadgeSetup.Render(" SETUP ")
	case "ready":
		return phaseBadgeReady.Render(" READY ")
	case "launch_failed":
		return phaseBadgeFailed.Render(" LAUNCH FAILED ")
	case "running":
		return phaseBadgeRunning.Render(" RUNNING ")
	case "evaluating":
		return phaseBadgeRunning.Render(" EVALUATING ")
	case "drift_detected":
		return phaseBadgeDrift.Render(" DRIFT DETECTED ")
	case "analyzing":
		return phaseBadgeAnalyzing.Render(" ANALYZING ")
	case "fix_ready":
		return phaseBadgeAnalyzing.Render(" FIX READY ")
	case "rerunning":
		return phaseBadgeRerunning.Render(" RERUNNING ")
	case "verified":
		return phaseBadgeVerified.Render(" FIX VERIFIED ")
	case "completed":
		return phaseBadgeCompleted.Render(" COMPLETED ")
	case "failed":
		return phaseBadgeFailed.Render(" FAILED ")
	case "stopped":
		return phaseBadgeStopped.Render(" STOPPED ")
	default:
		return phaseBadgeStopped.Render(" " + strings.ToUpper(phase) + " ")
	}
}

func renderCheck(c model.TrainCheckView) string {
	var icon, label, detail string

	switch c.Status {
	case "passed":
		icon = checkPassedStyle.Render(" ✓ ")
		label = checkPassedStyle.Render(c.Label)
		if c.Detail != "" {
			detail = " " + checkDetailStyle.Render(c.Detail)
		}
	case "failed":
		icon = checkFailedStyle.Render(" ✗ ")
		label = checkFailedStyle.Render(c.Label)
		if c.Detail != "" {
			detail = " " + checkFailedStyle.Render(c.Detail)
		}
	case "checking":
		icon = checkRunningStyle.Render(" ⟳ ")
		label = checkRunningStyle.Render(c.Label + "...")
	default:
		icon = checkPendingStyle.Render(" ○ ")
		label = checkPendingStyle.Render(c.Label)
	}

	return "  " + icon + " " + label + detail
}

func wrapText(s string, width int) []string {
	if width <= 0 || len(s) <= width {
		return []string{s}
	}

	var lines []string
	remaining := strings.TrimSpace(s)
	for len(remaining) > width {
		split := strings.LastIndex(remaining[:width], " ")
		if split <= 0 {
			split = width
		}
		lines = append(lines, strings.TrimSpace(remaining[:split]))
		remaining = strings.TrimSpace(remaining[split:])
	}
	if remaining != "" {
		lines = append(lines, remaining)
	}
	return lines
}

func renderDiffPreview(diff string, width, maxLines int) []string {
	lines := strings.Split(diff, "\n")
	if maxLines > 0 && len(lines) > maxLines {
		lines = append(lines[:maxLines], "...")
	}

	rendered := make([]string, 0, len(lines))
	for _, line := range lines {
		display := line
		if width > 0 && len(display) > width {
			display = display[:width-1] + "..."
		}
		style := checkDetailStyle
		switch {
		case strings.HasPrefix(line, "+"):
			style = checkPassedStyle
		case strings.HasPrefix(line, "-"):
			style = checkFailedStyle
		case strings.HasPrefix(line, "@@"):
			style = checkRunningStyle
		}
		rendered = append(rendered, style.Render(display))
	}
	return rendered
}

func renderHost(h model.TrainHostView) string {
	switch h.Status {
	case "connected":
		addr := ""
		if h.Address != "" {
			addr = " " + hostAddrStyle.Render(h.Address)
		}
		return "   " + hostConnectedStyle.Render("● "+h.Name) + addr
	case "connecting":
		addr := ""
		if h.Address != "" {
			addr = " " + hostAddrStyle.Render(h.Address)
		}
		return "   " + hostConnectingStyle.Render("◌ "+h.Name+"...") + addr
	case "failed":
		return "   " + checkFailedStyle.Render("✗ "+h.Name+" connection failed")
	default:
		return "   " + checkPendingStyle.Render("○ "+h.Name)
	}
}

func renderActionRow(tv model.TrainViewState) string {
	if len(tv.Actions) == 0 {
		return trainHintStyle.Render("  Setting up...")
	}

	var buttons []string
	for i, action := range tv.Actions {
		style := actionStyleFor(action, tv.ActionRowFocused && i == tv.FocusedAction)
		buttons = append(buttons, style.Render(action.Label))
	}

	row := "  " + strings.Join(buttons, "  ")
	hint := trainHintStyle.Render("  Esc focuses actions  i focuses chat  Arrow selects  Enter activates")
	return row + "\n" + hint
}

func actionStyleFor(action model.TrainActionItem, focused bool) lipgloss.Style {
	if action.Disabled {
		return actionDisabledStyle
	}

	isDanger := action.Action == model.ActionStop
	if focused {
		if isDanger {
			return actionDangerFocusedStyle
		}
		return actionFocusedStyle
	}

	if isDanger {
		return actionDangerStyle
	}
	return actionNormalStyle
}
