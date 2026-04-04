package panels

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/vigo999/mindspore-cli/ui/model"
)

// Style vars are populated by InitStyles() in styles.go.
var (
	trainTitleStyle         lipgloss.Style
	sectionHeaderStyle      lipgloss.Style
	phaseBadgeSetup         lipgloss.Style
	phaseBadgeReady         lipgloss.Style
	phaseBadgeRunning       lipgloss.Style
	phaseBadgeCompleted     lipgloss.Style
	phaseBadgeFailed        lipgloss.Style
	phaseBadgeStopped       lipgloss.Style
	phaseBadgeDrift         lipgloss.Style
	phaseBadgeAnalyzing     lipgloss.Style
	phaseBadgeRerunning     lipgloss.Style
	checkPassedStyle        lipgloss.Style
	checkFailedStyle        lipgloss.Style
	checkRunningStyle       lipgloss.Style
	checkPendingStyle       lipgloss.Style
	checkDetailStyle        lipgloss.Style
	trainDividerStyle       lipgloss.Style
	metricLabelStyle        lipgloss.Style
	actionNormalStyle       lipgloss.Style
	actionFocusedStyle      lipgloss.Style
	actionDangerStyle       lipgloss.Style
	actionDangerFocusedStyle lipgloss.Style
	actionDisabledStyle     lipgloss.Style
)

func RenderTrainSidebar(tv model.TrainWorkspaceState, width, height int) string {
	var sections []string

	title := trainTitleStyle.Render(fmt.Sprintf(" %s / %s", tv.Request.Model, tv.Request.Mode))
	sections = append(sections, title+"  "+workspaceStageBadge(tv.Stage), "")
	sections = append(sections, " "+sectionHeaderStyle.Render("Run Jobs"))
	sections = append(sections, renderRunNavigator(tv, width)...)

	if tv.Compare != nil && tv.Compare.Enabled {
		sections = append(sections, "")
		sections = append(sections, " "+sectionHeaderStyle.Render("Compare"))
		sections = append(sections, RenderCompareSummary(tv)...)
	}

	run := tv.ActiveRun()
	if run != nil && (run.CurrentIssue != nil || run.Issue != nil || len(run.AgentActions) > 0) {
		sections = append(sections, "")
		header := "Issue / Analysis"
		if tv.Focus == model.TrainPanelIssue {
			header += " [focused]"
		}
		sections = append(sections, " "+sectionHeaderStyle.Render(header))
		sections = append(sections, RenderTrainIssue(tv, width)...)
	}

	if setupLines := renderSidebarSetupSummary(tv, width); len(setupLines) > 0 {
		sections = append(sections, "")
		header := "Setup / Env"
		if tv.Focus == model.TrainPanelStatus {
			header += " [focused]"
		}
		sections = append(sections, " "+sectionHeaderStyle.Render(header))
		sections = append(sections, setupLines...)
	}

	content := strings.Join(sections, "\n")
	contentLines := strings.Split(content, "\n")
	actionLines := 0
	if len(tv.GlobalActions.Items) > 0 {
		actionLines = 2
	}
	gapNeeded := height - len(contentLines) - actionLines
	if gapNeeded > 0 {
		for i := 0; i < gapNeeded; i++ {
			sections = append(sections, "")
		}
	}

	if len(tv.GlobalActions.Items) > 0 {
		divider := trainDividerStyle.Render(" " + strings.Repeat("─", width-2))
		sections = append(sections, divider)
		sections = append(sections, renderActionRow(tv))
	}

	return clampPanelWidth(trimPanelHeight(strings.Join(sections, "\n"), height), width)
}

func RenderTrainStatus(tv model.TrainWorkspaceState, width, height int) string {
	run := tv.ActiveRun()
	if run == nil {
		return trimPanelHeight("", height)
	}

	var sections []string

	// Only show run title/meta/status during setup and ready phases.
	if run.Phase == model.TrainPhaseSetup || run.Phase == model.TrainPhaseReady || run.Phase == "" {
		title := run.Label
		if title == "" {
			title = run.ID
		}
		sections = append(sections, " "+trainTitleStyle.Render(title)+"  "+phaseBadge(run.Phase))
		meta := []string{}
		if run.Framework != "" {
			meta = append(meta, run.Framework)
		}
		if run.Device != "" {
			meta = append(meta, run.Device)
		}
		if run.TargetName != "" {
			meta = append(meta, run.TargetName)
		}
		if len(meta) > 0 {
			sections = append(sections, " "+checkDetailStyle.Render(strings.Join(meta, " · ")))
		}
		sections = append(sections, "")

		if run.ErrorMessage != "" {
			sections = append(sections, " "+checkFailedStyle.Render(run.ErrorMessage))
			sections = append(sections, "")
		}
	}

	if len(run.Metrics) > 0 {
		sections = append(sections, " "+sectionHeaderStyle.Render("Metrics"))
		sections = append(sections, renderMetricsSummary(run.Metrics))
	}

	return trimPanelHeight(strings.Join(sections, "\n"), height)
}

func renderRunNavigator(tv model.TrainWorkspaceState, width int) []string {
	lines := make([]string, 0, len(tv.Runs))
	activeID := tv.ActiveRunID
	focused := tv.Focus == model.TrainPanelRunList

	for _, run := range tv.Runs {
		marker := "○"
		style := checkPendingStyle
		switch run.Phase {
		case model.TrainPhaseRunning, model.TrainPhaseEvaluating:
			marker = "●"
			style = checkRunningStyle
		case model.TrainPhaseCompleted, model.TrainPhaseReady:
			marker = "✓"
			style = checkPassedStyle
		case model.TrainPhaseFailed, model.TrainPhaseDriftDetected:
			marker = "✗"
			style = checkFailedStyle
		}

		label := run.Label
		if label == "" {
			label = run.ID
		}
		line := "   " + style.Render(marker+" "+truncateRunText(label, width-8))
		if run.ID == activeID {
			line += " " + checkDetailStyle.Render("[active]")
			if focused {
				line = " " + actionFocusedStyle.Render(strings.TrimSpace(line))
			}
		}
		lines = append(lines, line)
	}

	if len(lines) == 0 {
		lines = append(lines, "   "+checkPendingStyle.Render("No runs"))
	}
	return lines
}

func renderMetricsSummary(metrics []model.MetricItem) string {
	parts := make([]string, 0, len(metrics))
	for _, metric := range metrics {
		if metric.Value == "" {
			continue
		}
		parts = append(parts, fmt.Sprintf("%s %s", metric.Name, metric.Value))
	}
	return "   " + metricLabelStyle.Render(strings.Join(parts, "  "))
}

func renderCheck(c model.ChecklistItem, width int) string {
	icon := "⟳"
	statusStyle := checkPendingStyle
	switch c.Status {
	case model.TrainCheckPass:
		icon = "✓"
		statusStyle = checkPassedStyle
	case model.TrainCheckFail:
		icon = "✗"
		statusStyle = checkFailedStyle
	case model.TrainCheckRunning:
		icon = "⟳"
		statusStyle = checkRunningStyle
	}
	name := displayCheckName(c.Name)
	detail := c.Summary
	if detail == "" {
		detail = "checking..."
		if c.Status == model.TrainCheckPass {
			detail = "ok"
		} else if c.Status == model.TrainCheckFail {
			detail = "failed"
		}
	}
	line := "   " + statusStyle.Render(icon) + " " + statusStyle.Render(fmt.Sprintf("%-14s", name)) + statusStyle.Render(truncateRunText(detail, width-20))
	return line
}

func renderSidebarSetupSummary(tv model.TrainWorkspaceState, width int) []string {
	run := tv.ActiveRun()
	if run == nil {
		return nil
	}

	lines := []string{}
	if target := activeTargetLine(tv, run); target != "" {
		sshLine := "   "
		maxTargetWidth := width - 14
		if strings.TrimSpace(run.StatusMessage) == "Fixing..." {
			sshLine += checkFailedStyle.Render("fixing... ")
			maxTargetWidth = width - 24
		}
		sshLine += metricLabelStyle.Render("ssh: ")
		sshLine += checkDetailStyle.Render(truncateRunText(target, maxTargetWidth))
		lines = append(lines, sshLine)
	}

	for _, item := range sidebarEnvItems(tv, run.ID) {
		lines = append(lines, "   "+metricLabelStyle.Render(item.label+": ")+checkDetailStyle.Render(truncateRunText(item.value, width-12)))
	}

	localChecks := tv.ChecksByGroup(run.ID, model.TrainCheckGroupLocal)
	if len(localChecks) > 0 {
		lines = append(lines, "")
		lines = append(lines, "   "+metricLabelStyle.Render("local checks"))
		for _, c := range localChecks {
			lines = append(lines, renderCheck(c, width))
		}
	}

	targetChecks := tv.ChecksByGroup(run.ID, model.TrainCheckGroupTarget)
	if len(targetChecks) > 0 {
		lines = append(lines, "")
		lines = append(lines, "   "+metricLabelStyle.Render("target checks"))
		for _, c := range targetChecks {
			lines = append(lines, renderCheck(c, width))
		}
		if strings.TrimSpace(run.StatusMessage) != "" && strings.TrimSpace(run.StatusMessage) != "Fixing..." {
			lines = append(lines, "   "+checkDetailStyle.Render(truncateRunText(run.StatusMessage, width-6)))
		}
	} else if strings.TrimSpace(run.StatusMessage) != "" && strings.TrimSpace(run.StatusMessage) != "Fixing..." {
		lines = append(lines, "")
		lines = append(lines, "   "+checkDetailStyle.Render(truncateRunText(run.StatusMessage, width-6)))
	}

	if tv.TrainPlan != nil {
		lines = append(lines, "")
		lines = append(lines, "   "+metricLabelStyle.Render("plan: ")+checkPassedStyle.Render("ready-to-start"))
	}

	return lines
}

type envItem struct {
	label string
	value string
}

func sidebarEnvItems(tv model.TrainWorkspaceState, runID string) []envItem {
	items := []envItem{}
	checks := append([]model.ChecklistItem{}, tv.ChecksByGroup(runID, model.TrainCheckGroupLocal)...)
	checks = append(checks, tv.ChecksByGroup(runID, model.TrainCheckGroupTarget)...)

	for _, check := range checks {
		if check.Status != model.TrainCheckPass {
			continue
		}
		switch check.Name {
		case "target_workdir":
			items = appendIfMissing(items, envItem{label: "workdir", value: check.Summary})
		case "target_aiframework":
			items = appendIfMissing(items, envItem{label: "model-lib", value: check.Summary})
		case "train_script":
			items = appendIfMissing(items, envItem{label: "script", value: check.Summary})
		case "base_model":
			items = appendIfMissing(items, envItem{label: "model", value: check.Summary})
		case "local_repo":
			items = appendIfMissing(items, envItem{label: "repo", value: check.Summary})
		case "local_aiframework":
			items = appendIfMissing(items, envItem{label: "local-lib", value: check.Summary})
		}
	}

	if tv.SetupContext.EnvKind != "" {
		items = appendIfMissing(items, envItem{label: "env", value: tv.SetupContext.EnvKind})
	}
	if tv.SetupContext.Workdir != "" {
		items = appendIfMissing(items, envItem{label: "workdir", value: tv.SetupContext.Workdir})
	}
	if tv.SetupContext.ScriptPath != "" {
		items = appendIfMissing(items, envItem{label: "script", value: tv.SetupContext.ScriptPath})
	}
	if tv.SetupContext.BaseModelRef != "" {
		items = appendIfMissing(items, envItem{label: "model", value: tv.SetupContext.BaseModelRef})
	}
	if tv.TrainPlan != nil && tv.TrainPlan.RepoSource != "" {
		items = appendIfMissing(items, envItem{label: "source", value: tv.TrainPlan.RepoSource})
	}

	return items
}

func activeTargetLine(tv model.TrainWorkspaceState, run *model.TrainRunState) string {
	targetName := run.TargetName
	if targetName == "" {
		targetName = tv.Request.TargetName
	}
	if targetName == "" {
		return ""
	}
	for _, host := range tv.Hosts {
		if host.Name == targetName {
			if host.Address != "" {
				return host.Name + " @ " + host.Address
			}
			return host.Name
		}
	}
	return targetName
}

func appendIfMissing(items []envItem, item envItem) []envItem {
	if strings.TrimSpace(item.value) == "" {
		return items
	}
	for _, existing := range items {
		if existing.label == item.label {
			return items
		}
	}
	return append(items, item)
}

func displayCheckName(name string) string {
	switch name {
	case "local_repo":
		return "repo"
	case "local_os":
		return "os"
	case "local_aiframework":
		return "libs"
	case "train_script":
		return "script"
	case "base_model":
		return "model"
	case "ssh":
		return "ssh connect"
	case "target_os":
		return "os"
	case "target_aiframework":
		return "libs"
	case "target_workdir":
		return "workdir"
	case "target_algo":
		return "script/config"
	case "target_gpu":
		return "gpu"
	case "target_npu":
		return "npu"
	default:
		return name
	}
}

func phaseBadge(phase model.TrainPhase) string {
	switch phase {
	case model.TrainPhaseSetup:
		return phaseBadgeSetup.Render(" SETUP ")
	case model.TrainPhaseReady:
		return phaseBadgeReady.Render(" READY ")
	case model.TrainPhaseRunning:
		return phaseBadgeRunning.Render(" RUNNING ")
	case model.TrainPhaseCompleted:
		return phaseBadgeCompleted.Render(" COMPLETED ")
	case model.TrainPhaseFailed:
		return phaseBadgeFailed.Render(" FAILED ")
	case model.TrainPhaseStopped:
		return phaseBadgeStopped.Render(" STOPPED ")
	case model.TrainPhaseDriftDetected:
		return phaseBadgeDrift.Render(" DRIFT ")
	case model.TrainPhaseAnalyzing:
		return phaseBadgeAnalyzing.Render(" ANALYZING ")
	case model.TrainPhaseFixing:
		return phaseBadgeAnalyzing.Render(" FIXING ")
	case model.TrainPhaseEvaluating:
		return phaseBadgeRerunning.Render(" EVALUATING ")
	default:
		return phaseBadgeStopped.Render(" IDLE ")
	}
}

func workspaceStageBadge(stage model.WorkspaceStage) string {
	switch stage {
	case model.StageSetup:
		return phaseBadgeSetup.Render(" WORKSPACE SETUP ")
	case model.StageReady:
		return phaseBadgeReady.Render(" WORKSPACE READY ")
	case model.StageRunning:
		return phaseBadgeRunning.Render(" WORKSPACE RUNNING ")
	case model.StageAnalyzing:
		return phaseBadgeAnalyzing.Render(" ANALYZING ")
	case model.StageFixing:
		return phaseBadgeAnalyzing.Render(" FIXING ")
	case model.StageDone:
		return phaseBadgeCompleted.Render(" DONE ")
	default:
		return phaseBadgeStopped.Render(" IDLE ")
	}
}

func renderActionRow(tv model.TrainWorkspaceState) string {
	if len(tv.GlobalActions.Items) == 0 {
		return ""
	}

	parts := make([]string, 0, len(tv.GlobalActions.Items))
	for i, action := range tv.GlobalActions.Items {
		style := actionStyleFor(action, tv.Focus == model.TrainPanelActions && i == tv.GlobalActions.SelectedIndex)
		parts = append(parts, style.Render(action.Label))
	}

	return " " + strings.Join(parts, " ")
}

func actionStyleFor(action model.TrainAction, focused bool) lipgloss.Style {
	if !action.Enabled {
		return actionDisabledStyle
	}
	if action.ID == "stop" || action.ID == "apply_fix" {
		if focused {
			return actionDangerFocusedStyle
		}
		return actionDangerStyle
	}
	if focused {
		return actionFocusedStyle
	}
	return actionNormalStyle
}

func wrapText(text string, width int) []string {
	if width <= 0 || len(text) <= width {
		return []string{text}
	}
	words := strings.Fields(text)
	if len(words) == 0 {
		return []string{text}
	}
	lines := []string{}
	current := words[0]
	for _, word := range words[1:] {
		if len(current)+1+len(word) > width {
			lines = append(lines, current)
			current = word
			continue
		}
		current += " " + word
	}
	lines = append(lines, current)
	return lines
}

func trimPanelHeight(content string, height int) string {
	lines := strings.Split(content, "\n")
	if len(lines) > height {
		// Show the bottom so the latest checks/items stay visible.
		lines = lines[len(lines)-height:]
	}
	for len(lines) < height {
		lines = append(lines, "")
	}
	return strings.Join(lines, "\n")
}

func truncateRunText(s string, maxLen int) string {
	if maxLen <= 0 || len(s) <= maxLen {
		return s
	}
	return s[:maxLen-1] + "…"
}

func clampPanelWidth(content string, width int) string {
	style := lipgloss.NewStyle().Width(width).MaxWidth(width)
	lines := strings.Split(content, "\n")
	for i := range lines {
		lines[i] = style.Render(lines[i])
	}
	return strings.Join(lines, "\n")
}
