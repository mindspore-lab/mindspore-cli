package panels

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/vigo999/ms-cli/ui/model"
)

var (
	panelTitleStyle = lipgloss.NewStyle().Bold(true).Padding(0, 1)
	panelBodyStyle  = lipgloss.NewStyle().Padding(0, 1)
	panelStubStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("244")).Padding(0, 1)

	runBarStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("252")).Padding(0, 1)
)

// ── Workspace status bar ────────────────────────────────────

// RenderWorkspaceStatusBar renders a horizontal stage progress indicator.
// All stages are shown; the current one is highlighted, past stages are
// dimmed green, future stages are gray.
func RenderWorkspaceStatusBar(stage model.WorkspaceStage, width int) string {
	stages := []model.WorkspaceStage{
		model.StageSetup,
		model.StageReady,
		model.StageRunning,
		model.StageAnalyzing,
		model.StageFixing,
		model.StageDone,
	}

	currentStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("114")).Bold(true)
	dimStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("240"))
	sepStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("238"))
	labelStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("252")).Bold(true)

	parts := make([]string, 0, len(stages)*2)
	for i, s := range stages {
		if s == stage {
			parts = append(parts, currentStyle.Render(string(s)))
		} else {
			parts = append(parts, dimStyle.Render(string(s)))
		}
		if i < len(stages)-1 {
			parts = append(parts, sepStyle.Render(" · "))
		}
	}

	line := "  " + labelStyle.Render("train workspace") + "  " + strings.Join(parts, "")
	return line
}

// ── Stacked layout API ──────────────────────────────────────

// TrainRunBarHeight returns the rendered height of the train job panel.
func TrainRunBarHeight(tv model.TrainWorkspaceState) int {
	n := len(tv.Runs)
	if n == 0 {
		n = 1
	}
	return n + 3 // run lines + title + top/bottom border
}

// StackedPanelHeights calculates heights for Status, Metrics, Logs in the
// stacked full-width layout.  available is the vertical budget shared
// between the three panels and the viewport.  Returns [status, metrics, logs].
func StackedPanelHeights(tv model.TrainWorkspaceState, available int) [3]int {
	ids := []model.TrainPanelID{
		model.TrainPanelStatus,
		model.TrainPanelMetrics,
		model.TrainPanelLogs,
	}

	// Count collapsed overhead.
	collapsedLines := 0
	expandedCount := 0
	for _, id := range ids {
		if tv.Panels[id] != nil && tv.Panels[id].Collapsed {
			collapsedLines += 3
		} else {
			expandedCount++
		}
	}

	// More panel space when fewer panels compete; viewport gets the rest.
	pct := 55
	if expandedCount == 1 {
		pct = 70
	} else if expandedCount == 2 {
		pct = 60
	}
	expandedSpace := available - collapsedLines
	panelBudget := collapsedLines
	if expandedCount > 0 && expandedSpace > 0 {
		panelBudget += expandedSpace * pct / 100
	}
	if panelBudget < 9 {
		panelBudget = 9
	}

	heights := allocatePanelHeights(tv, ids, panelBudget)
	return [3]int{heights[0], heights[1], heights[2]}
}

// RenderTrainRunBar renders the train job panel as a boxed panel.
func RenderTrainRunBar(tv model.TrainWorkspaceState, width, height int, focused bool) string {
	body := renderTrainJobBody(tv, maxInt(1, width-4))
	accent := lipgloss.Color("240")
	return renderPanelBox("train job", body, width, height, focused, false, accent)
}

func renderTrainJobBody(tv model.TrainWorkspaceState, width int) string {
	runs := tv.Runs
	if len(runs) == 0 {
		return checkPendingStyle.Render("no runs")
	}

	lines := make([]string, 0, len(runs))
	for i, run := range runs {
		marker := "○"
		markerStyle := checkPendingStyle
		switch run.Phase {
		case model.TrainPhaseRunning, model.TrainPhaseEvaluating:
			marker = "●"
			markerStyle = checkRunningStyle
		case model.TrainPhaseCompleted, model.TrainPhaseReady:
			marker = "✓"
			markerStyle = checkPassedStyle
		case model.TrainPhaseFailed, model.TrainPhaseDriftDetected:
			marker = "✗"
			markerStyle = checkFailedStyle
		}

		details := []string{fmt.Sprintf("run_id: %d", i+1)}
		mdl := tv.Request.Model
		if mdl != "" {
			details = append(details, "model: "+mdl)
		}
		method := tv.Request.Mode
		if method != "" {
			details = append(details, "method: "+method)
		}
		if run.TargetName != "" {
			details = append(details, "target: "+run.TargetName)
		}
		info := strings.Join(details, " · ")
		line := "   " + markerStyle.Render(marker) + " " + checkDetailStyle.Render(info)
		lines = append(lines, line)
	}

	return strings.Join(lines, "\n")
}

// RenderTrainActionStrip renders an inline action button row.
// Layout: separator + buttons + separator (visually splits from input below).
func RenderTrainActionStrip(tv model.TrainWorkspaceState, width int, focused bool) string {
	sep := trainDividerStyle.Render(strings.Repeat("─", width))
	if len(tv.GlobalActions.Items) == 0 {
		return sep + "\n" + checkPendingStyle.Render("  waiting for actions...") + "\n" + sep
	}

	parts := make([]string, 0, len(tv.GlobalActions.Items))
	for i, action := range tv.GlobalActions.Items {
		selected := focused && i == tv.GlobalActions.SelectedIndex
		style := actionStyleFor(action, selected)
		parts = append(parts, style.Render(action.Label))
	}

	return sep + "\n  " + strings.Join(parts, " ") + "\n" + sep
}

// RenderSelectionPopup renders a selection popup box string (without placement).
func RenderSelectionPopup(popup *model.SelectionPopup) string {
	titleStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("12")).Bold(true)
	normalStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("7"))
	selectedStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("14")).Bold(true)
	descStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("8"))
	hintStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("8")).Italic(true)

	var lines []string
	lines = append(lines, titleStyle.Render(popup.Title))
	lines = append(lines, "")
	for i, opt := range popup.Options {
		marker := "  "
		style := normalStyle
		if i == popup.Selected {
			marker = "> "
			style = selectedStyle
		}
		line := marker + style.Render(opt.Label)
		if opt.Desc != "" {
			line += " " + descStyle.Render(opt.Desc)
		}
		lines = append(lines, line)
	}
	lines = append(lines, "")
	lines = append(lines, hintStyle.Render("↑/↓ select · enter confirm · esc cancel"))

	content := strings.Join(lines, "\n")
	return lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("12")).
		Padding(0, 2).
		Render(content)
}

// RenderTrainWorkspacePanel renders a single boxed panel at the given size.
// Used for maximized view and stacked row rendering.
func RenderTrainWorkspacePanel(panel model.TrainPanelID, tv model.TrainWorkspaceState, width, height int) string {
	return renderPanel(panel, tv, width, height)
}

// ── Height allocation ───────────────────────────────────────

func allocatePanelHeights(tv model.TrainWorkspaceState, ids []model.TrainPanelID, total int) []int {
	weights := map[model.TrainPanelID]int{
		model.TrainPanelRunList: 24,
		model.TrainPanelIssue:   33,
		model.TrainPanelActions: 18,
		model.TrainPanelStatus:  38,
		model.TrainPanelMetrics: 26,
		model.TrainPanelLogs:    36,
	}
	heights := make([]int, len(ids))
	remaining := total
	remainingWeight := 0
	for _, id := range ids {
		if tv.Panels[id] != nil && tv.Panels[id].Collapsed {
			continue
		}
		remainingWeight += weights[id]
	}
	for i, id := range ids {
		if tv.Panels[id] != nil && tv.Panels[id].Collapsed {
			heights[i] = 3
			remaining -= 3
		}
	}
	if remaining < len(ids) {
		remaining = len(ids)
	}
	if remainingWeight == 0 {
		for i := range heights {
			if heights[i] == 0 {
				heights[i] = maxInt(3, remaining/(len(ids)-i))
			}
		}
		return rebalanceHeights(heights, total)
	}
	for i, id := range ids {
		if heights[i] != 0 {
			continue
		}
		h := remaining * weights[id] / remainingWeight
		if h < 4 {
			h = 4
		}
		heights[i] = h
	}
	return rebalanceHeights(heights, total)
}

func rebalanceHeights(heights []int, total int) []int {
	sum := 0
	for _, h := range heights {
		sum += h
	}
	for sum > total {
		reduced := false
		for i := len(heights) - 1; i >= 0 && sum > total; i-- {
			if heights[i] > 3 {
				heights[i]--
				sum--
				reduced = true
			}
		}
		if !reduced {
			for i := len(heights) - 1; i >= 0 && sum > total; i-- {
				if heights[i] > 1 {
					heights[i]--
					sum--
				}
			}
			break
		}
	}
	for sum < total && len(heights) > 0 {
		heights[len(heights)-1]++
		sum++
	}
	return heights
}

// ── Panel rendering ─────────────────────────────────────────

func renderPanel(id model.TrainPanelID, tv model.TrainWorkspaceState, width, height int) string {
	title := panelTitle(id, tv)
	bodyHeight := maxInt(1, height-3)
	bodyWidth := maxInt(1, width-4)

	body := panelBody(id, tv, bodyWidth, bodyHeight)
	state := tv.Panels[id]
	focused := state != nil && state.Focused
	collapsed := state != nil && state.Collapsed
	accent := panelAccent(id, tv)
	return renderPanelBoxWithScroll(title, body, width, height, focused, collapsed, accent, 0, 0)
}

func panelTitle(id model.TrainPanelID, tv model.TrainWorkspaceState) string {
	switch id {
	case model.TrainPanelRunList:
		return "run jobs"
	case model.TrainPanelIssue:
		return "issue"
	case model.TrainPanelActions:
		return "actions"
	case model.TrainPanelStatus:
		return "setup env"
	case model.TrainPanelMetrics:
		run := tv.ActiveRun()
		if run != nil {
			m := run.CurrentMetrics
			info := []string{}
			if m.TotalSteps > 0 {
				pct := float64(m.Step) / float64(m.TotalSteps) * 100
				info = append(info, fmt.Sprintf("step %d/%d %.0f%%", m.Step, m.TotalSteps, pct))
			}
			if m.Loss > 0 {
				info = append(info, fmt.Sprintf("loss %.4f", m.Loss))
			}
			if m.LR > 0 {
				info = append(info, fmt.Sprintf("lr %.1e", m.LR))
			}
			if m.Throughput > 0 {
				info = append(info, fmt.Sprintf("tput %.0f tok/s", m.Throughput))
			}
			if len(info) > 0 {
				return "metrics  " + strings.Join(info, " · ")
			}
		}
		return "metrics"
	case model.TrainPanelLogs:
		return "logs"
	default:
		return string(id)
	}
}

func panelAccent(id model.TrainPanelID, tv model.TrainWorkspaceState) lipgloss.Color {
	run := tv.ActiveRun()
	switch id {
	case model.TrainPanelIssue:
		if run != nil && (run.CurrentIssue != nil || run.Issue != nil) {
			return lipgloss.Color("196")
		}
		return lipgloss.Color("240")
	case model.TrainPanelActions:
		return lipgloss.Color("214")
	case model.TrainPanelStatus:
		return lipgloss.Color("114")
	case model.TrainPanelMetrics, model.TrainPanelLogs:
		return lipgloss.Color("240")
	default:
		return lipgloss.Color("240")
	}
}

func renderPanelBoxWithScroll(title, body string, width, height int, focused, collapsed bool, accent lipgloss.Color, totalLines, scrollOffset int) string {
	borderColor := lipgloss.Color("238")
	titleColor := lipgloss.Color("252")
	if focused {
		focusColor := accent
		if accent == lipgloss.Color("238") || accent == lipgloss.Color("240") {
			focusColor = lipgloss.Color("114")
		}
		borderColor = focusColor
		titleColor = focusColor
	}
	if collapsed {
		box := lipgloss.NewStyle().
			Width(width).
			Border(lipgloss.RoundedBorder()).
			BorderForeground(borderColor)
		content := lipgloss.NewStyle().
			Width(maxInt(1, width-4)).
			Height(1).
			Render(panelTitleStyle.Foreground(titleColor).Render(title) + " " + panelStubStyle.Render("[collapsed]"))
		return box.Render(content)
	}
	innerWidth := maxInt(1, width-4)
	innerHeight := maxInt(1, height-2)
	bodyH := maxInt(1, innerHeight-1)

	// When scrollbar is needed, reserve 1 column for it.
	hasScroll := totalLines > bodyH
	contentWidth := innerWidth
	if hasScroll {
		contentWidth = maxInt(1, innerWidth-1)
	}

	header := panelTitleStyle.Foreground(titleColor).Render(title)
	bodyBlock := panelBodyStyle.Width(contentWidth).Height(bodyH).Render(clampPanelWidth(trimPanelHeight(body, bodyH), contentWidth))

	if hasScroll {
		bodyBlock = attachScrollbar(bodyBlock, bodyH, totalLines, scrollOffset)
	}

	content := lipgloss.NewStyle().
		Width(innerWidth).
		Height(innerHeight).
		Render(lipgloss.JoinVertical(lipgloss.Left, header, bodyBlock))
	box := lipgloss.NewStyle().
		Width(width).
		Border(lipgloss.RoundedBorder()).
		BorderForeground(borderColor)
	return box.Render(content)
}

func renderPanelBox(title, body string, width, height int, focused, collapsed bool, accent lipgloss.Color) string {
	borderColor := lipgloss.Color("238")
	titleColor := lipgloss.Color("252")
	if focused {
		// Use accent if it's bright enough, otherwise fall back to green.
		focusColor := accent
		if accent == lipgloss.Color("238") || accent == lipgloss.Color("240") {
			focusColor = lipgloss.Color("114")
		}
		borderColor = focusColor
		titleColor = focusColor
	}
	if collapsed {
		box := lipgloss.NewStyle().
			Width(width).
			Border(lipgloss.RoundedBorder()).
			BorderForeground(borderColor)
		content := lipgloss.NewStyle().
			Width(maxInt(1, width-4)).
			Height(1).
			Render(panelTitleStyle.Foreground(titleColor).Render(title) + " " + panelStubStyle.Render("[collapsed]"))
		return box.Render(content)
	}
	innerWidth := maxInt(1, width-4)
	innerHeight := maxInt(1, height-2)
	header := panelTitleStyle.Foreground(titleColor).Render(title)
	bodyBlock := panelBodyStyle.Width(innerWidth).Height(maxInt(1, innerHeight-1)).Render(clampPanelWidth(trimPanelHeight(body, maxInt(1, innerHeight-1)), innerWidth))
	content := lipgloss.NewStyle().
		Width(innerWidth).
		Height(innerHeight).
		Render(lipgloss.JoinVertical(lipgloss.Left, header, bodyBlock))
	box := lipgloss.NewStyle().
		Width(width).
		Border(lipgloss.RoundedBorder()).
		BorderForeground(borderColor)
	return box.Render(content)
}

func panelBody(id model.TrainPanelID, tv model.TrainWorkspaceState, width, height int) string {
	switch id {
	case model.TrainPanelRunList:
		return renderRunListPanel(tv, width, height)
	case model.TrainPanelIssue:
		return strings.Join(RenderTrainIssue(tv, width), "\n")
	case model.TrainPanelActions:
		return renderActionsPanel(tv, width, height)
	case model.TrainPanelStatus:
		return renderStatusPanel(tv, width, height)
	case model.TrainPanelMetrics:
		return renderMetricsPanel(tv, width, height)
	case model.TrainPanelLogs:
		return renderLogsPanel(tv, width, height)
	case model.TrainPanelAgent:
		// Agent panel body is rendered separately via RenderAgentBox with viewport content.
		// When maximized, we show a placeholder indicating content is displayed via viewport.
		return ""
	default:
		return ""
	}
}

// ── Panel body renderers ────────────────────────────────────

func renderRunListPanel(tv model.TrainWorkspaceState, width, _ int) string {
	modelName := tv.Request.Model
	if modelName == "" {
		modelName = "train"
	}
	mode := tv.Request.Mode
	if mode == "" {
		mode = "workspace"
	}
	lines := []string{
		trainTitleStyle.Render(fmt.Sprintf("%s / %s", modelName, mode)),
		checkDetailStyle.Render(string(tv.Stage)),
		"",
	}
	lines = append(lines, renderRunNavigator(tv, width)...)
	return strings.Join(lines, "\n")
}

func renderActionsPanel(tv model.TrainWorkspaceState, width, _ int) string {
	if len(tv.GlobalActions.Items) == 0 {
		return checkPendingStyle.Render("No action available while checks are running.")
	}
	lines := make([]string, 0, len(tv.GlobalActions.Items))
	for i, action := range tv.GlobalActions.Items {
		prefix := "  "
		if tv.Focus == model.TrainPanelActions && i == tv.GlobalActions.SelectedIndex {
			prefix = ">"
		}
		lines = append(lines, prefix+" "+actionStyleFor(action, tv.Focus == model.TrainPanelActions && i == tv.GlobalActions.SelectedIndex).Width(maxInt(8, width-4)).Render(action.Label))
	}
	return strings.Join(lines, "\n")
}

func renderStatusPanel(tv model.TrainWorkspaceState, width, _ int) string {
	run := tv.ActiveRun()
	if run == nil {
		return checkPendingStyle.Render("No active run")
	}
	var lines []string
	// Only show run info header during setup/ready phases.
	if run.Phase == model.TrainPhaseSetup || run.Phase == model.TrainPhaseReady || run.Phase == "" {
		lines = append(lines,
			checkDetailStyle.Render(fmt.Sprintf("run: %s", run.Label)),
			checkDetailStyle.Render(fmt.Sprintf("phase: %s", run.Phase)),
		)
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
			lines = append(lines, checkDetailStyle.Render(strings.Join(meta, " · ")))
		}
		if run.StatusMessage != "" {
			lines = append(lines, "")
			lines = append(lines, checkDetailStyle.Render(truncateRunText(run.StatusMessage, width-2)))
		}
		if run.ErrorMessage != "" {
			lines = append(lines, checkFailedStyle.Render(truncateRunText(run.ErrorMessage, width-2)))
		}
	}

	localChecks := tv.ChecksByGroup(run.ID, model.TrainCheckGroupLocal)
	targetChecks := tv.ChecksByGroup(run.ID, model.TrainCheckGroupTarget)
	if len(localChecks) > 0 {
		lines = append(lines, "")
		lines = append(lines, metricLabelStyle.Render("local checks"))
		for _, c := range localChecks {
			lines = append(lines, strings.TrimSpace(renderCheck(c, width)))
		}
	}
	// Hide target checks while SSH is failed (SSH gates all target probes).
	sshBlocked := false
	for _, c := range targetChecks {
		if c.Name == "ssh" && c.Status == model.TrainCheckFail {
			sshBlocked = true
			break
		}
	}
	if len(targetChecks) > 0 && !sshBlocked {
		lines = append(lines, "")
		lines = append(lines, metricLabelStyle.Render("target checks"))
		for _, c := range targetChecks {
			lines = append(lines, strings.TrimSpace(renderCheck(c, width)))
		}
	} else if sshBlocked {
		// Only show the SSH check itself when it's blocking.
		lines = append(lines, "")
		lines = append(lines, metricLabelStyle.Render("target checks"))
		for _, c := range targetChecks {
			if c.Name == "ssh" {
				lines = append(lines, strings.TrimSpace(renderCheck(c, width)))
				break
			}
		}
	}
	if tv.TrainPlan != nil {
		lines = append(lines, "")
		lines = append(lines, checkPassedStyle.Render("plan ready-to-start"))
	}
	return strings.Join(lines, "\n")
}

func renderMetricsPanel(tv model.TrainWorkspaceState, width, height int) string {
	run := tv.ActiveRun()
	if run == nil {
		return checkPendingStyle.Render("Waiting for active run")
	}
	return RenderLossSparkline(run.LossSeries, width, height)
}

func renderLogsPanel(tv model.TrainWorkspaceState, width, height int) string {
	run := tv.ActiveRun()
	if run == nil {
		return ""
	}
	logs := run.Logs.Lines
	if len(logs) > height {
		logs = logs[len(logs)-height:]
	}
	lines := make([]string, 0, len(logs))
	for _, line := range logs {
		lines = append(lines, styleLogLine(line, width))
	}
	if len(lines) == 0 {
		lines = append(lines, checkPendingStyle.Render("Waiting for runtime output..."))
	}
	return strings.Join(lines, "\n")
}

// RenderAgentBox renders the agent viewport inside a boxed panel.
// Message colors are set at the source (red for errors, green for success).
// totalLines/offset enable the scrollbar when content overflows.
func RenderAgentBox(content string, width, height int, focused bool, totalLines, offset int, spinnerView string) string {
	borderColor := lipgloss.Color("238")
	titleColor := lipgloss.Color("252")
	if focused {
		accent := lipgloss.Color("114") // green, same as success
		borderColor = accent
		titleColor = accent
	}
	titleText := "agent"
	if spinnerView != "" {
		titleText = "agent " + spinnerView
	}
	// Show scroll position hint when not at bottom
	if totalLines > 0 && offset+height-2 < totalLines {
		remaining := totalLines - offset - (height - 2)
		if remaining < 0 {
			remaining = 0
		}
		titleText += fmt.Sprintf("  ↓%d more", remaining)
	}
	title := panelTitleStyle.Foreground(titleColor).Render(titleText)
	innerWidth := maxInt(1, width-4)
	innerHeight := maxInt(1, height-2)
	bodyHeight := maxInt(1, innerHeight-1)
	// Content is already scrolled by the viewport — clamp width to fit box.
	bodyContent := lipgloss.NewStyle().
		Width(innerWidth).
		Height(bodyHeight).
		Render(clampPanelWidth(content, innerWidth))
	inner := lipgloss.NewStyle().
		Width(innerWidth).
		Height(innerHeight).
		Render(lipgloss.JoinVertical(lipgloss.Left, title, bodyContent))
	box := lipgloss.NewStyle().
		Width(width).
		Border(lipgloss.RoundedBorder()).
		BorderForeground(borderColor)
	return box.Render(inner)
}

// renderScrollbar returns a vertical scrollbar column for the given dimensions.
// height is the visible area, totalLines is the content length, offset is the
// current scroll position.  Returns empty strings when all content is visible.
func renderScrollbar(height, totalLines, offset int) []string {
	if totalLines <= height || height <= 0 {
		return nil
	}
	trackStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("238"))
	thumbStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("252"))

	thumbSize := maxInt(1, height*height/totalLines)
	maxOffset := totalLines - height
	thumbPos := 0
	if maxOffset > 0 {
		thumbPos = offset * (height - thumbSize) / maxOffset
	}
	if thumbPos+thumbSize > height {
		thumbPos = height - thumbSize
	}

	bar := make([]string, height)
	for i := 0; i < height; i++ {
		if i >= thumbPos && i < thumbPos+thumbSize {
			bar[i] = thumbStyle.Render("┃")
		} else {
			bar[i] = trackStyle.Render("│")
		}
	}
	return bar
}

// attachScrollbar overlays a scrollbar on the right edge of content lines.
func attachScrollbar(content string, height, totalLines, offset int) string {
	bar := renderScrollbar(height, totalLines, offset)
	if bar == nil {
		return content
	}
	lines := strings.Split(content, "\n")
	// Pad or trim to height
	for len(lines) < height {
		lines = append(lines, "")
	}
	if len(lines) > height {
		lines = lines[:height]
	}
	for i := 0; i < height && i < len(bar); i++ {
		lines[i] = lines[i] + bar[i]
	}
	return strings.Join(lines, "\n")
}

// tailContent returns the last height lines of content (streaming view).
func tailContent(content string, height int) string {
	lines := strings.Split(content, "\n")
	if len(lines) > height {
		lines = lines[len(lines)-height:]
	}
	return strings.Join(lines, "\n")
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
