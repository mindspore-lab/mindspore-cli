package panels

import (
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/vigo999/ms-cli/ui/model"
)

var (
	trainPanelStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("240")).
			Padding(0, 1)

	trainTitleStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("87")).
			Bold(true)

	trainMutedStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("244"))

	trainKeyStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("248"))

	trainValueStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("252"))

	trainAxisStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("240"))

	trainGuideStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("238"))

	trainOverlapStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("220")).
				Bold(true)
)

var trainPalette = []lipgloss.Style{
	lipgloss.NewStyle().Foreground(lipgloss.Color("45")).Bold(true),
	lipgloss.NewStyle().Foreground(lipgloss.Color("205")).Bold(true),
	lipgloss.NewStyle().Foreground(lipgloss.Color("111")).Bold(true),
	lipgloss.NewStyle().Foreground(lipgloss.Color("220")).Bold(true),
}

type trainRightPanelLayout struct {
	ChartHeight int
	LogTitle    string
	LogCount    int
	ShowLegend  bool
}

type trainBrailleCell struct {
	bits     uint8
	series   int
	endpoint bool
}

type TrainLowerPanel struct {
	Title string
	Body  string
}

type TrainEmbeddedChatLayout struct {
	Active bool
	Title  string
	Width  int
	Height int
}

// RenderTrainDashboard renders the two-column /train dashboard.
func RenderTrainDashboard(d model.TrainDashboard, width, height int, copyMode bool, lowerPanel *TrainLowerPanel) string {
	if width < 48 {
		return trainMutedStyle.Render("Terminal width is too small for training dashboard.")
	}
	if height < 12 {
		return trainMutedStyle.Render("Terminal height is too small for training dashboard.")
	}

	leftWidth, rightWidth := resolveTrainDashboardColumnWidths(width)

	left := renderTrainLeftPanel(d, leftWidth, height, copyMode, lowerPanel)
	right := renderTrainRightPanel(d, rightWidth, height, copyMode)
	sep := lipgloss.NewStyle().Width(1).Render(" ")
	return lipgloss.JoinHorizontal(lipgloss.Top, left, sep, right)
}

// RenderTrainHintBar renders key hints for train dashboard mode.
func RenderTrainHintBar(width int, copyMode bool, retryEnabled bool, chatActive bool) string {
	divider := hintDividerStyle.Render(repeatChar("━", width))
	parts := []string{}
	if retryEnabled {
		parts = append(parts, hintKeyStyle.Render("ctrl+r")+" "+hintDescStyle.Render("retry"))
	}
	if chatActive {
		if len(parts) > 0 {
			parts = append(parts, hintSepStyle.Render("•"))
		}
		parts = append(parts,
			hintKeyStyle.Render("enter")+" "+hintDescStyle.Render("send"),
			hintSepStyle.Render("•"),
			hintKeyStyle.Render("/")+" "+hintDescStyle.Render("slash"),
		)
	}
	if len(parts) > 0 {
		parts = append(parts, hintSepStyle.Render("•"))
	}
	parts = append(parts,
		hintKeyStyle.Render("ctrl+y")+" "+hintDescStyle.Render("copy"),
		hintSepStyle.Render("•"),
		hintKeyStyle.Render("esc")+" "+hintDescStyle.Render("back chat"),
		hintSepStyle.Render("•"),
		hintKeyStyle.Render("ctrl+c")+" "+hintDescStyle.Render("quit"),
	)
	content := strings.Join(parts, " ")
	return divider + "\n" + hintTextStyle.Render(content)
}

func ResolveTrainEmbeddedChatLayout(d model.TrainDashboard, width, height int, copyMode bool) TrainEmbeddedChatLayout {
	if width < 48 || height < 12 || isTrainConnectionPhase(d) {
		return TrainEmbeddedChatLayout{}
	}

	leftWidth, _ := resolveTrainDashboardColumnWidths(width)
	innerWidth := leftWidth - 4
	if innerWidth < 20 {
		innerWidth = 20
	}
	bodyHeight := resolveTrainLeftLowerPanelHeight(height)

	return TrainEmbeddedChatLayout{
		Active: true,
		Title:  "Analysis Chat",
		Width:  innerWidth,
		Height: bodyHeight,
	}
}

func resolveTrainDashboardColumnWidths(width int) (int, int) {
	leftWidth := int(float64(width) * 0.44)
	if leftWidth < 36 {
		leftWidth = 36
	}
	if leftWidth > width-24 {
		leftWidth = width - 24
	}
	return leftWidth, width - leftWidth - 1
}

func renderTrainLeftPanel(d model.TrainDashboard, width, height int, copyMode bool, lowerPanel *TrainLowerPanel) string {
	if width < 4 || height < 4 {
		return trainMutedStyle.Render("panel too small")
	}
	innerWidth := width - 4
	if innerWidth < 20 {
		innerWidth = 20
	}

	lines := []string{
		trainTitleStyle.Render("Training Workflow"),
		renderKV("run_id", fallback(d.RunID, "-")),
		renderKV("status", renderStatusBadge(d.Status)),
		renderKV("uptime", formatUptime(d)),
		renderKV("hosts", fmt.Sprintf("%d", len(d.HostOrder))),
	}
	if copyMode {
		lines = append(lines, renderKV("copy", "frozen for mouse selection"))
	}
	lines = append(lines, "")
	lines = append(lines, renderTrainStageSection(d)...)

	lines = append(lines, "", trainTitleStyle.Render("Hosts"))
	for idx, hostName := range d.HostOrder {
		host := d.Hosts[hostName]
		if host == nil {
			continue
		}
		hostStyle := paletteStyle(idx)
		lines = append(lines, hostStyle.Render("● "+host.Name)+" "+renderHostStatus(host.Status))
		lines = append(lines, "  "+renderKV("model", fallback(host.Model, "-")))
		lines = append(lines, "  "+renderKV("steps", formatSteps(host.Step, host.TotalStep)))
		lines = append(lines, "  "+renderKV("loss", formatMaybeFloat(host.Loss, host.Step > 0 || host.Loss != 0, 5)))
		lines = append(lines, "  "+renderKV("throughput", formatMaybeFloat(host.Throughput, host.Throughput != 0, 2)))
		lines = append(lines, "  "+renderKV("grad_norm", formatMaybeFloat(host.GradNorm, host.GradNorm != 0, 4)))
		if host.LogPath != "" {
			lines = append(lines, "  "+renderKV("log", trimMiddle(host.LogPath, innerWidth-8)))
		}
		lines = append(lines, "")
	}

	content := fitLines(lines, height-2)
	if lowerPanel != nil && !isTrainConnectionPhase(d) {
		lowerHeight := resolveTrainLeftLowerPanelHeight(height)
		upperHeight := (height - 2) - (lowerHeight + 2)
		if upperHeight < 1 {
			upperHeight = 1
		}
		content = fitLines(lines, upperHeight)
		lowerTitle := strings.TrimSpace(lowerPanel.Title)
		if lowerTitle == "" {
			lowerTitle = "Analysis Chat"
		}
		lowerBody := strings.TrimRight(lowerPanel.Body, "\n")
		if strings.TrimSpace(lowerBody) == "" {
			lowerBody = trainMutedStyle.Render("Chat ready. Ask about the run, request code changes, or ask to rerun.")
		}
		content = append(content, "")
		content = append(content, trainTitleStyle.Render(lowerTitle))
		content = append(content, fitLines([]string{lowerBody}, lowerHeight)...)
		content = fitLines(content, height-2)
	}
	return trainPanelStyle.Width(width - 2).Height(height - 2).Render(strings.Join(content, "\n"))
}

func renderTrainRightPanel(d model.TrainDashboard, width, height int, copyMode bool) string {
	if width < 4 || height < 4 {
		return trainMutedStyle.Render("panel too small")
	}
	innerWidth := width - 4
	if innerWidth < 20 {
		innerWidth = 20
	}

	failureLines := renderFailureDetails(d, innerWidth)
	layout := resolveTrainRightPanelLayout(d, height, copyMode, len(failureLines))
	chart, xRange, yRange := renderLossChart(d, innerWidth, layout.ChartHeight)

	lines := []string{
		trainTitleStyle.Render("Loss Curve"),
		trainMutedStyle.Render(yRange),
		chart,
		trainMutedStyle.Render(xRange),
	}
	if copyMode {
		lines = append(lines, trainMutedStyle.Render("screen frozen: drag mouse to select text"))
	}
	if len(failureLines) > 0 {
		lines = append(lines, "")
		lines = append(lines, failureLines...)
	}
	if layout.ShowLegend {
		lines = append(lines,
			"",
			renderLegend(d, innerWidth),
		)
	}
	lines = append(lines,
		"",
		trainTitleStyle.Render(layout.LogTitle),
	)
	lines = append(lines, renderRecentLogs(d, innerWidth, layout.LogCount)...)

	content := fitLines(lines, height-2)
	return trainPanelStyle.Width(width - 2).Height(height - 2).Render(strings.Join(content, "\n"))
}

func resolveTrainLeftLowerPanelHeight(height int) int {
	bodyHeight := (height - 2) / 3
	if bodyHeight < 6 {
		bodyHeight = 6
	}
	if bodyHeight > 12 {
		bodyHeight = 12
	}
	return bodyHeight
}

func renderTrainStageSection(d model.TrainDashboard) []string {
	lines := []string{trainTitleStyle.Render("Stages")}
	if shouldCollapseTrainStages(d) {
		lines = append(lines, renderCollapsedTrainStageSummary(d)...)
		return lines
	}

	for _, stage := range d.StageOrder {
		label := d.StageLabels[stage]
		if label == "" {
			label = stage
		}
		icon, style := stageIconStyle(d.StageStatus[stage])
		stageLine := style.Render(icon) + " " + trainValueStyle.Render(label)
		lines = append(lines, stageLine)
	}
	return lines
}

func shouldCollapseTrainStages(d model.TrainDashboard) bool {
	return areAllTrainStagesSuccessful(d) || isTrainDashboardPhase(d)
}

func areAllTrainStagesSuccessful(d model.TrainDashboard) bool {
	if strings.TrimSpace(strings.ToLower(d.Status)) != "success" || len(d.StageOrder) == 0 {
		return false
	}
	for _, stage := range d.StageOrder {
		if d.StageStatus[stage] != model.TrainStageSuccess {
			return false
		}
	}
	return true
}

func isTrainDashboardPhase(d model.TrainDashboard) bool {
	if d.CurrentStage == "dashboard" {
		return true
	}
	switch d.StageStatus["dashboard"] {
	case model.TrainStageRunning, model.TrainStageSuccess:
		return true
	default:
		return false
	}
}

func renderCollapsedTrainStageSummary(d model.TrainDashboard) []string {
	if areAllTrainStagesSuccessful(d) {
		return []string{
			renderKV("summary", fmt.Sprintf("all %d stages succeeded", len(d.StageOrder))),
			renderKV("state", "workflow complete"),
		}
	}

	setupDone := 0
	setupTotal := 0
	for _, stage := range d.StageOrder {
		if stage == "dashboard" {
			continue
		}
		setupTotal++
		if d.StageStatus[stage] == model.TrainStageSuccess {
			setupDone++
		}
	}

	state := "realtime dashboard active"
	switch d.StageStatus["dashboard"] {
	case model.TrainStageSuccess:
		state = "realtime dashboard complete"
	case model.TrainStageFailed:
		state = "realtime dashboard failed"
	}

	return []string{
		renderKV("summary", fmt.Sprintf("%d/%d setup stages completed", setupDone, setupTotal)),
		renderKV("state", state),
	}
}

func resolveTrainRightPanelLayout(d model.TrainDashboard, height int, copyMode bool, failureLineCount int) trainRightPanelLayout {
	layout := trainRightPanelLayout{
		LogTitle:   "Recent Logs",
		LogCount:   4,
		ShowLegend: true,
	}

	if isTrainConnectionPhase(d) {
		chartHeight := (height - failureLineCount) / 4
		if copyMode {
			chartHeight--
		}
		if chartHeight < 6 {
			chartHeight = 6
		}
		if chartHeight > 8 {
			chartHeight = 8
		}
		layout.ChartHeight = chartHeight
		layout.LogTitle = "Connection Logs"
		layout.LogCount = 12
		layout.ShowLegend = false
		return layout
	}

	chartHeight := height - 14 - failureLineCount
	if copyMode {
		chartHeight--
	}
	if chartHeight < 8 {
		chartHeight = 8
	}
	layout.ChartHeight = chartHeight
	return layout
}

func isTrainConnectionPhase(d model.TrainDashboard) bool {
	status := strings.TrimSpace(strings.ToLower(d.Status))
	switch status {
	case "failed", "success", "stopped":
		return false
	}

	if d.CurrentStage == "dashboard" {
		return false
	}
	if d.StageStatus["dashboard"] == model.TrainStageRunning || d.StageStatus["dashboard"] == model.TrainStageSuccess {
		return false
	}

	switch d.CurrentStage {
	case "", "sync", "launch", "master", "stream":
		return true
	default:
		return false
	}
}

func renderLossChart(d model.TrainDashboard, width, height int) (string, string, string) {
	if width < 18 || height < 7 {
		return trainMutedStyle.Render("insufficient space"), "x(total step): -", "y(loss): -"
	}

	xMax := 0
	xTotalHint := 0
	maxLoss := 0.0
	hasPoint := false

	for _, hostName := range d.HostOrder {
		host := d.Hosts[hostName]
		if host != nil {
			if host.TotalStep > xTotalHint {
				xTotalHint = host.TotalStep
			}
			if host.Step > xMax {
				xMax = host.Step
			}
		}
		series := d.Series[hostName]
		for _, pt := range series {
			if pt.Step > xMax {
				xMax = pt.Step
			}
			if !isFinite(pt.Loss) {
				continue
			}
			hasPoint = true
			if pt.Loss > maxLoss {
				maxLoss = pt.Loss
			}
		}
	}

	if xMax <= 0 {
		xMax = resolveDynamicTrainXMax(xTotalHint)
	}

	plotRows := height - 1
	canvasRows := plotRows - 1
	if canvasRows < 3 {
		return trainMutedStyle.Render("insufficient space"), "x(total step): -", "y(loss): -"
	}

	yTicks, yStep, yMax := buildNiceFloatTicks(maxLoss, hasPoint, desiredYAxisTicks(canvasRows))
	yLabelWidth := 1
	for _, tick := range yTicks {
		label := formatAxisFloat(tick, yStep)
		if lipgloss.Width(label) > yLabelWidth {
			yLabelWidth = lipgloss.Width(label)
		}
	}
	plotWidth := width - yLabelWidth - 2
	if plotWidth < 10 {
		return trainMutedStyle.Render("insufficient space"), "x(total step): -", "y(loss): -"
	}

	cells := make([][]trainBrailleCell, canvasRows)
	for row := 0; row < canvasRows; row++ {
		cells[row] = make([]trainBrailleCell, plotWidth)
		for col := 0; col < plotWidth; col++ {
			cells[row][col].series = -2
		}
	}

	subWidth := plotWidth * 2
	subHeight := canvasRows * 4
	setSubpixel := func(x, y, series int, endpoint bool) {
		if x < 0 || x >= subWidth || y < 0 || y >= subHeight {
			return
		}
		row := y / 4
		col := x / 2
		cell := &cells[row][col]
		cell.bits |= brailleBit(x%2, y%4)
		switch {
		case cell.series == -2 || cell.series == series:
			cell.series = series
		case cell.series != series:
			cell.series = -1
		}
		if endpoint {
			cell.endpoint = true
		}
	}

	for hostIdx, hostName := range d.HostOrder {
		series := d.Series[hostName]
		prevX, prevY := 0, 0
		hasPrev := false
		for _, pt := range series {
			if !isFinite(pt.Loss) {
				continue
			}
			x := mapValueToIndex(float64(pt.Step), float64(xMax), subWidth)
			y := mapValueToReverseIndex(pt.Loss, yMax, subHeight)
			if hasPrev {
				drawLine(prevX, prevY, x, y, func(px, py int) {
					setSubpixel(px, py, hostIdx, false)
				})
			} else {
				setSubpixel(x, y, hostIdx, false)
			}
			prevX, prevY = x, y
			hasPrev = true
		}
		if hasPrev {
			setSubpixel(prevX, prevY, hostIdx, true)
		}
	}

	yTickRows := make(map[int]string, len(yTicks))
	for _, tick := range yTicks[1:] {
		row := mapValueToReverseIndex(tick, yMax, canvasRows)
		yTickRows[row] = formatAxisFloat(tick, yStep)
	}

	xTicks := buildXTicks(xMax, desiredXAxisTicks(plotWidth))
	xTickCols := make(map[int]struct{}, len(xTicks))
	for _, tick := range xTicks {
		xTickCols[mapValueToIndex(float64(tick), float64(xMax), plotWidth)] = struct{}{}
	}

	rows := make([]string, 0, height)
	for row := 0; row < canvasRows; row++ {
		label := yTickRows[row]
		rows = append(rows, renderTrainPlotRow(cells[row], yLabelWidth, label, label != ""))
	}
	rows = append(rows, renderTrainXAxisBaseline(plotWidth, yLabelWidth, formatAxisFloat(0, yStep), xTickCols))
	rows = append(rows, renderTrainXAxisLabels(plotWidth, yLabelWidth, xTicks, xMax))

	xRange := fmt.Sprintf("x(total step): 0 -> %d", xMax)
	yRange := fmt.Sprintf("y(loss): 0 -> %s", formatAxisFloat(yMax, yStep))
	return strings.Join(rows, "\n"), xRange, yRange
}

func resolveDynamicTrainXMax(totalHint int) int {
	switch {
	case totalHint > 0 && totalHint <= 100:
		return totalHint
	case totalHint > 0:
		return 100
	default:
		return 100
	}
}

func renderLegend(d model.TrainDashboard, width int) string {
	if len(d.HostOrder) == 0 {
		return trainMutedStyle.Render("No hosts.")
	}
	lines := []string{trainTitleStyle.Render("Legend")}
	for idx, hostName := range d.HostOrder {
		host := d.Hosts[hostName]
		if host == nil {
			continue
		}
		entry := fmt.Sprintf("● %s  loss=%s  step=%s",
			hostName,
			formatMaybeFloat(host.Loss, host.Step > 0 || host.Loss != 0, 5),
			formatSteps(host.Step, host.TotalStep),
		)
		lines = append(lines, paletteStyle(idx).Render(trimMiddle(entry, width)))
	}
	return strings.Join(lines, "\n")
}

func renderFailureDetails(d model.TrainDashboard, width int) []string {
	if strings.TrimSpace(strings.ToLower(d.Status)) != "failed" {
		return nil
	}

	lines := []string{}
	if strings.TrimSpace(d.FailedStage) == "" &&
		strings.TrimSpace(d.FailedHost) == "" &&
		strings.TrimSpace(d.FailedCommand) == "" &&
		strings.TrimSpace(d.Error) == "" {
		return lines
	}

	lines = append(lines, trainTitleStyle.Render("Failure"))
	if d.FailedStage != "" {
		lines = append(lines, renderKV("stage", d.FailedStage))
	}
	if d.FailedHost != "" {
		lines = append(lines, renderKV("host", d.FailedHost))
	}
	lines = append(lines, renderWrappedDetail("command", d.FailedCommand, width)...)
	lines = append(lines, renderWrappedDetail("error", d.Error, width)...)
	return lines
}

func renderRecentLogs(d model.TrainDashboard, width, count int) []string {
	if len(d.RecentLogs) == 0 {
		return []string{trainMutedStyle.Render("Waiting for logs...")}
	}
	if count <= 0 {
		count = 4
	}
	logs := d.RecentLogs
	if len(logs) > count {
		logs = logs[len(logs)-count:]
	}
	out := make([]string, 0, len(logs))
	for _, line := range logs {
		out = append(out, wrapPanelText(line, width)...)
	}
	return out
}

func stageIconStyle(status model.TrainStageStatus) (string, lipgloss.Style) {
	switch status {
	case model.TrainStageRunning:
		return "▶", lipgloss.NewStyle().Foreground(lipgloss.Color("81")).Bold(true)
	case model.TrainStageSuccess:
		return "✓", lipgloss.NewStyle().Foreground(lipgloss.Color("114")).Bold(true)
	case model.TrainStageFailed:
		return "✗", lipgloss.NewStyle().Foreground(lipgloss.Color("203")).Bold(true)
	default:
		return "○", lipgloss.NewStyle().Foreground(lipgloss.Color("240"))
	}
}

func renderStatusBadge(status string) string {
	status = strings.TrimSpace(strings.ToLower(status))
	if status == "" {
		status = "idle"
	}
	label := strings.ToUpper(status)
	var style lipgloss.Style
	switch status {
	case "running":
		style = lipgloss.NewStyle().Foreground(lipgloss.Color("81")).Bold(true)
	case "success":
		style = lipgloss.NewStyle().Foreground(lipgloss.Color("114")).Bold(true)
	case "failed":
		style = lipgloss.NewStyle().Foreground(lipgloss.Color("203")).Bold(true)
	case "stopped":
		style = lipgloss.NewStyle().Foreground(lipgloss.Color("214")).Bold(true)
	default:
		style = lipgloss.NewStyle().Foreground(lipgloss.Color("244")).Bold(true)
	}
	return style.Render(label)
}

func renderHostStatus(status model.TrainHostStatus) string {
	return renderStatusBadge(string(status))
}

func renderKV(key, value string) string {
	return trainKeyStyle.Render(key+":") + " " + trainValueStyle.Render(value)
}

func formatUptime(d model.TrainDashboard) string {
	if d.StartedAt.IsZero() {
		return "-"
	}
	end := time.Now()
	if !d.FinishedAt.IsZero() {
		end = d.FinishedAt
	}
	return end.Sub(d.StartedAt).Round(time.Second).String()
}

func formatSteps(step, total int) string {
	switch {
	case step <= 0 && total <= 0:
		return "-"
	case total > 0 && step > 0:
		return fmt.Sprintf("%d / %d", step, total)
	case total > 0:
		return fmt.Sprintf("- / %d", total)
	default:
		return fmt.Sprintf("%d", step)
	}
}

func formatMaybeFloat(v float64, ok bool, prec int) string {
	if !ok {
		return "-"
	}
	return fmt.Sprintf("%.*f", prec, v)
}

func renderWrappedDetail(label, value string, width int) []string {
	if strings.TrimSpace(value) == "" {
		return nil
	}

	lines := []string{trainKeyStyle.Render(label + ":")}
	wrapWidth := width - 2
	if wrapWidth < 8 {
		wrapWidth = width
	}
	for _, block := range strings.Split(strings.ReplaceAll(value, "\r\n", "\n"), "\n") {
		if strings.TrimSpace(block) == "" {
			continue
		}
		for _, line := range wrapPanelText(block, wrapWidth) {
			lines = append(lines, "  "+trainValueStyle.Render(line))
		}
	}
	return lines
}

func fitLines(lines []string, max int) []string {
	if max <= 0 {
		return []string{}
	}
	out := make([]string, 0, max)
	for _, block := range lines {
		for _, line := range strings.Split(block, "\n") {
			if len(out) == max {
				return out
			}
			out = append(out, line)
		}
	}
	for len(out) < max {
		out = append(out, "")
	}
	return out
}

func countPanelLines(block string) int {
	if block == "" {
		return 0
	}
	return len(strings.Split(block, "\n"))
}

func paletteStyle(i int) lipgloss.Style {
	if len(trainPalette) == 0 {
		return lipgloss.NewStyle()
	}
	return trainPalette[i%len(trainPalette)]
}

func desiredYAxisTicks(canvasRows int) int {
	switch {
	case canvasRows >= 12:
		return 6
	case canvasRows >= 8:
		return 5
	default:
		return 4
	}
}

func desiredXAxisTicks(plotWidth int) int {
	switch {
	case plotWidth >= 48:
		return 6
	case plotWidth >= 30:
		return 5
	default:
		return 4
	}
}

func buildNiceFloatTicks(maxValue float64, hasPoint bool, desired int) ([]float64, float64, float64) {
	if !hasPoint || maxValue <= 0 {
		return []float64{0, 0.5, 1}, 0.5, 1
	}
	if desired < 2 {
		desired = 2
	}

	step := niceAxisStep(maxValue / float64(desired-1))
	maxTick := math.Ceil(maxValue/step) * step
	if maxTick <= 0 {
		maxTick = 1
	}
	count := int(math.Round(maxTick/step)) + 1
	if count < 2 {
		count = 2
	}

	ticks := make([]float64, 0, count)
	for i := 0; i < count; i++ {
		ticks = append(ticks, float64(i)*step)
	}
	return ticks, step, maxTick
}

func niceAxisStep(raw float64) float64 {
	if raw <= 0 || math.IsNaN(raw) || math.IsInf(raw, 0) {
		return 1
	}

	exp := math.Floor(math.Log10(raw))
	scale := math.Pow(10, exp)
	fraction := raw / scale

	var niceFraction float64
	switch {
	case fraction <= 1:
		niceFraction = 1
	case fraction <= 2:
		niceFraction = 2
	case fraction <= 2.5:
		niceFraction = 2.5
	case fraction <= 5:
		niceFraction = 5
	default:
		niceFraction = 10
	}
	return niceFraction * scale
}

func formatAxisFloat(v, step float64) string {
	decimals := 0
	scaled := step
	for decimals < 6 && !almostEqual(scaled, math.Round(scaled)) {
		scaled *= 10
		decimals++
	}
	return fmt.Sprintf("%.*f", decimals, v)
}

func buildXTicks(maxValue, desired int) []int {
	if maxValue <= 0 {
		return []int{0, 1}
	}
	if desired < 2 {
		desired = 2
	}

	ticks := make([]int, 0, desired)
	last := -1
	for i := 0; i < desired; i++ {
		value := int(math.Round(float64(i) * float64(maxValue) / float64(desired-1)))
		if i == desired-1 {
			value = maxValue
		}
		if value <= last {
			continue
		}
		ticks = append(ticks, value)
		last = value
	}
	if ticks[len(ticks)-1] != maxValue {
		ticks = append(ticks, maxValue)
	}
	return ticks
}

func mapValueToIndex(value, maxValue float64, span int) int {
	if span <= 1 || maxValue <= 0 {
		return 0
	}
	ratio := value / maxValue
	if ratio < 0 {
		ratio = 0
	}
	if ratio > 1 {
		ratio = 1
	}
	return int(math.Round(ratio * float64(span-1)))
}

func mapValueToReverseIndex(value, maxValue float64, span int) int {
	if span <= 1 || maxValue <= 0 {
		return span - 1
	}
	ratio := value / maxValue
	if ratio < 0 {
		ratio = 0
	}
	if ratio > 1 {
		ratio = 1
	}
	return (span - 1) - int(math.Round(ratio*float64(span-1)))
}

func brailleBit(x, y int) uint8 {
	switch {
	case x == 0 && y == 0:
		return 0x01
	case x == 0 && y == 1:
		return 0x02
	case x == 0 && y == 2:
		return 0x04
	case x == 0 && y == 3:
		return 0x40
	case x == 1 && y == 0:
		return 0x08
	case x == 1 && y == 1:
		return 0x10
	case x == 1 && y == 2:
		return 0x20
	case x == 1 && y == 3:
		return 0x80
	default:
		return 0
	}
}

func brailleRune(bits uint8) rune {
	return rune(0x2800) + rune(bits)
}

func renderTrainPlotRow(row []trainBrailleCell, labelWidth int, tickLabel string, tickRow bool) string {
	var sb strings.Builder
	if tickRow {
		sb.WriteString(trainMutedStyle.Render(fmt.Sprintf("%*s ", labelWidth, tickLabel)))
		sb.WriteString(trainAxisStyle.Render("┤"))
	} else {
		sb.WriteString(strings.Repeat(" ", labelWidth+1))
		sb.WriteString(trainAxisStyle.Render("│"))
	}

	for _, cell := range row {
		switch {
		case cell.bits != 0:
			ch := brailleRune(cell.bits)
			if cell.endpoint {
				ch = '●'
			}
			if cell.series == -1 {
				sb.WriteString(trainOverlapStyle.Render(string(ch)))
			} else {
				sb.WriteString(paletteStyle(cell.series).Render(string(ch)))
			}
		case tickRow:
			sb.WriteString(trainGuideStyle.Render("⠄"))
		default:
			sb.WriteRune(' ')
		}
	}
	return sb.String()
}

func renderTrainXAxisBaseline(plotWidth, labelWidth int, zeroLabel string, tickCols map[int]struct{}) string {
	var sb strings.Builder
	sb.WriteString(trainMutedStyle.Render(fmt.Sprintf("%*s ", labelWidth, zeroLabel)))
	sb.WriteString(trainAxisStyle.Render("└"))
	for col := 0; col < plotWidth; col++ {
		ch := '─'
		if col > 0 {
			if _, ok := tickCols[col]; ok {
				ch = '┬'
			}
		}
		sb.WriteString(trainAxisStyle.Render(string(ch)))
	}
	return sb.String()
}

func renderTrainXAxisLabels(plotWidth, labelWidth int, ticks []int, maxValue int) string {
	labels := []rune(strings.Repeat(" ", plotWidth))
	lastEnd := -2

	for idx, tick := range ticks {
		label := fmt.Sprintf("%d", tick)
		labelWidth := len(label)
		col := mapValueToIndex(float64(tick), float64(maxValue), plotWidth)
		start := col - labelWidth/2
		if start < 0 {
			start = 0
		}
		if end := start + labelWidth; end > plotWidth {
			start = plotWidth - labelWidth
		}
		if start <= lastEnd {
			if idx == len(ticks)-1 {
				start = plotWidth - labelWidth
			} else {
				start = lastEnd + 2
			}
		}
		if start < 0 || start+labelWidth > plotWidth || start <= lastEnd {
			continue
		}
		for i, r := range label {
			labels[start+i] = r
		}
		lastEnd = start + labelWidth - 1
	}

	return strings.Repeat(" ", labelWidth+2) + trainMutedStyle.Render(string(labels))
}

func drawLine(x0, y0, x1, y1 int, paint func(x, y int)) {
	dx := absInt(x1 - x0)
	dy := -absInt(y1 - y0)
	sx := -1
	if x0 < x1 {
		sx = 1
	}
	sy := -1
	if y0 < y1 {
		sy = 1
	}
	err := dx + dy

	for {
		paint(x0, y0)
		if x0 == x1 && y0 == y1 {
			break
		}
		e2 := 2 * err
		if e2 >= dy {
			err += dy
			x0 += sx
		}
		if e2 <= dx {
			err += dx
			y0 += sy
		}
	}
}

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

func fallback(v, def string) string {
	if strings.TrimSpace(v) == "" {
		return def
	}
	return v
}

func trimMiddle(s string, max int) string {
	if max <= 0 {
		return ""
	}
	if lipgloss.Width(s) <= max {
		return s
	}
	runes := []rune(s)
	if len(runes) <= max {
		return string(runes)
	}
	if max <= 3 {
		return string(runes[:max])
	}
	head := (max - 1) / 2
	tail := max - 1 - head
	return string(runes[:head]) + "…" + string(runes[len(runes)-tail:])
}

func wrapPanelText(s string, width int) []string {
	if width <= 0 {
		return []string{""}
	}
	if lipgloss.Width(s) <= width {
		return []string{s}
	}

	var out []string
	remaining := strings.TrimSpace(s)
	indent := ""
	continuation := "  "

	for remaining != "" {
		limit := width - lipgloss.Width(indent)
		if limit <= 0 {
			limit = width
			indent = ""
		}
		if lipgloss.Width(remaining) <= limit {
			out = append(out, indent+remaining)
			break
		}

		cut := wrapBoundary(remaining, limit)
		head := strings.TrimRight(remaining[:cut], " ")
		if head == "" {
			head = remaining[:cut]
		}
		out = append(out, indent+head)
		remaining = strings.TrimLeft(remaining[cut:], " ")
		indent = continuation
	}

	if len(out) == 0 {
		return []string{""}
	}
	return out
}

func wrapBoundary(s string, width int) int {
	if width <= 0 {
		return 0
	}
	runes := []rune(s)
	if len(runes) <= width {
		return len(s)
	}

	lastSpace := -1
	for i := 0; i < width && i < len(runes); i++ {
		if runes[i] == ' ' || runes[i] == '\t' {
			lastSpace = i
		}
	}
	if lastSpace > 0 {
		return len(string(runes[:lastSpace]))
	}
	return len(string(runes[:width]))
}

func isFinite(v float64) bool {
	return !math.IsInf(v, 0) && !math.IsNaN(v)
}

func almostEqual(a, b float64) bool {
	return math.Abs(a-b) < 1e-12
}
