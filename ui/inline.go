package ui

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
	"github.com/mindspore-lab/mindspore-cli/ui/panels"
	"github.com/mindspore-lab/mindspore-cli/ui/theme"
)

var (
	bannerBoxStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("252")).
			Align(lipgloss.Left).
			Padding(1, 2)

	bannerTitleStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("39")).
				Bold(true)

	bannerLabelStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("244"))

	bannerValueStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("252"))

	cmdOutputStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("245"))

	cmdStderrStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("203"))

	metaStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("244")).
			Italic(true)
)

const liveShellPreviewOutputLines = 8

// maybePrintBanner prints the startup banner once, deferred until
// no modal popup is blocking the normal buffer.
func (a *App) maybePrintBanner() tea.Cmd {
	if a.bannerPrinted || a.bootActive || a.startupBannerSuppressed || a.setupPopup != nil || a.modelPicker != nil || a.sessionPicker != nil {
		return nil
	}
	a.bannerPrinted = true
	return tea.Sequence(
		tea.Println(RenderBanner(a.state.Version, a.state.WorkDir, a.state.RepoURL, a.state.Model.Name, a.state.Model.CtxMax)),
		a.signalHistoryReplayReady(),
	)
}

// RenderBanner renders the one-shot banner shown after boot in inline mode.
func RenderBanner(version, workDir, repoURL, modelName string, ctxMax int) string {
	ver := strings.TrimSpace(version)
	// Strip product name prefix (e.g. "MindSpore CLI. v0.5.0" → "v0.5.0")
	for _, prefix := range []string{"MindSpore CLI. ", "MindSpore CLI. "} {
		ver = strings.TrimPrefix(ver, prefix)
	}
	if ver == "" {
		ver = "dev"
	}
	title := bannerTitleStyle.Render("MindSpore CLI") + " " + bannerValueStyle.Render("("+ver+")")

	rows := []string{
		bannerRow("model", valueOrString(strings.TrimSpace(modelName), "unknown")),
		bannerRow("directory", valueOrString(shortenPath(strings.TrimSpace(workDir)), ".")),
	}

	body := title + "\n\n" + strings.Join(rows, "\n")
	return bannerBoxStyle.Render(body)
}

func bannerRow(label, value string) string {
	return bannerLabelStyle.Render(label+":") + " " + bannerValueStyle.Render(value)
}

func (a *App) signalHistoryReplayReady() tea.Cmd {
	if a == nil || a.userCh == nil {
		return nil
	}
	return func() tea.Msg {
		a.userCh <- historyReplayReadyToken
		return nil
	}
}

func formatContext(ctxMax int) string {
	if ctxMax <= 0 {
		return "-"
	}
	return strconv.Itoa(ctxMax) + " tokens"
}

func shortenPath(path string) string {
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		return path
	}
	if strings.HasPrefix(path, home) {
		return "~" + path[len(home):]
	}
	return path
}

func (a App) renderWidth() int {
	if a.width > 8 {
		return a.width - 4
	}
	return 96
}

func (a App) renderMainView() string {
	parts := []string{}
	if a.trainView.Active {
		parts = append(parts, panels.RenderTrainHUD(a.trainView, a.width, a.agentStatus()))
	}
	inputView := a.input.View()
	hintView := panels.RenderHintBar(a.state, a.width)
	if a.trainView.Active {
		hintView = panels.RenderTrainHUDHintBar(a.width)
	}
	queueBanner := ""
	if len(a.queuedInputs) > 0 {
		queueBanner = queueBannerStyle.Render("messages queued (press esc to interrupt)")
	}
	if a.permissionPrompt != nil {
		parts = append(parts, renderPermissionPromptPopup(a.permissionPrompt))
	} else {
		reservedParts := append([]string{}, parts...)
		if queueBanner != "" {
			reservedParts = append(reservedParts, queueBanner)
		}
		if status := a.activePreview(reservedParts, inputView, hintView); status != "" {
			parts = append(parts, "", status)
		}
	}
	if queueBanner != "" {
		parts = append(parts, queueBanner)
	}
	parts = append(parts, "", inputView, hintView)
	return trimViewHeight(lipgloss.JoinVertical(lipgloss.Left, parts...), a.height, false)
}

func (a App) activePreview(partsBeforeStatus []string, inputView, hintView string) string {
	if a.backgroundModelWork && a.state.WaitKind == model.WaitModel {
		a.thinking.SetText("Working...")
		teal := lipgloss.Color("#2DD4BF")
		a.thinking.SetStyle(
			lipgloss.NewStyle().Foreground(teal),
			lipgloss.NewStyle().Foreground(teal).Italic(true))
		return a.thinking.ViewWithTip()
	}
	if msg, ok := a.lastStreamingAgent(); ok {
		return a.fitStatusPreview(a.renderTranscriptMessage(msg), partsBeforeStatus, inputView, hintView)
	}
	if msg, ok := a.lastActiveTool(); ok {
		if strings.EqualFold(strings.TrimSpace(msg.ToolName), "Bash") {
			return a.renderShellActivePreview(msg, a.availableStatusLines(partsBeforeStatus, inputView, hintView))
		}
		return a.fitStatusPreview(a.renderTranscriptMessage(msg), partsBeforeStatus, inputView, hintView)
	}
	t := theme.Current
	if a.state.WaitKind == model.WaitModel {
		a.thinking.SetText("Working...")
		teal := lipgloss.Color("#2DD4BF")
		a.thinking.SetStyle(
			lipgloss.NewStyle().Foreground(teal),
			lipgloss.NewStyle().Foreground(teal).Italic(true))
		return a.thinking.ViewWithTip()
	}
	if a.state.WaitKind == model.WaitTool {
		label := "Running tool..."
		if msg, ok := a.lastActiveTool(); ok {
			label = "Running " + strings.TrimSpace(msg.ToolName) + "..."
		}
		a.thinking.SetText(label)
		a.thinking.SetStyle(
			lipgloss.NewStyle().Foreground(t.Warning),
			lipgloss.NewStyle().Foreground(t.Warning).Italic(true))
		return a.thinking.ViewWithTip()
	}
	// Fallback while deltas are buffered but no streaming message is available.
	a.deltaMu.Lock()
	hasDelta := a.deltaBuf.Len() > 0
	a.deltaMu.Unlock()
	if hasDelta {
		a.thinking.SetText("Responding...")
		a.thinking.SetStyle(
			lipgloss.NewStyle().Foreground(t.Success),
			lipgloss.NewStyle().Foreground(t.Success).Italic(true))
		return a.thinking.ViewWithTip()
	}
	return ""
}

func (a App) fitStatusPreview(content string, partsBeforeStatus []string, inputView, hintView string) string {
	return tailLines(content, a.availableStatusLines(partsBeforeStatus, inputView, hintView))
}

func (a App) availableStatusLines(partsBeforeStatus []string, inputView, hintView string) int {
	if a.height <= 0 {
		return 0
	}

	reserved := 0
	for _, part := range partsBeforeStatus {
		reserved += viewLineCount(part)
	}
	// Blank separator before the status block, plus the blank separator before the input.
	reserved += 2
	reserved += viewLineCount(inputView)
	reserved += viewLineCount(hintView)

	available := a.height - reserved
	if available < 1 {
		available = 1
	}
	return available
}

func (a App) renderShellActivePreview(msg model.Message, available int) string {
	if available <= 0 {
		return ""
	}

	header := panels.RenderToolCallHeader("Bash", strings.TrimSpace(msg.ToolArgs)) +
		" " + metaStyle.Render(a.shellActiveStatusText())
	if available == 1 {
		return header
	}

	outputSlots := available - 1
	if outputSlots > liveShellPreviewOutputLines {
		outputSlots = liveShellPreviewOutputLines
	}
	if outputSlots < 0 {
		outputSlots = 0
	}

	lines := renderShellPreviewOutputLines(msg.Content, outputSlots)

	parts := []string{header}
	parts = append(parts, lines...)
	return strings.Join(parts, "\n")
}

func (a App) shellActiveStatusText() string {
	text := "running, ctrl+o to expand..."
	if a.state.WaitKind == model.WaitTool && !a.state.WaitStartedAt.IsZero() {
		text += " " + model.FormatWaitDuration(a.currentWaitElapsed())
	}
	return text
}

func (a App) renderTranscriptMessage(msg model.Message) string {
	temp := model.NewState("", "", "", "", 0)
	temp.Messages = []model.Message{msg}
	temp.WaitKind = a.state.WaitKind
	temp.WaitStartedAt = a.state.WaitStartedAt
	return panels.RenderMessages(temp, a.thinking.View(), a.thinking.FrameView(), a.renderWidth(), a.trainView.Active)
}

func (a App) printMessage(msg model.Message) tea.Cmd {
	rendered := a.renderTranscriptMessage(msg)
	rendered = strings.TrimRight(rendered, "\n")
	if strings.TrimSpace(rendered) == "" {
		return nil
	}
	// Add blank line before block messages to separate logical groups.
	if msg.Kind == model.MsgAgent || msg.Kind == model.MsgTool {
		return tea.Sequence(tea.Println(""), tea.Println(rendered))
	}
	return tea.Println(rendered)
}

func (a App) historyMessagesForRender(expanded, includeLive bool) []model.Message {
	messages := make([]model.Message, 0, len(a.state.Messages))
	for _, msg := range a.state.Messages {
		if !includeLive && (msg.Pending || msg.Streaming) {
			continue
		}
		if msg.Kind == model.MsgTool && !expanded {
			msg = a.truncateToolForPrint(msg)
		}
		messages = append(messages, msg)
	}
	return messages
}

func (a App) renderTranscriptHistory(expanded, includeLive bool) string {
	history := a.historyMessagesForRender(expanded, includeLive)
	if len(history) == 0 {
		return ""
	}
	state := a.state
	state.Messages = history
	if !includeLive {
		state = state.WithThinking(false).ClearWait()
	}
	rendered := panels.RenderMessages(state, a.thinking.View(), a.thinking.FrameView(), a.renderWidth(), a.trainView.Active)
	return strings.TrimRight(rendered, "\n")
}

func (a App) reprintHistoryOnResize() tea.Cmd {
	if a.bootActive || a.modalAltScreen || a.startupBannerSuppressed || a.issueView.Active() {
		return nil
	}

	cmds := []tea.Cmd{
		tea.ClearScreen,
		tea.Printf("\x1b[3J"),
	}
	for _, line := range a.clearHeadingLines("") {
		cmds = append(cmds, tea.Println(line))
	}
	if history := a.renderTranscriptHistory(false, false); strings.TrimSpace(history) != "" {
		cmds = append(cmds, tea.Println(""), tea.Println(history))
	}
	return tea.Sequence(cmds...)
}

func (a App) printAgentDelta(delta string) tea.Cmd {
	// Buffer deltas silently. The live area shows a streaming indicator.
	// The full glamour-rendered message prints when AgentReply arrives.
	a.deltaMu.Lock()
	a.deltaBuf.WriteString(delta)
	a.deltaMu.Unlock()
	return nil
}

func (a App) flushDeltaBuf() tea.Cmd {
	a.deltaMu.Lock()
	content := a.deltaBuf.String()
	a.deltaBuf.Reset()
	a.deltaMu.Unlock()
	*a.deltaStarted = false
	if strings.TrimSpace(content) == "" {
		return nil
	}
	return a.printMessage(model.Message{Kind: model.MsgAgent, Content: content})
}

// formatAgentLines prefixes the first line of an agent reply block with ●
// and indents continuation lines to align with the text after the marker.
func (a App) formatAgentLines(text string) string {
	lines := strings.Split(text, "\n")
	for i := range lines {
		if !*a.deltaStarted {
			*a.deltaStarted = true
			lines[i] = "● " + lines[i]
		} else {
			lines[i] = "  " + lines[i]
		}
	}
	return strings.Join(lines, "\n")
}

func (a App) printUserInput(input string) tea.Cmd {
	input = strings.TrimSpace(input)
	if input == "" {
		return nil
	}
	cmd := a.printMessage(model.Message{Kind: model.MsgUser, Content: input})
	if cmd == nil {
		return nil
	}
	// Blank line before and after user input for visual separation.
	return tea.Sequence(tea.Println(""), cmd, tea.Println(""))
}

func (a App) printResolvedTool(ev model.Event) tea.Cmd {
	msg, ok := a.resolvedToolMessage(ev)
	if !ok {
		return nil
	}
	msg = a.truncateToolForPrint(msg)
	return a.printMessage(msg)
}

// truncateToolForPrint applies collapse/truncation policy to a tool message
// before printing. When toolsExpanded is true, content is returned unchanged.
func (a App) truncateToolForPrint(msg model.Message) model.Message {
	if msg.Kind != model.MsgTool || msg.Pending || msg.Streaming {
		return msg
	}
	if *a.toolsExpanded {
		return msg
	}
	if strings.EqualFold(strings.TrimSpace(msg.ToolName), "Read") {
		msg.Content = ""
		return msg
	}
	if msg.Display == model.DisplayCollapsed {
		msg.Content = collapsedToolDetails(msg.Content, collapsedPreviewLines(msg.ToolName))
		return msg
	}
	msg.Content = truncateToolContentForTool(msg.ToolName, msg.Content)
	return msg
}

func (a App) printShellHeader(ev model.Event) tea.Cmd {
	*a.cmdOutputStarted = false
	*a.cmdOutputLines = 0
	// CmdStarted event itself has no message; the command args live
	// in the tool message that ToolCallStart already created in state.
	args := strings.TrimSpace(ev.Message)
	if args == "" {
		if msg, ok := findToolMessage(a.state.Messages, ev.ToolCallID); ok {
			args = strings.TrimSpace(msg.ToolArgs)
		}
	}
	if args != "" && !strings.HasPrefix(args, "$ ") {
		args = "$ " + args
	}
	header := panels.RenderToolCallHeader("Bash", args)
	if strings.TrimSpace(header) == "" {
		return nil
	}
	return tea.Println(header)
}

func (a App) printShellFinished(ev model.Event, before []model.Message) tea.Cmd {
	// If we already streamed output via CmdOutput, the header was already
	// printed by printShellHeader — only show the summary line.
	prevMsg, hadPrev := findToolMessage(before, ev.ToolCallID)
	if *a.cmdOutputStarted || (hadPrev && strings.TrimSpace(prevMsg.Content) != "") {
		var cmds []tea.Cmd
		// Show truncation hint if lines were hidden during streaming.
		if !*a.toolsExpanded && *a.cmdOutputLines > collapsedPreviewMaxLines {
			omitted := *a.cmdOutputLines - collapsedPreviewMaxLines
			cmds = append(cmds, tea.Println(metaStyle.Render(
				fmt.Sprintf("     … +%d lines (ctrl+o to expand)", omitted))))
		}
		summary := strings.TrimSpace(ev.Summary)
		if summary != "" && summary != "completed" {
			cmds = append(cmds, tea.Println(metaStyle.Render("shell "+summary)))
		}
		return combineCmds(cmds...)
	}

	msg, ok := a.resolvedToolMessage(ev)
	if !ok {
		return nil
	}
	if strings.TrimSpace(msg.Content) == "(No output)" && strings.TrimSpace(msg.Summary) == "completed" {
		return nil
	}
	msg = a.truncateToolForPrint(msg)
	return a.printMessage(msg)
}

func (a App) printCommandOutput(chunk string) tea.Cmd {
	chunk = strings.ReplaceAll(chunk, "\r\n", "\n")
	chunk = strings.TrimSuffix(chunk, "\n")
	if strings.TrimSpace(chunk) == "" {
		return nil
	}

	// When tools are collapsed, limit streamed output to collapsedPreviewMaxLines.
	limit := collapsedPreviewMaxLines
	if *a.toolsExpanded {
		limit = -1 // no limit
	}

	lines := strings.Split(chunk, "\n")
	rendered := make([]string, 0, len(lines))
	for _, line := range lines {
		if limit >= 0 && *a.cmdOutputLines >= limit {
			*a.cmdOutputLines += len(lines) - len(rendered)
			break
		}
		style := cmdOutputStyle
		if strings.HasPrefix(strings.TrimSpace(line), "[stderr]") {
			style = cmdStderrStyle
		}
		prefix := "     "
		if !*a.cmdOutputStarted {
			*a.cmdOutputStarted = true
			prefix = "  ⎿  "
		}
		rendered = append(rendered, prefix+style.Render(line))
		*a.cmdOutputLines++
	}
	if len(rendered) == 0 {
		return nil
	}
	return tea.Println(strings.Join(rendered, "\n"))
}

func (a App) fallbackPrint(prevLen int) tea.Cmd {
	if prevLen < 0 || prevLen > len(a.state.Messages) {
		prevLen = len(a.state.Messages)
	}
	var cmds []tea.Cmd
	for i := prevLen; i < len(a.state.Messages); i++ {
		msg := a.state.Messages[i]
		if msg.Kind == model.MsgUser || msg.Pending || msg.Streaming {
			continue
		}
		cmds = append(cmds, a.printMessage(msg))
	}
	return combineCmds(cmds...)
}

func (a App) eventPrintCmd(ev model.Event, prevMessages []model.Message, suppressUserPrint bool) tea.Cmd {
	prevLen := len(prevMessages)

	switch ev.Type {
	case model.UserInput:
		if suppressUserPrint {
			return nil
		}
		return a.printUserInput(ev.Message)
	case model.AgentReply:
		// If deltas were already streamed, flush remaining buffer only.
		for i := len(prevMessages) - 1; i >= 0; i-- {
			if prevMessages[i].Kind == model.MsgAgent && prevMessages[i].Streaming {
				return a.flushDeltaBuf()
			}
		}
		return a.printMessage(model.Message{Kind: model.MsgAgent, Content: ev.Message, RawANSI: ev.RawANSI})
	case model.AgentReplyDelta:
		return a.printAgentDelta(ev.Message)
	case model.AgentBackgroundWork, model.AgentThinking, model.TaskDone, model.TokenUpdate:
		return nil
	case model.ToolCallStart:
		// Don't print pending tools to scrollback — they show in the live
		// area while running, and the resolved result prints when done.
		return nil
	case model.CmdStarted:
		// Don't print shell header to scrollback — print the full resolved
		// tool at CmdFinished so it shows ✓ like all other tools.
		*a.cmdOutputStarted = false
		*a.cmdOutputLines = 0
		return nil
	case model.CmdOutput:
		// Track output but don't print to scrollback.
		*a.cmdOutputStarted = true
		*a.cmdOutputLines++
		return nil
	case model.CmdFinished:
		return a.printResolvedTool(ev)
	case model.ToolRead, model.ToolGrep, model.ToolGlob, model.ToolEdit, model.ToolWrite, model.ToolSkill, model.ToolInterrupted, model.ToolWarning, model.ToolError, model.ToolReplay:
		return a.printResolvedTool(ev)
	case model.ClearScreen:
		return a.clearMessage(ev.Summary)
	default:
		return a.fallbackPrint(prevLen)
	}
}

func (a App) resolvedToolMessage(ev model.Event) (model.Message, bool) {
	if msg, ok := findToolMessage(a.state.Messages, ev.ToolCallID); ok {
		return msg, true
	}
	for i := len(a.state.Messages) - 1; i >= 0; i-- {
		if a.state.Messages[i].Kind == model.MsgTool && !a.state.Messages[i].Pending {
			return a.state.Messages[i], true
		}
	}
	return model.Message{}, false
}

func (a App) lastStableAgent() (model.Message, bool) {
	for i := len(a.state.Messages) - 1; i >= 0; i-- {
		msg := a.state.Messages[i]
		if msg.Kind == model.MsgAgent && !msg.Streaming {
			return msg, true
		}
	}
	return model.Message{}, false
}

func findToolMessage(messages []model.Message, toolCallID string) (model.Message, bool) {
	if idx := toolMessageIndex(messages, toolCallID); idx >= 0 {
		return messages[idx], true
	}
	return model.Message{}, false
}

func (a App) lastStreamingAgent() (model.Message, bool) {
	for i := len(a.state.Messages) - 1; i >= 0; i-- {
		msg := a.state.Messages[i]
		if msg.Kind == model.MsgAgent && msg.Streaming {
			return msg, true
		}
	}
	return model.Message{}, false
}

func (a App) lastActiveTool() (model.Message, bool) {
	for i := len(a.state.Messages) - 1; i >= 0; i-- {
		msg := a.state.Messages[i]
		if msg.Kind == model.MsgTool && (msg.Pending || msg.Streaming) {
			return msg, true
		}
	}
	return model.Message{}, false
}

func tailLines(content string, limit int) string {
	if limit <= 0 {
		return ""
	}
	lines := strings.Split(strings.TrimSpace(content), "\n")
	if len(lines) <= limit {
		return strings.Join(lines, "\n")
	}
	return strings.Join(lines[len(lines)-limit:], "\n")
}

func viewLineCount(content string) int {
	if content == "" {
		return 1
	}
	return len(strings.Split(content, "\n"))
}

func renderShellPreviewOutputLines(content string, limit int) []string {
	if limit <= 0 {
		return nil
	}

	content = strings.ReplaceAll(content, "\r\n", "\n")
	content = strings.TrimSuffix(content, "\n")
	if strings.TrimSpace(content) == "" {
		return nil
	}

	lines := strings.Split(content, "\n")
	if len(lines) > limit {
		lines = lines[len(lines)-limit:]
	}

	rendered := make([]string, 0, len(lines))
	for i, line := range lines {
		style := cmdOutputStyle
		if strings.HasPrefix(strings.TrimSpace(line), "[stderr]") {
			style = cmdStderrStyle
		}
		prefix := "     "
		if i == 0 {
			prefix = "  ⎿  "
		}
		rendered = append(rendered, prefix+style.Render(line))
	}
	return rendered
}

func combineCmds(cmds ...tea.Cmd) tea.Cmd {
	filtered := make([]tea.Cmd, 0, len(cmds))
	for _, cmd := range cmds {
		if cmd != nil {
			filtered = append(filtered, cmd)
		}
	}
	switch len(filtered) {
	case 0:
		return nil
	case 1:
		return filtered[0]
	default:
		return tea.Batch(filtered...)
	}
}

func (a App) clearHeadingLines(resumeHint string) []string {
	lines := []string{
		RenderBanner(a.state.Version, a.state.WorkDir, a.state.RepoURL, a.state.Model.Name, a.state.Model.CtxMax),
	}
	if hint := strings.TrimSpace(resumeHint); hint != "" {
		lines = append(lines, metaStyle.Render(hint))
	}
	return lines
}

func (a App) clearMessage(resumeHint string) tea.Cmd {
	cmds := []tea.Cmd{
		tea.ClearScreen,
		tea.Printf("\x1b[3J"),
	}
	for _, line := range a.clearHeadingLines(resumeHint) {
		cmds = append(cmds, tea.Println(line))
	}
	return tea.Sequence(cmds...)
}

func timestampLabel(now time.Time) string {
	return metaStyle.Render(fmt.Sprintf("[%s]", now.Format("15:04:05")))
}
