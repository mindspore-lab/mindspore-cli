package ui

import (
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/vigo999/ms-cli/ui/components"
	"github.com/vigo999/ms-cli/ui/model"
	"github.com/vigo999/ms-cli/ui/panels"

	tea "github.com/charmbracelet/bubbletea"
)

const (
	topBarHeight   = 3 // brand line + info line + divider
	chatLineHeight = 2
	hintBarHeight  = 2
	inputHeight    = 1
	verticalPad    = 2
)

var (
	chatLineStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("237"))
	trainSplitChar = lipgloss.NewStyle().Foreground(lipgloss.Color("240")).Render("│")
)

type trainFocusTarget int

const (
	trainFocusInput trainFocusTarget = iota
	trainFocusActions
)

// App is the TUI root model.
type App struct {
	state         model.State
	viewport      components.Viewport
	input         components.TextInput
	thinking      components.ThinkingSpinner
	width         int
	height        int
	eventCh       <-chan model.Event
	userCh        chan<- string // sends user input to the engine bridge
	lastInterrupt time.Time     // track last ctrl+c for double-press exit

	// Train mode
	trainView  model.TrainViewState
	trainFocus trainFocusTarget
}

// New creates a new App driven by the given event channel.
// userCh may be nil (demo mode) — user input won't be forwarded.
func New(ch <-chan model.Event, userCh chan<- string, version, workDir, repoURL, modelName string, ctxMax int) App {
	return App{
		state:    model.NewState(version, workDir, repoURL, modelName, ctxMax),
		input:    components.NewTextInput(),
		thinking: components.NewThinkingSpinner(),
		eventCh:  ch,
		userCh:   userCh,
	}
}

func (a App) waitForEvent() tea.Msg {
	ev, ok := <-a.eventCh
	if !ok {
		return model.Event{Type: model.Done}
	}
	return ev
}

func (a App) Init() tea.Cmd {
	return tea.Batch(
		a.thinking.Tick(),
		a.waitForEvent,
	)
}

func (a App) chatHeight() int {
	h := a.height - topBarHeight - chatLineHeight - hintBarHeight - inputHeight - verticalPad
	// Adjust for input height (including suggestions)
	inputH := a.input.Height()
	if inputH > 1 {
		h -= (inputH - 1)
	}
	if h < 1 {
		return 1
	}
	return h
}

func (a App) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd

	switch msg := msg.(type) {

	case tea.KeyMsg:
		return a.handleKey(msg)

	case tea.MouseMsg:
		if !a.state.MouseEnabled {
			return a, nil
		}
		var cmd tea.Cmd
		a.viewport, cmd = a.viewport.Update(msg)
		return a, cmd

	case tea.WindowSizeMsg:
		a.width = msg.Width
		a.height = msg.Height
		a.viewport = a.viewport.SetSize(a.chatWidth()-4, a.chatHeight())
		return a, nil

	case model.Event:
		return a.handleEvent(msg)

	default:
		var cmd tea.Cmd
		a.thinking, cmd = a.thinking.Update(msg)
		if cmd != nil {
			cmds = append(cmds, cmd)
		}
		a.updateViewport()
	}

	return a, tea.Batch(cmds...)
}

// chatWidth returns the width available for the chat/left panel.
func (a App) chatWidth() int {
	if a.trainView.Active {
		return a.width * 30 / 100
	}
	return a.width
}

func (a App) handleKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	// Check if we're in slash suggestion mode
	if a.input.IsSlashMode() {
		switch msg.String() {
		case "up", "down", "tab", "enter", "esc":
			// Let input handle these for suggestion navigation
			var cmd tea.Cmd
			a.input, cmd = a.input.Update(msg)
			// Recalculate chat height if suggestions changed
			a.viewport = a.viewport.SetSize(a.chatWidth()-4, a.chatHeight())
			return a, cmd
		}
	}

	if a.trainView.Active {
		switch msg.String() {
		case "esc":
			if a.trainFocus == trainFocusActions {
				return a, a.setTrainFocus(trainFocusInput)
			}
			if len(a.trainView.Actions) > 0 {
				return a, a.setTrainFocus(trainFocusActions)
			}
		}
	}

	// Train mode action navigation is driven by explicit focus, not input emptiness.
	if a.trainView.Active && a.trainFocus == trainFocusActions {
		switch msg.String() {
		case "i":
			return a, a.setTrainFocus(trainFocusInput)
		case "right", "down":
			if len(a.trainView.Actions) > 0 {
				a.trainView.FocusedAction = (a.trainView.FocusedAction + 1) % len(a.trainView.Actions)
				return a, nil
			}
		case "left", "up":
			if len(a.trainView.Actions) > 0 {
				a.trainView.FocusedAction--
				if a.trainView.FocusedAction < 0 {
					a.trainView.FocusedAction = len(a.trainView.Actions) - 1
				}
				return a, nil
			}
		case "enter":
			return a.handleTrainAction()
		default:
			if isTextEntryKey(msg) {
				focusCmd := a.setTrainFocus(trainFocusInput)
				var inputCmd tea.Cmd
				a.input, inputCmd = a.input.Update(msg)
				a.viewport = a.viewport.SetSize(a.chatWidth()-4, a.chatHeight())
				return a, tea.Batch(focusCmd, inputCmd)
			}
		}
	}

	switch msg.String() {
	case "ctrl+c":
		now := time.Now()
		// If last ctrl+c was within 1 second, quit
		if now.Sub(a.lastInterrupt) < time.Second {
			return a, tea.Quit
		}
		// Otherwise, cancel current input and show hint
		a.lastInterrupt = now
		a.input = a.input.Reset()
		a.state = a.state.WithMessage(model.Message{
			Kind:    model.MsgAgent,
			Content: "Interrupted. Press Ctrl+C again within 1 second to exit.",
		})
		a.updateViewport()
		return a, nil

	case "enter":
		// Don't process enter if in slash mode (handled above)
		if a.input.IsSlashMode() {
			var cmd tea.Cmd
			a.input, cmd = a.input.Update(msg)
			a.viewport = a.viewport.SetSize(a.chatWidth()-4, a.chatHeight())
			return a, cmd
		}

		val := a.input.Value()
		if val == "" {
			return a, nil
		}
		// Reset stats for new task
		a.state = a.state.ResetStats()
		a.state = a.state.WithThinking(false)
		a.state = a.state.WithMessage(model.Message{Kind: model.MsgUser, Content: val})
		a.input = a.input.Reset()
		a.viewport = a.viewport.SetSize(a.chatWidth()-4, a.chatHeight())
		a.updateViewport()
		if a.userCh != nil {
			select {
			case a.userCh <- val:
			default:
				// drop if buffer full — avoids freezing the UI
			}
		}
		return a, nil

	case "pgup", "pgdown", "home", "end":
		var cmd tea.Cmd
		a.viewport, cmd = a.viewport.Update(msg)
		return a, cmd

	case "up", "down":
		var cmd tea.Cmd
		a.viewport, cmd = a.viewport.Update(msg)
		return a, cmd

	default:
		var cmd tea.Cmd
		a.input, cmd = a.input.Update(msg)
		a.viewport = a.viewport.SetSize(a.chatWidth()-4, a.chatHeight())
		return a, cmd
	}
}

func (a App) handleEvent(ev model.Event) (tea.Model, tea.Cmd) {
	var eventCmd tea.Cmd

	switch ev.Type {
	case model.AgentThinking:
		a.state = a.state.WithThinking(true)
		a.state = a.state.WithMessage(model.Message{Kind: model.MsgThinking})

	case model.AgentReply:
		a.state = a.state.WithThinking(false)
		a.state = a.replaceThinking(model.Message{Kind: model.MsgAgent, Content: ev.Message})

	case model.CmdStarted:
		stats := a.state.Stats
		stats.Commands++
		a.state = a.state.WithStats(stats)
		a.state = a.state.WithMessage(model.Message{
			Kind:     model.MsgTool,
			ToolName: "Shell",
			Display:  model.DisplayExpanded,
			Content:  ev.Message,
		})

	case model.CmdOutput:
		a.state = a.appendToLastTool(ev.Message)

	case model.CmdFinished:
		// output already in the tool block

	case model.ToolRead:
		stats := a.state.Stats
		stats.FilesRead++
		a.state = a.state.WithStats(stats)
		a.state = a.state.WithMessage(model.Message{
			Kind: model.MsgTool, ToolName: "Read",
			Display: model.DisplayCollapsed, Content: ev.Message, Summary: ev.Summary,
		})

	case model.ToolGrep:
		stats := a.state.Stats
		stats.Searches++
		a.state = a.state.WithStats(stats)
		a.state = a.state.WithMessage(model.Message{
			Kind: model.MsgTool, ToolName: "Grep",
			Display: model.DisplayCollapsed, Content: ev.Message, Summary: ev.Summary,
		})

	case model.ToolGlob:
		stats := a.state.Stats
		stats.Searches++
		a.state = a.state.WithStats(stats)
		a.state = a.state.WithMessage(model.Message{
			Kind: model.MsgTool, ToolName: "Glob",
			Display: model.DisplayCollapsed, Content: ev.Message, Summary: ev.Summary,
		})

	case model.ToolEdit:
		stats := a.state.Stats
		stats.FilesEdited++
		a.state = a.state.WithStats(stats)
		a.state = a.state.WithMessage(model.Message{
			Kind: model.MsgTool, ToolName: "Edit",
			Display: model.DisplayExpanded, Content: ev.Message,
		})

	case model.ToolWrite:
		stats := a.state.Stats
		stats.FilesEdited++
		a.state = a.state.WithStats(stats)
		a.state = a.state.WithMessage(model.Message{
			Kind: model.MsgTool, ToolName: "Write",
			Display: model.DisplayExpanded, Content: ev.Message,
		})

	case model.ToolError:
		stats := a.state.Stats
		stats.Errors++
		a.state = a.state.WithStats(stats)
		a.state = a.state.WithMessage(model.Message{
			Kind: model.MsgTool, ToolName: ev.ToolName,
			Display: model.DisplayError, Content: ev.Message,
		})

	case model.AnalysisReady:
		a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: ev.Message})

	case model.TokenUpdate:
		mi := a.state.Model
		mi.CtxUsed = ev.CtxUsed
		mi.CtxMax = ev.CtxMax
		mi.TokensUsed = ev.TokensUsed
		a.state = a.state.WithModel(mi)

	case model.TaskUpdated:
		// no-op for now

	case model.ClearScreen:
		a.state.Messages = []model.Message{
			{Kind: model.MsgAgent, Content: ev.Message},
		}

	case model.ModelUpdate:
		mi := a.state.Model
		mi.Name = ev.Message
		a.state = a.state.WithModel(mi)

	case model.MouseModeToggle:
		enabled := a.state.MouseEnabled
		switch strings.ToLower(strings.TrimSpace(ev.Message)) {
		case "", "toggle":
			enabled = !enabled
		case "on", "enable", "enabled", "true", "1":
			enabled = true
		case "off", "disable", "disabled", "false", "0":
			enabled = false
		}
		a.state = a.state.WithMouseEnabled(enabled)
		if enabled {
			eventCmd = tea.EnableMouseCellMotion
		} else {
			eventCmd = tea.DisableMouse
		}

	// ── Train events ──────────────────────────────────────────

	case model.TrainModeOpen:
		a.handleTrainModeOpen(ev)

	case model.TrainModeClose:
		a.trainView = model.TrainViewState{}
		a.trainFocus = trainFocusInput
		a.input, _ = a.input.Focus()
		a.viewport = a.viewport.SetSize(a.chatWidth()-4, a.chatHeight())

	case model.TrainSetup:
		a.handleTrainSetup(ev)

	case model.TrainConnect:
		a.handleTrainConnect(ev)

	case model.TrainReady:
		a.trainView.UpdatePhase("ready")
		a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: ev.Message})

	case model.TrainStarted:
		a.handleTrainStarted(ev)

	case model.TrainLogLine:
		a.handleTrainLogLine(ev)

	case model.TrainMetric:
		a.handleTrainMetric(ev)

	case model.TrainDone:
		a.handleTrainDone(ev)

	case model.TrainError:
		a.trainView.UpdatePhase("failed")
		a.state = a.state.WithMessage(model.Message{
			Kind: model.MsgTool, ToolName: "Train",
			Display: model.DisplayError, Content: ev.Message,
		})

	case model.TrainLaunchFailed:
		a.handleTrainLaunchFailed(ev)

	// ── Phase 2 events ──────────────────────────────────────

	case model.TrainEvalStarted:
		a.trainView.UpdatePhase("evaluating")

	case model.TrainEvalCompleted:
		if ev.Train != nil {
			a.trainView.Compare = &model.TrainCompareView{
				BaselineAcc:  ev.Train.BaselineAcc,
				CandidateAcc: ev.Train.CandidateAcc,
				Drift:        ev.Train.Drift,
				Status:       "evaluated",
			}
		}

	case model.TrainDriftDetected:
		a.trainView.UpdatePhase("drift_detected")
		// Reset issue for new problem cycle
		a.trainView.Issue = nil
		if ev.Train != nil && a.trainView.Compare != nil {
			a.trainView.Compare.Status = "mismatch"
		}
		a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: ev.Message})

	case model.TrainAnalyzing:
		a.trainView.UpdatePhase("analyzing")

	case model.TrainAnalysisReady:
		a.trainView.UpdatePhase("fix_ready")
		if ev.Train != nil {
			a.trainView.Issue = &model.TrainIssueView{
				Type:       ev.Train.IssueType,
				Title:      ev.Train.IssueTitle,
				Detail:     ev.Train.IssueDetail,
				Confidence: ev.Train.Confidence,
				FixSummary: ev.Train.FixSummary,
				DiffText:   ev.Train.DiffText,
			}
			// Re-derive actions now that Issue is set
			a.trainView.UpdatePhase("fix_ready")
		}

	case model.TrainFixApplied:
		// Fix log lines go to the NPU lane via LogLine events

	case model.TrainRerunStarted:
		a.trainView.UpdatePhase("rerunning")
		if ev.Train != nil {
			a.trainView.NPULane.Status = "rerunning"
			a.trainView.NPULane.RunLabel = ev.Train.RunLabel
			// Reset NPU lane series for the new run
			a.trainView.NPULane.LossSeries = nil
			a.trainView.NPULane.Metrics = model.TrainMetricsView{}
		}
		a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: ev.Message})

	case model.TrainVerified:
		a.trainView.UpdatePhase("verified")
		if ev.Train != nil && a.trainView.Compare != nil {
			a.trainView.Compare.CandidateAcc = ev.Train.CandidateAcc
			a.trainView.Compare.Drift = ev.Train.Drift
			a.trainView.Compare.Status = "verified"
		}
		a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: ev.Message})

	case model.Done:
		return a, tea.Quit
	}

	a.updateViewport()
	if eventCmd != nil {
		return a, tea.Batch(eventCmd, a.waitForEvent)
	}
	return a, a.waitForEvent
}

// handleTrainAction executes the currently focused action button.
func (a App) handleTrainAction() (tea.Model, tea.Cmd) {
	if a.trainView.FocusedAction >= len(a.trainView.Actions) {
		return a, nil
	}
	action := a.trainView.Actions[a.trainView.FocusedAction]
	if action.Disabled {
		return a, nil
	}

	// Send the action as text input to the engine bridge
	var input string
	switch action.Action {
	case model.ActionStart:
		input = "start"
	case model.ActionStop:
		input = "stop"
	case model.ActionRetry:
		input = "retry"
	case model.ActionClose:
		input = "exit"
	case model.ActionAnalyze:
		input = "analyze"
	case model.ActionApplyFix:
		input = "apply fix"
	case model.ActionViewDiff:
		input = "view diff"
	}

	if input != "" && a.userCh != nil {
		select {
		case a.userCh <- input:
		default:
		}
	}
	return a, nil
}

func (a *App) setTrainFocus(target trainFocusTarget) tea.Cmd {
	a.trainFocus = target
	a.trainView.ActionRowFocused = target == trainFocusActions
	if target == trainFocusActions {
		a.input = a.input.Blur()
		return nil
	}
	var cmd tea.Cmd
	a.input, cmd = a.input.Focus()
	return cmd
}

func isTextEntryKey(msg tea.KeyMsg) bool {
	if len(msg.Runes) > 0 {
		return true
	}
	switch msg.String() {
	case "backspace", "delete", "space":
		return true
	default:
		return false
	}
}

// ── Train event helpers ──────────────────────────────────────

func (a *App) handleTrainModeOpen(ev model.Event) {
	mdl, method := "", ""
	if ev.Train != nil {
		mdl = ev.Train.Model
		method = ev.Train.Method
	}
	a.trainView = model.NewTrainViewState(mdl, method)
	a.trainFocus = trainFocusInput
	a.trainView.ActionRowFocused = false
	a.input, _ = a.input.Focus()
	// Resize chat viewport for split layout
	a.viewport = a.viewport.SetSize(a.chatWidth()-4, a.chatHeight())
}

func (a *App) handleTrainSetup(ev model.Event) {
	if ev.Train == nil {
		return
	}
	for i := range a.trainView.Checks {
		if a.trainView.Checks[i].Name == ev.Train.Check {
			a.trainView.Checks[i].Status = ev.Train.Status
			a.trainView.Checks[i].Detail = ev.Train.Detail
			return
		}
	}
}

func (a *App) handleTrainConnect(ev model.Event) {
	if ev.Train == nil {
		return
	}
	// Update existing host or append new one
	for i := range a.trainView.Hosts {
		if a.trainView.Hosts[i].Name == ev.Train.Host {
			a.trainView.Hosts[i].Status = ev.Train.Status
			a.trainView.Hosts[i].Address = ev.Train.Address
			return
		}
	}
	a.trainView.Hosts = append(a.trainView.Hosts, model.TrainHostView{
		Name:    ev.Train.Host,
		Address: ev.Train.Address,
		Status:  ev.Train.Status,
	})
}

func (a *App) handleTrainStarted(ev model.Event) {
	a.trainView.UpdatePhase("running")
	if ev.Train != nil {
		lane := a.trainView.LaneByID(ev.Train.Lane)
		if lane != nil {
			lane.Status = "running"
			lane.RunLabel = ev.Train.RunLabel
		}
	}
}

func (a *App) handleTrainLogLine(ev model.Event) {
	if ev.Train != nil && ev.Train.Lane != "" {
		lane := a.trainView.LaneByID(ev.Train.Lane)
		if lane != nil {
			lane.AppendLog(ev.Message)
			return
		}
	}
	// Global log — append to both lanes
	a.trainView.GPULane.AppendLog(ev.Message)
	a.trainView.NPULane.AppendLog(ev.Message)
}

func (a *App) handleTrainMetric(ev model.Event) {
	if ev.Train == nil {
		return
	}
	lane := a.trainView.LaneByID(ev.Train.Lane)
	if lane == nil {
		return
	}
	lane.Metrics = model.TrainMetricsView{
		Step:       ev.Train.Step,
		TotalSteps: ev.Train.TotalSteps,
		Loss:       ev.Train.Loss,
		LR:         ev.Train.LR,
		Throughput: ev.Train.Throughput,
	}
	lane.LossSeries = append(lane.LossSeries,
		model.TrainPoint{Step: ev.Train.Step, Value: ev.Train.Loss})
}

func (a *App) handleTrainDone(ev model.Event) {
	if ev.Train != nil {
		lane := a.trainView.LaneByID(ev.Train.Lane)
		if lane != nil {
			lane.Status = "completed"
		}
	}

	// Check if the other lane has already failed — if so, transition to launch_failed
	if a.trainView.GPULane.Status == "completed" && a.trainView.NPULane.Status == "failed" {
		a.trainView.UpdatePhase("launch_failed")
	} else if a.trainView.NPULane.Status == "completed" && a.trainView.GPULane.Status == "failed" {
		a.trainView.UpdatePhase("launch_failed")
	}

	a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: ev.Message})
}

func (a *App) handleTrainLaunchFailed(ev model.Event) {
	// Set NPU lane to failed
	a.trainView.NPULane.Status = "failed"
	// Set issue
	if ev.Train != nil {
		a.trainView.Issue = &model.TrainIssueView{
			Type:   ev.Train.IssueType,
			Title:  ev.Train.IssueTitle,
			Detail: ev.Train.IssueDetail,
		}
	}
	// Phase stays "running" until GPU completes, but we show the failure
	// If GPU is already completed, set phase to launch_failed
	if a.trainView.GPULane.Status == "completed" {
		a.trainView.UpdatePhase("launch_failed")
	}
	a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: ev.Message})
}

// ── Rendering ────────────────────────────────────────────────

func (a App) replaceThinking(m model.Message) model.State {
	msgs := make([]model.Message, 0, len(a.state.Messages))
	for _, msg := range a.state.Messages {
		if msg.Kind != model.MsgThinking {
			msgs = append(msgs, msg)
		}
	}
	msgs = append(msgs, m)
	next := a.state
	next.Messages = msgs
	return next
}

func (a App) appendToLastTool(line string) model.State {
	msgs := make([]model.Message, len(a.state.Messages))
	copy(msgs, a.state.Messages)

	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Kind == model.MsgTool {
			msgs[i] = model.Message{
				Kind:     model.MsgTool,
				ToolName: msgs[i].ToolName,
				Display:  msgs[i].Display,
				Content:  msgs[i].Content + "\n" + line,
			}
			break
		}
	}

	next := a.state
	next.Messages = msgs
	return next
}

func (a *App) updateViewport() {
	content := panels.RenderMessages(a.state, a.thinking.View())
	a.viewport = a.viewport.SetContent(content)
}

func (a App) chatLine() string {
	w := a.chatWidth()
	return chatLineStyle.Render(strings.Repeat("─", w))
}

func (a App) View() string {
	topBar := panels.RenderTopBar(a.state, a.width)

	if a.trainView.Active {
		return a.renderTrainLayout(topBar)
	}

	line := a.chatLine()
	chat := a.viewport.View()
	input := "  " + a.input.View()
	hintBar := panels.RenderHintBar(a.width)

	return lipgloss.JoinVertical(lipgloss.Left,
		topBar,
		line,
		chat,
		line,
		input,
		hintBar,
	)
}

func (a App) renderTrainLayout(topBar string) string {
	leftWidth := a.chatWidth()
	rightWidth := a.width - leftWidth - 1 // 1 col for vertical separator
	contentHeight := a.chatHeight()

	// Left: control/compare/actions panel
	leftPanel := panels.RenderTrainSetup(a.trainView, leftWidth, contentHeight)

	// Right: two lane panels side by side
	rightFull := a.renderDualLanes(rightWidth, contentHeight)

	// Join left and right with vertical separator
	sep := strings.Repeat(trainSplitChar+"\n", contentHeight)
	if len(sep) > 0 {
		sep = sep[:len(sep)-1]
	}
	body := lipgloss.JoinHorizontal(lipgloss.Top, leftPanel, sep, rightFull)

	input := "  " + a.input.View()
	hintBar := panels.RenderHintBar(a.width)

	return lipgloss.JoinVertical(lipgloss.Left,
		topBar,
		body,
		chatLineStyle.Render(strings.Repeat("─", a.width)),
		input,
		hintBar,
	)
}

// renderDualLanes renders the right side as two lane panels side by side.
func (a App) renderDualLanes(width, height int) string {
	// Split right side: GPU lane (left) | separator | NPU lane (right)
	laneWidth := (width - 1) / 2 // 1 col for separator
	if laneWidth < 15 {
		laneWidth = 15
	}
	npuWidth := width - laneWidth - 1

	gpuPanel := panels.RenderLanePanel(a.trainView.GPULane, laneWidth, height)
	npuPanel := panels.RenderLanePanel(a.trainView.NPULane, npuWidth, height)

	// Vertical separator between lanes
	sep := strings.Repeat(trainSplitChar+"\n", height)
	if len(sep) > 0 {
		sep = sep[:len(sep)-1]
	}

	return lipgloss.JoinHorizontal(lipgloss.Top, gpuPanel, sep, npuPanel)
}
