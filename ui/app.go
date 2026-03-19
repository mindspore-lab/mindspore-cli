package ui

import (
	"fmt"
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
	bootDuration   = 2 * time.Second
	bootTickRate   = 80 * time.Millisecond
	maxToolLines   = 120
	maxToolRunes   = 12000
)

var (
	chatLineStyle     = lipgloss.NewStyle().Foreground(lipgloss.Color("237"))
	trainErrorStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("196"))
	trainSuccessStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("114"))
	trainWorkingStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("220"))
)

// agentMsg formats an agent message with a status marker and fixed-width source prefix.
// done=true → "✓ source      : msg", done=false → "⟳ source      : msg".
// Agent names are right-padded to 12 chars so messages align vertically.
func agentMsg(source, msg string, done bool) string {
	marker := "⟳"
	if done {
		marker = "✓"
	}
	// Strip existing "agent-name: " prefix from msg to avoid duplication.
	if source != "" && strings.HasPrefix(msg, source+": ") {
		msg = strings.TrimPrefix(msg, source+": ")
	}
	if source != "" {
		return fmt.Sprintf("%s %-12s: %s", marker, source, msg)
	}
	return fmt.Sprintf("%s %s", marker, msg)
}

var (
	diffAddStyle     = lipgloss.NewStyle().Foreground(lipgloss.Color("114")) // green
	diffRemoveStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("196")) // red
	diffHunkStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("12"))  // blue
	diffFileStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("220")).Bold(true)
	diffContextStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("244")) // dim
	diffSummaryStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("252")).Bold(true)
)

// formatDiffLine colorizes a single diff line for the agent panel.
func formatDiffLine(line string) string {
	indent := "               " // align with agent message content
	switch {
	case strings.HasPrefix(line, "---") || strings.HasPrefix(line, "+++"):
		return indent + diffFileStyle.Render(line)
	case strings.HasPrefix(line, "@@"):
		return indent + diffHunkStyle.Render(line)
	case strings.HasPrefix(line, "+"):
		return indent + diffAddStyle.Render(line)
	case strings.HasPrefix(line, "-"):
		return indent + diffRemoveStyle.Render(line)
	case strings.Contains(line, "files changed"):
		return indent + diffSummaryStyle.Render(line)
	case line == "":
		return ""
	default:
		return indent + diffContextStyle.Render(line)
	}
}

// evSource extracts ActionSource from a train event, or returns fallback.
func evSource(data *model.TrainEventData, fallback string) string {
	if data != nil && data.ActionSource != "" {
		return data.ActionSource
	}
	return fallback
}

type bootDoneMsg struct{}
type bootTickMsg struct{}

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
	mouseEnabled  bool

	// Train mode
	trainView  model.TrainViewState
	trainFocus model.TrainPanelID

	bootActive    bool
	bootHighlight int
}

// New creates a new App driven by the given event channel.
// userCh may be nil — user input won't be forwarded.
func New(ch <-chan model.Event, userCh chan<- string, version, workDir, repoURL, modelName string, ctxMax int) App {
	return App{
		state:      model.NewState(version, workDir, repoURL, modelName, ctxMax),
		input:      components.NewTextInput(),
		thinking:   components.NewThinkingSpinner(),
		eventCh:    ch,
		userCh:     userCh,
		bootActive: true,
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
		tea.Tick(bootTickRate, func(time.Time) tea.Msg {
			return bootTickMsg{}
		}),
		tea.Tick(bootDuration, func(time.Time) tea.Msg {
			return bootDoneMsg{}
		}),
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

func (a App) trainBodyHeight() int {
	h := a.height - topBarHeight - 1 - hintBarHeight - a.input.Height()
	if h < 1 {
		return 1
	}
	return h
}

func (a App) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd

	switch msg := msg.(type) {

	case tea.KeyMsg:
		if a.bootActive {
			return a, nil
		}
		m, cmd := a.handleKey(msg)
		return m, a.ensureWaitForEvent(cmd)

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
		if a.trainView.Active {
			a.resizeTrainViewport()
		} else {
			a.viewport = a.viewport.SetSize(a.chatWidth()-4, a.chatHeight())
		}
		return a, nil

	case bootTickMsg:
		if !a.bootActive {
			return a, nil
		}
		a.bootHighlight++
		return a, tea.Tick(bootTickRate, func(time.Time) tea.Msg {
			return bootTickMsg{}
		})

	case bootDoneMsg:
		a.bootActive = false
		a.updateViewport()
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

// ensureWaitForEvent wraps a cmd to always include waitForEvent,
// so the UI keeps listening for backend events after key presses.
func (a App) ensureWaitForEvent(cmd tea.Cmd) tea.Cmd {
	if cmd == nil {
		return a.waitForEvent
	}
	return tea.Batch(cmd, a.waitForEvent)
}

// chatWidth returns the width available for the chat panel.
// In the stacked train layout the viewport is full-width.
func (a App) chatWidth() int {
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

	// Train mode focus cycling
	if a.trainView.Active {
		switch msg.String() {
		case "tab":
			return a, a.cycleTrainFocus(1)
		case "shift+tab":
			return a, a.cycleTrainFocus(-1)
		case "c":
			// Collapse only boxed panels (Status, Metrics, Logs).
			switch a.trainFocus {
			case model.TrainPanelStatus, model.TrainPanelMetrics, model.TrainPanelLogs:
				a.trainView.TogglePanelCollapse(a.trainFocus)
				a.resizeTrainViewport()
			}
			return a, nil
		case "z":
			a.trainView.TogglePanelMaximize(a.trainFocus)
			a.resizeTrainViewport()
			return a, nil
		case "esc":
			if a.trainFocus != model.TrainPanelActions {
				return a, a.setTrainFocusPanel(model.TrainPanelActions)
			}
			return a, nil
		}
	}

	// Selection popup navigation
	if a.trainView.Active && a.trainView.SelectionPopup != nil {
		switch msg.String() {
		case "up", "left":
			p := a.trainView.SelectionPopup
			p.Selected--
			if p.Selected < 0 {
				p.Selected = len(p.Options) - 1
			}
			return a, nil
		case "down", "right":
			p := a.trainView.SelectionPopup
			p.Selected = (p.Selected + 1) % len(p.Options)
			return a, nil
		case "enter":
			p := a.trainView.SelectionPopup
			selected := p.Options[p.Selected]
			a.trainView.SelectionPopup = nil
			var input string
			switch p.ActionID {
			case "add_algo_feature":
				input = "add algo-feature " + selected.ID
			case "add_perf_feature":
				input = "add perf-feature " + selected.ID
			}
			if input != "" && a.userCh != nil {
				select {
				case a.userCh <- input:
				default:
				}
			}
			return a, nil
		case "esc":
			a.trainView.SelectionPopup = nil
			return a, nil
		}
		return a, nil
	}

	// Train mode action navigation
	if a.trainView.Active && a.trainFocus == model.TrainPanelActions {
		switch msg.String() {
		case "right", "down":
			if len(a.trainView.GlobalActions.Items) > 0 {
				a.trainView.GlobalActions.SelectedIndex = (a.trainView.GlobalActions.SelectedIndex + 1) % len(a.trainView.GlobalActions.Items)
				return a, nil
			}
		case "left", "up":
			if len(a.trainView.GlobalActions.Items) > 0 {
				a.trainView.GlobalActions.SelectedIndex--
				if a.trainView.GlobalActions.SelectedIndex < 0 {
					a.trainView.GlobalActions.SelectedIndex = len(a.trainView.GlobalActions.Items) - 1
				}
				return a, nil
			}
		case "enter":
			// If user typed text, send it to backend instead of firing button.
			if val := a.input.Value(); val != "" {
				a.state = a.state.ResetStats()
				a.state = a.state.WithThinking(false)
				a.state = a.state.WithMessage(model.Message{Kind: model.MsgUser, Content: val})
				a.input = a.input.Reset()
				a.viewport = a.viewport.SetSize(a.chatWidth()-4, a.chatHeight())
				a.updateViewport()
				// Confirmation words fire the current button action.
				if isConfirmation(val) {
					return a.handleTrainAction()
				}
				if a.userCh != nil {
					select {
					case a.userCh <- val:
					default:
					}
				}
				return a, nil
			}
			return a.handleTrainAction()
		}
	}

	if a.trainView.Active && a.trainFocus == model.TrainPanelRunList {
		switch msg.String() {
		case "down", "right":
			a.trainView.SelectNextRun()
			return a, nil
		case "up", "left":
			a.trainView.SelectPrevRun()
			return a, nil
		case "enter":
			return a, a.setTrainFocusPanel(model.TrainPanelStatus)
		}
	}

	if a.trainView.Active && a.trainFocus == model.TrainPanelStatus {
		switch msg.String() {
		case "enter", "esc":
			return a, a.setTrainFocusPanel(model.TrainPanelActions)
		}
	}

	if a.trainView.Active && a.trainFocus == model.TrainPanelMetrics {
		switch msg.String() {
		case "enter", "esc":
			return a, a.setTrainFocusPanel(model.TrainPanelActions)
		}
	}

	// Train mode agent panel scrolling
	if a.trainView.Active && a.trainFocus == model.TrainPanelAgent {
		switch msg.String() {
		case "up", "k":
			a.viewport.Model.LineUp(1)
			return a, nil
		case "down", "j":
			a.viewport.Model.LineDown(1)
			return a, nil
		case "pgup":
			a.viewport.Model.HalfViewUp()
			return a, nil
		case "pgdown":
			a.viewport.Model.HalfViewDown()
			return a, nil
		case "home", "g":
			a.viewport.Model.GotoTop()
			return a, nil
		case "end", "G":
			a.viewport.Model.GotoBottom()
			return a, nil
		case "enter", "esc":
			return a, a.setTrainFocusPanel(model.TrainPanelActions)
		}
	}

	// Train mode logs panel scrolling
	if a.trainView.Active && a.trainFocus == model.TrainPanelLogs {
		switch msg.String() {
		case "up", "down", "pgup", "pgdown":
			// TODO: implement log viewport scrolling
			return a, nil
		case "enter":
			return a, a.setTrainFocusPanel(model.TrainPanelActions)
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
		if !a.hasThinkingMessage() {
			a.state = a.state.WithMessage(model.Message{Kind: model.MsgThinking})
		}

	case model.AgentReply:
		a.state = a.state.WithThinking(false)
		content := ev.Message
		if ev.Train != nil && ev.Train.IsDiff {
			content = formatDiffLine(ev.Message)
		} else if ev.Train != nil && ev.Train.ActionSource != "" {
			content = agentMsg(ev.Train.ActionSource, ev.Message, false)
		}
		a.state = a.replaceThinking(model.Message{Kind: model.MsgAgent, Content: content})

	case model.ToolCallStart:
		a.state = a.state.WithThinking(false)
		a.state = a.replaceThinking(a.pendingToolMessage(ev))

	case model.CmdStarted:
		stats := a.state.Stats
		stats.Commands++
		a.state = a.state.WithStats(stats)
		a.state = a.resolveToolEvent(ev, model.Message{
			Kind:     model.MsgTool,
			ToolName: "Shell",
			Display:  model.DisplayExpanded,
			Content:  truncateToolContent(ev.Message),
		})

	case model.CmdOutput:
		a.state = a.appendToLastTool(ev.Message)

	case model.CmdFinished:
		// output already in the tool block

	case model.ToolRead:
		stats := a.state.Stats
		stats.FilesRead++
		a.state = a.state.WithStats(stats)
		a.state = a.resolveToolEvent(ev, model.Message{
			Kind: model.MsgTool, ToolName: "Read",
			Display: model.DisplayCollapsed, Content: ev.Message, Summary: ev.Summary,
		})

	case model.ToolGrep:
		stats := a.state.Stats
		stats.Searches++
		a.state = a.state.WithStats(stats)
		a.state = a.resolveToolEvent(ev, model.Message{
			Kind: model.MsgTool, ToolName: "Grep",
			Display: model.DisplayCollapsed, Content: ev.Message, Summary: ev.Summary,
		})

	case model.ToolGlob:
		stats := a.state.Stats
		stats.Searches++
		a.state = a.state.WithStats(stats)
		a.state = a.resolveToolEvent(ev, model.Message{
			Kind: model.MsgTool, ToolName: "Glob",
			Display: model.DisplayCollapsed, Content: ev.Message, Summary: ev.Summary,
		})

	case model.ToolEdit:
		stats := a.state.Stats
		stats.FilesEdited++
		a.state = a.state.WithStats(stats)
		a.state = a.resolveToolEvent(ev, model.Message{
			Kind: model.MsgTool, ToolName: "Edit",
			Display: model.DisplayExpanded, Content: truncateToolContent(ev.Message),
		})

	case model.ToolWrite:
		stats := a.state.Stats
		stats.FilesEdited++
		a.state = a.state.WithStats(stats)
		a.state = a.resolveToolEvent(ev, model.Message{
			Kind: model.MsgTool, ToolName: "Write",
			Display: model.DisplayExpanded, Content: truncateToolContent(ev.Message),
		})

	case model.ToolSkill:
		a.state = a.resolveToolEvent(ev, model.Message{
			Kind: model.MsgTool, ToolName: "Skill",
			Display: model.DisplayCollapsed, Content: ev.Message, Summary: ev.Summary,
		})

	case model.ToolError:
		stats := a.state.Stats
		stats.Errors++
		a.state = a.state.WithStats(stats)
		a.state = a.resolveToolEvent(ev, model.Message{
			Kind: model.MsgTool, ToolName: displayToolName(ev.ToolName),
			Display: model.DisplayError, Content: truncateToolContent(ev.Message),
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
		a.trainFocus = model.TrainPanelActions
		a.input, _ = a.input.Focus()
		a.viewport = a.viewport.SetSize(a.chatWidth()-4, a.chatHeight())

	case model.TrainSetup:
		a.handleTrainSetup(ev)

	case model.TrainConnect:
		a.handleTrainConnect(ev)

	case model.TrainPlanReady:
		if ev.Train != nil {
			a.trainView.SetupContext = model.SetupContext{
				LocalReady:   true,
				TargetReady:  true,
				RepoPath:     ev.Train.RepoPath,
				ScriptPath:   ev.Train.ScriptPath,
				BaseModelRef: ev.Train.BaseModelRef,
				ConfigPath:   ev.Train.ConfigPath,
				EnvKind:      ev.Train.EnvKind,
				Workdir:      ev.Train.Workdir,
				TargetName:   valueOr(ev.Train.Host, a.trainView.Request.TargetName),
			}
			a.trainView.TrainPlan = &model.TrainPlan{
				ID:         ev.Train.PlanID,
				RunID:      trainEventRunID(ev.Train),
				Framework:  valueOr(a.ensureTrainRun(ev.Train).Framework, "PyTorch"),
				RepoSource: ev.Train.RepoSource,
				ScriptPath: ev.Train.ScriptPath,
				BaseModel:  ev.Train.BaseModelRef,
				ConfigPath: ev.Train.ConfigPath,
				EnvKind:    ev.Train.EnvKind,
				Workdir:    ev.Train.Workdir,
				TargetName: valueOr(ev.Train.Host, a.trainView.Request.TargetName),
				Ready:      true,
			}
			a.trainView.RunConfig = &model.RunConfig{
				RunID:      trainEventRunID(ev.Train),
				Model:      valueOr(a.trainView.Request.Model, "bootstrap-model"),
				Method:     valueOr(a.trainView.Request.Mode, "lora"),
				Framework:  valueOr(a.ensureTrainRun(ev.Train).Framework, "PyTorch"),
				Device:     valueOr(a.ensureTrainRun(ev.Train).Device, "Ascend"),
				TargetName: valueOr(ev.Train.Host, a.trainView.Request.TargetName),
				ScriptPath: ev.Train.ScriptPath,
				ConfigPath: ev.Train.ConfigPath,
			}
		}
		a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: agentMsg(evSource(ev.Train, "setup-helper"), ev.Message, true)})

	case model.TrainReady:
		a.trainView.SetStage(model.StageReady)
		a.trainView.SetRunPhase(trainEventRunID(ev.Train), model.TrainPhaseReady)
		if run := a.ensureTrainRun(ev.Train); run != nil {
			run.StatusMessage = ev.Message
		}
		rid := trainEventRunID(ev.Train)
		a.trainView.SetAgentActions(rid, nil)
		if r := a.trainView.RunByID(rid); r != nil {
			r.CurrentIssue = nil
		}
		a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: trainSuccessStyle.Render(agentMsg(evSource(ev.Train, ""), ev.Message, true))})

	case model.TrainStarted:
		a.handleTrainStarted(ev)

	case model.TrainIssueDetected:
		if ev.Train != nil {
			stage := a.trainView.Stage // keep current stage by default
			switch mapIssueKind(ev.Train.IssueType) {
			case model.IssueBootstrap:
				stage = model.StageSetup
			case model.IssueFailure:
				a.trainView.SetRunPhase(trainEventRunID(ev.Train), model.TrainPhaseFailed)
				stage = a.trainView.Stage // use whatever SetRunPhase set
			}
			a.trainView.SetIssue(model.IssueRecord{
				ID:      valueOr(ev.Train.IssueID, "issue-"+trainEventRunID(ev.Train)),
				RunID:   trainEventRunID(ev.Train),
				Kind:    mapIssueKind(ev.Train.IssueType),
				Phase:   string(a.trainView.Stage),
				Summary: valueOr(ev.Message, ev.Train.IssueDetail),
				Signature: map[string]any{
					"type": ev.Train.IssueType,
				},
				Details: map[string]any{
					"title":  ev.Train.IssueTitle,
					"detail": ev.Train.IssueDetail,
				},
			})
			a.trainView.SetStage(stage)
			// Mark the SSH check as failed in the checklist so the setup env panel
			// shows it red during repair (before emitProbeResult, which we skip).
			if ev.Train.IssueID == "bootstrap-target-ssh" {
				a.trainView.UpsertCheck(trainEventRunID(ev.Train), model.ChecklistItem{
					Group:    model.TrainCheckGroupTarget,
					Name:     "ssh",
					Status:   model.TrainCheckFail,
					Summary:  ev.Train.IssueDetail,
					Critical: true,
				})
			}
		}
		if ev.Message != "" {
			a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: trainErrorStyle.Render(agentMsg(evSource(ev.Train, "observer"), ev.Message, false))})
		}

	case model.TrainLogLine:
		a.handleTrainLogLine(ev)

	case model.TrainMetric:
		a.handleTrainMetric(ev)

	case model.TrainDone:
		a.handleTrainDone(ev)

	case model.TrainStopped:
		a.trainView.SetStage(model.StageDone)
		runID := trainEventRunID(ev.Train)
		a.trainView.SetRunPhase(runID, model.TrainPhaseStopped)
		if run := a.trainView.RunByID(runID); run != nil {
			run.StatusMessage = ev.Message
		}
		a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: trainErrorStyle.Render(agentMsg(evSource(ev.Train, "observer"), ev.Message, false))})

	case model.TrainError:
		a.trainView.SetRunPhase(trainEventRunID(ev.Train), model.TrainPhaseFailed)
		if run := a.ensureTrainRun(ev.Train); run != nil {
			run.ErrorMessage = ev.Message
		}
		a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: trainErrorStyle.Render(agentMsg(evSource(ev.Train, "observer"), ev.Message, false))})

	// ── Phase 2 events ──────────────────────────────────────

	case model.TrainEvalStarted:
		a.trainView.SetStage(model.StageRunning)
		a.trainView.SetRunPhase(trainEventRunID(ev.Train), model.TrainPhaseEvaluating)

	case model.TrainEvalCompleted:
		if ev.Train != nil {
			if a.trainView.Compare == nil {
				a.trainView.Compare = &model.CompareViewState{}
			}
			a.trainView.Compare = &model.CompareViewState{
				Enabled:      true,
				LeftRunID:    compareLeftRunID(a.trainView),
				RightRunID:   compareRightRunID(a.trainView),
				BaselineAcc:  ev.Train.BaselineAcc,
				CandidateAcc: ev.Train.CandidateAcc,
				Drift:        ev.Train.Drift,
				Status:       "evaluated",
			}
			a.trainView.Panels[model.TrainPanelCompare].Collapsed = false
		}

	case model.TrainDriftDetected:
		a.trainView.SetStage(model.StageAnalyzing)
		a.trainView.SetRunPhase(trainEventRunID(ev.Train), model.TrainPhaseDriftDetected)
		if ev.Train != nil {
			a.trainView.SetIssue(model.IssueRecord{
				ID:      valueOr(ev.Train.IssueID, "issue-"+trainEventRunID(ev.Train)),
				RunID:   trainEventRunID(ev.Train),
				Kind:    model.IssueAccuracy,
				Phase:   string(a.trainView.Stage),
				Summary: ev.Message,
			})
		}
		if ev.Train != nil && a.trainView.Compare != nil {
			a.trainView.Compare.Status = "mismatch"
		}
		a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: trainErrorStyle.Render(agentMsg(evSource(ev.Train, "observer"), ev.Message, false))})

	case model.TrainAnalysisStarted:
		a.trainView.SetStage(model.StageAnalyzing)
		a.trainView.SetRunPhase(trainEventRunID(ev.Train), model.TrainPhaseAnalyzing)
		a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: agentMsg(evSource(ev.Train, ""), ev.Message, false)})

	case model.TrainAnalyzing:
		a.trainView.SetStage(model.StageAnalyzing)
		a.trainView.SetRunPhase(trainEventRunID(ev.Train), model.TrainPhaseAnalyzing)

	case model.TrainActionSuggested:
		if ev.Train != nil {
			if valueOr(ev.Train.ActionID, "") == "repair-ssh-connectivity" {
				if run := a.ensureTrainRun(ev.Train); run != nil {
					run.StatusMessage = "Fixing..."
				}
				a.trainView.SetStage(model.StageSetup)
				a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: trainWorkingStyle.Render(agentMsg("setup-helper", "fixing ssh connectivity...", false))})
				break
			}
			if valueOr(ev.Train.ActionID, "") == "install-missing-libs" {
				if run := a.ensureTrainRun(ev.Train); run != nil {
					run.StatusMessage = "Installing..."
				}
				a.trainView.SetStage(model.StageSetup)
				a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: trainWorkingStyle.Render(agentMsg("setup-helper", "installing missing library...", false))})
				break
			}
			a.trainView.SetAgentActions(trainEventRunID(ev.Train), []model.AgentAction{
				{
					ID:     valueOr(ev.Train.ActionID, "suggested-action"),
					RunID:  trainEventRunID(ev.Train),
					Kind:   model.AgentActionKind(ev.Train.ActionKind),
					Label:  valueOr(ev.Train.ActionLabel, valueOr(ev.Train.FixSummary, "Suggested action")),
					Source: valueOr(ev.Train.ActionSource, "analysis"),
				},
			})
			if ev.Message != "" {
				a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: trainWorkingStyle.Render(agentMsg(evSource(ev.Train, ""), ev.Message, false))})
			}
			if mapIssueKind(ev.Train.IssueType) == model.IssueBootstrap {
				a.trainView.SetStage(model.StageSetup)
			}
		}

	case model.TrainAnalysisReady:
		a.trainView.SetRunPhase(trainEventRunID(ev.Train), model.TrainPhaseReady)
		a.trainView.SetStage(model.StageAnalyzing) // override: analysis is done but fix not yet applied
		if ev.Train != nil {
			rid := trainEventRunID(ev.Train)
			if r := a.trainView.RunByID(rid); r != nil {
				r.Issue = &model.TrainIssueView{
					Type:       ev.Train.IssueType,
					Title:      ev.Train.IssueTitle,
					Detail:     ev.Train.IssueDetail,
					Confidence: ev.Train.Confidence,
					FixSummary: ev.Train.FixSummary,
					DiffText:   ev.Train.DiffText,
				}
			}
			a.trainView.SetAgentActions(rid, []model.AgentAction{
				{
					ID:     valueOr(ev.Train.ActionID, "apply-fix"),
					RunID:  rid,
					Kind:   mapActionKind(ev.Train.IssueType),
					Label:  valueOr(ev.Train.ActionLabel, valueOr(ev.Train.FixSummary, "Apply fix")),
					Source: valueOr(ev.Train.ActionSource, "analysis"),
				},
			})
		}

	case model.TrainFixApplied:
		// Fix is done — clear agent actions, mark fix applied, set to ready so user can rerun.
		rid := trainEventRunID(ev.Train)
		if run := a.trainView.EnsureRun(rid, "", "", "", "", ""); run != nil {
			run.FixApplied = true
			run.AgentActions = nil // clear so RefreshActions shows "rerun" not "apply fix"
			run.StatusMessage = ev.Message
		}
		a.trainView.SetStage(model.StageReady)
		a.trainView.SetRunPhase(rid, model.TrainPhaseReady)
		if ev.Message != "" {
			a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: trainSuccessStyle.Render(agentMsg(evSource(ev.Train, ""), ev.Message, true))})
		}

	case model.TrainActionApplied:
		if ev.Train != nil && mapIssueKind(ev.Train.IssueType) == model.IssueBootstrap {
			// Stay at StageSetup so the setup env panel remains expanded.
			a.trainView.SetStage(model.StageSetup)
			a.trainView.SetAgentActions(trainEventRunID(ev.Train), nil)
			actionID := valueOr(ev.Train.ActionID, "")
			if run := a.ensureTrainRun(ev.Train); run != nil {
				// Preserve the status flag so handleTrainSetup knows what's being repaired.
				if actionID == "install-missing-libs" {
					run.StatusMessage = "Installing..."
				}
				// SSH keeps "Fixing..." (set by TrainActionSuggested)
			}
			// Show download/install progress in agent panel.
			if actionID == "install-missing-libs" && ev.Message != "" {
				a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: trainWorkingStyle.Render(agentMsg(evSource(ev.Train, "setup-helper"), ev.Message, false))})
			}
		} else {
			rid := trainEventRunID(ev.Train)
			a.trainView.SetRunPhase(rid, model.TrainPhaseFixing)
			a.trainView.SetAgentActions(rid, nil)
			if run := a.trainView.EnsureRun(rid, "", "", "", "", ""); run != nil {
				run.StatusMessage = ev.Message
			}
			a.trainView.SetStage(model.StageFixing)
			if ev.Message != "" {
				a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: trainWorkingStyle.Render(agentMsg(evSource(ev.Train, ""), ev.Message, false))})
			}
		}

	case model.TrainRerunStarted:
		a.trainView.SetStage(model.StageRunning)
		a.trainView.SetRunPhase(trainEventRunID(ev.Train), model.TrainPhaseRunning)
		if run := a.ensureTrainRun(ev.Train); run != nil {
			run.RunLabel = ev.Train.RunLabel
			run.LossSeries = nil
			run.Metrics = nil
			run.CurrentMetrics = model.TrainMetricsView{}
			run.Logs.Lines = nil
		}
		a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: agentMsg(evSource(ev.Train, ""), ev.Message, false)})

	case model.TrainVerified:
		a.trainView.SetStage(model.StageDone)
		a.trainView.SetRunPhase(trainEventRunID(ev.Train), model.TrainPhaseCompleted)
		if ev.Train != nil && a.trainView.Compare != nil {
			a.trainView.Compare.CandidateAcc = ev.Train.CandidateAcc
			a.trainView.Compare.Drift = ev.Train.Drift
			a.trainView.Compare.Status = "verified"
		}
		a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: trainSuccessStyle.Render(agentMsg(evSource(ev.Train, ""), ev.Message, true))})

	case model.Done:
		return a, tea.Quit
	}

	// Keep App.trainFocus in sync with model focus (SetRunPhase/SetStage
	// may call SetFocus internally) and resize viewport for stacked layout.
	if a.trainView.Active {
		a.trainFocus = a.trainView.Focus
		a.resizeTrainViewport()
	}

	a.updateViewport()
	if eventCmd != nil {
		return a, tea.Batch(eventCmd, a.waitForEvent)
	}
	return a, a.waitForEvent
}

// handleTrainAction executes the currently focused action button.
func (a App) handleTrainAction() (tea.Model, tea.Cmd) {
	if a.trainView.GlobalActions.SelectedIndex >= len(a.trainView.GlobalActions.Items) {
		return a, nil
	}
	action := a.trainView.GlobalActions.Items[a.trainView.GlobalActions.SelectedIndex]
	if !action.Enabled {
		return a, nil
	}

	// Send the action as text input to the engine bridge
	var input string
	switch action.ID {
	case "start", "rerun":
		input = "start"
	case "stop":
		input = "stop"
	case "retry":
		input = "retry"
	case "close":
		input = "exit"
	case "diagnose":
		input = "analyze"
	case "apply_fix":
		input = "apply fix"
	case "analyze_perf":
		input = "analyze perf"
	case "add_algo_feature":
		a.trainView.SelectionPopup = &model.SelectionPopup{
			Title:    "select algo-feature",
			ActionID: "add_algo_feature",
			Options: []model.SelectionOption{
				{ID: "mhc", Label: "MHC", Desc: "multi-head cascaded attention"},
				{ID: "flash-attn", Label: "Flash Attention", Desc: "memory-efficient fused attention"},
				{ID: "sparse-attn", Label: "Sparse Attention", Desc: "block-sparse attention pattern"},
				{ID: "lora-plus", Label: "LoRA+", Desc: "differential learning rate for A/B"},
				{ID: "galore", Label: "GaLore", Desc: "gradient low-rank projection"},
				{ID: "ddpm-noise", Label: "DDPM Noise Schedule", Desc: "denoising diffusion noise scheduling"},
				{ID: "dpo", Label: "DPO", Desc: "direct preference optimization alignment"},
				{ID: "rope-scaling", Label: "RoPE Scaling", Desc: "rotary position embedding extrapolation"},
				{ID: "moe-routing", Label: "MoE Routing", Desc: "mixture-of-experts dynamic routing"},
			},
		}
		return a, nil
	case "add_perf_feature":
		a.trainView.SelectionPopup = &model.SelectionPopup{
			Title:    "select perf-feature",
			ActionID: "add_perf_feature",
			Options: []model.SelectionOption{
				{ID: "fa2", Label: "Flash Attention v2", Desc: "fused IO-aware attention kernel"},
				{ID: "fused-adam", Label: "Fused Adam", Desc: "single-kernel adam optimizer"},
				{ID: "gradient-ckpt", Label: "Gradient Checkpointing", Desc: "trade compute for memory"},
				{ID: "bf16-mixed", Label: "BF16 Mixed Precision", Desc: "bfloat16 forward + fp32 grads"},
				{ID: "graph-mod", Label: "Graph Mode", Desc: "static graph compilation for NPU"},
				{ID: "comm-overlap", Label: "Communication Overlap", Desc: "overlap allreduce with backward pass"},
				{ID: "zero-offload", Label: "ZeRO Offload", Desc: "offload optimizer states to CPU"},
				{ID: "sequence-parallel", Label: "Sequence Parallel", Desc: "split sequence across devices"},
				{ID: "selective-recompute", Label: "Selective Recompute", Desc: "recompute only attention activations"},
			},
		}
		return a, nil
	case "view_diff":
		input = "view diff"
	case "inspect_logs":
		return a, a.setTrainFocusPanel(model.TrainPanelLogs)
	default:
		// AgentAction buttons (e.g. "fix-dsa-op") → route as "apply fix".
		input = "apply fix"
	}

	if input != "" && a.userCh != nil {
		select {
		case a.userCh <- input:
		default:
		}
	}
	return a, nil
}

// ── Focus management ─────────────────────────────────────────

func (a *App) setTrainFocusPanel(panel model.TrainPanelID) tea.Cmd {
	a.trainFocus = panel
	a.trainView.SetFocus(panel)
	// Panel focus controls navigation ownership only; the input remains available
	// whenever focus is not on the action row so users can keep typing commands.
	if panel == model.TrainPanelActions {
		a.input = a.input.Blur()
		return nil
	}
	var cmd tea.Cmd
	a.input, cmd = a.input.Focus()
	return cmd
}

func (a *App) cycleTrainFocus(direction int) tea.Cmd {
	order := []model.TrainPanelID{
		model.TrainPanelRunList,
		model.TrainPanelStatus,
		model.TrainPanelMetrics,
		model.TrainPanelLogs,
		model.TrainPanelAgent,
		model.TrainPanelActions,
	}
	current := 0
	for i, panel := range order {
		if panel == a.trainFocus {
			current = i
			break
		}
	}
	next := order[(current+direction+len(order))%len(order)]
	return a.setTrainFocusPanel(next)
}

// ── Train event helpers ──────────────────────────────────────

func (a *App) handleTrainModeOpen(ev model.Event) {
	mdl, method := "", ""
	if ev.Train != nil {
		mdl = ev.Train.Model
		method = ev.Train.Method
	}
	if a.trainView.Active && ev.Train != nil && ev.Train.RunID != "" {
		run := a.ensureTrainRun(ev.Train)
		if run != nil {
			run.Phase = model.TrainPhaseSetup
			run.StatusMessage = "Running setup checks..."
			if strings.TrimSpace(ev.Train.RawInput) == "" {
				run.Label = "Bootstrap Run"
			} else {
				run.Label = formatWorkspaceRunLabel(run.ID, ev.Train.RawInput)
			}
			a.trainView.SetActiveRun(run.ID)
			a.trainFocus = a.trainView.Focus
		}
		return
	}
	a.trainView = *model.NewTrainViewState()
	a.trainView.Active = true
	a.trainView.Request = model.TrainRequestSummary{
		RawInput: strings.TrimSpace(valueOr(ev.Train.RawInput, mdl+" "+method)),
		Model:    mdl,
		Mode:     method,
	}
	a.trainView.SetRunPhase("primary", model.TrainPhaseSetup)
	a.trainView.SetStage(model.StageSetup)
	label := "run-1"
	if ev.Train != nil && strings.TrimSpace(ev.Train.RawInput) != "" {
		label = formatWorkspaceRunLabel("primary", ev.Train.RawInput)
	} else if strings.TrimSpace(mdl) == "" && strings.TrimSpace(method) == "" {
		label = "Bootstrap Run"
	}
	run := a.trainView.EnsureRun("primary", label, "PyTorch", "Ascend", "", "primary")
	run.StatusMessage = "Running setup checks..."
	a.trainFocus = a.trainView.Focus
	a.input, _ = a.input.Focus()
	// Resize viewport for stacked layout.
	a.resizeTrainViewport()
}

func (a *App) handleTrainSetup(ev model.Event) {
	if ev.Train == nil {
		return
	}
	run := a.ensureTrainRun(ev.Train)
	if run == nil {
		return
	}
	if run.StatusMessage == "Fixing..." && ev.Train.Check == "ssh" && ev.Train.Status == "passed" {
		run.StatusMessage = ""
		a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: trainSuccessStyle.Render(agentMsg("setup-helper", "ssh connectivity repaired", true))})
	}
	if run.StatusMessage == "Installing..." && ev.Train.Check == "libs" && ev.Train.Status == "passed" {
		run.StatusMessage = ""
		a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: trainSuccessStyle.Render(agentMsg("setup-helper", "missing library installed successfully", true))})
	}
	// Skip checklist update for post-repair failures — the original probe result
	// is re-emitted after auto-resolve returns, but we don't want the UI
	// to briefly show the check as failed again before the recovery EventCheckPassed arrives.
	isPostRepairSSHFail := run.StatusMessage == "Fixing..." && ev.Train.Check == "ssh" &&
		(ev.Train.Status == "failed" || ev.Train.Status == "fail")
	if !isPostRepairSSHFail {
		a.trainView.UpsertCheck(run.ID, model.ChecklistItem{
			Group:    mapTrainGroup(ev.Train.Scope),
			Name:     ev.Train.Check,
			Status:   mapTrainStatus(ev.Train.Status),
			Summary:  ev.Train.Detail,
			Critical: ev.Train.Critical,
		})
	}
	// Post failures to agent viewport, but suppress for checks that have
	// dedicated auto-resolve flows (SSH, libs) — those already show messages
	// via TrainIssueDetected. Also suppress if already being repaired.
	if (ev.Train.Status == "failed" || ev.Train.Status == "fail") &&
		run.StatusMessage != "Fixing..." && run.StatusMessage != "Installing..." &&
		ev.Train.Check != "ssh" && ev.Train.Check != "libs" {
		checkName := displayCheckNameFromEvent(ev.Train.Check)
		msg := fmt.Sprintf("%s failed: %s", checkName, ev.Train.Detail)
		a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: trainErrorStyle.Render(agentMsg("setup-helper", msg, false))})
	}
}

func (a *App) handleTrainConnect(ev model.Event) {
	if ev.Train == nil {
		return
	}
	// Don't clear "Fixing..." here — let handleTrainSetup clear it when ssh passes,
	// so the guard suppresses the post-repair CheckFailed message.
	// Update existing host or append new one
	isNew := true
	for i := range a.trainView.Hosts {
		if a.trainView.Hosts[i].Name == ev.Train.Host {
			a.trainView.Hosts[i].Status = ev.Train.Status
			a.trainView.Hosts[i].Address = ev.Train.Address
			isNew = false
			break
		}
	}
	if isNew {
		a.trainView.Hosts = append(a.trainView.Hosts, model.TrainHostView{
			Name:    ev.Train.Host,
			Address: ev.Train.Address,
			Status:  ev.Train.Status,
		})
		a.trainView.Request.TargetName = ev.Train.Host
		if run := a.ensureTrainRun(ev.Train); run != nil && run.TargetName == "" {
			run.TargetName = ev.Train.Host
		}
	}
	// Connection failures are reported via TrainIssueDetected; no separate message here.
}

func (a *App) handleTrainStarted(ev model.Event) {
	run := a.ensureTrainRun(ev.Train)
	if run == nil {
		return
	}
	a.trainView.SetRunPhase(run.ID, model.TrainPhaseRunning)
	a.trainView.SetStage(model.StageRunning)
	run.StatusMessage = ev.Message
	run.RunLabel = ev.Train.RunLabel
	a.trainView.SetActiveRun(run.ID)
}

func (a *App) handleTrainLogLine(ev model.Event) {
	a.trainView.AppendLog(trainEventRunID(ev.Train), ev.Message)
	// Auto-expand logs panel so the user sees new output.
	if p := a.trainView.Panels[model.TrainPanelLogs]; p != nil && p.Collapsed {
		p.Collapsed = false
	}
}

func (a *App) handleTrainMetric(ev model.Event) {
	if ev.Train == nil {
		return
	}
	run := a.ensureTrainRun(ev.Train)
	if run == nil {
		return
	}
	// Auto-expand metrics panel so the user sees live updates.
	if p := a.trainView.Panels[model.TrainPanelMetrics]; p != nil && p.Collapsed {
		p.Collapsed = false
	}
	run.CurrentMetrics = model.TrainMetricsView{
		Step:       ev.Train.Step,
		TotalSteps: ev.Train.TotalSteps,
		Loss:       ev.Train.Loss,
		LR:         ev.Train.LR,
		Throughput: ev.Train.Throughput,
	}
	a.trainView.UpsertMetric(run.ID, "step", formatMetricValue("step", ev.Train))
	a.trainView.UpsertMetric(run.ID, "loss", formatMetricValue("loss", ev.Train))
	a.trainView.UpsertMetric(run.ID, "lr", formatMetricValue("lr", ev.Train))
	a.trainView.UpsertMetric(run.ID, "throughput", formatMetricValue("throughput", ev.Train))
	run.LossSeries = append(run.LossSeries,
		model.TrainPoint{Step: ev.Train.Step, Value: ev.Train.Loss})
}

func (a *App) handleTrainDone(ev model.Event) {
	runID := trainEventRunID(ev.Train)
	a.trainView.SetRunPhase(runID, model.TrainPhaseCompleted)
	a.trainView.SetStage(model.StageDone)
	if run := a.trainView.RunByID(runID); run != nil {
		run.StatusMessage = ev.Message
	}
}

func mapTrainStatus(status string) model.TrainCheckStatus {
	switch status {
	case "passed", "pass":
		return model.TrainCheckPass
	case "failed", "fail":
		return model.TrainCheckFail
	case "checking":
		return model.TrainCheckRunning
	default:
		return model.TrainCheckPending
	}
}

func mapTrainGroup(scope string) model.TrainCheckGroup {
	if scope == string(model.TrainCheckGroupTarget) {
		return model.TrainCheckGroupTarget
	}
	return model.TrainCheckGroupLocal
}

func mapIssueKind(issueType string) model.IssueKind {
	switch issueType {
	case "bootstrap":
		return model.IssueBootstrap
	case "failure", "runtime":
		return model.IssueFailure
	case "accuracy":
		return model.IssueAccuracy
	case "performance":
		return model.IssuePerformance
	default:
		return model.IssueFailure
	}
}

func mapActionKind(issueType string) model.AgentActionKind {
	switch issueType {
	case "accuracy":
		return model.ActionApplyPatch
	case "performance":
		return model.ActionChangeConfig
	default:
		return model.ActionChangeEnv
	}
}

func formatMetricValue(name string, data *model.TrainEventData) string {
	switch name {
	case "step":
		return fmt.Sprintf("%d/%d", data.Step, data.TotalSteps)
	case "loss":
		return fmt.Sprintf("%.4f", data.Loss)
	case "lr":
		return fmt.Sprintf("%.1e", data.LR)
	case "throughput":
		return fmt.Sprintf("%.0f tok/s", data.Throughput)
	default:
		return ""
	}
}

func trainEventRunID(data *model.TrainEventData) string {
	if data == nil {
		return "primary"
	}
	if data.RunID != "" {
		return data.RunID
	}
	switch data.Lane {
	case "gpu":
		return "torch_npu"
	case "npu":
		return "mindspore_npu"
	default:
		return "primary"
	}
}

func (a *App) ensureTrainRun(data *model.TrainEventData) *model.TrainRunState {
	runID := trainEventRunID(data)
	label, framework, device, targetName, role := inferRunMeta(runID, data, a.trainView.Request.TargetName)
	run := a.trainView.EnsureRun(runID, label, framework, device, targetName, role)
	if run.TargetName == "" {
		run.TargetName = targetName
	}
	return run
}

func inferRunMeta(runID string, data *model.TrainEventData, defaultTarget string) (label, framework, device, targetName, role string) {
	if data != nil && strings.TrimSpace(data.RawInput) != "" {
		label = data.RawInput
	}
	switch runID {
	case "torch_npu":
		return valueOr(label, "Torch / NPU"), "PyTorch", "Ascend", valueOr(dataHost(data), "torch-npu-910b-0"), "baseline"
	case "mindspore_npu":
		return valueOr(label, "MindSpore / NPU"), "MindSpore", "Ascend", valueOr(dataHost(data), "mindspore-npu-910b-0"), "candidate"
	default:
		target := defaultTarget
		if data != nil && data.Host != "" {
			target = data.Host
		}
		fallback := formatWorkspaceRunLabel(runID, "")
		if runID != "primary" {
			fallback = formatWorkspaceRunLabel(runID, "")
		}
		return valueOr(label, fallback), "PyTorch", "Ascend", target, "primary"
	}
}

func dataHost(data *model.TrainEventData) string {
	if data == nil {
		return ""
	}
	return data.Host
}

// isConfirmation returns true if the input is a confirmation word
// that should fire the current focused button action.
func isConfirmation(input string) bool {
	switch strings.ToLower(strings.TrimSpace(input)) {
	case "yes", "ok", "do it", "go ahead", "confirm", "sure", "yep", "y":
		return true
	}
	return false
}

func valueOr(v, fallback string) string {
	if strings.TrimSpace(v) != "" {
		return v
	}
	return fallback
}

func displayCheckNameFromEvent(name string) string {
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
		return "ssh"
	case "target_os":
		return "target os"
	case "target_aiframework":
		return "target libs"
	case "target_workdir":
		return "workdir"
	case "target_algo":
		return "script/config"
	case "target_gpu":
		return "gpu"
	case "target_npu":
		return "npu"
	case "code_source":
		return "code source"
	case "runtime_env":
		return "runtime env"
	default:
		return name
	}
}

func formatWorkspaceRunLabel(runID, rawInput string) string {
	index := "1"
	if runID != "" && runID != "primary" {
		index = strings.TrimPrefix(runID, "run-")
		if index == "" || index == runID {
			index = runID
		}
	}
	base := "run-" + index
	rawInput = strings.TrimSpace(rawInput)
	if rawInput == "" {
		return base
	}
	return base + " [" + rawInput + "]"
}

func compareLeftRunID(tv model.TrainWorkspaceState) string {
	runs := compareRuns(tv)
	if len(runs) > 0 {
		return runs[0].ID
	}
	return ""
}

func compareRightRunID(tv model.TrainWorkspaceState) string {
	runs := compareRuns(tv)
	if len(runs) > 1 {
		return runs[1].ID
	}
	return ""
}

func compareRuns(tv model.TrainWorkspaceState) []model.TrainRunState {
	var baseline *model.TrainRunState
	var candidate *model.TrainRunState
	nonPrimary := make([]model.TrainRunState, 0, len(tv.Runs))

	for i := range tv.Runs {
		run := tv.Runs[i]
		switch run.Role {
		case "baseline":
			if baseline == nil {
				baseline = &run
			}
		case "candidate":
			if candidate == nil {
				candidate = &run
			}
		}
		if run.Role != "primary" {
			nonPrimary = append(nonPrimary, run)
		}
	}

	if baseline != nil && candidate != nil {
		return []model.TrainRunState{*baseline, *candidate}
	}
	if len(nonPrimary) >= 2 {
		return nonPrimary[:2]
	}
	return tv.Runs
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

func (a App) hasThinkingMessage() bool {
	for i := len(a.state.Messages) - 1; i >= 0; i-- {
		if a.state.Messages[i].Kind == model.MsgThinking {
			return true
		}
	}
	return false
}

func (a App) appendToLastTool(line string) model.State {
	msgs := make([]model.Message, len(a.state.Messages))
	copy(msgs, a.state.Messages)

	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Kind == model.MsgTool {
			content := msgs[i].Content
			if content == "" {
				content = line
			} else {
				content += "\n" + line
			}
			msgs[i] = model.Message{
				Kind:     model.MsgTool,
				ToolName: msgs[i].ToolName,
				Display:  msgs[i].Display,
				Content:  truncateToolContent(content),
				Summary:  msgs[i].Summary,
				Pending:  false,
			}
			break
		}
	}

	next := a.state
	next.Messages = msgs
	return next
}

func (a App) pendingToolMessage(ev model.Event) model.Message {
	toolName := displayToolName(ev.ToolName)
	summary := "running..."
	display := model.DisplayCollapsed
	switch ev.ToolName {
	case "shell":
		display = model.DisplayExpanded
		summary = "running command..."
	case "edit", "write":
		display = model.DisplayExpanded
		summary = "applying changes..."
	case "load_skill":
		toolName = "Skill"
		summary = "loading skill..."
	}
	content := ev.Message
	if ev.ToolName == "shell" && !strings.HasPrefix(strings.TrimSpace(content), "$ ") {
		content = "$ " + content
	}
	return model.Message{
		Kind:     model.MsgTool,
		ToolName: toolName,
		Display:  display,
		Content:  content,
		Summary:  summary,
		Pending:  true,
	}
}

func (a App) resolveToolEvent(ev model.Event, fallback model.Message) model.State {
	msgs := make([]model.Message, len(a.state.Messages))
	copy(msgs, a.state.Messages)

	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Kind != model.MsgTool || !msgs[i].Pending {
			continue
		}
		msgs[i] = finalizeToolMessage(msgs[i], ev)
		next := a.state
		next.Messages = msgs
		return next
	}

	fallback.Pending = false
	next := a.state
	next.Messages = append(msgs, fallback)
	return next
}

func finalizeToolMessage(pending model.Message, ev model.Event) model.Message {
	switch ev.Type {
	case model.CmdStarted:
		return model.Message{
			Kind:     model.MsgTool,
			ToolName: valueOrString(pending.ToolName, "Shell"),
			Display:  model.DisplayExpanded,
			Content:  truncateToolContent(ev.Message),
			Summary:  ev.Summary,
		}
	case model.ToolEdit, model.ToolWrite:
		return model.Message{
			Kind:     model.MsgTool,
			ToolName: pending.ToolName,
			Display:  model.DisplayExpanded,
			Content:  truncateToolContent(ev.Message),
			Summary:  ev.Summary,
		}
	case model.ToolRead, model.ToolGrep, model.ToolGlob, model.ToolSkill:
		content := pending.Content
		if strings.TrimSpace(content) == "" {
			content = ev.Message
		}
		return model.Message{
			Kind:     model.MsgTool,
			ToolName: pending.ToolName,
			Display:  model.DisplayCollapsed,
			Content:  content,
			Summary:  firstNonEmpty(ev.Summary, pending.Summary),
		}
	case model.ToolError:
		toolName := pending.ToolName
		if toolName == "" {
			toolName = displayToolName(ev.ToolName)
		}
		return model.Message{
			Kind:     model.MsgTool,
			ToolName: toolName,
			Display:  model.DisplayError,
			Content:  truncateToolContent(ev.Message),
		}
	default:
		return pending
	}
}

func displayToolName(name string) string {
	switch strings.TrimSpace(name) {
	case "read":
		return "Read"
	case "grep":
		return "Grep"
	case "glob":
		return "Glob"
	case "edit":
		return "Edit"
	case "write":
		return "Write"
	case "shell":
		return "Shell"
	case "load_skill":
		return "Skill"
	default:
		if name == "" {
			return "Tool"
		}
		return name
	}
}

func truncateToolContent(content string) string {
	content = strings.ReplaceAll(content, "\r\n", "\n")
	runes := []rune(content)
	truncatedByRunes := false
	if len(runes) > maxToolRunes {
		runes = runes[:maxToolRunes]
		content = string(runes)
		truncatedByRunes = true
	}

	lines := strings.Split(content, "\n")
	truncatedByLines := false
	if len(lines) > maxToolLines {
		lines = lines[:maxToolLines]
		content = strings.Join(lines, "\n")
		truncatedByLines = true
	}

	if !truncatedByRunes && !truncatedByLines {
		return content
	}

	var parts []string
	if truncatedByLines {
		parts = append(parts, fmt.Sprintf("%d lines", maxToolLines))
	}
	if truncatedByRunes {
		parts = append(parts, fmt.Sprintf("%d chars", maxToolRunes))
	}
	return content + "\n[ui truncated after " + strings.Join(parts, ", ") + "]"
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func valueOrString(value, fallback string) string {
	if strings.TrimSpace(value) == "" {
		return fallback
	}
	return value
}

// agentStatus returns the spinner text for the current agent phase, or "" if idle.
func (a *App) agentStatus() string {
	if !a.trainView.Active {
		if a.state.IsThinking {
			return "thinking..."
		}
		return ""
	}
	run := a.trainView.ActiveRun()
	if run == nil {
		return ""
	}
	switch run.Phase {
	case model.TrainPhaseSetup:
		return "setting up..."
	case model.TrainPhaseRunning:
		return "training..."
	case model.TrainPhaseAnalyzing:
		return "analyzing..."
	case model.TrainPhaseFixing:
		return "applying fix..."
	case model.TrainPhaseEvaluating:
		return "evaluating..."
	}
	return ""
}

func (a *App) updateViewport() {
	// Check if user is at (or near) bottom before updating content.
	atBottom := a.viewport.AtBottom() || a.viewport.TotalLines() <= a.viewport.Model.Height
	width := a.viewport.Model.Width
	if width <= 0 {
		width = a.chatWidth() - 4
	}
	if width < 1 {
		width = 1
	}
	content := panels.RenderMessages(a.state, a.thinking.View(), width, a.trainView.Active)
	a.viewport = a.viewport.SetContent(content)
	// Only auto-scroll to bottom if user hasn't scrolled up.
	if atBottom {
		a.viewport.Model.GotoBottom()
	}
}

func (a App) chatLine() string {
	w := a.chatWidth()
	return chatLineStyle.Render(strings.Repeat("─", w))
}

func (a App) View() string {
	if a.bootActive {
		return panels.RenderBootScreen(a.width, a.height, a.bootHighlight)
	}

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

// trainAvailableHeight returns the vertical budget shared between the
// three train panels (Status, Metrics, Logs) and the agent viewport.
func (a App) trainAvailableHeight() int {
	// Fixed costs: topBar(3) + stageBar(1) + runBar(dynamic) + actionStrip(3) + input + hintBar(2)
	runBarH := panels.TrainRunBarHeight(a.trainView)
	inputH := a.input.Height()
	fixed := topBarHeight + 1 + runBarH + 3 + inputH + hintBarHeight
	avail := a.height - fixed
	if avail < 6 {
		return 6
	}
	return avail
}

// resizeTrainViewport recalculates the viewport size for the stacked layout.
// Must be called from Update() paths that change panel collapse/phase/size.
func (a *App) resizeTrainViewport() {
	if !a.trainView.Active {
		a.viewport = a.viewport.SetSize(a.width-4, a.chatHeight())
		return
	}
	avail := a.trainAvailableHeight()
	ph := panels.StackedPanelHeights(a.trainView, avail)
	vpH := avail - ph[0] - ph[1] - ph[2] - 2 // -2 for agent box border
	if vpH < 3 {
		vpH = 3
	}
	a.viewport = a.viewport.SetSize(a.width-4, vpH)
}

func (a App) renderTrainLayout(topBar string) string {
	w := a.width

	// Maximized panel — full screen.
	if panel, ok := a.trainView.MaximizedPanel(); ok {
		body := panels.RenderTrainWorkspacePanel(panel, a.trainView, w, a.trainBodyHeight())
		input := "  " + a.input.View()
		hintBar := panels.RenderTrainHintBar(w, a.trainFocus, true)
		return trimViewHeight(lipgloss.JoinVertical(lipgloss.Left,
			topBar,
			body,
			chatLineStyle.Render(strings.Repeat("─", w)),
			input,
			hintBar,
		), a.height)
	}

	// ── Stacked full-width layout ────────────────────────────
	avail := a.trainAvailableHeight()
	ph := panels.StackedPanelHeights(a.trainView, avail)
	vpH := avail - ph[0] - ph[1] - ph[2] - 2 // -2 for agent box border
	if vpH < 3 {
		vpH = 3
	}

	stageBar := panels.RenderWorkspaceStatusBar(a.trainView.Stage, w)
	runBarH := panels.TrainRunBarHeight(a.trainView)
	runBar := panels.RenderTrainRunBar(a.trainView, w, runBarH, a.trainFocus == model.TrainPanelRunList)
	statusPanel := panels.RenderTrainWorkspacePanel(model.TrainPanelStatus, a.trainView, w, ph[0])
	metricsPanel := panels.RenderTrainWorkspacePanel(model.TrainPanelMetrics, a.trainView, w, ph[1])
	logsPanel := panels.RenderTrainWorkspacePanel(model.TrainPanelLogs, a.trainView, w, ph[2])

	// Agent message viewport in a boxed panel.
	vpContent := a.viewport.View()
	agentSpinner := ""
	if status := a.agentStatus(); status != "" {
		agentSpinner = a.thinking.FrameView() + " " + status
	}
	agentBox := panels.RenderAgentBox(vpContent, w, vpH+2, a.trainFocus == model.TrainPanelAgent, a.viewport.TotalLines(), a.viewport.YOffset(), agentSpinner)

	actionStrip := panels.RenderTrainActionStrip(a.trainView, w, a.trainFocus == model.TrainPanelActions)
	input := "  " + a.input.View()
	hintBar := panels.RenderTrainHintBar(w, a.trainFocus)

	layout := trimViewHeight(lipgloss.JoinVertical(lipgloss.Left,
		topBar,
		stageBar,
		runBar,
		statusPanel,
		metricsPanel,
		logsPanel,
		agentBox,
		actionStrip,
		input,
		hintBar,
	), a.height)

	if a.trainView.SelectionPopup != nil {
		layout = overlayPopup(layout, panels.RenderSelectionPopup(a.trainView.SelectionPopup), w, a.height)
	}

	return layout
}

func trimViewHeight(content string, height int) string {
	if height <= 0 {
		return content
	}
	lines := strings.Split(content, "\n")
	if len(lines) > height {
		lines = lines[:height]
	}
	for len(lines) < height {
		lines = append(lines, "")
	}
	return strings.Join(lines, "\n")
}

// overlayPopup centers a popup box on top of existing rendered content.
func overlayPopup(bg, popup string, width, height int) string {
	bgLines := strings.Split(bg, "\n")
	popupLines := strings.Split(popup, "\n")

	popupH := len(popupLines)
	startY := (height - popupH) / 2
	if startY < 0 {
		startY = 0
	}

	for len(bgLines) < height {
		bgLines = append(bgLines, "")
	}

	for i, pLine := range popupLines {
		y := startY + i
		if y >= len(bgLines) {
			break
		}
		pW := lipgloss.Width(pLine)
		padLeft := (width - pW) / 2
		if padLeft < 0 {
			padLeft = 0
		}
		bgLines[y] = strings.Repeat(" ", padLeft) + pLine
	}

	if len(bgLines) > height {
		bgLines = bgLines[:height]
	}
	return strings.Join(bgLines, "\n")
}
