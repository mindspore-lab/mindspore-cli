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
	topBarHeight              = 1 // brand line only
	chatLineHeight            = 0
	hintBarHeight             = 1
	inputHeight               = 1
	bottomSafePadding         = 2
	verticalPad               = 2
	bootDuration              = 2 * time.Second
	bootTickRate              = 80 * time.Millisecond
	bootReadyToken            = "__boot_ready__"
	maxToolLines              = 120
	maxToolRunes              = 12000
	interruptQueuedTrainToken = "__interrupt_queued_train__"
	hintRotateInterval        = 4 * time.Second
	interruptActiveTaskToken  = "__interrupt_active_task__"
)

var (
	chatLineStyle     = lipgloss.NewStyle().Foreground(lipgloss.Color("237"))
	trainErrorStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("196"))
	trainSuccessStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("114"))
	trainWorkingStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("220"))
	queueBannerStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("220")).PaddingLeft(2)
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
	source := fallback
	if data != nil && data.ActionSource != "" {
		source = data.ActionSource
	}
	if source == "setup-helper" {
		return "setup-agent"
	}
	return source
}

type bootDoneMsg struct{}
type bootTickMsg struct{}
type hintRotateMsg struct{}

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
	trainView     model.TrainViewState
	trainFocus    model.TrainPanelID
	bugView       model.BugViewState
	issueView     model.IssueViewState
	bootActive    bool
	bootHighlight int
	queuedInputs  []string
	hintNoteIndex int // which hint note to show (0=skills, 1=release)
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
	if a.userCh != nil {
		select {
		case a.userCh <- bootReadyToken:
		default:
		}
	}
	return tea.Batch(
		a.thinking.Tick(),
		tea.Tick(bootTickRate, func(time.Time) tea.Msg {
			return bootTickMsg{}
		}),
		tea.Tick(bootDuration, func(time.Time) tea.Msg {
			return bootDoneMsg{}
		}),
		tea.Tick(hintRotateInterval, func(time.Time) tea.Msg {
			return hintRotateMsg{}
		}),
		a.waitForEvent,
	)
}

func (a *App) refreshHintNote() {
	notes := make([]string, 0, 2)
	if a.state.SkillsNote != "" {
		notes = append(notes, a.state.SkillsNote)
	}
	if a.state.ReleaseNote != "" {
		notes = append(notes, a.state.ReleaseNote)
	}
	if len(notes) == 0 {
		a.state.HintNote = ""
		return
	}
	a.state.HintNote = notes[a.hintNoteIndex%len(notes)]
}

func (a App) chatHeight() int {
	h := a.height - topBarHeight - chatLineHeight - hintBarHeight - a.input.Height()
	h -= a.activeHUDHeight()
	h -= a.queueBannerHeight()
	h -= bottomSafePadding
	if h < 1 {
		return 1
	}
	return h
}

func (a App) queueBannerHeight() int {
	if len(a.queuedInputs) == 0 {
		return 0
	}
	return 1
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
		var cmd tea.Cmd
		a.viewport, cmd = a.viewport.Update(msg)
		return a, cmd

	case tea.WindowSizeMsg:
		a.width = msg.Width
		a.height = msg.Height
		a.resizeActiveLayout()
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

	case hintRotateMsg:
		a.hintNoteIndex++
		a.refreshHintNote()
		return a, tea.Tick(hintRotateInterval, func(time.Time) tea.Msg {
			return hintRotateMsg{}
		})

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

func (a *App) resizeInput() {
	inputWidth := a.chatWidth() - 4
	if inputWidth < 1 {
		inputWidth = 1
	}
	a.input = a.input.SetWidth(inputWidth)
}

func (a *App) resizeActiveLayout() {
	a.resizeInput()
	a.viewport = a.viewport.SetSize(a.chatWidth()-4, a.chatHeight())
}

func (a App) handleKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	if msg.String() == "ctrl+c" {
		now := time.Now()
		if now.Sub(a.lastInterrupt) < time.Second {
			return a, tea.Quit
		}
		a.lastInterrupt = now
		if a.userCh != nil {
			select {
			case a.userCh <- interruptActiveTaskToken:
			default:
			}
		}
		a.state = a.clearThinking()
		a.input = a.input.Reset()
		a.resizeActiveLayout()
		a.state = a.state.WithMessage(model.Message{
			Kind:    model.MsgAgent,
			Content: "Interrupt requested. Press Ctrl+C again within 1 second to exit.",
		})
		a.updateViewport()
		return a, nil
	}

	if a.issueView.Active() {
		return a.handleIssueKey(msg)
	}

	if a.bugView.Active() {
		return a.handleBugKey(msg)
	}

	// Check if we're in slash suggestion mode
	if a.input.IsSlashMode() {
		switch msg.String() {
		case "tab", "esc":
			var cmd tea.Cmd
			a.input, cmd = a.input.Update(msg)
			a.resizeActiveLayout()
			return a, cmd
		case "up", "down":
			// Only capture for suggestions if there are visible candidates
			if a.input.HasSuggestions() {
				var cmd tea.Cmd
				a.input, cmd = a.input.Update(msg)
				return a, cmd
			}
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
				input = "/train add algo-feature " + selected.ID
			case "add_perf_feature":
				input = "/train add perf-feature " + selected.ID
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

	if a.trainView.Active && strings.TrimSpace(a.input.Value()) == "" && len(a.trainView.GlobalActions.Items) > 0 {
		switch msg.String() {
		case "tab", "right":
			a.selectTrainAction(1)
			return a, nil
		case "shift+tab", "left":
			a.selectTrainAction(-1)
			return a, nil
		}
	}

	switch msg.String() {
	case "esc":
		if len(a.queuedInputs) > 0 && a.trainView.Active && a.isTrainBusy() && strings.TrimSpace(a.input.Value()) == "" && a.userCh != nil {
			select {
			case a.userCh <- "/train exit":
			default:
			}
			return a, nil
		}
		var cmd tea.Cmd
		a.input, cmd = a.input.Update(msg)
		a.resizeActiveLayout()
		return a, cmd

	case "ctrl+j", "shift+enter":
		var cmd tea.Cmd
		a.input, cmd = a.input.Update(msg)
		a.resizeActiveLayout()
		return a, cmd

	case "enter":
		// Don't process enter if in slash mode (handled above)
		if a.input.IsSlashMode() {
			var cmd tea.Cmd
			a.input, cmd = a.input.Update(msg)
			a.resizeActiveLayout()
			return a, cmd
		}

		val := strings.TrimSpace(a.input.Value())
		if val == "" {
			if a.trainView.Active && len(a.trainView.GlobalActions.Items) > 0 {
				return a.handleTrainAction()
			}
			return a, nil
		}
		if a.shouldQueueInput(val) {
			a.queuedInputs = append(a.queuedInputs, val)
			a.input = a.input.PushHistory(val)
			a.input = a.input.Reset()
			a.resizeActiveLayout()
			return a, nil
		}
		// Reset stats for new task
		a.state = a.state.ResetStats()
		a.state = a.state.WithThinking(false)
		if !strings.HasPrefix(val, "/") {
			a.state = a.state.WithMessage(model.Message{Kind: model.MsgUser, Content: val})
		}
		a.input = a.input.PushHistory(val)
		a.input = a.input.Reset()
		a.resizeActiveLayout()
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
		if a.input.CanNavigateHistory(msg.String()) {
			if msg.String() == "up" {
				a.input = a.input.PrevHistory()
			} else {
				a.input = a.input.NextHistory()
			}
			a.resizeActiveLayout()
			return a, nil
		}
		var cmd tea.Cmd
		a.input, cmd = a.input.Update(msg)
		a.resizeActiveLayout()
		return a, cmd

	default:
		var cmd tea.Cmd
		a.input, cmd = a.input.Update(msg)
		a.resizeActiveLayout()
		return a, cmd
	}
}

func (a App) shouldQueueInput(val string) bool {
	if strings.TrimSpace(val) == "" {
		return false
	}
	return a.trainView.Active && a.isTrainBusy()
}

func (a App) isTrainBusy() bool {
	if !a.trainView.Active {
		return false
	}
	run := a.trainView.ActiveRun()
	if run == nil {
		return false
	}
	switch run.Phase {
	case model.TrainPhaseSetup, model.TrainPhaseRunning, model.TrainPhaseAnalyzing, model.TrainPhaseFixing, model.TrainPhaseEvaluating:
		return true
	default:
		return false
	}
}

func (a App) maybeDispatchQueuedInput() App {
	if len(a.queuedInputs) == 0 || a.isTrainBusy() || a.userCh == nil {
		return a
	}
	next := a.queuedInputs[0]
	a.queuedInputs = append([]string{}, a.queuedInputs[1:]...)
	a.state = a.state.ResetStats()
	a.state = a.state.WithThinking(false)
	a.state = a.state.WithMessage(model.Message{Kind: model.MsgUser, Content: next})
	select {
	case a.userCh <- next:
	default:
	}
	a.resizeActiveLayout()
	return a
}

func (a App) handleEvent(ev model.Event) (tea.Model, tea.Cmd) {
	a = a.applyUsageSnapshot(ev)

	var eventCmd tea.Cmd

	switch ev.Type {
	case model.UserInput:
		a.state = a.state.WithMessage(model.Message{Kind: model.MsgUser, Content: ev.Message})
	case model.IssueIndexOpen:
		a.openIssueIndex(ev.IssueView)

	case model.IssueDetailOpen:
		a.openIssueDetail(ev.IssueView)

	case model.BugIndexOpen:
		a.openBugIndex(ev.BugView)

	case model.BugDetailOpen:
		a.openBugDetail(ev.BugView)

	case model.AgentThinking:
		a.state = a.state.WithThinking(true)

	case model.AgentReply:
		a.state = a.state.WithThinking(false)
		a.input = a.input.ClearSlashMode()
		content := ev.Message
		if ev.Train != nil && ev.Train.IsDiff {
			content = formatDiffLine(ev.Message)
		} else if ev.Train != nil && ev.Train.ActionSource != "" {
			content = agentMsg(evSource(ev.Train, ""), ev.Message, false)
		}
		a.state = a.finalizeAgentMessage(content)

	case model.AgentReplyDelta:
		a.state = a.state.WithThinking(false)
		a.state = a.appendToStreamingAgent(ev.Message)

	case model.ToolCallStart:
		a.state = a.state.WithThinking(false)
		a.state = a.commitStreamingAgent()
		a.state = a.state.WithMessage(a.pendingToolMessage(ev))

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
		msg := model.Message{
			Kind:     model.MsgTool,
			ToolName: displayToolName(ev.ToolName),
			Display:  model.DisplayCollapsed,
			Content:  ev.Message,
			Summary:  ev.Summary,
		}
		if strings.TrimSpace(ev.ToolName) == "load_skill" {
			a.state = a.resolveToolEvent(ev, msg)
		} else {
			a.state = a.state.WithMessage(msg)
		}

	case model.ToolError:
		a.state = a.clearThinking()
		stats := a.state.Stats
		stats.Errors++
		a.state = a.state.WithStats(stats)
		a.state = a.resolveToolEvent(ev, model.Message{
			Kind: model.MsgTool, ToolName: displayToolName(ev.ToolName),
			Display: model.DisplayError, Content: truncateToolContent(ev.Message),
		})

	case model.ToolReplay:
		a.state = a.state.WithMessage(replayToolMessage(ev))

	case model.AnalysisReady:
		a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: ev.Message})

	case model.TokenUpdate:
		// usage snapshot is applied before the event switch

	case model.TaskUpdated:
		// no-op for now

	case model.ClearScreen:
		a.state.Messages = []model.Message{
			{Kind: model.MsgAgent, Content: ev.Message},
		}

	case model.ModelUpdate:
		mi := a.state.Model
		mi.Name = ev.Message
		if ev.CtxMax > 0 {
			mi.CtxMax = ev.CtxMax
		}
		a.state = a.state.WithModel(mi)

	case model.IssueUserUpdate:
		a.state = a.state.WithIssueUser(ev.Message)

	case model.ReleaseNoteUpdate:
		a.state.ReleaseNote = ev.Message
		a.refreshHintNote()

	case model.SkillsNoteUpdate:
		a.state.SkillsNote = ev.Message
		a.refreshHintNote()

	// ── Train events ──────────────────────────────────────────

	case model.TrainModeOpen:
		a.handleTrainModeOpen(ev)

	case model.TrainModeClose:
		a.trainView = model.TrainViewState{}
		a.trainFocus = model.TrainPanelActions
		a.input, _ = a.input.Focus()
		a.resizeActiveLayout()

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
				Dataset:    a.trainView.Request.Dataset,
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
		if summary := a.renderTrainSetupSummary(trainEventRunID(ev.Train)); summary != "" {
			a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: summary})
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
	// may call SetFocus internally) and keep the unified layout sized correctly.
	if a.trainView.Active {
		a.trainFocus = a.trainView.Focus
		a.resizeActiveLayout()
	}

	a = a.maybeDispatchQueuedInput()
	a.updateViewport()
	if eventCmd != nil {
		return a, tea.Batch(eventCmd, a.waitForEvent)
	}
	return a, a.waitForEvent
}

func (a App) applyUsageSnapshot(ev model.Event) App {
	if ev.CtxMax <= 0 {
		return a
	}

	mi := a.state.Model
	mi.CtxUsed = ev.CtxUsed
	mi.CtxMax = ev.CtxMax
	mi.TokensUsed = ev.TokensUsed
	a.state = a.state.WithModel(mi)
	return a
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
		input = "/train start"
	case "stop":
		input = "/train stop"
	case "retry":
		input = "/train retry"
	case "close":
		input = "/train exit"
	case "diagnose":
		input = "/train analyze"
	case "apply_fix":
		input = "/train apply fix"
	case "analyze_perf":
		input = "/train analyze perf"
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
		input = "/train view diff"
	case "inspect_logs":
		a.state = a.state.WithMessage(model.Message{
			Kind:    model.MsgAgent,
			Content: "runtime logs now stream in the shared chat area",
		})
		return a, nil
	default:
		// AgentAction buttons (e.g. "fix-dsa-op") → route as "apply fix".
		input = "/train apply fix"
	}

	if input != "" && a.userCh != nil {
		select {
		case a.userCh <- input:
		default:
		}
	}
	return a, nil
}

func (a *App) selectTrainAction(delta int) {
	if len(a.trainView.GlobalActions.Items) == 0 {
		return
	}
	next := a.trainView.GlobalActions.SelectedIndex + delta
	for next < 0 {
		next += len(a.trainView.GlobalActions.Items)
	}
	a.trainView.GlobalActions.SelectedIndex = next % len(a.trainView.GlobalActions.Items)
}

// ── Train event helpers ──────────────────────────────────────

func (a *App) handleTrainModeOpen(ev model.Event) {
	mdl, method := "", ""
	if ev.Train != nil {
		mdl = ev.Train.Model
		method = ev.Train.Method
	}
	if !a.trainView.Active && len(a.trainView.Runs) > 0 {
		a.trainView.Active = true
		if strings.TrimSpace(mdl) != "" {
			a.trainView.Request.Model = mdl
		}
		if strings.TrimSpace(method) != "" {
			a.trainView.Request.Mode = method
		}
		if ev.Train != nil && strings.TrimSpace(ev.Train.RawInput) != "" {
			a.trainView.Request.RawInput = strings.TrimSpace(ev.Train.RawInput)
		}
		if ev.Train != nil && strings.TrimSpace(ev.Train.RunID) != "" {
			a.trainView.SetActiveRun(ev.Train.RunID)
		}
		a.trainFocus = a.trainView.Focus
		a.input, _ = a.input.Focus()
		a.resizeActiveLayout()
		return
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
	dataset := ""
	if ev.Train != nil {
		dataset = parseTrainDataset(ev.Train.RawInput)
	}
	a.trainView.Request = model.TrainRequestSummary{
		RawInput: strings.TrimSpace(valueOr(ev.Train.RawInput, mdl+" "+method)),
		Model:    mdl,
		Mode:     method,
		Dataset:  dataset,
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
	a.resizeActiveLayout()
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
		a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: trainSuccessStyle.Render(agentMsg("setup-agent", "ssh connectivity repaired", true))})
	}
	if run.StatusMessage == "Installing..." && ev.Train.Check == "libs" && ev.Train.Status == "passed" {
		run.StatusMessage = ""
		a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: trainSuccessStyle.Render(agentMsg("setup-agent", "missing library installed successfully", true))})
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
	if msg, style := renderTrainSetupStreamMessage(ev.Train); msg != "" {
		content := agentMsg("setup-agent", msg, style != "working")
		switch style {
		case "success":
			content = trainSuccessStyle.Render(content)
		case "error":
			content = trainErrorStyle.Render(content)
		default:
			content = trainWorkingStyle.Render(content)
		}
		a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: content})
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
	if msg, style := renderTrainConnectStreamMessage(ev.Train); msg != "" {
		content := agentMsg("setup-agent", msg, style != "working")
		switch style {
		case "success":
			content = trainSuccessStyle.Render(content)
		case "error":
			content = trainErrorStyle.Render(content)
		default:
			content = trainWorkingStyle.Render(content)
		}
		a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: content})
	}
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

func (a App) appendToStreamingAgent(delta string) model.State {
	if delta == "" {
		return a.state
	}

	msgs := make([]model.Message, len(a.state.Messages))
	copy(msgs, a.state.Messages)

	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Kind == model.MsgAgent && msgs[i].Streaming {
			msgs[i].Content += delta
			next := a.state
			next.Messages = msgs
			return next
		}
	}

	msgs = append(msgs, model.Message{
		Kind:      model.MsgAgent,
		Content:   delta,
		Streaming: true,
	})
	next := a.state
	next.Messages = msgs
	return next
}

func (a App) finalizeAgentMessage(content string) model.State {
	msgs := make([]model.Message, len(a.state.Messages))
	copy(msgs, a.state.Messages)

	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Kind == model.MsgAgent && msgs[i].Streaming {
			msgs[i].Content = content
			msgs[i].Streaming = false
			next := a.state
			next.Messages = msgs
			return next
		}
	}

	next := a.state
	next.Messages = append(msgs, model.Message{Kind: model.MsgAgent, Content: content})
	return next
}

func (a App) commitStreamingAgent() model.State {
	msgs := make([]model.Message, len(a.state.Messages))
	copy(msgs, a.state.Messages)

	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Kind == model.MsgAgent && msgs[i].Streaming {
			msgs[i].Streaming = false
			next := a.state
			next.Messages = msgs
			return next
		}
	}
	return a.state
}

func (a App) clearThinking() model.State {
	return a.state.WithThinking(false)
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

func replayToolMessage(ev model.Event) model.Message {
	display := model.DisplayCollapsed
	content := ev.Message

	switch strings.TrimSpace(ev.ToolName) {
	case "shell", "edit", "write":
		display = model.DisplayExpanded
		content = truncateToolContent(ev.Message)
	}

	return model.Message{
		Kind:     model.MsgTool,
		ToolName: displayToolName(ev.ToolName),
		Display:  display,
		Content:  content,
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

func renderTrainSetupStreamMessage(data *model.TrainEventData) (string, string) {
	if data == nil {
		return "", ""
	}
	checkName := displayCheckNameFromEvent(data.Check)
	switch strings.ToLower(strings.TrimSpace(data.Status)) {
	case "checking":
		return fmt.Sprintf("checking %s...", checkName), "working"
	case "passed", "pass":
		if strings.TrimSpace(data.Detail) != "" {
			return fmt.Sprintf("%s ok: %s", checkName, data.Detail), "success"
		}
		return fmt.Sprintf("%s ok", checkName), "success"
	case "failed", "fail":
		if strings.TrimSpace(data.Detail) != "" {
			return fmt.Sprintf("%s failed: %s", checkName, data.Detail), "error"
		}
		return fmt.Sprintf("%s failed", checkName), "error"
	default:
		return "", ""
	}
}

func renderTrainConnectStreamMessage(data *model.TrainEventData) (string, string) {
	if data == nil {
		return "", ""
	}
	host := strings.TrimSpace(data.Host)
	addr := strings.TrimSpace(data.Address)
	target := host
	if target == "" {
		target = "target host"
	}
	if addr != "" {
		target += " (" + addr + ")"
	}
	switch strings.ToLower(strings.TrimSpace(data.Status)) {
	case "connecting":
		return "connecting to " + target + "...", "working"
	case "connected":
		return "connected to " + target, "success"
	default:
		return "", ""
	}
}

func (a *App) renderTrainSetupSummary(runID string) string {
	run := a.trainView.RunByID(runID)
	if run == nil {
		return ""
	}
	lines := []string{"setup summary"}
	if header := a.trainSetupSummaryHeader(run); header != "" {
		lines = append(lines, header, "")
	}
	lines = append(lines, "local checks")
	lines = append(lines, formatTrainCheckSummary(a.trainView.ChecksByGroup(run.ID, model.TrainCheckGroupLocal))...)
	lines = append(lines, "")
	lines = append(lines, "target checks")
	lines = append(lines, formatTrainCheckSummary(a.trainView.ChecksByGroup(run.ID, model.TrainCheckGroupTarget))...)
	return renderTrainSummaryBox(lines)
}

func (a *App) trainSetupSummaryHeader(run *model.TrainRunState) string {
	parts := []string{}
	if run != nil && strings.TrimSpace(run.ID) != "" {
		parts = append(parts, "run_id: "+run.ID)
	}
	if machine := appTrainMachineValue(a.trainView, run); strings.TrimSpace(machine) != "" && machine != "-" {
		parts = append(parts, "machine: "+machine)
	}
	if modelName := appTrainModelValue(a.trainView); strings.TrimSpace(modelName) != "" {
		parts = append(parts, "model: "+modelName)
	}
	return strings.Join(parts, " | ")
}

func formatTrainCheckSummary(items []model.ChecklistItem) []string {
	if len(items) == 0 {
		return []string{"  - no checks recorded"}
	}
	lines := make([]string, 0, len(items))
	for _, item := range items {
		label := displayCheckNameFromEvent(item.Name)
		summary := strings.TrimSpace(item.Summary)
		if summary == "" {
			summary = string(item.Status)
		}
		lines = append(lines, fmt.Sprintf("  %s %s: %s", trainCheckStatusMarker(item.Status), label, summary))
	}
	return lines
}

func trainCheckStatusMarker(status model.TrainCheckStatus) string {
	switch status {
	case model.TrainCheckPass:
		return "[x]"
	case model.TrainCheckFail:
		return "[!]"
	case model.TrainCheckRunning:
		return "[~]"
	default:
		return "[ ]"
	}
}

func renderTrainSummaryBox(lines []string) string {
	width := 0
	for _, line := range lines {
		if w := lipgloss.Width(line); w > width {
			width = w
		}
	}
	if width < 24 {
		width = 24
	}
	boxed := make([]string, 0, len(lines)+2)
	boxed = append(boxed, "╭"+strings.Repeat("─", width+2)+"╮")
	for _, line := range lines {
		pad := width - lipgloss.Width(line)
		if pad < 0 {
			pad = 0
		}
		boxed = append(boxed, "│ "+line+strings.Repeat(" ", pad)+" │")
	}
	boxed = append(boxed, "╰"+strings.Repeat("─", width+2)+"╯")
	return strings.Join(boxed, "\n")
}

func appTrainMachineValue(tv model.TrainWorkspaceState, run *model.TrainRunState) string {
	target := ""
	switch {
	case tv.RunConfig != nil && strings.TrimSpace(tv.RunConfig.TargetName) != "":
		target = tv.RunConfig.TargetName
	case run != nil && strings.TrimSpace(run.TargetName) != "":
		target = run.TargetName
	case strings.TrimSpace(tv.SetupContext.TargetName) != "":
		target = tv.SetupContext.TargetName
	case strings.TrimSpace(tv.Request.TargetName) != "":
		target = tv.Request.TargetName
	}
	device := ""
	switch {
	case tv.RunConfig != nil && strings.TrimSpace(tv.RunConfig.Device) != "":
		device = tv.RunConfig.Device
	case run != nil && strings.TrimSpace(run.Device) != "":
		device = run.Device
	}
	device = appNormalizeTrainDevice(device)
	switch {
	case target != "" && device != "":
		return target + " " + device
	case target != "":
		return target
	case device != "":
		return device
	default:
		return ""
	}
}

func appTrainModelValue(tv model.TrainWorkspaceState) string {
	if tv.RunConfig != nil && strings.TrimSpace(tv.RunConfig.Model) != "" {
		return tv.RunConfig.Model
	}
	return strings.TrimSpace(tv.Request.Model)
}

func appNormalizeTrainDevice(device string) string {
	switch strings.ToLower(strings.TrimSpace(device)) {
	case "ascend", "npu":
		return "npu"
	case "cuda", "gpu", "nvidia":
		return "gpu"
	}
	if strings.TrimSpace(device) == "" {
		return ""
	}
	return strings.ToLower(strings.TrimSpace(device))
}

func parseTrainDataset(rawInput string) string {
	fields := strings.Fields(strings.TrimSpace(rawInput))
	if len(fields) < 3 {
		return ""
	}
	return strings.Join(fields[2:], " ")
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

func (a App) activeHUDHeight() int {
	if a.trainView.Active {
		return lipgloss.Height(panels.RenderTrainHUD(a.trainView, a.width, a.agentStatus()))
	}
	return 0
}

func (a App) chatLine() string {
	w := a.chatWidth()
	return chatLineStyle.Render(strings.Repeat("─", w))
}

func (a App) View() string {
	if a.bootActive {
		return panels.RenderBootScreen(a.width, a.height, a.bootHighlight)
	}
	if a.issueView.Active() {
		return a.renderIssueView()
	}
	if a.bugView.Active() {
		return a.renderBugView()
	}

	topBar := panels.RenderTopBar(a.state, a.width)
	chat := a.viewport.View()
	queueBanner := ""
	if len(a.queuedInputs) > 0 {
		queueBanner = queueBannerStyle.Render("messages queued (press esc to interrupt)")
	}
	input := a.input.View()
	hintBar := panels.RenderHintBar(a.state, a.width)

	parts := []string{topBar}
	if a.trainView.Active {
		parts = append(parts, panels.RenderTrainHUD(a.trainView, a.width, a.agentStatus()))
		hintBar = panels.RenderTrainHUDHintBar(a.width)
	}
	parts = append(parts,
		chat,
	)
	if queueBanner != "" {
		parts = append(parts, queueBanner)
	}
	parts = append(parts,
		input,
		hintBar,
	)
	for i := 0; i < bottomSafePadding; i++ {
		parts = append(parts, "")
	}

	layout := lipgloss.JoinVertical(lipgloss.Left, parts...)

	if a.trainView.Active && a.trainView.SelectionPopup != nil {
		layout = overlayPopup(layout, panels.RenderSelectionPopup(a.trainView.SelectionPopup), a.width, a.height)
	}

	return trimViewHeight(layout, a.height)
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
