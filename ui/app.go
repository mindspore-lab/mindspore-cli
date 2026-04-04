package ui

import (
	"fmt"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/mindspore-lab/mindspore-cli/ui/components"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
	"github.com/mindspore-lab/mindspore-cli/ui/panels"
	"github.com/mindspore-lab/mindspore-cli/ui/theme"

	tea "github.com/charmbracelet/bubbletea"
)

const (
	topBarHeight                    = 1 // brand line only
	chatLineHeight                  = 0
	hintBarHeight                   = 1
	inputHeight                     = 1
	bottomSafePadding               = 2
	verticalPad                     = 2
	bootDuration                    = 2 * time.Second
	bootTickRate                    = 80 * time.Millisecond
	defaultToolMaxRunes             = 12000
	writeEditPreviewHeadLines       = 5
	writeEditPreviewTailLines       = 0
	shellPreviewHeadLines           = 5
	shellPreviewTailLines           = 0
	errorPreviewHeadLines           = 5
	errorPreviewTailLines           = 0
	defaultPreviewHeadLines         = 5
	defaultPreviewTailLines         = 0
	collapsedPreviewMaxLines        = 3
	bootReadyToken                  = "__boot_ready__"
	historyReplayReadyToken         = "__history_replay_ready__"
	maxToolLines                    = 120
	maxToolRunes                    = 12000
	interruptQueuedTrainToken       = "__interrupt_queued_train__"
	interruptActiveTaskToken        = "__interrupt_active_task__"
	internalPermissionsActionPrefix = "\x00permissions:"
	modelSetupToken                 = "__model_setup"
)

// Style vars are populated by InitStyles() below.
var (
	chatLineStyle     lipgloss.Style
	trainErrorStyle   lipgloss.Style
	trainSuccessStyle lipgloss.Style
	trainWorkingStyle lipgloss.Style
	queueBannerStyle  lipgloss.Style
	atFileCandidateRE = regexp.MustCompile(`^[A-Za-z0-9._/\\-]+$`)
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
	diffAddStyle     lipgloss.Style
	diffRemoveStyle  lipgloss.Style
	diffHunkStyle    lipgloss.Style
	diffFileStyle    lipgloss.Style
	diffContextStyle lipgloss.Style
	diffSummaryStyle lipgloss.Style
)

// InitStyles rebuilds the package-level style vars from theme.Current.
func InitStyles() {
	t := theme.Current
	chatLineStyle = lipgloss.NewStyle().Foreground(t.SelectionBG)
	trainErrorStyle = lipgloss.NewStyle().Foreground(t.Error)
	trainSuccessStyle = lipgloss.NewStyle().Foreground(t.Success)
	trainWorkingStyle = lipgloss.NewStyle().Foreground(t.Warning)
	queueBannerStyle = lipgloss.NewStyle().Foreground(t.Warning).PaddingLeft(2)
	diffAddStyle = lipgloss.NewStyle().Foreground(t.Success)
	diffRemoveStyle = lipgloss.NewStyle().Foreground(t.Error)
	diffHunkStyle = lipgloss.NewStyle().Foreground(t.Accent)
	diffFileStyle = lipgloss.NewStyle().Foreground(t.Warning).Bold(true)
	diffContextStyle = lipgloss.NewStyle().Foreground(t.TextSecondary)
	diffSummaryStyle = lipgloss.NewStyle().Foreground(t.TextPrimary).Bold(true)
}

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

type permissionPromptState struct {
	title    string
	message  string
	options  []model.PermissionOption
	selected int
}

type permissionsViewState struct {
	tab          int
	search       string
	searchCursor int
	selected     int
	allow        []string
	ask          []string
	deny         []string
	dialogMode   permissionsDialogMode
	dialogInput  string
	dialogCursor int
	dialogChoice int
	dialogTarget string
	dialogSource string
	dialogRule   string
}

type permissionsDialogMode int

const (
	permissionsDialogNone permissionsDialogMode = iota
	permissionsDialogAddRule
	permissionsDialogChooseRuleScope
	permissionsDialogDeleteRule
)

// App is the TUI root model.
type App struct {
	state            model.State
	viewport         components.Viewport
	input            components.TextInput
	thinking         components.ThinkingSpinner
	width            int
	height           int
	eventCh          <-chan model.Event
	userCh           chan<- string // sends user input to the engine bridge
	lastInterrupt    time.Time     // track last ctrl+c for double-press exit
	mouseEnabled     bool
	replayWait       *model.ReplayWaitData
	modalAltScreen   bool
	deltaMu          *sync.Mutex
	deltaBuf         *strings.Builder // buffers agent deltas until a full line is ready
	deltaStarted     *bool            // true after the first agent delta line is printed
	eventListening   *int32           // atomic flag: 1 = waitForEvent goroutine is active
	cmdOutputStarted *bool            // true after first shell output line is printed
	cmdOutputLines   *int             // lines printed so far for current shell command
	followBottom     bool
	unreadCount      int
	lastMsgCount     int

	// Train mode
	trainView     model.TrainViewState
	trainFocus    model.TrainPanelID
	bugView       model.BugViewState
	issueView     model.IssueViewState
	bootActive    bool
	bootHighlight int
	bannerPrinted bool
	queuedInputs  []string

	permissionPrompt *permissionPromptState
	permissionsView  *permissionsViewState
	toolsExpanded    *bool
	modelPicker      *model.SelectionPopup
	setupPopup       *model.SetupPopup
	appendHistoryFn  func(string)

	// Tool output viewer (alt-screen overlay, toggled via Ctrl+O)
	toolOutputView *toolOutputViewState
}

// toolOutputViewState holds state for the alt-screen tool output viewer.
type toolOutputViewState struct {
	toolCallID string
	msg        model.Message
	scrollOff  int // vertical scroll offset (line index)
}

// New creates a new App driven by the given event channel.
// userCh may be nil — user input won't be forwarded.
func New(ch <-chan model.Event, userCh chan<- string, version, workDir, repoURL, modelName string, ctxMax int) App {
	return App{
		state:            model.NewState(version, workDir, repoURL, modelName, ctxMax),
		input:            components.NewTextInput().WithFileSuggestions(workDir),
		thinking:         components.NewThinkingSpinner(),
		eventCh:          ch,
		userCh:           userCh,
		bootActive:       true,
		deltaMu:          &sync.Mutex{},
		deltaBuf:         &strings.Builder{},
		eventListening:   new(int32),
		deltaStarted:     new(bool),
		cmdOutputStarted: new(bool),
		cmdOutputLines:   new(int),
		toolsExpanded:    new(bool),
		followBottom:     true,
	}
}

// NewReplay creates a TUI instance that starts directly in chat view for playback.
func NewReplay(ch <-chan model.Event, userCh chan<- string, version, workDir, repoURL, modelName string, ctxMax int) App {
	app := New(ch, userCh, version, workDir, repoURL, modelName, ctxMax)
	app.bootActive = false
	return app
}

// SeedInputHistory preloads persisted prompt history into the current composer.
func (a App) SeedInputHistory(values []string) App {
	a.input = a.input.SeedHistory(values)
	return a
}

// WithInputHistoryAppender installs the persistence hook for submitted prompts.
func (a App) WithInputHistoryAppender(fn func(string)) App {
	a.appendHistoryFn = fn
	return a
}

func (a App) rememberInput(value string) App {
	var added bool
	a.input, added = a.input.RecordHistory(value)
	if added && a.appendHistoryFn != nil {
		a.appendHistoryFn(value)
	}
	return a
}
func (a App) waitForEvent() tea.Msg {
	// Prevent multiple goroutines from reading the event channel
	// concurrently — that causes non-deterministic event ordering.
	if !atomic.CompareAndSwapInt32(a.eventListening, 0, 1) {
		// Another goroutine is already listening; yield a no-op.
		return nil
	}
	ev, ok := <-a.eventCh
	atomic.StoreInt32(a.eventListening, 0)
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
		a.waitForEvent,
	)
}

func (a App) chatHeight() int {
	h := a.height - a.persistentTopBarHeight() - chatLineHeight - hintBarHeight - a.input.Height()
	h -= a.activeHUDHeight()
	h -= a.queueBannerHeight()
	h -= a.bottomPaddingHeight()
	if h < 1 {
		return 1
	}
	return h
}

func (a App) desiredChatHeight(contentLines int) int {
	maxHeight := a.chatHeight()
	if contentLines < 1 {
		contentLines = 1
	}
	if contentLines > maxHeight {
		return maxHeight
	}
	return contentLines
}

func (a App) persistentTopBarHeight() int {
	return 0
}

func (a App) bottomPaddingHeight() int {
	return 0
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
		a.syncViewportScrollState()
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
		return a, a.maybePrintBanner()

	case model.Event:
		return a.handleEvent(msg)

	default:
		var cmd tea.Cmd
		a.thinking, cmd = a.thinking.Update(msg)
		a.thinking.Elapsed = a.currentWaitElapsed()
		if cmd != nil {
			cmds = append(cmds, cmd)
		}
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

// chatWidth returns the width available for the chat area.
func (a App) chatWidth() int {
	return a.width
}

func (a *App) resizeInput() {
	inputWidth := a.chatWidth() - 4
	if inputWidth < 1 {
		inputWidth = 1
	}
	a.input = a.input.SetWidth(inputWidth)
	a.input = a.input.SetMaxVisibleRows(a.maxComposerEditorRows())
}

// maxComposerEditorRows returns the maximum number of editor rows the
// composer may display before it becomes internally scrollable.
// It subtracts all fixed and dynamic layout overhead from the terminal
// height, ensuring the total view never exceeds the terminal.
func (a App) maxComposerEditorRows() int {
	if a.height <= 0 {
		return 0 // unknown terminal size → no cap
	}
	avail := a.height
	avail -= a.persistentTopBarHeight()
	avail -= chatLineHeight
	avail -= hintBarHeight
	avail -= a.activeHUDHeight()
	avail -= a.queueBannerHeight()
	avail -= a.bottomPaddingHeight()
	avail -= 1 // empty-line separator before the input
	rows := avail - a.input.ReservedHeight()
	if rows < 1 {
		return 1
	}
	return rows
}

func (a *App) resizeActiveLayout() {
	a.resizeInput()
	a.viewport = a.viewport.SetSize(a.chatWidth()-4, a.chatHeight())
	a.syncViewportScrollState()
}

func (a *App) wantsModalAltScreen() bool {
	if a == nil {
		return false
	}
	return a.modelPicker != nil || a.setupPopup != nil || a.toolOutputView != nil
}

func (a *App) syncModalAltScreen() tea.Cmd {
	if a == nil {
		return nil
	}

	wants := a.wantsModalAltScreen()
	switch {
	case wants && !a.modalAltScreen:
		a.modalAltScreen = true
		return tea.EnterAltScreen
	case !wants && a.modalAltScreen:
		a.modalAltScreen = false
		return tea.ExitAltScreen
	default:
		return nil
	}
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
		a.replayWait = nil
		a.state = a.clearThinking()
		a.input = a.input.Reset()
		a.resizeActiveLayout()
		interruptMsg := model.Message{
			Kind:    model.MsgAgent,
			Content: "Interrupt requested. Press Ctrl+C again within 1 second to exit.",
		}
		a.state = a.state.WithMessage(interruptMsg)
		return a, a.printMessage(interruptMsg)
	}

	// Tool output viewer intercepts all keys when active.
	if a.toolOutputView != nil {
		return a.handleToolOutputViewKey(msg)
	}

	if a.issueView.Active() {
		return a.handleIssueKey(msg)
	}

	if msg.String() == "ctrl+o" {
		if a.toolOutputView != nil {
			a.toolOutputView = nil
		} else {
			if toolMsg, ok := a.latestToolMessage(); ok {
				a.toolOutputView = &toolOutputViewState{
					toolCallID: toolMsg.ToolCallID,
					msg:        toolMsg,
				}
			}
		}
		return a, a.syncModalAltScreen()
	}

	if a.bugView.Active() {
		return a.handleBugKey(msg)
	}

	if a.permissionPrompt != nil {
		switch msg.String() {
		case "up", "left":
			if len(a.permissionPrompt.options) > 0 {
				a.permissionPrompt.selected--
				if a.permissionPrompt.selected < 0 {
					a.permissionPrompt.selected = len(a.permissionPrompt.options) - 1
				}
			}
			return a, nil
		case "down", "right", "tab":
			if len(a.permissionPrompt.options) > 0 {
				a.permissionPrompt.selected = (a.permissionPrompt.selected + 1) % len(a.permissionPrompt.options)
			}
			return a, nil
		case "enter":
			if len(a.permissionPrompt.options) > 0 {
				input := a.permissionPrompt.options[a.permissionPrompt.selected].Input
				a.permissionPrompt = nil
				if a.userCh != nil {
					select {
					case a.userCh <- input:
					default:
					}
				}
			}
			return a, nil
		case "esc":
			a.permissionPrompt = nil
			if a.userCh != nil {
				select {
				case a.userCh <- "esc":
				default:
				}
			}
			return a, nil
		default:
			return a, nil
		}
	}

	if a.permissionsView != nil {
		if a.permissionsView.dialogMode != permissionsDialogNone {
			switch a.permissionsView.dialogMode {
			case permissionsDialogAddRule:
				switch msg.String() {
				case "enter":
					if rule := strings.TrimSpace(a.permissionsView.dialogInput); rule != "" {
						a.permissionsView.dialogRule = rule
						a.permissionsView.dialogMode = permissionsDialogChooseRuleScope
						a.permissionsView.dialogChoice = 0
						return a, nil
					}
					return a, nil
				case "backspace":
					a.permissionsView.dialogInput, a.permissionsView.dialogCursor = deleteRuneBeforeCursor(a.permissionsView.dialogInput, a.permissionsView.dialogCursor)
					return a, nil
				case "delete":
					a.permissionsView.dialogInput, a.permissionsView.dialogCursor = deleteRuneAtCursor(a.permissionsView.dialogInput, a.permissionsView.dialogCursor)
					return a, nil
				case "left":
					a.permissionsView.dialogCursor = moveCursorLeft(a.permissionsView.dialogCursor)
					return a, nil
				case "right":
					a.permissionsView.dialogCursor = moveCursorRight(a.permissionsView.dialogInput, a.permissionsView.dialogCursor)
					return a, nil
				case "home", "ctrl+a":
					a.permissionsView.dialogCursor = 0
					return a, nil
				case "end", "ctrl+e":
					a.permissionsView.dialogCursor = len([]rune(a.permissionsView.dialogInput))
					return a, nil
				case "esc":
					a.permissionsView.dialogMode = permissionsDialogNone
					a.permissionsView.dialogInput = ""
					a.permissionsView.dialogCursor = 0
					return a, nil
				default:
					if msg.Type == tea.KeyRunes {
						a.permissionsView.dialogInput, a.permissionsView.dialogCursor = insertRunesAtCursor(a.permissionsView.dialogInput, a.permissionsView.dialogCursor, msg.Runes)
					} else if msg.Type == tea.KeySpace {
						a.permissionsView.dialogInput, a.permissionsView.dialogCursor = insertRunesAtCursor(a.permissionsView.dialogInput, a.permissionsView.dialogCursor, []rune{' '})
					}
					return a, nil
				}
			case permissionsDialogChooseRuleScope:
				switch msg.String() {
				case "up", "left":
					a.permissionsView.dialogChoice--
					if a.permissionsView.dialogChoice < 0 {
						a.permissionsView.dialogChoice = 1
					}
					return a, nil
				case "down", "right", "tab":
					a.permissionsView.dialogChoice = (a.permissionsView.dialogChoice + 1) % 2
					return a, nil
				case "enter":
					cmd, ok := permissionsRuleToAddCommand(a.permissionsView.tab, a.permissionsView.dialogRule, permissionScopeByChoice(a.permissionsView.dialogChoice))
					if ok && a.userCh != nil {
						a.permissionsView = nil
						select {
						case a.userCh <- cmd:
						default:
						}
					}
					return a, nil
				case "esc":
					a.permissionsView.dialogMode = permissionsDialogAddRule
					a.permissionsView.dialogChoice = 0
					return a, nil
				default:
					return a, nil
				}
			case permissionsDialogDeleteRule:
				switch msg.String() {
				case "up", "left":
					a.permissionsView.dialogChoice--
					if a.permissionsView.dialogChoice < 0 {
						a.permissionsView.dialogChoice = 1
					}
					return a, nil
				case "down", "right", "tab":
					a.permissionsView.dialogChoice = (a.permissionsView.dialogChoice + 1) % 2
					return a, nil
				case "enter":
					yes := a.permissionsView.dialogChoice == 0
					if !yes {
						a.permissionsView.dialogMode = permissionsDialogNone
						return a, nil
					}
					var (
						cmd string
						ok  bool
					)
					cmd, ok = permissionsRemoveCommandForItem(a.permissionsView.tab, a.permissionsView.dialogTarget)
					a.permissionsView = nil
					if ok && a.userCh != nil {
						select {
						case a.userCh <- cmd:
						default:
						}
					}
					return a, nil
				case "esc":
					a.permissionsView.dialogMode = permissionsDialogNone
					return a, nil
				default:
					return a, nil
				}
			}
		}

		switch msg.String() {
		case "shift+tab":
			a.permissionsView.tab = (a.permissionsView.tab + 2) % 3
			a.permissionsView.selected = 0
			return a, nil
		case "tab":
			a.permissionsView.tab = (a.permissionsView.tab + 1) % 3
			a.permissionsView.selected = 0
			return a, nil
		case "left":
			a.permissionsView.searchCursor = moveCursorLeft(a.permissionsView.searchCursor)
			return a, nil
		case "right":
			a.permissionsView.searchCursor = moveCursorRight(a.permissionsView.search, a.permissionsView.searchCursor)
			return a, nil
		case "up":
			items := permissionsFilteredItems(a.permissionsView)
			if len(items) > 0 {
				a.permissionsView.selected--
				if a.permissionsView.selected < 0 {
					a.permissionsView.selected = len(items) - 1
				}
			}
			return a, nil
		case "down":
			items := permissionsFilteredItems(a.permissionsView)
			if len(items) > 0 {
				a.permissionsView.selected = (a.permissionsView.selected + 1) % len(items)
			}
			return a, nil
		case "enter":
			items := permissionsFilteredItems(a.permissionsView)
			if len(items) == 0 {
				return a, nil
			}
			selected := items[a.permissionsView.selected]
			if selected == "Add a new rule…" {
				a.permissionsView.dialogMode = permissionsDialogAddRule
				a.permissionsView.dialogInput = ""
				a.permissionsView.dialogCursor = 0
				a.permissionsView.dialogRule = ""
				return a, nil
			}
			a.permissionsView.dialogMode = permissionsDialogDeleteRule
			a.permissionsView.dialogChoice = 0
			a.permissionsView.dialogTarget = selected
			a.permissionsView.dialogSource = "From project local settings"
			return a, nil
		case "backspace":
			a.permissionsView.search, a.permissionsView.searchCursor = deleteRuneBeforeCursor(a.permissionsView.search, a.permissionsView.searchCursor)
			a.permissionsView.selected = 0
			return a, nil
		case "delete":
			a.permissionsView.search, a.permissionsView.searchCursor = deleteRuneAtCursor(a.permissionsView.search, a.permissionsView.searchCursor)
			a.permissionsView.selected = 0
			return a, nil
		case "home", "ctrl+a":
			a.permissionsView.searchCursor = 0
			return a, nil
		case "end", "ctrl+e":
			a.permissionsView.searchCursor = len([]rune(a.permissionsView.search))
			return a, nil
		case " ":
			a.permissionsView.search, a.permissionsView.searchCursor = insertRunesAtCursor(a.permissionsView.search, a.permissionsView.searchCursor, []rune{' '})
			a.permissionsView.selected = 0
			return a, nil
		case "esc":
			a.permissionsView = nil
			return a, nil
		default:
			if msg.Type == tea.KeyRunes {
				a.permissionsView.search, a.permissionsView.searchCursor = insertRunesAtCursor(a.permissionsView.search, a.permissionsView.searchCursor, msg.Runes)
				a.permissionsView.selected = 0
				return a, nil
			}
			if msg.Type == tea.KeySpace {
				a.permissionsView.search, a.permissionsView.searchCursor = insertRunesAtCursor(a.permissionsView.search, a.permissionsView.searchCursor, []rune{' '})
				a.permissionsView.selected = 0
				return a, nil
			}
			return a, nil
		}
	}

	// Check if we're in slash suggestion mode
	if a.input.HasSuggestions() {
		switch msg.String() {
		case "tab", "esc", "enter":
			var cmd tea.Cmd
			a.input, cmd = a.input.Update(msg)
			a.resizeActiveLayout()
			return a, cmd
		case "up", "down":
			var cmd tea.Cmd
			a.input, cmd = a.input.Update(msg)
			return a, cmd
		}
	}

	// Multi-step model setup popup navigation
	if a.setupPopup != nil {
		switch a.setupPopup.Screen {
		case model.SetupScreenModeSelect:
			switch msg.String() {
			case "up", "left":
				a.setupPopup.MoveModeSelection(-1)
				return a, nil
			case "down", "right":
				a.setupPopup.MoveModeSelection(1)
				return a, nil
			case "enter":
				if a.setupPopup.ModeSelected == 0 {
					a.setupPopup.Screen = model.SetupScreenPresetPicker
				} else {
					a.setupPopup.Screen = model.SetupScreenEnvInfo
				}
				return a, nil
			case "esc":
				if a.setupPopup.CanEscape {
					a.setupPopup = nil
				}
				return a, a.syncModalAltScreen()
			}
		case model.SetupScreenPresetPicker:
			switch msg.String() {
			case "up", "left":
				a.setupPopup.MovePresetSelection(-1)
				return a, nil
			case "down", "right":
				a.setupPopup.MovePresetSelection(1)
				return a, nil
			case "enter":
				opt := a.setupPopup.PresetOptions[a.setupPopup.PresetSelected]
				if !opt.Disabled {
					a.setupPopup.SelectedPreset = opt
					a.setupPopup.Screen = model.SetupScreenTokenInput
					a.setupPopup.TokenError = ""
				}
				return a, nil
			case "esc":
				a.setupPopup.Screen = model.SetupScreenModeSelect
				return a, nil
			}
		case model.SetupScreenTokenInput:
			switch msg.String() {
			case "enter":
				if a.userCh != nil && strings.TrimSpace(a.setupPopup.TokenValue) != "" {
					cmd := fmt.Sprintf("%s %s %s", modelSetupToken,
						a.setupPopup.SelectedPreset.ID,
						strings.TrimSpace(a.setupPopup.TokenValue))
					select {
					case a.userCh <- cmd:
					default:
					}
				}
				return a, nil
			case "esc":
				a.setupPopup.Screen = model.SetupScreenPresetPicker
				return a, nil
			case "backspace":
				runes := []rune(a.setupPopup.TokenValue)
				if len(runes) > 0 {
					a.setupPopup.TokenValue = string(runes[:len(runes)-1])
				}
				return a, nil
			default:
				if msg.Type == tea.KeyRunes {
					a.setupPopup.TokenValue += string(msg.Runes)
				} else if msg.Type == tea.KeySpace {
					// Don't add spaces to tokens
				}
				return a, nil
			}
		case model.SetupScreenEnvInfo:
			if msg.String() == "esc" {
				a.setupPopup.Screen = model.SetupScreenModeSelect
				return a, nil
			}
			return a, nil
		}
		return a, nil
	}

	// Selection popup navigation
	if a.modelPicker != nil || (a.trainView.Active && a.trainView.SelectionPopup != nil) {
		p := a.trainView.SelectionPopup
		if a.modelPicker != nil {
			p = a.modelPicker
		}
		if p == nil || len(p.Options) == 0 {
			return a, nil
		}
		switch msg.String() {
		case "up", "left":
			p.Selected--
			if p.Selected < 0 {
				p.Selected = len(p.Options) - 1
			}
			return a, nil
		case "down", "right":
			p.Selected = (p.Selected + 1) % len(p.Options)
			return a, nil
		case "enter":
			selected := p.Options[p.Selected]
			a.trainView.SelectionPopup = nil
			a.modelPicker = nil
			var input string
			switch p.ActionID {
			case "add_algo_feature":
				input = "/train add algo-feature " + selected.ID
			case "add_perf_feature":
				input = "/train add perf-feature " + selected.ID
			case "model_picker":
				input = "/model " + selected.ID
			}
			if input != "" && a.userCh != nil {
				select {
				case a.userCh <- input:
				default:
				}
			}
			return a, a.syncModalAltScreen()
		case "esc":
			a.trainView.SelectionPopup = nil
			a.modelPicker = nil
			exitAlt := a.syncModalAltScreen()
			banner := a.maybePrintBanner()
			if exitAlt != nil && banner != nil {
				return a, tea.Sequence(exitAlt, banner)
			}
			return a, combineCmds(exitAlt, banner)
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
		// HasSuggestions means slash mode currently has visible candidates, so
		// Enter accepts the current suggestion instead of submitting or newline.
		if a.input.HasSuggestions() {
			var cmd tea.Cmd
			a.input, cmd = a.input.Update(msg)
			a.resizeActiveLayout()
			return a, cmd
		}
		// ConsumeEscapedEnter turns "\+Enter" into a newline by consuming the
		// backslash immediately before the cursor.
		if updated, consumed := a.input.ConsumeEscapedEnter(); consumed {
			a.input = updated
			a.resizeActiveLayout()
			return a, nil
		}
		// IsSlashMode can still be true after candidates disappear, so let the
		// input keep handling Enter inside slash mode instead of submitting.
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
			a = a.rememberInput(val)
			a.input = a.input.Reset()
			a.resizeActiveLayout()
			return a, a.printUserInput(val)
		}
		// Reset stats for new task
		a.state = a.state.ResetStats()
		a.replayWait = nil
		a.state = a.clearThinking()
		if !strings.HasPrefix(val, "/") && !shouldDeferUserEcho(val) {
			a.state = a.state.WithMessage(model.Message{Kind: model.MsgUser, Content: val})
			a.state = a.startWait(model.WaitModel)
		}
		a = a.rememberInput(val)
		a.input = a.input.Reset()
		a.resizeActiveLayout()
		printCmd := a.printUserInput(val)
		if a.userCh != nil {
			select {
			case a.userCh <- val:
			default:
				// drop if buffer full — avoids freezing the UI
			}
		}
		return a, printCmd

	case "home", "end":
		var cmd tea.Cmd
		a.input.Model, cmd = a.input.Model.Update(msg)
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
	a.replayWait = nil
	a.state = a.clearThinking()
	if !strings.HasPrefix(strings.TrimSpace(next), "/") && !shouldDeferUserEcho(next) {
		a.state = a.state.WithMessage(model.Message{Kind: model.MsgUser, Content: next})
		a.state = a.startWait(model.WaitModel)
	}
	select {
	case a.userCh <- next:
	default:
	}
	a.resizeActiveLayout()
	return a
}

func (a App) handleEvent(ev model.Event) (tea.Model, tea.Cmd) {
	a = a.applyUsageSnapshot(ev)
	prevMessages := append([]model.Message(nil), a.state.Messages...)

	var eventCmd tea.Cmd

	switch ev.Type {
	case model.UserInput:
		if last := len(a.state.Messages) - 1; last >= 0 && a.state.Messages[last].Kind == model.MsgUser {
			msgs := append([]model.Message{}, a.state.Messages...)
			msgs[last].Content = ev.Message
			a.state.Messages = msgs
		} else {
			a.state = a.state.WithMessage(model.Message{Kind: model.MsgUser, Content: ev.Message})
		}
	case model.IssueIndexOpen:
		a.openIssueIndex(ev.IssueView)

	case model.IssueDetailOpen:
		a.openIssueDetail(ev.IssueView)

	case model.BugIndexOpen:
		a.openBugIndex(ev.BugView)

	case model.BugDetailOpen:
		a.openBugDetail(ev.BugView)

	case model.TaskDone:
		a.replayWait = nil
		a.state = a.clearThinking()

	case model.AgentThinking:
		a.replayWait = ev.ReplayWait
		a.state = a.startWait(model.WaitModel)

	case model.AgentReply:
		a.replayWait = nil
		a.state = a.clearThinking()
		a.input = a.input.ClearSlashMode()
		content := ev.Message
		if ev.Train != nil && ev.Train.IsDiff {
			content = formatDiffLine(ev.Message)
		} else if ev.Train != nil && ev.Train.ActionSource != "" {
			content = agentMsg(evSource(ev.Train, ""), ev.Message, false)
		}
		a.state = a.finalizeAgentMessage(model.Message{Kind: model.MsgAgent, Content: content, RawANSI: ev.RawANSI})

	case model.ContextNotice:
		a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: ev.Message, Display: model.DisplayNotice})

	case model.AgentReplyDelta:
		a.replayWait = nil
		// Keep WaitStartedAt so the elapsed timer continues through "Responding...".
		a.state = a.state.WithThinking(false)
		a.state = a.appendToStreamingAgent(ev.Message)

	case model.PermissionPrompt:
		a.replayWait = nil
		a.state = a.clearThinking()
		a.permissionPrompt = toPermissionPromptState(ev)

	case model.PermissionsView:
		a.replayWait = nil
		a.state = a.clearThinking()
		a.permissionsView = toPermissionsViewState(ev)

	case model.ToolCallStart:
		a.replayWait = ev.ReplayWait
		a.state = a.startWait(model.WaitTool)
		a.state = a.commitStreamingAgent()
		a.state = a.state.WithMessage(a.pendingToolMessage(ev))

	case model.CmdStarted:
		stats := a.state.Stats
		stats.Commands++
		a.state = a.state.WithStats(stats)
		a.state = a.resolveToolEvent(ev, model.Message{
			Kind:       model.MsgTool,
			ToolName:   "Bash",
			ToolCallID: ev.ToolCallID,
			Display:    model.DisplayCollapsed,
			Streaming:  true,
		})

	case model.CmdOutput:
		a.state = a.appendToolOutput(ev)

	case model.CmdFinished:
		a.replayWait = nil
		a.state = a.clearThinking()
		a.state = a.resolveToolEvent(ev, model.Message{
			Kind:       model.MsgTool,
			ToolName:   "Bash",
			ToolCallID: ev.ToolCallID,
			Display:    model.DisplayCollapsed,
			Content:    ev.Message,
			Summary:    ev.Summary,
		})

	case model.ToolRead:
		a.replayWait = nil
		a.state = a.clearThinking()
		stats := a.state.Stats
		stats.FilesRead++
		a.state = a.state.WithStats(stats)
		a.state = a.resolveToolEvent(ev, model.Message{
			Kind: model.MsgTool, ToolName: "Read", ToolArgs: ev.Message,
			Display: model.DisplayCollapsed, Content: ev.Message, Summary: ev.Summary,
		})

	case model.ToolGrep:
		a.replayWait = nil
		a.state = a.clearThinking()
		stats := a.state.Stats
		stats.Searches++
		a.state = a.state.WithStats(stats)
		a.state = a.resolveToolEvent(ev, model.Message{
			Kind: model.MsgTool, ToolName: "Grep", ToolArgs: ev.Message,
			Display: model.DisplayCollapsed, Content: ev.Message, Summary: ev.Summary,
		})

	case model.ToolGlob:
		a.replayWait = nil
		a.state = a.clearThinking()
		stats := a.state.Stats
		stats.Searches++
		a.state = a.state.WithStats(stats)
		a.state = a.resolveToolEvent(ev, model.Message{
			Kind: model.MsgTool, ToolName: "Glob", ToolArgs: ev.Message,
			Display: model.DisplayCollapsed, Content: ev.Message, Summary: ev.Summary,
		})

	case model.ToolEdit:
		a.replayWait = nil
		a.state = a.clearThinking()
		stats := a.state.Stats
		stats.FilesEdited++
		a.state = a.state.WithStats(stats)
		a.state = a.resolveToolEvent(ev, model.Message{
			Kind: model.MsgTool, ToolName: "Edit", ToolArgs: ev.Message,
			Display: model.DisplayExpanded, Content: ev.Message,
		})

	case model.ToolWrite:
		a.replayWait = nil
		a.state = a.clearThinking()
		stats := a.state.Stats
		stats.FilesEdited++
		a.state = a.state.WithStats(stats)
		a.state = a.resolveToolEvent(ev, model.Message{
			Kind: model.MsgTool, ToolName: "Write", ToolArgs: ev.Message,
			Display: model.DisplayExpanded, Content: ev.Message,
		})

	case model.ToolSkill:
		a.replayWait = nil
		a.state = a.clearThinking()
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

	case model.ToolWarning:
		a.replayWait = nil
		a.state = a.clearThinking()
		a.state = a.resolveToolEvent(ev, model.Message{
			Kind: model.MsgTool, ToolName: displayToolName(ev.ToolName), ToolArgs: ev.Message,
			Display: model.DisplayWarning, Content: ev.Message,
		})

	case model.ToolError:
		a.replayWait = nil
		a.state = a.clearThinking()
		stats := a.state.Stats
		stats.Errors++
		a.state = a.state.WithStats(stats)
		a.state = a.resolveToolEvent(ev, model.Message{
			Kind: model.MsgTool, ToolName: displayToolName(ev.ToolName), ToolArgs: ev.Message,
			Display: model.DisplayError, Content: ev.Message,
		})

	case model.ToolReplay:
		a.replayWait = nil
		a.state = a.resolveReplayToolEvent(ev)

	case model.AnalysisReady:
		a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: ev.Message})

	case model.TokenUpdate:
		// usage snapshot is applied before the event switch

	case model.TaskUpdated:
		// no-op for now

	case model.ClearScreen:
		a.replayWait = nil
		a.state = a.state.ClearWait()
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

	case model.ModelPickerOpen:
		if ev.Popup != nil {
			cp := *ev.Popup
			cp.Options = append([]model.SelectionOption(nil), ev.Popup.Options...)
			a.modelPicker = &cp
			eventCmd = combineCmds(eventCmd, a.syncModalAltScreen())
		}

	case model.ModelSetupOpen:
		if ev.SetupPopup != nil {
			cp := *ev.SetupPopup
			cp.PresetOptions = append([]model.SelectionOption(nil), ev.SetupPopup.PresetOptions...)
			a.setupPopup = &cp
			eventCmd = combineCmds(eventCmd, a.syncModalAltScreen())
		}

	case model.ModelSetupClose:
		a.setupPopup = nil
		exitAlt := a.syncModalAltScreen()
		banner := a.maybePrintBanner()
		if exitAlt != nil && banner != nil {
			eventCmd = combineCmds(eventCmd, tea.Sequence(exitAlt, banner))
		} else {
			eventCmd = combineCmds(eventCmd, exitAlt, banner)
		}

	case model.ModelSetupTokenError:
		if a.setupPopup != nil {
			a.setupPopup.TokenError = ev.Message
		}

	case model.IssueUserUpdate:
		a.state = a.state.WithIssueUser(ev.Message)

	case model.SkillsNoteUpdate:
		a.state.SkillsNote = ev.Message

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
		a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: trainSuccessStyle.Render(agentMsg(evSource(ev.Train, ""), ev.Message, true)), RawANSI: true})

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
			a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: trainErrorStyle.Render(agentMsg(evSource(ev.Train, "observer"), ev.Message, false)), RawANSI: true})
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
		a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: trainErrorStyle.Render(agentMsg(evSource(ev.Train, "observer"), ev.Message, false)), RawANSI: true})

	case model.TrainError:
		a.trainView.SetRunPhase(trainEventRunID(ev.Train), model.TrainPhaseFailed)
		if run := a.ensureTrainRun(ev.Train); run != nil {
			run.ErrorMessage = ev.Message
		}
		a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: trainErrorStyle.Render(agentMsg(evSource(ev.Train, "observer"), ev.Message, false)), RawANSI: true})

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
		a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: trainErrorStyle.Render(agentMsg(evSource(ev.Train, "observer"), ev.Message, false)), RawANSI: true})

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
				a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: trainWorkingStyle.Render(agentMsg("setup-helper", "fixing ssh connectivity...", false)), RawANSI: true})
				break
			}
			if valueOr(ev.Train.ActionID, "") == "install-missing-libs" {
				if run := a.ensureTrainRun(ev.Train); run != nil {
					run.StatusMessage = "Installing..."
				}
				a.trainView.SetStage(model.StageSetup)
				a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: trainWorkingStyle.Render(agentMsg("setup-helper", "installing missing library...", false)), RawANSI: true})
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
				a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: trainWorkingStyle.Render(agentMsg(evSource(ev.Train, ""), ev.Message, false)), RawANSI: true})
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
			a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: trainSuccessStyle.Render(agentMsg(evSource(ev.Train, ""), ev.Message, true)), RawANSI: true})
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
				a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: trainWorkingStyle.Render(agentMsg(evSource(ev.Train, "setup-helper"), ev.Message, false)), RawANSI: true})
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
				a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: trainWorkingStyle.Render(agentMsg(evSource(ev.Train, ""), ev.Message, false)), RawANSI: true})
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
		a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: trainSuccessStyle.Render(agentMsg(evSource(ev.Train, ""), ev.Message, true)), RawANSI: true})

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
	printCmd := a.eventPrintCmd(ev, prevMessages)
	eventCmd = combineCmds(eventCmd, printCmd, a.maybePrintBanner())
	if eventCmd != nil {
		// Sequence ensures tea.Println output is processed before
		// waitForEvent picks up the next event, preventing race
		// conditions with streaming shell output ordering.
		return a, tea.Sequence(eventCmd, a.waitForEvent)
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
		a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: trainSuccessStyle.Render(agentMsg("setup-agent", "ssh connectivity repaired", true)), RawANSI: true})
	}
	if run.StatusMessage == "Installing..." && ev.Train.Check == "libs" && ev.Train.Status == "passed" {
		run.StatusMessage = ""
		a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: trainSuccessStyle.Render(agentMsg("setup-agent", "missing library installed successfully", true)), RawANSI: true})
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
		a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: content, RawANSI: true})
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
		a.state = a.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: content, RawANSI: true})
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

func (a App) finalizeAgentMessage(msg model.Message) model.State {
	msgs := make([]model.Message, len(a.state.Messages))
	copy(msgs, a.state.Messages)

	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Kind == model.MsgAgent && msgs[i].Streaming {
			msgs[i].Content = msg.Content
			msgs[i].RawANSI = msg.RawANSI
			msgs[i].Streaming = false
			next := a.state
			next.Messages = msgs
			return next
		}
	}

	next := a.state
	next.Messages = append(msgs, msg)
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
	return a.state.WithThinking(false).ClearWait()
}

func (a App) startWait(kind model.WaitKind) model.State {
	next := a.state
	if next.WaitKind != kind || next.WaitStartedAt.IsZero() {
		next = next.WithWait(kind, time.Now())
	}
	return next.WithThinking(kind == model.WaitModel)
}

func (a App) thinkingStatusText() string {
	text := "Thinking..."
	if a.state.WaitKind != model.WaitModel || a.state.WaitStartedAt.IsZero() {
		return text
	}
	return text + " " + model.FormatWaitDuration(a.currentWaitElapsed())
}

func (a App) currentWaitElapsed() time.Duration {
	if a.state.WaitStartedAt.IsZero() {
		return 0
	}

	elapsed := time.Since(a.state.WaitStartedAt)
	if a.replayWait == nil ||
		a.replayWait.OriginalDuration <= 0 ||
		a.replayWait.SimulatedDuration <= 0 ||
		a.replayWait.SimulatedDuration >= a.replayWait.OriginalDuration {
		return elapsed
	}

	if elapsed >= a.replayWait.SimulatedDuration {
		return a.replayWait.OriginalDuration
	}

	scaled := float64(elapsed) * float64(a.replayWait.OriginalDuration) / float64(a.replayWait.SimulatedDuration)
	if scaled < 1 {
		return time.Nanosecond
	}
	return time.Duration(scaled)
}

func (a App) appendToolOutput(ev model.Event) model.State {
	msgs := make([]model.Message, len(a.state.Messages))
	copy(msgs, a.state.Messages)

	idx := toolMessageIndex(msgs, ev.ToolCallID)
	if idx < 0 {
		for i := len(msgs) - 1; i >= 0; i-- {
			if msgs[i].Kind == model.MsgTool {
				idx = i
				break
			}
		}
	}
	if idx >= 0 {
		content := msgs[idx].Content
		if content == "" {
			content = ev.Message
		} else {
			content += "\n" + ev.Message
		}
		msgs[idx].Content = content
		msgs[idx].Pending = false
		msgs[idx].Streaming = true
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
	body := content
	if ev.ToolName == "shell" {
		body = ""
	}
	return model.Message{
		Kind:       model.MsgTool,
		ToolName:   toolName,
		ToolCallID: ev.ToolCallID,
		ToolArgs:   content,
		Display:    display,
		Content:    body,
		Summary:    summary,
		Pending:    true,
	}
}

func (a App) resolveToolEvent(ev model.Event, fallback model.Message) model.State {
	msgs := make([]model.Message, len(a.state.Messages))
	copy(msgs, a.state.Messages)

	if idx := pendingToolMessageIndex(msgs, ev.ToolCallID); idx >= 0 {
		msgs[idx] = finalizeToolMessage(msgs[idx], ev)
		next := a.state
		next.Messages = msgs
		return next
	}

	if idx := toolMessageIndex(msgs, ev.ToolCallID); idx >= 0 {
		msgs[idx] = finalizeToolMessage(msgs[idx], ev)
		next := a.state
		next.Messages = msgs
		return next
	}

	for i := len(msgs) - 1; i >= 0; i-- {
		if !isPendingToolMessage(msgs[i]) {
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

func (a App) resolveReplayToolEvent(ev model.Event) model.State {
	msgs := make([]model.Message, len(a.state.Messages))
	copy(msgs, a.state.Messages)

	if idx := pendingReplayToolMessageIndex(msgs, ev); idx >= 0 {
		msgs[idx] = finalizeToolReplayMessage(msgs[idx], ev)
		next := a.state
		next.Messages = msgs
		return next
	}

	next := a.state
	next.Messages = append(msgs, replayToolMessage(ev))
	return next
}

func pendingToolMessageIndex(msgs []model.Message, toolCallID string) int {
	toolCallID = strings.TrimSpace(toolCallID)
	if toolCallID == "" {
		return -1
	}
	for i := len(msgs) - 1; i >= 0; i-- {
		if !isPendingToolMessage(msgs[i]) {
			continue
		}
		if strings.TrimSpace(msgs[i].ToolCallID) == toolCallID {
			return i
		}
	}
	return -1
}

func toolMessageIndex(msgs []model.Message, toolCallID string) int {
	toolCallID = strings.TrimSpace(toolCallID)
	if toolCallID == "" {
		return -1
	}
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Kind != model.MsgTool {
			continue
		}
		if strings.TrimSpace(msgs[i].ToolCallID) == toolCallID {
			return i
		}
	}
	return -1
}

func pendingReplayToolMessageIndex(msgs []model.Message, ev model.Event) int {
	if idx := pendingToolMessageIndex(msgs, ev.ToolCallID); idx >= 0 {
		return idx
	}

	toolName := displayToolName(ev.ToolName)
	for i := len(msgs) - 1; i >= 0; i-- {
		if !isPendingToolMessage(msgs[i]) {
			continue
		}
		if strings.TrimSpace(msgs[i].ToolName) == toolName {
			return i
		}
	}
	return -1
}

func isPendingToolMessage(msg model.Message) bool {
	return msg.Kind == model.MsgTool && msg.Pending
}

func finalizeToolMessage(pending model.Message, ev model.Event) model.Message {
	switch ev.Type {
	case model.CmdStarted:
		return model.Message{
			Kind:       model.MsgTool,
			ToolName:   valueOrString(pending.ToolName, "Bash"),
			ToolCallID: valueOrString(pending.ToolCallID, ev.ToolCallID),
			ToolArgs:   valueOrString(pending.ToolArgs, ev.Message),
			Display:    model.DisplayCollapsed,
			Content:    ev.Message,
			Summary:    ev.Summary,
			Streaming:  true,
		}
	case model.CmdFinished:
		return model.Message{
			Kind:       model.MsgTool,
			ToolName:   valueOrString(pending.ToolName, "Bash"),
			ToolCallID: valueOrString(pending.ToolCallID, ev.ToolCallID),
			ToolArgs:   valueOrString(pending.ToolArgs, pending.Content),
			Display:    model.DisplayCollapsed,
			Content:    ev.Message,
			Summary:    ev.Summary,
		}
	case model.ToolEdit, model.ToolWrite:
		return model.Message{
			Kind:       model.MsgTool,
			ToolName:   pending.ToolName,
			ToolCallID: valueOrString(pending.ToolCallID, ev.ToolCallID),
			ToolArgs:   valueOrString(pending.ToolArgs, pending.Content),
			Display:    model.DisplayExpanded,
			Content:    ev.Message,
			Summary:    ev.Summary,
		}
	case model.ToolRead:
		return model.Message{
			Kind:       model.MsgTool,
			ToolName:   pending.ToolName,
			ToolCallID: valueOrString(pending.ToolCallID, ev.ToolCallID),
			ToolArgs:   valueOrString(pending.ToolArgs, ev.Message),
			Display:    model.DisplayCollapsed,
			Content:    "",
			Summary:    firstNonEmpty(ev.Summary, pending.Summary),
		}
	case model.ToolGrep, model.ToolGlob, model.ToolSkill:
		return model.Message{
			Kind:       model.MsgTool,
			ToolName:   pending.ToolName,
			ToolCallID: valueOrString(pending.ToolCallID, ev.ToolCallID),
			ToolArgs:   valueOrString(pending.ToolArgs, ev.Message),
			Display:    model.DisplayCollapsed,
			Content:    ev.Message,
			Summary:    firstNonEmpty(ev.Summary, pending.Summary),
		}
	case model.ToolWarning:
		toolName := pending.ToolName
		if toolName == "" {
			toolName = displayToolName(ev.ToolName)
		}
		return model.Message{
			Kind:       model.MsgTool,
			ToolName:   toolName,
			ToolCallID: valueOrString(pending.ToolCallID, ev.ToolCallID),
			ToolArgs:   valueOrString(pending.ToolArgs, pending.Content),
			Display:    model.DisplayWarning,
			Content:    ev.Message,
		}
	case model.ToolError:
		toolName := pending.ToolName
		if toolName == "" {
			toolName = displayToolName(ev.ToolName)
		}
		return model.Message{
			Kind:       model.MsgTool,
			ToolName:   toolName,
			ToolCallID: valueOrString(pending.ToolCallID, ev.ToolCallID),
			ToolArgs:   valueOrString(pending.ToolArgs, pending.Content),
			Display:    model.DisplayError,
			Content:    ev.Message,
		}
	default:
		return pending
	}
}

func finalizeToolReplayMessage(pending model.Message, ev model.Event) model.Message {
	msg := replayToolMessage(ev)
	msg.ToolArgs = valueOrString(pending.ToolArgs, pending.Content)
	msg.ToolCallID = pending.ToolCallID
	return msg
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
		return "Bash"
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
	case "edit", "write":
		display = model.DisplayExpanded
		content = truncateToolContentForTool(ev.ToolName, ev.Message)
	case "shell":
		content = truncateToolContentForTool(ev.ToolName, ev.Message)
	}

	return model.Message{
		Kind:       model.MsgTool,
		ToolName:   displayToolName(ev.ToolName),
		ToolCallID: ev.ToolCallID,
		Display:    display,
		Content:    content,
	}
}

func truncateToolContentForTool(toolName, content string) string {
	content = strings.ReplaceAll(content, "\r\n", "\n")
	if strings.TrimSpace(content) == "" {
		return content
	}
	headLines, tailLines := toolPreviewPolicy(toolName)
	return truncateToolContentWithPolicy(content, headLines, tailLines, defaultToolMaxRunes)
}

func toolPreviewPolicy(toolName string) (headLines, tailLines int) {
	switch strings.ToLower(strings.TrimSpace(toolName)) {
	case "write", "edit":
		return writeEditPreviewHeadLines, writeEditPreviewTailLines
	case "shell":
		return shellPreviewHeadLines, shellPreviewTailLines
	case "tool", "engine":
		return errorPreviewHeadLines, errorPreviewTailLines
	default:
		return defaultPreviewHeadLines, defaultPreviewTailLines
	}
}

func truncateToolContentWithPolicy(content string, headLines, tailLines, maxRunes int) string {
	originalLines := strings.Split(content, "\n")
	omittedLines := 0
	truncatedByRunes := false

	runes := []rune(content)
	if len(runes) > maxRunes {
		content = string(runes[:maxRunes])
		truncatedByRunes = true
	}

	lines := strings.Split(content, "\n")
	visible := lines
	if headLines >= 0 && tailLines >= 0 && len(lines) > headLines+tailLines && len(lines) > headLines {
		head := append([]string{}, lines[:headLines]...)
		tail := []string{}
		if tailLines > 0 && tailLines < len(lines)-headLines {
			tail = append([]string{}, lines[len(lines)-tailLines:]...)
		}
		visible = append(head, tail...)
		omittedLines = len(lines) - len(visible)
	}

	if truncatedByRunes && len(originalLines) > len(lines) {
		omittedLines += len(originalLines) - len(lines)
	}
	if !truncatedByRunes && omittedLines <= 0 {
		return strings.Join(visible, "\n")
	}

	if omittedLines < 1 {
		omittedLines = 1
	}
	visible = append(visible, fmt.Sprintf("… +%d lines (ctrl+o to expand)", omittedLines))
	return strings.Join(visible, "\n")
}

func collapsedToolDetails(content string, maxLines int) string {
	content = strings.ReplaceAll(content, "\r\n", "\n")
	lines := strings.Split(strings.TrimSpace(content), "\n")
	filtered := make([]string, 0, len(lines))
	for _, line := range lines {
		if strings.TrimSpace(line) == "" {
			continue
		}
		filtered = append(filtered, line)
	}
	if len(filtered) == 0 {
		return ""
	}
	if maxLines <= 0 || len(filtered) <= maxLines {
		return strings.Join(filtered, "\n")
	}
	visible := append([]string{}, filtered[:maxLines]...)
	visible = append(visible, fmt.Sprintf("… +%d lines (ctrl+o to expand)", len(filtered)-maxLines))
	return strings.Join(visible, "\n")
}

func collapsedPreviewLines(toolName string) int {
	switch strings.ToLower(strings.TrimSpace(toolName)) {
	case "read":
		return 0
	case "skill":
		return 2
	default:
		return collapsedPreviewMaxLines
	}
}

// ── Tool output viewer (alt-screen overlay) ────────────────────────

var (
	toolViewTitleStyle = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("39"))
	toolViewHintStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("244")).Italic(true)
	toolViewLineStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("240"))
)

// handleToolOutputViewKey handles keys while the tool output viewer is open.
func (a App) handleToolOutputViewKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	v := a.toolOutputView
	contentHeight := a.toolOutputContentHeight()

	switch msg.String() {
	case "ctrl+o", "q", "esc":
		a.toolOutputView = nil
		return a, a.syncModalAltScreen()
	case "up", "k":
		if v.scrollOff > 0 {
			v.scrollOff--
		}
	case "down", "j":
		if v.scrollOff < contentHeight-a.toolOutputViewportHeight() {
			v.scrollOff++
		}
	case "pgup", "b":
		v.scrollOff -= a.toolOutputViewportHeight()
		if v.scrollOff < 0 {
			v.scrollOff = 0
		}
	case "pgdown", "f", " ":
		v.scrollOff += a.toolOutputViewportHeight()
		max := contentHeight - a.toolOutputViewportHeight()
		if max < 0 {
			max = 0
		}
		if v.scrollOff > max {
			v.scrollOff = max
		}
	case "home", "g":
		v.scrollOff = 0
	case "end", "G":
		max := contentHeight - a.toolOutputViewportHeight()
		if max < 0 {
			max = 0
		}
		v.scrollOff = max
	}
	return a, nil
}

func (a App) toolOutputViewportHeight() int {
	// 3 lines reserved: title bar, bottom divider, hint bar
	h := a.height - 3
	if h < 1 {
		h = 1
	}
	return h
}

func (a App) latestToolMessage() (model.Message, bool) {
	for i := len(a.state.Messages) - 1; i >= 0; i-- {
		msg := a.state.Messages[i]
		if msg.Kind == model.MsgTool {
			return msg, true
		}
	}
	return model.Message{}, false
}

func (a App) toolOutputMessage() model.Message {
	if a.toolOutputView == nil {
		return model.Message{}
	}
	if toolCallID := strings.TrimSpace(a.toolOutputView.toolCallID); toolCallID != "" {
		if msg, ok := findToolMessage(a.state.Messages, toolCallID); ok {
			return msg
		}
	}
	return a.toolOutputView.msg
}

func (a App) toolOutputContentLines() []string {
	if a.toolOutputView == nil {
		return nil
	}
	content := strings.TrimSpace(a.toolOutputMessage().Content)
	if content == "" {
		return []string{"(no output)"}
	}
	return strings.Split(content, "\n")
}

func (a App) toolOutputContentHeight() int {
	return len(a.toolOutputContentLines())
}

// renderToolOutputView renders the full-screen tool output viewer.
func (a App) renderToolOutputView() string {
	v := a.toolOutputView
	if v == nil {
		return ""
	}
	msg := a.toolOutputMessage()
	w := a.width
	if w < 10 {
		w = 10
	}
	vpHeight := a.toolOutputViewportHeight()
	allLines := a.toolOutputContentLines()
	totalLines := len(allLines)

	// Title bar
	toolName := strings.TrimSpace(msg.ToolName)
	toolArgs := strings.TrimSpace(msg.ToolArgs)
	if toolArgs == "" {
		toolArgs = strings.TrimSpace(msg.Summary)
	}
	title := toolViewTitleStyle.Render(fmt.Sprintf(" %s(%s)", toolName, toolArgs))
	lineInfo := toolViewHintStyle.Render(fmt.Sprintf(" %d/%d ", v.scrollOff+1, totalLines))
	titlePad := w - lipgloss.Width(title) - lipgloss.Width(lineInfo)
	if titlePad < 0 {
		titlePad = 0
	}
	titleBar := title + strings.Repeat(" ", titlePad) + lineInfo

	// Visible content slice
	end := v.scrollOff + vpHeight
	if end > totalLines {
		end = totalLines
	}
	start := v.scrollOff
	if start > totalLines {
		start = totalLines
	}
	visible := allLines[start:end]

	// Pad to fill viewport height
	for len(visible) < vpHeight {
		visible = append(visible, "~")
	}

	// Bottom bar
	divider := toolViewLineStyle.Render(strings.Repeat("─", w))
	hint := toolViewHintStyle.Render(" ctrl+o/q/esc: close  j/k: scroll  pgup/pgdn: page  g/G: top/bottom")

	parts := []string{titleBar, divider}
	parts = append(parts, visible...)
	parts = append(parts, divider, hint)

	return strings.Join(parts, "\n")
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
	atBottom := a.viewport.AtBottom() || a.viewport.TotalLines() <= a.viewport.VisibleHeight()
	width := a.viewport.Model.Width
	if width <= 0 {
		width = a.chatWidth() - 4
	}
	if width < 1 {
		width = 1
	}
	content := panels.RenderMessages(a.state, a.thinking.View(), a.thinking.FrameView(), width, a.trainView.Active)
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
	if !a.modalAltScreen {
		return a.renderMainView()
	}

	// Tool output viewer — full-screen scrollable view of tool output.
	if a.toolOutputView != nil {
		return a.renderToolOutputView()
	}

	// Temporary alt screen for modal popups — render only the popup on a blank backdrop.
	blank := strings.Repeat("\n", a.height-1)
	if a.trainView.Active && a.trainView.SelectionPopup != nil {
		return overlayPopup(blank, panels.RenderSelectionPopup(a.trainView.SelectionPopup), a.width, a.height)
	}
	if a.modelPicker != nil {
		return overlayPopup(blank, panels.RenderSelectionPopup(a.modelPicker), a.width, a.height)
	}
	if a.setupPopup != nil {
		return overlayPopup(blank, panels.RenderSetupPopup(a.setupPopup), a.width, a.height)
	}
	return blank
}

func (a *App) syncViewportScrollState() {
	if a.viewport.AtBottom() {
		a.followBottom = true
		a.unreadCount = 0
		return
	}
	a.followBottom = false
}

func (a *App) syncUnreadState(prevCount int, wasAtBottom bool) {
	currentCount := len(a.state.Messages)
	if currentCount > prevCount && !wasAtBottom {
		a.unreadCount += currentCount - prevCount
	}
	if a.viewport.AtBottom() {
		a.unreadCount = 0
		a.followBottom = true
	}
}

func trimViewHeight(content string, height int, fill bool) string {
	if height <= 0 {
		return content
	}
	lines := strings.Split(content, "\n")
	if len(lines) > height {
		lines = lines[:height]
	}
	if fill {
		for len(lines) < height {
			lines = append(lines, "")
		}
	}
	if !fill {
		for len(lines) > 0 && lines[len(lines)-1] == "" {
			lines = lines[:len(lines)-1]
		}
		if len(lines) == 0 {
			return ""
		}
	}
	return strings.Join(lines, "\n")
}

func shouldDeferUserEcho(input string) bool {
	for _, token := range strings.Fields(input) {
		if isAtFileCandidateToken(token) {
			return true
		}
	}
	return false
}

func isAtFileCandidateToken(token string) bool {
	switch {
	case token == "":
		return false
	case strings.HasPrefix(token, "@@"):
		return false
	case !strings.HasPrefix(token, "@") || len(token) == 1:
		return false
	}
	return atFileCandidateRE.MatchString(token[1:])
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

func toPermissionPromptState(ev model.Event) *permissionPromptState {
	if ev.Permission == nil {
		return &permissionPromptState{
			title:    "Permission required",
			message:  strings.TrimSpace(ev.Message),
			options:  []model.PermissionOption{{Input: "1", Label: "1. Yes"}, {Input: "2", Label: "2. Allow for this session"}, {Input: "3", Label: "3. No"}},
			selected: 0,
		}
	}

	options := ev.Permission.Options
	if len(options) == 0 {
		options = []model.PermissionOption{{Input: "1", Label: "1. Yes"}, {Input: "3", Label: "3. No"}}
	}
	selected := ev.Permission.DefaultIndex
	if selected < 0 || selected >= len(options) {
		selected = 0
	}
	return &permissionPromptState{
		title:    valueOrString(ev.Permission.Title, "Permission required"),
		message:  strings.TrimSpace(valueOrString(ev.Permission.Message, ev.Message)),
		options:  options,
		selected: selected,
	}
}

func renderPermissionPromptPopup(p *permissionPromptState) string {
	t := theme.Current
	golden := lipgloss.Color("#E5A100")
	titleStyle := lipgloss.NewStyle().Foreground(golden).Bold(true)
	detailStyle := lipgloss.NewStyle().Foreground(t.TextPrimary)
	selectedStyle := lipgloss.NewStyle().Foreground(golden).Bold(true)
	unselectedStyle := lipgloss.NewStyle().Foreground(t.TextPrimary)
	hintStyle := lipgloss.NewStyle().Foreground(t.TextPrimary)

	lines := []string{titleStyle.Render(p.title)}
	if strings.TrimSpace(p.message) != "" {
		lines = append(lines, detailStyle.Render(p.message))
	}
	lines = append(lines, "")
	for i, opt := range p.options {
		prefix := "  "
		style := unselectedStyle
		if i == p.selected {
			prefix = "❯ "
			style = selectedStyle
		}
		lines = append(lines, prefix+style.Render(opt.Label))
	}
	lines = append(lines, "", hintStyle.Render("↑/↓ select · enter confirm · esc cancel"))

	// Find max visible width to ensure uniform box sizing.
	content := strings.Join(lines, "\n")
	maxW := 0
	for _, line := range lines {
		if w := lipgloss.Width(line); w > maxW {
			maxW = w
		}
	}
	return lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(golden).
		Padding(0, 1).
		Width(maxW + 2). // +2 for padding
		Render(content)
}

func toPermissionsViewState(ev model.Event) *permissionsViewState {
	data := ev.Permissions
	if data == nil {
		data = &model.PermissionsViewData{}
	}
	return &permissionsViewState{
		tab:          0,
		search:       "",
		searchCursor: 0,
		selected:     0,
		allow:        append([]string{}, data.Allow...),
		ask:          append([]string{}, data.Ask...),
		deny:         append([]string{}, data.Deny...),
	}
}

func permissionsFilteredItems(v *permissionsViewState) []string {
	items := []string{}
	var source []string
	switch v.tab {
	case 0:
		items = append(items, "Add a new rule…")
		source = v.allow
	case 1:
		items = append(items, "Add a new rule…")
		source = v.ask
	case 2:
		items = append(items, "Add a new rule…")
		source = v.deny
	}
	items = append(items, source...)
	query := strings.TrimSpace(strings.ToLower(v.search))
	if query == "" {
		return items
	}
	filtered := make([]string, 0, len(items))
	for _, it := range items {
		if strings.Contains(strings.ToLower(it), query) {
			filtered = append(filtered, it)
		}
	}
	return filtered
}

func permissionsLevelByTab(tab int) string {
	switch tab {
	case 0:
		return "allow_always"
	case 1:
		return "ask"
	case 2:
		return "deny"
	default:
		return "ask"
	}
}

func permissionScopeByChoice(choice int) string {
	switch choice {
	case 1:
		return "user"
	default:
		return "project"
	}
}

func permissionsRuleToAddCommand(tab int, raw, scope string) (string, bool) {
	rule := strings.TrimSpace(raw)
	if rule == "" {
		return "", false
	}
	level := permissionsLevelByTab(tab)
	scope = strings.ToLower(strings.TrimSpace(scope))
	if scope == "" {
		scope = "project"
	}
	if scope != "project" && scope != "user" {
		return "", false
	}
	return internalPermissionsActionPrefix + "add " + level + " " + rule + " --scope " + scope, true
}

func moveCursorLeft(cursor int) int {
	if cursor <= 0 {
		return 0
	}
	return cursor - 1
}

func moveCursorRight(s string, cursor int) int {
	n := len([]rune(s))
	if cursor >= n {
		return n
	}
	return cursor + 1
}

func insertRunesAtCursor(s string, cursor int, add []rune) (string, int) {
	r := []rune(s)
	if cursor < 0 {
		cursor = 0
	}
	if cursor > len(r) {
		cursor = len(r)
	}
	out := make([]rune, 0, len(r)+len(add))
	out = append(out, r[:cursor]...)
	out = append(out, add...)
	out = append(out, r[cursor:]...)
	return string(out), cursor + len(add)
}

func deleteRuneBeforeCursor(s string, cursor int) (string, int) {
	r := []rune(s)
	if cursor <= 0 || len(r) == 0 {
		if cursor < 0 {
			return s, 0
		}
		return s, cursor
	}
	if cursor > len(r) {
		cursor = len(r)
	}
	out := make([]rune, 0, len(r)-1)
	out = append(out, r[:cursor-1]...)
	out = append(out, r[cursor:]...)
	return string(out), cursor - 1
}

func deleteRuneAtCursor(s string, cursor int) (string, int) {
	r := []rune(s)
	if len(r) == 0 || cursor < 0 || cursor >= len(r) {
		if cursor < 0 {
			return s, 0
		}
		if cursor > len(r) {
			return s, len(r)
		}
		return s, cursor
	}
	out := make([]rune, 0, len(r)-1)
	out = append(out, r[:cursor]...)
	out = append(out, r[cursor+1:]...)
	return string(out), cursor
}

func renderDialogInputWithCursor(input string, cursor int, placeholder string) string {
	if input == "" {
		return placeholder
	}
	r := []rune(input)
	if len(r) == 0 {
		return placeholder
	}
	if cursor < 0 {
		cursor = 0
	}
	if cursor > len(r) {
		cursor = len(r)
	}

	visible := func(ch rune) string {
		if ch == ' ' {
			return "\u00a0"
		}
		return string(ch)
	}

	var b strings.Builder
	for i, ch := range r {
		s := visible(ch)
		if i == cursor {
			b.WriteString(lipgloss.NewStyle().Reverse(true).Render(s))
			continue
		}
		b.WriteString(s)
	}
	if cursor == len(r) {
		b.WriteString(lipgloss.NewStyle().Reverse(true).Render("\u00a0"))
	}
	return b.String()
}

func permissionRuleDescription(rule string) string {
	rule = strings.TrimSpace(rule)
	if rule == "" {
		return "The permission rule"
	}
	if idx := strings.Index(rule, "("); idx > 0 && strings.HasSuffix(rule, ")") {
		tool := strings.TrimSpace(rule[:idx])
		spec := strings.TrimSpace(rule[idx+1 : len(rule)-1])
		switch strings.ToLower(tool) {
		case "bash":
			if spec != "" {
				return "The Bash command " + spec
			}
		case "webfetch":
			if spec != "" {
				return "The WebFetch request " + spec
			}
		}
		return fmt.Sprintf("The %s rule %s", tool, spec)
	}
	return "The " + rule + " permission rule"
}

func permissionsRemoveCommandForItem(tab int, item string) (string, bool) {
	it := strings.TrimSpace(item)
	if it == "" || it == "Add a new rule…" {
		return "", false
	}
	lower := strings.ToLower(it)
	if strings.HasPrefix(lower, "bash(") && strings.HasSuffix(lower, ")") {
		cmd := strings.TrimSpace(it[len("Bash(") : len(it)-1])
		if cmd == "" {
			return "", false
		}
		return internalPermissionsActionPrefix + "remove command " + cmd, true
	}
	if strings.HasPrefix(lower, "path(") && strings.HasSuffix(lower, ")") {
		pattern := strings.TrimSpace(it[len("Path(") : len(it)-1])
		if pattern == "" {
			return "", false
		}
		return internalPermissionsActionPrefix + "remove path " + pattern, true
	}
	if strings.HasPrefix(lower, "edit(") && strings.HasSuffix(lower, ")") {
		pattern := strings.TrimSpace(it[len("Edit(") : len(it)-1])
		if pattern == "" {
			return "", false
		}
		return internalPermissionsActionPrefix + "remove path " + pattern, true
	}
	if strings.HasPrefix(it, "/") {
		return "", false
	}
	return internalPermissionsActionPrefix + "remove tool " + it, true
}

func renderPermissionsViewPopup(v *permissionsViewState) string {
	tabs := []string{"Allow", "Ask", "Deny"}
	header := fmt.Sprintf("Permissions:  %s  (tab/shift+tab to cycle)", strings.Join(tabs, "   "))

	headerStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("252")).Bold(true)
	selectedTabStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("14")).Bold(true)
	normalStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("7"))
	selectedStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("14")).Bold(true)
	hintStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("8")).Italic(true)
	dimStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("8"))

	tabRendered := make([]string, len(tabs))
	for i, tab := range tabs {
		if i == v.tab {
			tabRendered[i] = selectedTabStyle.Render(tab)
		} else {
			tabRendered[i] = tab
		}
	}
	header = fmt.Sprintf("Permissions:  %s  (tab/shift+tab to cycle)", strings.Join(tabRendered, "   "))

	modeMsg := "Claude Code won't ask before using allowed tools."
	switch v.tab {
	case 1:
		modeMsg = "Claude Code will always ask for confirmation before using these tools."
	case 2:
		modeMsg = "Claude Code will always reject requests to use denied tools."
	}
	searchValue := renderDialogInputWithCursor(v.search, v.searchCursor, "Search…")
	searchBox := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("244")).
		Padding(0, 1).
		Render("⌕ " + searchValue)

	items := permissionsFilteredItems(v)
	if v.dialogMode != permissionsDialogNone {
		return renderPermissionsDialog(v)
	}

	lines := []string{headerStyle.Render(header), "", normalStyle.Render(modeMsg), searchBox, ""}
	if len(items) == 0 {
		lines = append(lines, dimStyle.Render("No rules matched your search."))
	} else {
		for i, item := range items {
			prefix := "  "
			style := normalStyle
			if i == v.selected {
				prefix = "❯ "
				style = selectedStyle
			}
			lines = append(lines, prefix+style.Render(fmt.Sprintf("%d. %s", i+1, item)))
		}
	}
	lines = append(lines, "", hintStyle.Render("Press ↑↓ to navigate · Enter to select · Type to search · ←/→ move cursor · Esc to cancel"))

	return strings.Join(lines, "\n")
}

func renderPermissionsDialog(v *permissionsViewState) string {
	titleStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("252")).Bold(true)
	normalStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("7"))
	selectedStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("14")).Bold(true)

	switch v.dialogMode {
	case permissionsDialogAddRule:
		title := "Add allow permission rule"
		switch v.tab {
		case 1:
			title = "Add ask permission rule"
		case 2:
			title = "Add deny permission rule"
		}
		input := renderDialogInputWithCursor(v.dialogInput, v.dialogCursor, "Enter permission rule…")
		box := lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("244")).
			Padding(0, 1).
			Render(input)
		lines := []string{
			titleStyle.Render(title),
			"",
			normalStyle.Render("Permission rules are a tool name, optionally followed by a specifier in parentheses."),
			normalStyle.Render("e.g., WebFetch(domain:example.com) or Bash(ls -la)"),
			"",
			box,
			"",
			lipgloss.NewStyle().Foreground(lipgloss.Color("8")).Italic(true).Render("Enter to continue · ←/→ move cursor · Esc to cancel"),
		}
		return lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("244")).
			Padding(0, 1).
			Render(strings.Join(lines, "\n"))
	case permissionsDialogChooseRuleScope:
		title := "Add allow permission rule"
		switch v.tab {
		case 1:
			title = "Add ask permission rule"
		case 2:
			title = "Add deny permission rule"
		}
		rule := strings.TrimSpace(v.dialogRule)
		if rule == "" {
			rule = strings.TrimSpace(v.dialogInput)
		}
		desc := permissionRuleDescription(rule)
		opt1Prefix, opt2Prefix := "  ", "  "
		opt1Style, opt2Style := normalStyle, normalStyle
		switch v.dialogChoice {
		case 1:
			opt2Prefix, opt2Style = "❯ ", selectedStyle
		default:
			opt1Prefix, opt1Style = "❯ ", selectedStyle
		}
		lines := []string{
			titleStyle.Render(title),
			"",
			normalStyle.Render("  " + rule),
			normalStyle.Render("  " + desc),
			"",
			normalStyle.Render("Where should this rule be saved?"),
			opt1Prefix + opt1Style.Render("1. Project settings          Saved in .mscli/permissions.json"),
			opt2Prefix + opt2Style.Render("2. User settings             Saved in ~/.mscli/permissions.json"),
			"",
			lipgloss.NewStyle().Foreground(lipgloss.Color("8")).Italic(true).Render("Enter to confirm · Esc to cancel"),
		}
		return lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("244")).
			Padding(0, 1).
			Render(strings.Join(lines, "\n"))
	case permissionsDialogDeleteRule:
		title := "Delete allowed tool?"
		if v.tab == 1 {
			title = "Delete ask tool?"
		} else if v.tab == 2 {
			title = "Delete denied tool?"
		}
		yesPrefix := "  "
		noPrefix := "  "
		yesStyle := normalStyle
		noStyle := normalStyle
		if v.dialogChoice == 0 {
			yesPrefix = "❯ "
			yesStyle = selectedStyle
		} else {
			noPrefix = "❯ "
			noStyle = selectedStyle
		}
		lines := []string{
			titleStyle.Render(title),
			"",
			normalStyle.Render("  " + v.dialogTarget),
		}
		if strings.TrimSpace(v.dialogSource) != "" {
			lines = append(lines, normalStyle.Render("  "+v.dialogSource))
		}
		lines = append(lines,
			"",
			normalStyle.Render("Are you sure you want to delete this permission rule?"),
			"",
			yesPrefix+yesStyle.Render("1. Yes"),
			noPrefix+noStyle.Render("2. No"),
			"",
			lipgloss.NewStyle().Foreground(lipgloss.Color("8")).Italic(true).Render("Esc to cancel"),
		)
		return lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("244")).
			Padding(0, 1).
			Render(strings.Join(lines, "\n"))
	default:
		return ""
	}
}
