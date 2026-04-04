package model

import (
	"fmt"
	"time"

	"github.com/mindspore-lab/mindspore-cli/internal/bugs"
	issuepkg "github.com/mindspore-lab/mindspore-cli/internal/issues"
)

// TaskInfo represents a task in the task pool.
type TaskInfo struct {
	ID   string
	Name string
}

// ModelInfo holds LLM model metadata for the top bar.
type ModelInfo struct {
	Name       string
	CtxUsed    int
	CtxMax     int
	TokensUsed int
}

// MessageKind distinguishes chat message types.
type MessageKind int

const (
	MsgUser MessageKind = iota
	MsgAgent
	MsgTool
)

// WaitKind identifies the current wait state shown in the chat UI.
type WaitKind int

const (
	WaitNone WaitKind = iota
	WaitModel
	WaitTool
)

// DisplayMode controls how a tool message is rendered.
type DisplayMode int

const (
	DisplayExpanded  DisplayMode = iota // full output shown (Shell user-cmd, Edit, Write)
	DisplayCollapsed                    // 1-line summary (Read, Grep, Glob, agent-internal Shell)
	DisplayWarning                      // expanded + yellow highlight
	DisplayError                        // expanded + red highlight
	DisplayNotice                       // non-reply agent text (context notices, etc.)
)

// Message is a single entry in the chat stream.
type Message struct {
	Kind       MessageKind
	Content    string
	RawANSI    bool
	ToolName   string
	ToolCallID string
	ToolArgs   string
	Display    DisplayMode
	Summary    string // shown when collapsed, e.g. "5 matches", "23 files"
	Pending    bool
	Streaming  bool
}

// EventType identifies the kind of UI event.
type EventType string

const (
	TaskUpdated      EventType = "TaskUpdated"
	ToolCallStart    EventType = "ToolCallStart"
	CmdStarted       EventType = "CmdStarted"
	CmdOutput        EventType = "CmdOutput"
	CmdFinished      EventType = "CmdFinished"
	AnalysisReady    EventType = "AnalysisReady"
	AgentReply       EventType = "AgentReply"
	AgentReplyDelta  EventType = "AgentReplyDelta"
	PermissionPrompt EventType = "PermissionPrompt"
	PermissionsView  EventType = "PermissionsView"
	AgentThinking    EventType = "AgentThinking"
	ContextNotice    EventType = "ContextNotice"
	UserInput        EventType = "UserInput"
	ToolReplay       EventType = "ToolReplay"
	TokenUpdate      EventType = "TokenUpdate"
	ToolRead         EventType = "ToolRead"
	ToolGrep         EventType = "ToolGrep"
	ToolGlob         EventType = "ToolGlob"
	ToolEdit         EventType = "ToolEdit"
	ToolWrite        EventType = "ToolWrite"
	ToolSkill        EventType = "ToolSkill"
	ToolWarning      EventType = "ToolWarning"
	ToolError        EventType = "ToolError"
	ClearScreen      EventType = "ClearScreen"
	ModelUpdate      EventType = "ModelUpdate"
	ModelPickerOpen      EventType = "ModelPickerOpen"
	ModelSetupOpen       EventType = "ModelSetupOpen"
	ModelSetupClose      EventType = "ModelSetupClose"
	ModelSetupTokenError EventType = "ModelSetupTokenError"
	MouseModeToggle  EventType = "MouseModeToggle"
	IssueUserUpdate  EventType = "IssueUserUpdate"
	SkillsNoteUpdate EventType = "SkillsNoteUpdate"
	TaskDone         EventType = "TaskDone"
	Done             EventType = "Done"
)

// Event is sent from the agent loop to the TUI.
// Implements tea.Msg so Bubble Tea can route it.
type Event struct {
	Type        EventType
	Task        string
	Message     string
	RawANSI     bool
	ToolName    string
	ToolCallID  string
	Summary     string
	ReplayWait  *ReplayWaitData
	CtxUsed     int
	CtxMax      int
	TokensUsed  int
	Train       *TrainEventData // non-nil for train events only
	Project     *ProjectStatusView
	Permission  *PermissionPromptData
	Permissions *PermissionsViewData
	Popup      *SelectionPopup // non-nil for popup events only
	SetupPopup *SetupPopup    // non-nil for model setup popup events
	BugView     *BugEventData   // non-nil for bug view events only
	IssueView   *IssueEventData // non-nil for issue view events only
	Bug         *bugs.Bug       // reserved for lightweight bug payloads
	Issue       *issuepkg.Issue // reserved for lightweight issue payloads
}

// ReplayWaitData lets replay fast-forward the UI timer while using shorter real delays.
type ReplayWaitData struct {
	OriginalDuration  time.Duration
	SimulatedDuration time.Duration
}

// PermissionPromptData describes a structured permission prompt for interactive UI rendering.
type PermissionPromptData struct {
	Title        string
	Message      string
	Options      []PermissionOption
	DefaultIndex int
}

type PermissionOption struct {
	// Input is the token sent back to backend permission handler, e.g. "1", "2", "3", "esc".
	Input string
	Label string
}

// PermissionsViewData is the payload for interactive /permissions view.
type PermissionsViewData struct {
	Allow       []string
	Ask         []string
	Deny        []string
	RuleSources map[string]string
}

// TaskStats tracks execution statistics for the current task.
type TaskStats struct {
	Commands    int // shell commands executed
	FilesRead   int // files read
	FilesEdited int // files edited/written
	Searches    int // grep/glob operations
	Errors      int // errors encountered
}

// State is the central UI state.
type State struct {
	Version          string
	Tasks            []TaskInfo
	ActiveTask       int
	Model            ModelInfo
	Messages         []Message
	ShowTaskSelector bool
	WorkDir          string
	RepoURL          string
	Stats            TaskStats // current task statistics
	IsThinking       bool      // whether AI is currently thinking
	WaitKind         WaitKind
	WaitStartedAt    time.Time
	WaitElapsed      time.Duration
	MouseEnabled     bool   // whether mouse mode is enabled (for scrolling)
	IssueUser        string // logged-in bug server user
	SkillsNote       string // skills repo status for hint bar
}

// NewState returns an initial empty state.
func NewState(version, workDir, repoURL, modelName string, ctxMax int) State {
	if modelName == "" {
		modelName = "No model (/model to configure)"
	}
	if ctxMax == 0 {
		ctxMax = 128000 // Default for models like gpt-4o
	}
	return State{
		Version: version,
		Tasks:   []TaskInfo{},
		Model: ModelInfo{
			Name:   modelName,
			CtxMax: ctxMax,
		},
		WorkDir:      workDir,
		RepoURL:      repoURL,
		Stats:        TaskStats{},
		IsThinking:   false,
		WaitKind:     WaitNone,
		MouseEnabled: true, // default to enabled for scroll wheel
	}
}

// WithTask returns a new State with the given task added.
func (s State) WithTask(t TaskInfo) State {
	return State{
		Version:          s.Version,
		Tasks:            append(append([]TaskInfo{}, s.Tasks...), t),
		ActiveTask:       s.ActiveTask,
		Model:            s.Model,
		Messages:         s.Messages,
		ShowTaskSelector: s.ShowTaskSelector,
		WorkDir:          s.WorkDir,
		RepoURL:          s.RepoURL,
		Stats:            s.Stats,
		IsThinking:       s.IsThinking,
		WaitKind:         s.WaitKind,
		WaitStartedAt:    s.WaitStartedAt,
		MouseEnabled:     s.MouseEnabled,
		IssueUser:        s.IssueUser,
		SkillsNote:       s.SkillsNote,
	}
}

// WithMessage returns a new State with the given message appended.
func (s State) WithMessage(m Message) State {
	return State{
		Version:          s.Version,
		Tasks:            s.Tasks,
		ActiveTask:       s.ActiveTask,
		Model:            s.Model,
		Messages:         append(append([]Message{}, s.Messages...), m),
		ShowTaskSelector: s.ShowTaskSelector,
		WorkDir:          s.WorkDir,
		RepoURL:          s.RepoURL,
		Stats:            s.Stats,
		IsThinking:       s.IsThinking,
		WaitKind:         s.WaitKind,
		WaitStartedAt:    s.WaitStartedAt,
		MouseEnabled:     s.MouseEnabled,
		IssueUser:        s.IssueUser,
		SkillsNote:       s.SkillsNote,
	}
}

// WithModel returns a new State with updated model info.
func (s State) WithModel(m ModelInfo) State {
	return State{
		Version:          s.Version,
		Tasks:            s.Tasks,
		ActiveTask:       s.ActiveTask,
		Model:            m,
		Messages:         s.Messages,
		ShowTaskSelector: s.ShowTaskSelector,
		WorkDir:          s.WorkDir,
		RepoURL:          s.RepoURL,
		Stats:            s.Stats,
		IsThinking:       s.IsThinking,
		WaitKind:         s.WaitKind,
		WaitStartedAt:    s.WaitStartedAt,
		MouseEnabled:     s.MouseEnabled,
		IssueUser:        s.IssueUser,
		SkillsNote:       s.SkillsNote,
	}
}

// WithStats returns a new State with updated stats.
func (s State) WithStats(stats TaskStats) State {
	return State{
		Version:          s.Version,
		Tasks:            s.Tasks,
		ActiveTask:       s.ActiveTask,
		Model:            s.Model,
		Messages:         s.Messages,
		ShowTaskSelector: s.ShowTaskSelector,
		WorkDir:          s.WorkDir,
		RepoURL:          s.RepoURL,
		Stats:            stats,
		IsThinking:       s.IsThinking,
		WaitKind:         s.WaitKind,
		WaitStartedAt:    s.WaitStartedAt,
		MouseEnabled:     s.MouseEnabled,
		IssueUser:        s.IssueUser,
		SkillsNote:       s.SkillsNote,
	}
}

// WithThinking returns a new State with updated thinking status.
func (s State) WithThinking(thinking bool) State {
	return State{
		Version:          s.Version,
		Tasks:            s.Tasks,
		ActiveTask:       s.ActiveTask,
		Model:            s.Model,
		Messages:         s.Messages,
		ShowTaskSelector: s.ShowTaskSelector,
		WorkDir:          s.WorkDir,
		RepoURL:          s.RepoURL,
		Stats:            s.Stats,
		IsThinking:       thinking,
		WaitKind:         s.WaitKind,
		WaitStartedAt:    s.WaitStartedAt,
		MouseEnabled:     s.MouseEnabled,
		IssueUser:        s.IssueUser,
		SkillsNote:       s.SkillsNote,
	}
}

// WithWait returns a new State with updated wait metadata.
func (s State) WithWait(kind WaitKind, startedAt time.Time) State {
	return State{
		Version:          s.Version,
		Tasks:            s.Tasks,
		ActiveTask:       s.ActiveTask,
		Model:            s.Model,
		Messages:         s.Messages,
		ShowTaskSelector: s.ShowTaskSelector,
		WorkDir:          s.WorkDir,
		RepoURL:          s.RepoURL,
		Stats:            s.Stats,
		IsThinking:       s.IsThinking,
		WaitKind:         kind,
		WaitStartedAt:    startedAt,
		MouseEnabled:     s.MouseEnabled,
		IssueUser:        s.IssueUser,
		SkillsNote:       s.SkillsNote,
	}
}

// ClearWait returns a new State without an active wait.
func (s State) ClearWait() State {
	return s.WithWait(WaitNone, time.Time{})
}

// ResetStats returns a new State with reset stats.
func (s State) ResetStats() State {
	return State{
		Version:          s.Version,
		Tasks:            s.Tasks,
		ActiveTask:       s.ActiveTask,
		Model:            s.Model,
		Messages:         s.Messages,
		ShowTaskSelector: s.ShowTaskSelector,
		WorkDir:          s.WorkDir,
		RepoURL:          s.RepoURL,
		Stats:            TaskStats{},
		IsThinking:       s.IsThinking,
		WaitKind:         s.WaitKind,
		WaitStartedAt:    s.WaitStartedAt,
		MouseEnabled:     s.MouseEnabled,
		IssueUser:        s.IssueUser,
		SkillsNote:       s.SkillsNote,
	}
}

// WithIssueUser returns a new State with updated issue user.
func (s State) WithIssueUser(user string) State {
	return State{
		Version:          s.Version,
		Tasks:            s.Tasks,
		ActiveTask:       s.ActiveTask,
		Model:            s.Model,
		Messages:         s.Messages,
		ShowTaskSelector: s.ShowTaskSelector,
		WorkDir:          s.WorkDir,
		RepoURL:          s.RepoURL,
		Stats:            s.Stats,
		IsThinking:       s.IsThinking,
		WaitKind:         s.WaitKind,
		WaitStartedAt:    s.WaitStartedAt,
		MouseEnabled:     s.MouseEnabled,
		IssueUser:        user,
		SkillsNote:       s.SkillsNote,
	}
}

// WithMouseEnabled returns a new State with updated mouse mode.
func (s State) WithMouseEnabled(enabled bool) State {
	return State{
		Version:          s.Version,
		Tasks:            s.Tasks,
		ActiveTask:       s.ActiveTask,
		Model:            s.Model,
		Messages:         s.Messages,
		ShowTaskSelector: s.ShowTaskSelector,
		WorkDir:          s.WorkDir,
		RepoURL:          s.RepoURL,
		Stats:            s.Stats,
		IsThinking:       s.IsThinking,
		WaitKind:         s.WaitKind,
		WaitStartedAt:    s.WaitStartedAt,
		MouseEnabled:     enabled,
		IssueUser:        s.IssueUser,
		SkillsNote:       s.SkillsNote,
	}
}

// FormatWaitDuration renders elapsed wait time for the UI in human-readable form.
func FormatWaitDuration(d time.Duration) string {
	if d < 0 {
		d = 0
	}
	totalSeconds := int(d / time.Second)
	if totalSeconds < 60 {
		return fmt.Sprintf("%ds", totalSeconds)
	}
	minutes := totalSeconds / 60
	seconds := totalSeconds % 60
	if seconds == 0 {
		return fmt.Sprintf("%dm", minutes)
	}
	return fmt.Sprintf("%dm %ds", minutes, seconds)
}
