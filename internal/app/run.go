package app

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/vigo999/ms-cli/agent/loop"
	"github.com/vigo999/ms-cli/integrations/llm"
	"github.com/vigo999/ms-cli/internal/version"
	"github.com/vigo999/ms-cli/ui"
	"github.com/vigo999/ms-cli/ui/model"
)

const provideAPIKeyFirstMsg = "provide api key first"
const interruptActiveTaskToken = "__interrupt_active_task__"
const internalPermissionsActionPrefix = "\x00permissions:"

// Run parses CLI args, wires dependencies, and starts the application.
func Run(args []string) error {
	if len(args) > 0 && (args[0] == "--version" || args[0] == "-v") {
		fmt.Println(version.Version)
		return nil
	}

	cfg, err := parseBootstrapConfig(args)
	if err != nil {
		return err
	}

	app, err := Wire(cfg)
	if err != nil {
		return err
	}

	return app.run()
}

// run starts the TUI.
func (a *Application) run() error {
	go cleanUpdateTmp()
	if checkAndPromptUpdate() {
		return nil
	}
	err := a.runReal()
	resumeHint := a.exitResumeHint()
	if a.session != nil {
		_ = a.session.Close()
	}
	if err == nil && resumeHint != "" {
		fmt.Fprintln(os.Stdout, resumeHint)
	}
	return err
}

func (a *Application) runReal() error {
	userCh := make(chan string, 8)
	tui := ui.New(a.EventCh, userCh, Version, a.WorkDir, a.RepoURL, a.Config.Model.Model, a.Config.Context.Window)
	p := tea.NewProgram(tui, tuiProgramOptions()...)

	// Emit saved login so the topbar shows the user immediately.
	if a.issueUser != "" {
		a.EventCh <- model.Event{Type: model.IssueUserUpdate, Message: a.issueUser}
	}

	go a.replayHistory()
	go a.inputLoop(userCh)
	if a.permissionSettingsIssue != nil {
		a.emitPermissionSettingsPrompt("")
	}

	_, err := p.Run()
	close(userCh)
	return err
}

func tuiProgramOptions(extra ...tea.ProgramOption) []tea.ProgramOption {
	opts := []tea.ProgramOption{
		tea.WithAltScreen(),
	}
	return append(opts, extra...)
}

func (a *Application) inputLoop(userCh <-chan string) {
	for input := range userCh {
		a.processInput(input)
	}
}

func (a *Application) processInput(input string) {
	if strings.HasPrefix(input, internalPermissionsActionPrefix) {
		payload := strings.TrimPrefix(input, internalPermissionsActionPrefix)
		a.cmdPermissionsInternal(strings.Fields(payload))
		return
	}

	trimmed := strings.TrimSpace(input)
	if trimmed == "" {
		return
	}

	if trimmed == bootReadyToken {
		a.startDeferredStartup()
		return
	}

	if a.permissionSettingsIssue != nil {
		a.handlePermissionSettingsPromptInput(trimmed)
		return
	}

	if trimmed == interruptQueuedTrainToken {
		a.interruptQueuedTrain()
		return
	}

	if trimmed == interruptActiveTaskToken {
		a.interruptActiveTasks()
		return
	}

	if a.permissionUI != nil && a.permissionUI.HandleInput(trimmed) {
		return
	}

	if strings.HasPrefix(trimmed, "/") {
		a.handleCommand(trimmed)
		return
	}

	go a.runTask(trimmed)
}

func (a *Application) handlePermissionSettingsPromptInput(input string) {
	if a == nil || a.permissionSettingsIssue == nil {
		return
	}
	switch strings.ToLower(strings.TrimSpace(input)) {
	case "1", "y", "yes", "exit":
		a.permissionSettingsIssue = nil
		if a.EventCh != nil {
			a.EventCh <- model.Event{Type: model.Done}
		}
	case "2", "c", "continue":
		a.permissionSettingsIssue = nil
		if a.EventCh != nil {
			a.EventCh <- model.Event{
				Type:    model.AgentReply,
				Message: "Continuing without the invalid permission settings file.",
			}
		}
	default:
		a.emitPermissionSettingsPrompt("Please choose 1 or 2.")
	}
}

func (a *Application) runTask(description string) {
	emit := func(ev model.Event) { a.EventCh <- ev }
	persistSnapshot := func() {
		if err := a.persistSessionSnapshot(); err != nil {
			a.emitToolError("session", "Failed to persist session snapshot: %v", err)
		}
	}
	if err := a.activateSessionPersistence(); err != nil {
		a.emitToolError("session", "Failed to start session persistence: %v", err)
		return
	}

	if !a.llmReady {
		if err := a.recordUnavailableTurn(description, provideAPIKeyFirstMsg); err != nil {
			a.emitToolError("context", "Failed to record local turn: %v", err)
			return
		}
		persistSnapshot()
		emit(model.Event{
			Type:    model.AgentReply,
			Message: provideAPIKeyFirstMsg,
		})
		return
	}

	task := loop.Task{
		ID:          generateTaskID(),
		Description: description,
	}
	ctx, runID := a.beginTaskRun()
	defer a.finishTaskRun(runID)

	err := a.Engine.RunWithContextStream(ctx, task, func(ev loop.Event) {
		uiEvent := convertLoopEvent(ev)
		if uiEvent != nil {
			emit(*uiEvent)
		}
	})
	if errors.Is(err, context.Canceled) {
		persistSnapshot()
		return
	}
	if err != nil {
		errMsg := err.Error()
		if strings.Contains(errMsg, "timeout") || strings.Contains(errMsg, "deadline") {
			errMsg = fmt.Sprintf("%s\n\nTip: The request timed out. Try:\n  1. Run /compact to reduce context size\n  2. Start a new conversation with /clear\n  3. Increase timeout in config (model.timeout_sec)", errMsg)
		}
		emit(model.Event{
			Type:     model.ToolError,
			ToolName: "Engine",
			Message:  errMsg,
		})
		persistSnapshot()
		return
	}
	persistSnapshot()
}

func (a *Application) beginTaskRun() (context.Context, uint64) {
	ctx, cancel := context.WithCancel(context.Background())

	a.taskMu.Lock()
	defer a.taskMu.Unlock()

	a.taskRunID++
	runID := a.taskRunID
	if a.taskCancels == nil {
		a.taskCancels = map[uint64]context.CancelFunc{}
	}
	a.taskCancels[runID] = cancel
	return ctx, runID
}

func (a *Application) finishTaskRun(runID uint64) {
	a.taskMu.Lock()
	defer a.taskMu.Unlock()

	if len(a.taskCancels) == 0 {
		return
	}
	delete(a.taskCancels, runID)
	if len(a.taskCancels) == 0 {
		a.taskCancels = nil
	}
}

func (a *Application) interruptActiveTasks() bool {
	a.taskMu.Lock()
	if len(a.taskCancels) == 0 {
		a.taskMu.Unlock()
		return false
	}

	cancels := make([]context.CancelFunc, 0, len(a.taskCancels))
	for _, cancel := range a.taskCancels {
		if cancel != nil {
			cancels = append(cancels, cancel)
		}
	}
	a.taskCancels = nil
	a.taskMu.Unlock()

	for _, cancel := range cancels {
		cancel()
	}
	return true
}

func (a *Application) replayHistory() {
	for _, ev := range a.replayBacklog {
		a.EventCh <- ev
	}
	if len(a.replayBacklog) == 0 || a.ctxManager == nil {
		return
	}
	usage := a.ctxManager.TokenUsage()
	if usage.Max <= 0 {
		return
	}
	a.EventCh <- model.Event{
		Type:    model.TokenUpdate,
		CtxUsed: usage.Current,
		CtxMax:  usage.Max,
	}
}

func (a *Application) addContextMessages(msgs ...llm.Message) error {
	if a == nil || a.ctxManager == nil {
		return nil
	}

	for _, msg := range msgs {
		if err := a.ctxManager.AddMessage(msg); err != nil {
			return err
		}
	}
	return nil
}

func (a *Application) persistSessionSnapshot() error {
	if a == nil || a.session == nil || a.ctxManager == nil {
		return nil
	}
	return a.session.SaveSnapshot(a.currentSystemPrompt(), a.ctxManager.GetNonSystemMessages())
}

func (a *Application) activateSessionPersistence() error {
	if a == nil || a.session == nil {
		return nil
	}
	return a.session.Activate()
}

func (a *Application) exitResumeHint() string {
	if a == nil || a.session == nil || !a.session.HasPersistedDialogue() {
		return ""
	}

	sessionID := strings.TrimSpace(a.session.ID())
	if sessionID == "" {
		return ""
	}
	return fmt.Sprintf("Resume this session with: ms-cli resume %s", sessionID)
}

func (a *Application) recordUnavailableTurn(userInput, assistantReply string) error {
	if err := a.addContextMessages(
		llm.NewUserMessage(userInput),
		llm.NewAssistantMessage(assistantReply),
	); err != nil {
		return err
	}
	if a.session == nil {
		return nil
	}
	if err := a.session.AppendUserInput(userInput); err != nil {
		return err
	}
	if err := a.session.AppendAssistant(assistantReply); err != nil {
		return err
	}
	return nil
}

func (a *Application) currentSystemPrompt() string {
	if a == nil || a.ctxManager == nil {
		return ""
	}
	if msg := a.ctxManager.GetSystemPrompt(); msg != nil {
		return msg.Content
	}
	return ""
}

func (a *Application) emitToolError(toolName, format string, args ...any) {
	if a == nil || a.EventCh == nil {
		return
	}
	a.EventCh <- model.Event{
		Type:     model.ToolError,
		ToolName: toolName,
		Message:  fmt.Sprintf(format, args...),
	}
}

func parseBootstrapConfig(args []string) (BootstrapConfig, error) {
	if len(args) > 0 && args[0] == "resume" {
		fs := flag.NewFlagSet("ms-cli resume", flag.ContinueOnError)
		fs.SetOutput(os.Stderr)
		url := fs.String("url", "", "LLM API base URL")
		modelFlag := fs.String("model", "", "Model name")
		apiKey := fs.String("api-key", "", "API key")
		if err := fs.Parse(args[1:]); err != nil {
			return BootstrapConfig{}, err
		}
		rest := fs.Args()
		if len(rest) > 1 {
			return BootstrapConfig{}, fmt.Errorf("usage: ms-cli resume [sess_xxx]")
		}
		cfg := BootstrapConfig{
			URL:    *url,
			Model:  *modelFlag,
			Key:    *apiKey,
			Resume: true,
		}
		if len(rest) == 1 {
			cfg.ResumeSessionID = rest[0]
		}
		return cfg, nil
	}

	fs := flag.NewFlagSet("ms-cli", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	url := fs.String("url", "", "LLM API base URL")
	modelFlag := fs.String("model", "", "Model name")
	apiKey := fs.String("api-key", "", "API key")

	if err := fs.Parse(args); err != nil {
		return BootstrapConfig{}, err
	}
	if len(fs.Args()) > 0 {
		return BootstrapConfig{}, fmt.Errorf("unknown subcommand: %s", fs.Args()[0])
	}

	return BootstrapConfig{
		URL:   *url,
		Model: *modelFlag,
		Key:   *apiKey,
	}, nil
}

var loopEventTypeMap = map[string]model.EventType{
	"ToolCallStart":   model.ToolCallStart,
	"AgentReply":      model.AgentReply,
	"AgentReplyDelta": model.AgentReplyDelta,
	"AgentThinking":   model.AgentThinking,
	"ToolRead":        model.ToolRead,
	"ToolGrep":        model.ToolGrep,
	"ToolGlob":        model.ToolGlob,
	"ToolEdit":        model.ToolEdit,
	"ToolWrite":       model.ToolWrite,
	"ToolSkill":       model.ToolSkill,
	"ToolError":       model.ToolError,
	"CmdStarted":      model.CmdStarted,
	"AnalysisReady":   model.AnalysisReady,
	"TokenUpdate":     model.TokenUpdate,
	"TaskFailed":      model.ToolError,
}

// convertLoopEvent maps loop.Event -> UI model.Event.
func convertLoopEvent(ev loop.Event) *model.Event {
	uiType, ok := loopEventTypeMap[ev.Type]
	if !ok {
		if ev.Type == "TaskCompleted" || ev.Type == "TaskStarted" {
			return nil
		}
		if ev.Message != "" {
			return &model.Event{Type: model.AgentReply, Message: ev.Message}
		}
		return nil
	}

	return &model.Event{
		Type:       uiType,
		Message:    ev.Message,
		ToolName:   ev.ToolName,
		Summary:    ev.Summary,
		CtxUsed:    ev.CtxUsed,
		CtxMax:     ev.CtxMax,
		TokensUsed: ev.TokensUsed,
	}
}

func generateTaskID() string {
	return time.Now().Format("20060102-150405-000")
}
