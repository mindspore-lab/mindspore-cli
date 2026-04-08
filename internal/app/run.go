package app

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	agentctx "github.com/mindspore-lab/mindspore-cli/agent/context"
	"github.com/mindspore-lab/mindspore-cli/agent/loop"
	"github.com/mindspore-lab/mindspore-cli/agent/session"
	"github.com/mindspore-lab/mindspore-cli/integrations/llm"
	"github.com/mindspore-lab/mindspore-cli/internal/version"
	"github.com/mindspore-lab/mindspore-cli/ui"
	"github.com/mindspore-lab/mindspore-cli/ui/components"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
	"github.com/mindspore-lab/mindspore-cli/ui/panels"
	"github.com/mindspore-lab/mindspore-cli/ui/render"
	"github.com/mindspore-lab/mindspore-cli/ui/theme"
)

const provideAPIKeyFirstMsg = "LLM unavailable: provide api key first, or /login and switch to free model."
const interruptActiveTaskToken = "__interrupt_active_task__"
const internalPermissionsActionPrefix = "\x00permissions:"
const historyReplayReadyToken = "__history_replay_ready__"

var waitReplayDelay = func(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return nil
	}
	timer := time.NewTimer(d)
	defer timer.Stop()

	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

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
	if err := theme.Apply(a.Config.UI.Theme); err != nil {
		return fmt.Errorf("theme: %w", err)
	}
	panels.InitStyles()
	components.InitStyles()
	render.InitStyles()
	ui.InitStyles()

	userCh := make(chan string, 8)
	tui := ui.New(a.EventCh, userCh, Version, a.WorkDir, a.RepoURL, a.Config.Model.Model, a.Config.Context.Window)
	if a.replayOnly || a.startupSessionPicker != nil {
		tui = ui.NewReplay(a.EventCh, userCh, Version, a.WorkDir, a.RepoURL, a.Config.Model.Model, a.Config.Context.Window)
	}
	if a.startupSessionPicker != nil {
		tui = tui.WithStartupBannerSuppressed()
	}
	if history, err := loadInputHistoryForWorkdir(a.WorkDir); err == nil {
		tui = tui.SeedInputHistory(history)
	}
	tui = tui.WithInputHistoryAppender(func(text string) {
		_ = appendInputHistory(a.WorkDir, text)
	})
	p := tea.NewProgram(tui, tuiProgramOptions()...)

	// Emit saved login so the topbar shows the user immediately.
	if a.issueUser != "" {
		a.EventCh <- model.Event{Type: model.IssueUserUpdate, Message: a.issueUser}
	}

	go a.inputLoop(userCh)
	if !a.deferHistoryReplay {
		a.startReplayHistory()
	}
	if a.permissionSettingsIssue != nil && !a.replayOnly {
		a.emitPermissionSettingsPrompt("")
	}
	if a.needsSetupPopup && a.startupSessionPicker == nil && !a.deferHistoryReplay && !a.replayOnly {
		a.emitModelSetupPopup(false) // canEscape=false on first boot
	}
	if a.startupSessionPicker != nil && a.permissionSettingsIssue == nil {
		a.openSessionPicker(a.startupSessionPicker.Mode, a.startupSessionPicker.ReplaySpeed)
		a.startupSessionPicker = nil
	}

	_, err := p.Run()
	close(userCh)
	return err
}

func tuiProgramOptions(extra ...tea.ProgramOption) []tea.ProgramOption {
	return extra
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
		if a.replayOnly {
			return
		}
		a.startDeferredStartup()
		return
	}
	if trimmed == historyReplayReadyToken {
		if a.deferHistoryReplay {
			a.startReplayHistory()
		}
		return
	}

	if a.replayOnly {
		if trimmed == interruptActiveTaskToken {
			a.interruptReplay()
		}
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

	if strings.HasPrefix(trimmed, modelSetupToken+" ") {
		parts := strings.Fields(trimmed)
		a.cmdModelSetup(parts[1:])
		return
	}

	if strings.HasPrefix(trimmed, "/") {
		a.handleCommand(trimmed)
		return
	}

	expanded, err := a.expandInputText(trimmed)
	if err != nil {
		a.emitInputExpansionError(err)
		return
	}
	a.EventCh <- model.Event{Type: model.UserInput, Message: expanded}

	go a.runTask(expanded)
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
		if a.startupSessionPicker != nil {
			a.openSessionPicker(a.startupSessionPicker.Mode, a.startupSessionPicker.ReplaySpeed)
			a.startupSessionPicker = nil
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
			emit(model.Event{
				Type:     model.ToolWarning,
				ToolName: "Engine",
				Message:  errMsg,
			})
			persistSnapshot()
			return
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
	if a.replayOnly {
		defer a.finishReplayMode()
	}
	if len(a.replayTimeline) > 0 {
		a.replayHistoryTimeline()
		return
	}
	for _, ev := range a.replayBacklog {
		a.EventCh <- ev
	}
	if len(a.replayBacklog) == 0 || a.ctxManager == nil {
		return
	}
	a.emitTokenUsageSnapshot()
}

func (a *Application) finishReplayMode() {
	if a == nil {
		return
	}
	a.replayOnly = false
	a.replayTimeline = nil
	a.replayBacklog = nil
}

func (a *Application) startReplayHistory() {
	if a == nil {
		return
	}
	if !a.historyReplayStarted.CompareAndSwap(false, true) {
		return
	}
	go a.replayHistory()
}

func (a *Application) replayHistoryTimeline() {
	if a == nil {
		return
	}

	ctx, cancel := context.WithCancel(context.Background())
	a.setReplayCancel(cancel)
	defer a.clearReplayCancel()

	var previousFrame *session.ReplayFrame
	for i, frame := range a.replayTimeline {
		if i > 0 {
			delay := a.replayDelayBetween(previousFrame, frame)
			if delay > 0 {
				if err := waitReplayDelay(ctx, a.scaledReplayDelay(delay)); err != nil {
					return
				}
			}
		}

		select {
		case <-ctx.Done():
			return
		case a.EventCh <- frame.Event:
		}
		current := frame
		previousFrame = &current
	}

	if len(a.replayTimeline) == 0 || a.ctxManager == nil {
		return
	}
	a.emitTokenUsageSnapshot()
}

func (a *Application) scaledReplayDelay(delay time.Duration) time.Duration {
	if delay <= 0 {
		return 0
	}
	speed := replaySpeedOrDefault(a.replaySpeed)
	scaled := float64(delay) / speed
	if scaled < 1 {
		return time.Nanosecond
	}
	return time.Duration(math.Round(scaled))
}

func (a *Application) replayDelayBetween(previous *session.ReplayFrame, current session.ReplayFrame) time.Duration {
	if previous == nil {
		return 0
	}
	if shouldSkipReplayDelay(previous.Event.Type, current.Event.Type) {
		return 0
	}
	return current.Timestamp.Sub(previous.Timestamp)
}

func shouldSkipReplayDelay(previousType, currentType model.EventType) bool {
	if currentType != model.UserInput {
		return false
	}
	switch previousType {
	case model.AgentReply, model.AgentReplyDelta:
		return true
	default:
		return false
	}
}

func (a *Application) setReplayCancel(cancel context.CancelFunc) {
	if a == nil {
		return
	}
	a.taskMu.Lock()
	defer a.taskMu.Unlock()
	a.replayCancel = cancel
}

func (a *Application) clearReplayCancel() {
	if a == nil {
		return
	}
	a.taskMu.Lock()
	defer a.taskMu.Unlock()
	a.replayCancel = nil
}

func (a *Application) interruptReplay() bool {
	if a == nil {
		return false
	}

	a.taskMu.Lock()
	cancel := a.replayCancel
	a.taskMu.Unlock()
	if cancel == nil {
		return false
	}

	cancel()
	return true
}

func (a *Application) emitTokenUsageSnapshot() {
	if a == nil || a.EventCh == nil || a.ctxManager == nil {
		return
	}
	details := a.displayTokenUsageDetails()
	if details.ContextWindow <= 0 {
		return
	}
	a.EventCh <- model.Event{
		Type:    model.TokenUpdate,
		CtxUsed: details.Current,
		CtxMax:  details.ContextWindow,
	}
}

func (a *Application) displayTokenUsageDetails() agentctx.TokenUsageDetails {
	if a == nil || a.ctxManager == nil {
		return agentctx.TokenUsageDetails{}
	}

	details := a.ctxManager.TokenUsageDetails()
	if details.Source != agentctx.TokenUsageSourceLocalEstimate {
		return details
	}
	if len(a.ctxManager.GetNonSystemMessages()) > 0 {
		return details
	}

	details.Current = 0
	details.LocalEstimatedTotal = 0
	details.Available = details.ContextWindow - details.Reserved
	return details
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
	return a.session.SaveSnapshotWithUsage(
		a.currentSystemPrompt(),
		a.ctxManager.GetNonSystemMessages(),
		providerUsageSnapshotFromDetails(a.ctxManager.TokenUsageDetails()),
	)
}

func (a *Application) noteLiveLLMActivity() error {
	if a == nil || a.session == nil {
		return nil
	}
	if a.sessionLLMActivity.Load() {
		return nil
	}
	if err := a.session.Activate(); err != nil {
		return err
	}
	a.ensureSessionPermissionStore()
	a.sessionLLMActivity.Store(true)
	return nil
}

func (a *Application) exitResumeHint() string {
	if a == nil || a.replayOnly || a.session == nil {
		return ""
	}
	if !a.sessionLLMActivity.Load() {
		return ""
	}
	return cliResumeHintForSession(a.session.ID())
}

func cliResumeHintForSession(sessionID string) string {
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" {
		return ""
	}
	return fmt.Sprintf("Resume the previous conversation with: mscli resume %s", sessionID)
}

func inlineResumeHintForSession(sessionID string) string {
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" {
		return ""
	}
	return fmt.Sprintf("Resume the previous conversation with: /resume %s", sessionID)
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
	if len(args) > 0 && args[0] == "replay" {
		fs := flag.NewFlagSet("mindspore-cli replay", flag.ContinueOnError)
		fs.SetOutput(os.Stderr)
		debug := fs.Bool("debug", false, "Dump raw LLM requests/responses into the session directory")
		if err := fs.Parse(args[1:]); err != nil {
			return BootstrapConfig{}, err
		}
		target, speed, err := parseReplayTargetAndSpeed(fs.Args())
		if err != nil {
			return BootstrapConfig{}, err
		}
		return BootstrapConfig{
			Replay:          true,
			ReplaySessionID: target,
			ReplaySpeed:     speed,
			Debug:           *debug,
		}, nil
	}

	if len(args) > 0 && args[0] == "resume" {
		fs := flag.NewFlagSet("mscli resume", flag.ContinueOnError)
		fs.SetOutput(os.Stderr)
		url := fs.String("url", "", "LLM API base URL")
		modelFlag := fs.String("model", "", "Model name")
		apiKey := fs.String("api-key", "", "API key")
		debug := fs.Bool("debug", false, "Dump raw LLM requests/responses into the session directory")
		if err := fs.Parse(args[1:]); err != nil {
			return BootstrapConfig{}, err
		}
		rest := fs.Args()
		if len(rest) > 1 {
			return BootstrapConfig{}, fmt.Errorf("usage: mscli resume [sess_xxx]")
		}
		cfg := BootstrapConfig{
			URL:    *url,
			Model:  *modelFlag,
			Key:    *apiKey,
			Debug:  *debug,
			Resume: true,
		}
		if len(rest) == 1 {
			cfg.ResumeSessionID = rest[0]
		}
		return cfg, nil
	}

	fs := flag.NewFlagSet("mscli", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	url := fs.String("url", "", "LLM API base URL")
	modelFlag := fs.String("model", "", "Model name")
	apiKey := fs.String("api-key", "", "API key")
	debug := fs.Bool("debug", false, "Dump raw LLM requests/responses into the session directory")

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
		Debug: *debug,
	}, nil
}

func parseReplayTargetAndSpeed(args []string) (string, float64, error) {
	usageErr := func() error {
		return fmt.Errorf("usage: mindspore-cli replay [sess_xxx|trajectory.json|trajectory.jsonl] [speed]")
	}

	switch len(args) {
	case 0:
		return "", 1, nil
	case 1:
		if speed, ok := parseReplaySpeed(args[0]); ok {
			return "", speed, nil
		}
		return args[0], 1, nil
	case 2:
		speed, ok := parseReplaySpeed(args[1])
		if !ok {
			return "", 0, usageErr()
		}
		return args[0], speed, nil
	default:
		return "", 0, usageErr()
	}
}

func parseReplaySpeed(raw string) (float64, bool) {
	raw = strings.TrimSpace(strings.TrimSuffix(strings.ToLower(raw), "x"))
	if raw == "" {
		return 0, false
	}
	speed, err := strconv.ParseFloat(raw, 64)
	if err != nil || speed <= 0 {
		return 0, false
	}
	return speed, true
}

var loopEventTypeMap = map[string]model.EventType{
	"ToolCallStart":       model.ToolCallStart,
	"AgentReply":          model.AgentReply,
	"AgentReplyDelta":     model.AgentReplyDelta,
	"AgentBackgroundWork": model.AgentBackgroundWork,
	"AgentThinking":       model.AgentThinking,
	"ContextCompacted":    model.ContextNotice,
	"ToolRead":            model.ToolRead,
	"ToolGrep":            model.ToolGrep,
	"ToolGlob":            model.ToolGlob,
	"ToolEdit":            model.ToolEdit,
	"ToolWrite":           model.ToolWrite,
	"ToolSkill":           model.ToolSkill,
	"ToolError":           model.ToolError,
	"ToolInterrupted":     model.ToolInterrupted,
	"CmdStarted":          model.CmdStarted,
	"CmdOutput":           model.CmdOutput,
	"CmdFinished":         model.CmdFinished,
	"AnalysisReady":       model.AnalysisReady,
	"TokenUpdate":         model.TokenUpdate,
	"TaskFailed":          model.ToolError,
}

// convertLoopEvent maps loop.Event -> UI model.Event.
func convertLoopEvent(ev loop.Event) *model.Event {
	uiType, ok := loopEventTypeMap[ev.Type]
	if !ok {
		if ev.Type == "TaskStarted" {
			return nil
		}
		if ev.Type == "TaskCompleted" {
			return &model.Event{Type: model.TaskDone}
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
		ToolCallID: ev.ToolCallID,
		Summary:    ev.Summary,
		Meta:       ev.Meta,
		CtxUsed:    ev.CtxUsed,
		CtxMax:     ev.CtxMax,
		TokensUsed: ev.TokensUsed,
	}
}

func generateTaskID() string {
	return time.Now().Format("20060102-150405-000")
}
