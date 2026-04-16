package loop

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"time"

	ctxmanager "github.com/mindspore-lab/mindspore-cli/agent/context"
	"github.com/mindspore-lab/mindspore-cli/integrations/llm"
	"github.com/mindspore-lab/mindspore-cli/permission"
	"github.com/mindspore-lab/mindspore-cli/tools"
)

// EngineConfig holds engine configuration.
type EngineConfig struct {
	MaxIterations  int
	ContextWindow  int
	MaxTokens      *int
	Temperature    *float32
	TimeoutPerTurn time.Duration
	SystemPrompt   string
}

// Engine runs the ReAct loop: LLM → tool call → LLM → done.
type Engine struct {
	config      EngineConfig
	provider    llm.Provider
	tools       *tools.Registry
	ctxManager  *ctxmanager.Manager
	permission  permission.PermissionService
	recorder    *TrajectoryRecorder
	debugDumper *llm.DebugDumper
}

// TrajectoryRecorder records runtime conversation events for persistence.
type TrajectoryRecorder struct {
	RecordUserInput     func(string) error
	RecordAssistant     func(string) error
	RecordToolCall      func(llm.ToolCall) error
	RecordToolResult    func(llm.ToolCall, string) error
	RecordSkillActivate func(string) error
	PersistSnapshot     func() error
}

// NewEngine creates a new engine.
func NewEngine(cfg EngineConfig, provider llm.Provider, tools *tools.Registry) *Engine {
	if cfg.SystemPrompt == "" {
		cfg.SystemPrompt = DefaultSystemPrompt()
	}

	engine := &Engine{
		config:   cfg,
		provider: provider,
		tools:    tools,
	}

	managerCfg := ctxmanager.DefaultManagerConfig()
	if cfg.ContextWindow > 0 {
		managerCfg.ContextWindow = cfg.ContextWindow
	}
	managerCfg.ReserveTokens = 4000
	engine.ctxManager = ctxmanager.NewManager(managerCfg)
	engine.ctxManager.SetSystemPrompt(cfg.SystemPrompt)
	engine.permission = permission.NewNoOpPermissionService()

	return engine
}

// SetContextManager sets the context manager.
func (e *Engine) SetContextManager(cm *ctxmanager.Manager) {
	if cm == nil {
		return
	}
	if cm.GetSystemPrompt() == nil {
		switch {
		case e.ctxManager != nil && e.ctxManager.GetSystemPrompt() != nil:
			cm.SetSystemPrompt(e.ctxManager.GetSystemPrompt().Content)
		case e.config.SystemPrompt != "":
			cm.SetSystemPrompt(e.config.SystemPrompt)
		}
	}
	e.ctxManager = cm
}

// SetPermissionService sets the permission service.
func (e *Engine) SetPermissionService(ps permission.PermissionService) {
	e.permission = ps
}

// SetTrajectoryRecorder records runtime conversation events for persistence.
func (e *Engine) SetTrajectoryRecorder(recorder *TrajectoryRecorder) {
	e.recorder = recorder
}

// SetLLMDebugDumper enables raw request/response dumping for LLM calls.
func (e *Engine) SetLLMDebugDumper(dumper *llm.DebugDumper) {
	e.debugDumper = dumper
}

// ToolNames returns the names of registered tools.
func (e *Engine) ToolNames() []string {
	toolList := e.tools.List()
	names := make([]string, len(toolList))
	for i, t := range toolList {
		names[i] = t.Name()
	}
	return names
}

// Run executes a task and returns events.
func (e *Engine) Run(task Task) ([]Event, error) {
	return e.RunWithContext(context.Background(), task)
}

// RunWithContext executes the ReAct loop for a task.
func (e *Engine) RunWithContext(ctx context.Context, task Task) ([]Event, error) {
	return e.runWithContext(ctx, task, nil)
}

// RunWithContextStream executes the ReAct loop and emits events as they occur.
func (e *Engine) RunWithContextStream(ctx context.Context, task Task, sink func(Event)) error {
	_, err := e.runWithContext(ctx, task, sink)
	return err
}

func (e *Engine) runWithContext(ctx context.Context, task Task, sink func(Event)) ([]Event, error) {
	exec := &executor{
		engine:    e,
		task:      task,
		events:    make([]Event, 0),
		sink:      sink,
		startTime: time.Now(),
	}
	return exec.run(ctx)
}

// executor manages a single ReAct loop run.
type executor struct {
	engine     *Engine
	task       Task
	events     []Event
	iterCount  int
	startTime  time.Time
	totalUsage llm.Usage
	sink       func(Event)

	responsesPreviousID string
	responsesFollowup   []llm.Message
}

type contextCompactionNotice struct {
	BeforeTokens int
	AfterTokens  int
}

func (ex *executor) run(ctx context.Context) ([]Event, error) {
	notice, err := ex.addContextMessage(llm.NewUserMessage(ex.task.Description))
	if err != nil {
		ex.addEvent(NewEvent(EventTaskFailed, fmt.Sprintf("Persist message error: %v", err)))
		return ex.events, err
	}
	if ex.engine.recorder != nil && ex.engine.recorder.RecordUserInput != nil {
		if err := ex.engine.recorder.RecordUserInput(ex.task.Description); err != nil {
			ex.addEvent(NewEvent(EventTaskFailed, fmt.Sprintf("Persist message error: %v", err)))
			return ex.events, err
		}
	}
	if err := ex.persistSnapshot(); err != nil {
		ex.addEvent(NewEvent(EventTaskFailed, fmt.Sprintf("Persist snapshot error: %v", err)))
		return ex.events, err
	}
	ex.emitContextCompactionNotice(notice)
	ex.addEvent(NewEvent(EventTaskStarted, fmt.Sprintf("Task: %s", ex.task.Description)))

	completed := false
	for ex.engine.config.MaxIterations == 0 || ex.iterCount < ex.engine.config.MaxIterations {
		ex.iterCount++
		ex.addEvent(NewEvent(EventAgentThinking, ""))

		if err := ctx.Err(); err != nil {
			return ex.events, err
		}

		resp, err := ex.callLLM(ctx)
		if err != nil {
			return ex.events, err
		}

		ex.trackUsage(resp.Usage)

		continueLoop, err := ex.handleResponse(ctx, resp)
		if err != nil {
			if errors.Is(err, context.Canceled) {
				return ex.events, err
			}
			ex.addEvent(NewEvent(EventTaskFailed, fmt.Sprintf("Handle response error: %v", err)))
			return ex.events, err
		}
		if !continueLoop {
			completed = true
			break
		}
	}

	if completed {
		ex.addEvent(NewEvent(EventTaskCompleted, "Task completed successfully"))
	} else if ex.engine.config.MaxIterations > 0 && ex.iterCount >= ex.engine.config.MaxIterations {
		ex.addEvent(NewEvent(EventTaskFailed, "Task exceeded maximum iterations."))
	} else {
		ex.addEvent(NewEvent(EventTaskCompleted, "Task completed successfully"))
	}

	return ex.events, nil
}

func (ex *executor) callLLM(ctx context.Context) (*llm.CompletionResponse, error) {
	timeout := ex.engine.config.TimeoutPerTurn
	if timeout == 0 {
		timeout = 5 * time.Minute
	}

	llmCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	ex.sanitizeToolPairsBeforeRequest()

	req := &llm.CompletionRequest{
		Messages:    ex.requestMessages(),
		Tools:       ex.engine.tools.ToLLMTools(),
		Temperature: ex.engine.config.Temperature,
		MaxTokens:   ex.engine.config.MaxTokens,
	}

	if ex.usesResponsesChain() && ex.responsesPreviousID != "" {
		llmCtx = llm.WithPreviousResponseID(llmCtx, ex.responsesPreviousID)
	}
	if ex.engine.debugDumper != nil {
		llmCtx = llm.WithDebugDumper(llmCtx, ex.engine.debugDumper)
	}

	resp, err := ex.streamCompletion(llmCtx, req)
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(ctx.Err(), context.Canceled) || errors.Is(llmCtx.Err(), context.Canceled) {
			return nil, context.Canceled
		}
		if ctx.Err() == context.DeadlineExceeded || llmCtx.Err() == context.DeadlineExceeded {
			return nil, fmt.Errorf("request timeout: %w", err)
		}
		errMsg := fmt.Sprintf("LLM error: %v", err)
		ex.addEvent(NewEvent(EventTaskFailed, errMsg))
		return nil, fmt.Errorf("LLM completion: %w", err)
	}

	return resp, nil
}

func (ex *executor) sanitizeToolPairsBeforeRequest() {
	if ex.engine == nil || ex.engine.ctxManager == nil {
		return
	}

	messages := ex.engine.ctxManager.GetNonSystemMessages()
	valid := validToolCallIDs(messages)
	sanitized, report := sanitizeMessagesForValidToolCallIDs(messages, valid)
	if report.changed() {
		ex.engine.ctxManager.SetNonSystemMessages(sanitized)
		valid = validToolCallIDs(sanitized)
	}

	if ex.usesResponsesChain() && ex.responsesPreviousID != "" && len(ex.responsesFollowup) > 0 {
		sanitizedFollowup, followupReport := sanitizeMessagesForValidToolCallIDs(ex.responsesFollowup, valid)
		ex.responsesFollowup = sanitizedFollowup
		if len(ex.responsesFollowup) == 0 {
			ex.responsesPreviousID = ""
		}
		if !report.changed() && followupReport.changed() {
			report = followupReport
		}
	}

	if !report.changed() {
		return
	}

	ev := NewEvent(EventToolError, report.warningMessage())
	ev.ToolName = "context"
	ex.addEvent(ev)
}

func (ex *executor) requestMessages() []llm.Message {
	if !ex.usesResponsesChain() || ex.responsesPreviousID == "" || len(ex.responsesFollowup) == 0 {
		return ex.engine.ctxManager.GetMessages()
	}

	msgs := make([]llm.Message, 0, len(ex.responsesFollowup)+1)
	if system := ex.engine.ctxManager.GetSystemPrompt(); system != nil {
		msgs = append(msgs, *system)
	}
	msgs = append(msgs, ex.responsesFollowup...)
	return msgs
}

func (ex *executor) usesResponsesChain() bool {
	return ex.engine.provider != nil && ex.engine.provider.Name() == string(llm.ProviderOpenAIResponses)
}

func (ex *executor) streamCompletion(ctx context.Context, req *llm.CompletionRequest) (*llm.CompletionResponse, error) {
	iter, err := ex.engine.provider.CompleteStream(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("stream completion: %w", err)
	}
	defer iter.Close()

	resp := &llm.CompletionResponse{}
	for {
		chunk, nextErr := iter.Next()
		if chunk != nil {
			ex.applyStreamChunk(resp, chunk)
		}
		if nextErr != nil {
			if nextErr == io.EOF {
				break
			}
			return nil, nextErr
		}
	}

	if resp.FinishReason == "" {
		if len(resp.ToolCalls) > 0 {
			resp.FinishReason = llm.FinishToolCalls
		} else {
			resp.FinishReason = llm.FinishStop
		}
	}

	return resp, nil
}

func (ex *executor) applyStreamChunk(resp *llm.CompletionResponse, chunk *llm.StreamChunk) {
	if chunk.ID != "" {
		resp.ID = chunk.ID
	}
	if chunk.Model != "" {
		resp.Model = chunk.Model
	}
	if chunk.Content != "" {
		resp.Content += chunk.Content
		ex.addEvent(NewEvent(EventAgentReplyDelta, chunk.Content))
	}
	if chunk.BackgroundWork {
		ex.addEvent(NewEvent(EventAgentBackgroundWork, ""))
	}
	if len(chunk.ToolCalls) > 0 {
		resp.ToolCalls = make([]llm.ToolCall, len(chunk.ToolCalls))
		copy(resp.ToolCalls, chunk.ToolCalls)
	}
	if chunk.FinishReason != "" {
		resp.FinishReason = chunk.FinishReason
	}
	if chunk.Usage != nil {
		resp.Usage = *chunk.Usage
	}
}

func (ex *executor) handleResponse(ctx context.Context, resp *llm.CompletionResponse) (bool, error) {
	notice, err := ex.addContextMessage(llm.Message{
		Role:      "assistant",
		Content:   resp.Content,
		ToolCalls: resp.ToolCalls,
	})
	if err != nil {
		return false, err
	}
	if ex.engine.recorder != nil {
		if strings.TrimSpace(resp.Content) != "" && ex.engine.recorder.RecordAssistant != nil {
			if err := ex.engine.recorder.RecordAssistant(resp.Content); err != nil {
				return false, err
			}
		}
		for _, tc := range resp.ToolCalls {
			if ex.engine.recorder.RecordToolCall != nil {
				if err := ex.engine.recorder.RecordToolCall(tc); err != nil {
					return false, err
				}
			}
		}
	}
	if err := ex.persistSnapshot(); err != nil {
		return false, err
	}
	ex.emitContextCompactionNotice(notice)

	if ex.usesResponsesChain() && strings.TrimSpace(resp.ID) != "" {
		ex.responsesPreviousID = strings.TrimSpace(resp.ID)
		ex.responsesFollowup = nil
	}

	if resp.Content != "" {
		ex.addEvent(NewEvent(EventAgentReply, resp.Content))
	}

	if len(resp.ToolCalls) > 0 {
		for _, tc := range resp.ToolCalls {
			if err := ex.executeToolCall(ctx, tc); err != nil {
				return false, err
			}
		}
		return true, nil
	}

	return false, nil
}

func (ex *executor) executeToolCall(ctx context.Context, tc llm.ToolCall) error {
	toolName := tc.Function.Name

	tool, ok := ex.engine.tools.Get(toolName)
	if !ok {
		errMsg := fmt.Sprintf("Tool not found: %s", toolName)
		notice, err := ex.addToolResultWithFallback(tc.ID, errMsg)
		if err != nil {
			return err
		}
		if err := ex.persistSnapshot(); err != nil {
			return err
		}
		ex.emitContextCompactionNotice(notice)
		ex.addEvent(NewEvent(EventToolError, errMsg))
		return nil
	}

	// Check permission before emitting ToolCallStart so the UI does not
	// show "running command..." while actually waiting for user approval.
	action := extractAction(toolName, tc.Function.Arguments)
	path := extractPathArg(tc.Function.Arguments)
	granted, err := ex.engine.permission.Request(ctx, toolName, action, path)
	if err != nil {
		return err
	}
	if !granted {
		toolResultMsg := "The user doesn't want to proceed with this tool use. The tool use was rejected (eg. if it was a file edit, the new_string was NOT written to the file). STOP what you are doing and ask the user how to proceed."
		notice, err := ex.addToolResultWithFallback(tc.ID, toolResultMsg)
		if err != nil {
			return err
		}
		if err := ex.persistSnapshot(); err != nil {
			return err
		}
		ex.emitContextCompactionNotice(notice)
		errEv := NewEvent(EventToolError, "rejected by user")
		errEv.ToolName = toolName
		errEv.ToolCallID = tc.ID
		ex.addEvent(errEv)
		return nil
	}

	startEv := NewEvent(EventToolCallStart, describeToolCall(toolName, tc.Function.Arguments))
	startEv.ToolName = toolName
	startEv.ToolCallID = tc.ID
	ex.addEvent(startEv)

	// Execute
	var result *tools.Result
	streamingTool, canStream := tool.(tools.StreamingTool)
	if canStream {
		result, err = streamingTool.ExecuteStream(ctx, tc.Function.Arguments, func(update tools.StreamEvent) {
			ex.addStreamingToolEvent(toolName, tc.ID, update)
		})
	} else {
		result, err = tool.Execute(ctx, tc.Function.Arguments)
	}
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(ctx.Err(), context.Canceled) {
			if interruptErr := ex.handleInterruptedToolCall(tc, ""); interruptErr != nil {
				return interruptErr
			}
			return context.Canceled
		}
		errMsg := fmt.Sprintf("Tool execution error: %v", err)
		notice, err := ex.addToolResultWithFallback(tc.ID, errMsg)
		if err != nil {
			return err
		}
		if err := ex.persistSnapshot(); err != nil {
			return err
		}
		ex.emitContextCompactionNotice(notice)
		ex.addEvent(NewEvent(EventToolError, errMsg))
		return nil
	}

	if result.Error != nil {
		if errors.Is(result.Error, context.Canceled) || errors.Is(ctx.Err(), context.Canceled) {
			if interruptErr := ex.handleInterruptedToolCall(tc, result.Content); interruptErr != nil {
				return interruptErr
			}
			return context.Canceled
		}
		errMsg := result.Error.Error()
		notice, err := ex.addToolResultWithFallback(tc.ID, errMsg)
		if err != nil {
			return err
		}
		if err := ex.persistSnapshot(); err != nil {
			return err
		}
		ex.emitContextCompactionNotice(notice)
		ex.addEvent(NewEvent(EventToolError, fmt.Sprintf("Tool %s failed: %s", toolName, errMsg)))
		return nil
	}
	if errors.Is(ctx.Err(), context.Canceled) {
		if interruptErr := ex.handleInterruptedToolCall(tc, result.Content); interruptErr != nil {
			return interruptErr
		}
		return context.Canceled
	}

	notice, err := ex.addToolResultWithFallback(tc.ID, result.Content)
	if err != nil {
		return err
	}
	if toolName == "load_skill" && ex.engine.recorder != nil && ex.engine.recorder.RecordSkillActivate != nil {
		if skillName := skillNameFromToolCall(tc); skillName != "" {
			if err := ex.engine.recorder.RecordSkillActivate(skillName); err != nil {
				return err
			}
		}
	}
	if err := ex.persistSnapshot(); err != nil {
		return err
	}
	ex.emitContextCompactionNotice(notice)
	ex.addToolEvent(toolName, tc.ID, result)
	return nil
}

func (ex *executor) handleInterruptedToolCall(tc llm.ToolCall, partialOutput string) error {
	content := interruptedToolResultContent(partialOutput)
	notice, err := ex.addToolResultWithFallback(tc.ID, content)
	if err != nil {
		return err
	}
	if err := ex.persistSnapshot(); err != nil {
		return err
	}
	ex.emitContextCompactionNotice(notice)

	ev := NewEvent(EventToolInterrupted, strings.TrimSpace(partialOutput))
	ev.ToolName = tc.Function.Name
	ev.ToolCallID = tc.ID
	ev.Summary = "interrupted"
	ex.addEvent(ev)
	return nil
}

func interruptedToolResultContent(partialOutput string) string {
	lines := []string{
		"tool interrupted",
		"status: interrupted",
		"reason: user requested cancellation",
		"result: incomplete; do not treat this tool call as successful",
	}
	partialOutput = strings.TrimSpace(partialOutput)
	if partialOutput == "" {
		lines = append(lines, "partial_output: none")
		return strings.Join(lines, "\n")
	}
	lines = append(lines, "partial_output:")
	lines = append(lines, partialOutput)
	return strings.Join(lines, "\n")
}

var toolEventMap = map[string]string{
	"read":       EventToolRead,
	"grep":       EventToolGrep,
	"list_dir":   EventToolListDir,
	"glob":       EventToolGlob,
	"edit":       EventToolEdit,
	"write":      EventToolWrite,
	"shell":      EventCmdFinished,
	"load_skill": EventToolSkill,
}

func (ex *executor) addStreamingToolEvent(toolName, toolCallID string, update tools.StreamEvent) {
	var eventType string
	switch update.Type {
	case tools.StreamEventStarted:
		if toolName != "shell" {
			return
		}
		eventType = EventCmdStarted
	case tools.StreamEventOutput:
		if toolName != "shell" {
			return
		}
		eventType = EventCmdOutput
	default:
		return
	}

	ev := NewEvent(eventType, update.Message)
	ev.ToolName = toolName
	ev.ToolCallID = toolCallID
	ev.Summary = update.Summary
	ex.addEvent(ev)
}

func (ex *executor) addToolEvent(toolName, toolCallID string, result *tools.Result) {
	eventType := EventToolStarted
	if t, ok := toolEventMap[toolName]; ok {
		eventType = t
	}
	ev := NewEvent(eventType, result.Content)
	ev.ToolName = toolName
	ev.ToolCallID = toolCallID
	ev.Summary = result.Summary
	ev.Meta = result.Meta
	ex.addEvent(ev)
}

func (ex *executor) addEvent(ev Event) {
	usage := ex.engine.ctxManager.TokenUsage()
	ev.CtxUsed = usage.Current
	ev.CtxMax = usage.ContextWindow
	ev.TokensUsed = ex.totalUsage.TotalTokens
	ex.events = append(ex.events, ev)
	if ex.sink != nil {
		ex.sink(ev)
	}
}

func (ex *executor) trackUsage(u llm.Usage) {
	ex.totalUsage.PromptTokens += u.PromptTokens
	ex.totalUsage.CompletionTokens += u.CompletionTokens
	ex.totalUsage.TotalTokens += u.TotalTokens
}

func (ex *executor) persistSnapshot() error {
	if ex.engine.recorder == nil || ex.engine.recorder.PersistSnapshot == nil {
		return nil
	}
	return ex.engine.recorder.PersistSnapshot()
}

func (ex *executor) addContextMessage(msg llm.Message) (*contextCompactionNotice, error) {
	beforeUsage := ex.engine.ctxManager.TokenUsage()
	beforeCompactCount := ex.engine.ctxManager.CompactCount()
	if err := ex.engine.ctxManager.AddMessage(msg); err != nil {
		return nil, err
	}
	afterCompactCount := ex.engine.ctxManager.CompactCount()
	if afterCompactCount <= beforeCompactCount {
		return nil, nil
	}
	afterUsage := ex.engine.ctxManager.TokenUsage()
	return &contextCompactionNotice{
		BeforeTokens: beforeUsage.Current,
		AfterTokens:  afterUsage.Current,
	}, nil
}

func (ex *executor) emitContextCompactionNotice(notice *contextCompactionNotice) {
	if notice == nil {
		return
	}
	ex.addEvent(NewEvent(
		EventContextCompacted,
		fmt.Sprintf("Context compacted automatically: %d -> %d tokens.", notice.BeforeTokens, notice.AfterTokens),
	))
}

func (ex *executor) addToolResult(callID, content string) (*contextCompactionNotice, error) {
	msg := llm.NewToolMessage(callID, content)
	notice, err := ex.addContextMessage(msg)
	if err != nil {
		return nil, err
	}
	if ex.usesResponsesChain() && ex.responsesPreviousID != "" {
		ex.responsesFollowup = append(ex.responsesFollowup, msg)
	}
	if ex.engine.recorder != nil && ex.engine.recorder.RecordToolResult != nil {
		var toolCall llm.ToolCall
		toolCall.ID = callID
		if tc := ex.findToolCall(callID); tc != nil {
			toolCall = *tc
		}
		if err := ex.engine.recorder.RecordToolResult(toolCall, content); err != nil {
			return nil, err
		}
	}
	return notice, nil
}

func (ex *executor) addToolResultWithFallback(callID, content string) (*contextCompactionNotice, error) {
	notice, err := ex.addToolResult(callID, content)
	if err != nil {
		fallback := fmt.Sprintf("tool result replaced due to context limit: %v", err)
		fallbackNotice, fallbackErr := ex.addToolResult(callID, fallback)
		if fallbackErr != nil {
			return nil, fmt.Errorf("persist tool result fallback: %w (original error: %v)", fallbackErr, err)
		}
		if err := ex.persistSnapshot(); err != nil {
			return nil, err
		}
		ex.addEvent(NewEvent(EventToolError, fallback))
		return fallbackNotice, nil
	}
	return notice, nil
}

func (ex *executor) findToolCall(callID string) *llm.ToolCall {
	for _, msg := range ex.engine.ctxManager.GetNonSystemMessages() {
		if msg.Role != "assistant" {
			continue
		}
		for _, tc := range msg.ToolCalls {
			if tc.ID == callID {
				copy := tc
				return &copy
			}
		}
	}
	return nil
}

func skillNameFromToolCall(tc llm.ToolCall) string {
	if tc.Function.Name != "load_skill" {
		return ""
	}

	var args struct {
		Name string `json:"name"`
	}
	if err := json.Unmarshal(tc.Function.Arguments, &args); err != nil {
		return ""
	}
	return strings.TrimSpace(args.Name)
}

func extractAction(toolName string, raw json.RawMessage) string {
	if toolName == "shell" {
		var args struct {
			Command string `json:"command"`
		}
		if err := json.Unmarshal(raw, &args); err == nil {
			if cmd := strings.TrimSpace(args.Command); cmd != "" {
				return cmd
			}
		}
	}
	return string(raw)
}

func extractPathArg(raw json.RawMessage) string {
	var params map[string]any
	if err := json.Unmarshal(raw, &params); err != nil {
		return ""
	}
	for _, key := range []string{"path", "file_path"} {
		if v, ok := params[key].(string); ok {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

func describeToolCall(toolName string, raw json.RawMessage) string {
	var params map[string]any
	_ = json.Unmarshal(raw, &params)

	getString := func(keys ...string) string {
		for _, key := range keys {
			if v, ok := params[key].(string); ok {
				v = strings.TrimSpace(v)
				if v != "" {
					return v
				}
			}
		}
		return ""
	}

	switch toolName {
	case "shell":
		return getString("command")
	case "read", "edit", "write":
		return getString("path", "file_path")
	case "list_dir":
		return getString("path")
	case "grep":
		pattern := getString("pattern")
		path := getString("path")
		switch {
		case pattern != "" && path != "":
			return fmt.Sprintf("%q in %s", pattern, path)
		case pattern != "":
			return pattern
		default:
			return path
		}
	case "glob":
		pattern := getString("pattern")
		path := getString("path")
		switch {
		case pattern != "" && path != "":
			return fmt.Sprintf("%s in %s", pattern, path)
		case pattern != "":
			return pattern
		default:
			return path
		}
	case "load_skill":
		return getString("name")
	default:
		preview := strings.TrimSpace(string(raw))
		if preview == "" {
			return toolName
		}
		return preview
	}
}

func DefaultSystemPrompt() string {
	return `You are MindSpore CLI, an AI agent for AI infrastructure and model training workflows.

You help ML engineers and AI infra developers get training jobs running, diagnose failures, align results, migrate model code, and improve performance. You focus on training-task-oriented workflows rather than general-purpose code generation.

You have access to the following tools:
- read: Read file contents
- list_dir: List directory contents and structure
- write: Create or overwrite files
- edit: Edit files by replacing text
- grep: Search for patterns in files
- glob: Find files matching patterns
- shell: Execute shell commands
- load_skill: Load a skill's detailed instructions. Call this when the user's task matches an available skill listed below.

Guidelines:
1. Use tools to gather information before making changes
2. Always read files before editing them
3. Make minimal, focused changes
4. Prefer dedicated tools over shell whenever a dedicated tool can do the job
5. Run tests with shell to verify changes
6. If the user rejects a tool permission request, do not try to bypass that denial with another tool or shell pattern. Stop and ask the user how to proceed.
7. Before any write call, verify arguments contain BOTH "path" and "content"; if either is missing, do not call write yet.
8. Never call write with empty JSON arguments ({}).
9. When a user describes a training problem (failure, accuracy, performance), load the appropriate diagnosis skill.
10. When a user asks to migrate or port a model, load migrate-agent.

IMPORTANT: When you have gathered enough information to answer the user's question, you MUST provide your final answer directly WITHOUT using any more tools. Do not keep calling tools indefinitely - provide a clear, concise response once you have the information needed.

When making edits, ensure the old_string matches exactly (including whitespace and newlines).`
}

func defaultSystemPrompt() string {
	return DefaultSystemPrompt()
}
