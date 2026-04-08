package loop

import (
	"context"
	"encoding/json"
	"io"
	"strings"
	"sync"
	"testing"

	ctxmanager "github.com/mindspore-lab/mindspore-cli/agent/context"
	"github.com/mindspore-lab/mindspore-cli/integrations/llm"
	"github.com/mindspore-lab/mindspore-cli/tools"
)

type scriptedStreamProvider struct {
	mu        sync.Mutex
	responses []*llm.CompletionResponse
}

func (p *scriptedStreamProvider) Name() string {
	return "scripted"
}

func (p *scriptedStreamProvider) Complete(context.Context, *llm.CompletionRequest) (*llm.CompletionResponse, error) {
	return nil, io.EOF
}

func (p *scriptedStreamProvider) CompleteStream(context.Context, *llm.CompletionRequest) (llm.StreamIterator, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if len(p.responses) == 0 {
		return &scriptedStreamIterator{}, nil
	}

	resp := p.responses[0]
	p.responses = p.responses[1:]

	return &scriptedStreamIterator{
		chunks: []llm.StreamChunk{{
			Content:      resp.Content,
			ToolCalls:    append([]llm.ToolCall(nil), resp.ToolCalls...),
			FinishReason: resp.FinishReason,
			Usage:        &resp.Usage,
		}},
	}, nil
}

func (p *scriptedStreamProvider) SupportsTools() bool {
	return true
}

func (p *scriptedStreamProvider) AvailableModels() []llm.ModelInfo {
	return nil
}

type scriptedStreamIterator struct {
	chunks []llm.StreamChunk
	index  int
}

func (it *scriptedStreamIterator) Next() (*llm.StreamChunk, error) {
	if it.index >= len(it.chunks) {
		return nil, io.EOF
	}
	chunk := it.chunks[it.index]
	it.index++
	return &chunk, nil
}

func (it *scriptedStreamIterator) Close() error {
	return nil
}

type stubTool struct {
	name    string
	content string
	summary string
}

func (t stubTool) Name() string {
	return t.name
}

func (t stubTool) Description() string {
	return "stub tool"
}

func (t stubTool) Schema() llm.ToolSchema {
	return llm.ToolSchema{Type: "object"}
}

func (t stubTool) Execute(context.Context, json.RawMessage) (*tools.Result, error) {
	return &tools.Result{Content: t.content, Summary: t.summary}, nil
}

type streamingStubTool struct {
	stubTool
	updates []tools.StreamEvent
}

func (t streamingStubTool) ExecuteStream(ctx context.Context, raw json.RawMessage, emit func(tools.StreamEvent)) (*tools.Result, error) {
	if emit != nil {
		emit(tools.StreamEvent{Type: tools.StreamEventStarted})
		for _, update := range t.updates {
			emit(update)
		}
	}
	return t.Execute(ctx, raw)
}

type cancelAwareStreamingStubTool struct {
	stubTool
	started chan struct{}
}

func (t cancelAwareStreamingStubTool) ExecuteStream(ctx context.Context, raw json.RawMessage, emit func(tools.StreamEvent)) (*tools.Result, error) {
	if emit != nil {
		emit(tools.StreamEvent{Type: tools.StreamEventStarted})
		emit(tools.StreamEvent{Type: tools.StreamEventOutput, Message: t.content})
	}
	select {
	case <-t.started:
	default:
		close(t.started)
	}
	<-ctx.Done()
	return tools.StringResultWithSummary(t.content, "interrupted"), nil
}

func newPersistenceRecorder(log *[]string) *TrajectoryRecorder {
	last := ""
	appendLog := func(entry string) {
		*log = append(*log, entry)
	}

	return &TrajectoryRecorder{
		RecordUserInput: func(string) error {
			last = "user"
			appendLog(last)
			return nil
		},
		RecordAssistant: func(string) error {
			last = "assistant"
			appendLog(last)
			return nil
		},
		RecordToolCall: func(tc llm.ToolCall) error {
			last = "tool_call:" + tc.Function.Name
			appendLog(last)
			return nil
		},
		RecordToolResult: func(tc llm.ToolCall, _ string) error {
			last = "tool_result:" + tc.Function.Name
			appendLog(last)
			return nil
		},
		RecordSkillActivate: func(skillName string) error {
			last = "skill:" + skillName
			appendLog(last)
			return nil
		},
		PersistSnapshot: func() error {
			appendLog("snapshot:" + last)
			return nil
		},
	}
}

func requireOrder(t *testing.T, log []string, entries ...string) {
	t.Helper()

	next := 0
	for _, entry := range entries {
		found := -1
		for i := next; i < len(log); i++ {
			if log[i] == entry {
				found = i
				next = i + 1
				break
			}
		}
		if found == -1 {
			t.Fatalf("expected log to contain %q after index %d, got %v", entry, next, log)
		}
	}
}

func TestRunPersistsSnapshotBeforeStreamingTaskEvents(t *testing.T) {
	provider := &scriptedStreamProvider{
		responses: []*llm.CompletionResponse{{
			Content:      "ok",
			FinishReason: llm.FinishStop,
		}},
	}
	engine := NewEngine(EngineConfig{
		MaxIterations: 1,
		ContextWindow: 4096,
	}, provider, tools.NewRegistry())

	var log []string
	engine.SetTrajectoryRecorder(newPersistenceRecorder(&log))

	err := engine.RunWithContextStream(context.Background(), Task{
		ID:          "persist-before-ui",
		Description: "say ok",
	}, func(ev Event) {
		log = append(log, "ui:"+ev.Type)
	})
	if err != nil {
		t.Fatalf("RunWithContextStream failed: %v", err)
	}

	requireOrder(t, log, "user", "snapshot:user", "ui:TaskStarted")
	requireOrder(t, log, "assistant", "snapshot:assistant", "ui:AgentReply")
}

func TestRunPersistsToolResultBeforeToolRender(t *testing.T) {
	args, err := json.Marshal(map[string]string{"path": "README.md"})
	if err != nil {
		t.Fatalf("marshal tool args: %v", err)
	}

	provider := &scriptedStreamProvider{
		responses: []*llm.CompletionResponse{
			{
				ToolCalls: []llm.ToolCall{{
					ID:   "call-read-1",
					Type: "function",
					Function: llm.ToolCallFunc{
						Name:      "read",
						Arguments: args,
					},
				}},
				FinishReason: llm.FinishToolCalls,
			},
			{
				Content:      "done",
				FinishReason: llm.FinishStop,
			},
		},
	}

	registry := tools.NewRegistry()
	registry.MustRegister(stubTool{name: "read", content: "file contents", summary: "1 line"})

	engine := NewEngine(EngineConfig{
		MaxIterations: 2,
		ContextWindow: 4096,
	}, provider, registry)

	var log []string
	engine.SetTrajectoryRecorder(newPersistenceRecorder(&log))

	err = engine.RunWithContextStream(context.Background(), Task{
		ID:          "persist-tool-result",
		Description: "read the file",
	}, func(ev Event) {
		log = append(log, "ui:"+ev.Type)
	})
	if err != nil {
		t.Fatalf("RunWithContextStream failed: %v", err)
	}

	requireOrder(t, log, "tool_call:read", "snapshot:tool_call:read", "ui:ToolCallStart")
	requireOrder(t, log, "tool_result:read", "snapshot:tool_result:read", "ui:ToolRead")
}

func TestRunShellStreamingEmitsLiveCommandEvents(t *testing.T) {
	args, err := json.Marshal(map[string]string{"command": "printf 'line-1\\nline-2\\n'"})
	if err != nil {
		t.Fatalf("marshal tool args: %v", err)
	}

	provider := &scriptedStreamProvider{
		responses: []*llm.CompletionResponse{
			{
				ToolCalls: []llm.ToolCall{{
					ID:   "call-shell-1",
					Type: "function",
					Function: llm.ToolCallFunc{
						Name:      "shell",
						Arguments: args,
					},
				}},
				FinishReason: llm.FinishToolCalls,
			},
			{
				Content:      "done",
				FinishReason: llm.FinishStop,
			},
		},
	}

	registry := tools.NewRegistry()
	registry.MustRegister(streamingStubTool{
		stubTool: stubTool{name: "shell", content: "line-1\nline-2", summary: "completed"},
		updates: []tools.StreamEvent{
			{Type: tools.StreamEventOutput, Message: "line-1"},
			{Type: tools.StreamEventOutput, Message: "line-2"},
		},
	})

	engine := NewEngine(EngineConfig{
		MaxIterations: 2,
		ContextWindow: 4096,
	}, provider, registry)

	var events []Event
	err = engine.RunWithContextStream(context.Background(), Task{
		ID:          "stream-shell-events",
		Description: "run a shell command",
	}, func(ev Event) {
		switch ev.Type {
		case EventToolCallStart, EventCmdStarted, EventCmdOutput, EventCmdFinished:
			events = append(events, ev)
		}
	})
	if err != nil {
		t.Fatalf("RunWithContextStream failed: %v", err)
	}

	if len(events) != 5 {
		t.Fatalf("event count = %d, want 5 (%#v)", len(events), events)
	}

	wantTypes := []string{
		EventToolCallStart,
		EventCmdStarted,
		EventCmdOutput,
		EventCmdOutput,
		EventCmdFinished,
	}
	for i, want := range wantTypes {
		if got := events[i].Type; got != want {
			t.Fatalf("events[%d].Type = %q, want %q", i, got, want)
		}
		if got := events[i].ToolCallID; got != "call-shell-1" {
			t.Fatalf("events[%d].ToolCallID = %q, want call-shell-1", i, got)
		}
	}

	if got := events[4].Summary; got != "completed" {
		t.Fatalf("final summary = %q, want completed", got)
	}
	if got := events[4].Message; got != "line-1\nline-2" {
		t.Fatalf("final message = %q, want full shell output", got)
	}
}

func TestRunPersistsInterruptedToolResultBeforeInterruptedRender(t *testing.T) {
	args, err := json.Marshal(map[string]string{"command": "sleep 10"})
	if err != nil {
		t.Fatalf("marshal tool args: %v", err)
	}

	provider := &scriptedStreamProvider{
		responses: []*llm.CompletionResponse{{
			ToolCalls: []llm.ToolCall{{
				ID:   "call-shell-interrupt-1",
				Type: "function",
				Function: llm.ToolCallFunc{
					Name:      "shell",
					Arguments: args,
				},
			}},
			FinishReason: llm.FinishToolCalls,
		}},
	}

	registry := tools.NewRegistry()
	started := make(chan struct{})
	registry.MustRegister(cancelAwareStreamingStubTool{
		stubTool: stubTool{name: "shell", content: "partial line"},
		started:  started,
	})

	engine := NewEngine(EngineConfig{
		MaxIterations: 1,
		ContextWindow: 4096,
	}, provider, registry)

	var log []string
	var events []Event
	engine.SetTrajectoryRecorder(newPersistenceRecorder(&log))

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		<-started
		cancel()
	}()

	err = engine.RunWithContextStream(ctx, Task{
		ID:          "interrupt-shell",
		Description: "run shell then interrupt",
	}, func(ev Event) {
		log = append(log, "ui:"+ev.Type)
		events = append(events, ev)
	})
	if err == nil || !strings.Contains(strings.ToLower(err.Error()), "canceled") {
		t.Fatalf("RunWithContextStream error = %v, want context canceled", err)
	}

	requireOrder(t, log, "tool_call:shell", "snapshot:tool_call:shell", "ui:ToolCallStart")
	requireOrder(t, log, "tool_result:shell", "snapshot:tool_result:shell", "ui:ToolInterrupted")

	var interrupted *Event
	for i := range events {
		if events[i].Type == EventToolInterrupted {
			interrupted = &events[i]
			break
		}
	}
	if interrupted == nil {
		t.Fatalf("expected ToolInterrupted event, got %#v", events)
	}
	if got, want := interrupted.ToolCallID, "call-shell-interrupt-1"; got != want {
		t.Fatalf("ToolInterrupted ToolCallID = %q, want %q", got, want)
	}
	if got, want := interrupted.Summary, "interrupted"; got != want {
		t.Fatalf("ToolInterrupted summary = %q, want %q", got, want)
	}
	if got, want := interrupted.Message, "partial line"; got != want {
		t.Fatalf("ToolInterrupted message = %q, want %q", got, want)
	}

	messages := engine.ctxManager.GetNonSystemMessages()
	var toolResult string
	for _, msg := range messages {
		if msg.Role == "tool" && msg.ToolCallID == "call-shell-interrupt-1" {
			toolResult = msg.Content
		}
	}
	if !strings.Contains(toolResult, "status: interrupted") {
		t.Fatalf("tool result missing interrupted status, got:\n%s", toolResult)
	}
	if !strings.Contains(toolResult, "reason: user requested cancellation") {
		t.Fatalf("tool result missing cancellation reason, got:\n%s", toolResult)
	}
	if !strings.Contains(toolResult, "partial_output:") {
		t.Fatalf("tool result missing partial output section, got:\n%s", toolResult)
	}
	if !strings.Contains(toolResult, "partial line") {
		t.Fatalf("tool result missing streamed partial output, got:\n%s", toolResult)
	}
}

func TestRunPersistsSnapshotBeforeContextCompactionNotice(t *testing.T) {
	provider := &scriptedStreamProvider{
		responses: []*llm.CompletionResponse{{
			Content:      "ok",
			FinishReason: llm.FinishStop,
		}},
	}
	engine := NewEngine(EngineConfig{
		MaxIterations: 1,
		ContextWindow: 100,
	}, provider, tools.NewRegistry())

	cm := ctxmanager.NewManager(ctxmanager.ManagerConfig{
		ContextWindow:       100,
		ReserveTokens:       10,
		CompactionThreshold: 0.9,
		EnableSmartCompact:  false,
	})
	cm.SetSystemPrompt("system")
	for i := 0; i < 3; i++ {
		if err := cm.AddMessage(llm.NewUserMessage(strings.Repeat("x", 80))); err != nil {
			t.Fatalf("preload AddMessage #%d failed: %v", i+1, err)
		}
	}
	engine.SetContextManager(cm)

	var log []string
	var compactMessage string
	engine.SetTrajectoryRecorder(newPersistenceRecorder(&log))

	err := engine.RunWithContextStream(context.Background(), Task{
		ID:          "persist-context-compaction",
		Description: strings.Repeat("y", 40),
	}, func(ev Event) {
		log = append(log, "ui:"+ev.Type)
		if ev.Type == EventContextCompacted {
			compactMessage = ev.Message
		}
	})
	if err != nil {
		t.Fatalf("RunWithContextStream failed: %v", err)
	}

	requireOrder(t, log, "user", "snapshot:user", "ui:ContextCompacted", "ui:TaskStarted")
	if !strings.Contains(compactMessage, "Context compacted automatically:") {
		t.Fatalf("context compaction message = %q, want automatic compaction summary", compactMessage)
	}
}

func TestRunPersistsToolErrorBeforeErrorRender(t *testing.T) {
	args, err := json.Marshal(map[string]string{"path": "missing.txt"})
	if err != nil {
		t.Fatalf("marshal tool args: %v", err)
	}

	provider := &scriptedStreamProvider{
		responses: []*llm.CompletionResponse{
			{
				ToolCalls: []llm.ToolCall{{
					ID:   "call-missing-1",
					Type: "function",
					Function: llm.ToolCallFunc{
						Name:      "missing_tool",
						Arguments: args,
					},
				}},
				FinishReason: llm.FinishToolCalls,
			},
			{
				Content:      "done",
				FinishReason: llm.FinishStop,
			},
		},
	}

	engine := NewEngine(EngineConfig{
		MaxIterations: 2,
		ContextWindow: 4096,
	}, provider, tools.NewRegistry())

	var log []string
	engine.SetTrajectoryRecorder(newPersistenceRecorder(&log))

	err = engine.RunWithContextStream(context.Background(), Task{
		ID:          "persist-tool-error",
		Description: "use the missing tool",
	}, func(ev Event) {
		log = append(log, "ui:"+ev.Type)
	})
	if err != nil {
		t.Fatalf("RunWithContextStream failed: %v", err)
	}

	// ToolCallStart is emitted after permission check, so missing tools
	// skip it entirely — verify error handling order instead.
	requireOrder(t, log, "tool_call:missing_tool", "snapshot:tool_call:missing_tool")
	requireOrder(t, log, "tool_result:missing_tool", "snapshot:tool_result:missing_tool", "ui:ToolError")
}
