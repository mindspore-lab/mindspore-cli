package loop

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/vigo999/ms-cli/integrations/llm"
	"github.com/vigo999/ms-cli/tools"
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

	requireOrder(t, log, "tool_call:missing_tool", "snapshot:tool_call:missing_tool", "ui:ToolCallStart")
	requireOrder(t, log, "tool_result:missing_tool", "snapshot:tool_result:missing_tool", "ui:ToolError")
}

type blockingCancelTool struct {
	name    string
	started chan struct{}
}

func (t blockingCancelTool) Name() string {
	return t.name
}

func (t blockingCancelTool) Description() string {
	return "blocking cancel tool"
}

func (t blockingCancelTool) Schema() llm.ToolSchema {
	return llm.ToolSchema{Type: "object"}
}

func (t blockingCancelTool) Execute(ctx context.Context, _ json.RawMessage) (*tools.Result, error) {
	select {
	case <-t.started:
	default:
		close(t.started)
	}
	<-ctx.Done()
	return nil, ctx.Err()
}

func TestRunAddsInterruptedToolResultOnCanceledToolCall(t *testing.T) {
	args, err := json.Marshal(map[string]string{"path": "README.md"})
	if err != nil {
		t.Fatalf("marshal tool args: %v", err)
	}

	provider := &scriptedStreamProvider{
		responses: []*llm.CompletionResponse{{
			ToolCalls: []llm.ToolCall{{
				ID:   "call-cancel-1",
				Type: "function",
				Function: llm.ToolCallFunc{
					Name:      "read",
					Arguments: args,
				},
			}},
			FinishReason: llm.FinishToolCalls,
		}},
	}

	started := make(chan struct{})
	registry := tools.NewRegistry()
	registry.MustRegister(blockingCancelTool{name: "read", started: started})

	engine := NewEngine(EngineConfig{
		MaxIterations: 2,
		ContextWindow: 4096,
	}, provider, registry)

	ctx, cancel := context.WithCancel(context.Background())
	var events []Event
	done := make(chan error, 1)
	go func() {
		done <- engine.RunWithContextStream(ctx, Task{
			ID:          "cancel-tool",
			Description: "read file",
		}, func(ev Event) {
			events = append(events, ev)
		})
	}()

	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for tool execution to start")
	}
	cancel()

	select {
	case runErr := <-done:
		if !errors.Is(runErr, context.Canceled) {
			t.Fatalf("RunWithContextStream error = %v, want context canceled", runErr)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for cancellation")
	}

	msgs := engine.ctxManager.GetNonSystemMessages()
	if len(msgs) < 3 {
		t.Fatalf("expected at least 3 non-system messages, got %d", len(msgs))
	}

	last := msgs[len(msgs)-1]
	if last.Role != "tool" {
		t.Fatalf("last role = %q, want tool", last.Role)
	}
	if last.ToolCallID != "call-cancel-1" {
		t.Fatalf("last tool_call_id = %q, want %q", last.ToolCallID, "call-cancel-1")
	}
	if !strings.Contains(strings.ToLower(last.Content), "interrupted") {
		t.Fatalf("last tool result = %q, want interrupted marker", last.Content)
	}

	foundInterruptedEvent := false
	for _, ev := range events {
		if ev.Type == EventToolError && strings.Contains(strings.ToLower(ev.Message), "interrupted by user") {
			foundInterruptedEvent = true
			break
		}
	}
	if !foundInterruptedEvent {
		t.Fatalf("expected interrupted tool error event, got events: %#v", events)
	}
}

func TestRunBackfillsInterruptedResultsForUnexecutedToolCalls(t *testing.T) {
	args, err := json.Marshal(map[string]string{"path": "README.md"})
	if err != nil {
		t.Fatalf("marshal tool args: %v", err)
	}

	provider := &scriptedStreamProvider{
		responses: []*llm.CompletionResponse{{
			ToolCalls: []llm.ToolCall{
				{
					ID:   "call-cancel-1",
					Type: "function",
					Function: llm.ToolCallFunc{
						Name:      "read",
						Arguments: args,
					},
				},
				{
					ID:   "call-cancel-2",
					Type: "function",
					Function: llm.ToolCallFunc{
						Name:      "grep",
						Arguments: args,
					},
				},
			},
			FinishReason: llm.FinishToolCalls,
		}},
	}

	started := make(chan struct{})
	registry := tools.NewRegistry()
	registry.MustRegister(blockingCancelTool{name: "read", started: started})
	registry.MustRegister(stubTool{name: "grep", content: "should-not-run", summary: "1 line"})

	engine := NewEngine(EngineConfig{
		MaxIterations: 2,
		ContextWindow: 4096,
	}, provider, registry)

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		done <- engine.RunWithContextStream(ctx, Task{
			ID:          "cancel-two-tools",
			Description: "read and grep",
		}, nil)
	}()

	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for first tool execution to start")
	}
	cancel()

	select {
	case runErr := <-done:
		if !errors.Is(runErr, context.Canceled) {
			t.Fatalf("RunWithContextStream error = %v, want context canceled", runErr)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for cancellation")
	}

	msgs := engine.ctxManager.GetNonSystemMessages()
	toolResultsByID := map[string]string{}
	for _, msg := range msgs {
		if msg.Role != "tool" {
			continue
		}
		toolResultsByID[msg.ToolCallID] = msg.Content
	}

	for _, id := range []string{"call-cancel-1", "call-cancel-2"} {
		content, ok := toolResultsByID[id]
		if !ok {
			t.Fatalf("missing tool result for %s in messages: %#v", id, msgs)
		}
		if !strings.Contains(strings.ToLower(content), "interrupted") {
			t.Fatalf("tool result for %s = %q, want interrupted marker", id, content)
		}
	}
}
