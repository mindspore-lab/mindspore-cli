package loop

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	ctxmanager "github.com/mindspore-lab/mindspore-cli/agent/context"
	"github.com/mindspore-lab/mindspore-cli/integrations/llm"
)

type responsesCaptureProvider struct {
	captureProvider
}

func (p *responsesCaptureProvider) Name() string {
	return string(llm.ProviderOpenAIResponses)
}

func TestCallLLMSanitizesUnpairedToolMessagesBeforeRequest(t *testing.T) {
	provider := &captureProvider{}
	engine := newEngineForContextTests(provider)

	cm := ctxmanager.NewManager(ctxmanager.ManagerConfig{
		ContextWindow: 8000,
		ReserveTokens: 4000,
	})
	engine.SetContextManager(cm)

	validArgs, err := json.Marshal(map[string]string{"path": "README.md"})
	if err != nil {
		t.Fatalf("marshal valid args: %v", err)
	}
	invalidArgs, err := json.Marshal(map[string]string{"command": "pwd"})
	if err != nil {
		t.Fatalf("marshal invalid args: %v", err)
	}

	if err := cm.AddMessage(llm.Message{
		Role:    "assistant",
		Content: "keep this assistant content",
		ToolCalls: []llm.ToolCall{
			{
				ID:   "call_keep",
				Type: "function",
				Function: llm.ToolCallFunc{
					Name:      "read",
					Arguments: validArgs,
				},
			},
			{
				ID:   "call_drop",
				Type: "function",
				Function: llm.ToolCallFunc{
					Name:      "shell",
					Arguments: invalidArgs,
				},
			},
		},
	}); err != nil {
		t.Fatalf("AddMessage assistant failed: %v", err)
	}
	if err := cm.AddMessage(llm.NewToolMessage("call_keep", "README contents")); err != nil {
		t.Fatalf("AddMessage tool keep failed: %v", err)
	}
	if err := cm.AddMessage(llm.NewToolMessage("call_orphan", "orphan tool result")); err != nil {
		t.Fatalf("AddMessage tool orphan failed: %v", err)
	}

	ex := &executor{engine: engine}
	if _, err := ex.callLLM(context.Background()); err != nil {
		t.Fatalf("callLLM failed: %v", err)
	}

	if provider.lastReq == nil {
		t.Fatal("expected provider to receive request")
	}
	if len(provider.lastReq.Messages) != 3 {
		t.Fatalf("provider message count = %d, want 3", len(provider.lastReq.Messages))
	}

	assistant := provider.lastReq.Messages[1]
	if assistant.Role != "assistant" {
		t.Fatalf("assistant role = %q, want assistant", assistant.Role)
	}
	if assistant.Content != "keep this assistant content" {
		t.Fatalf("assistant content = %q, want preserved content", assistant.Content)
	}
	if len(assistant.ToolCalls) != 1 {
		t.Fatalf("assistant tool call count = %d, want 1", len(assistant.ToolCalls))
	}
	if got := assistant.ToolCalls[0].ID; got != "call_keep" {
		t.Fatalf("assistant tool call id = %q, want call_keep", got)
	}

	toolMsg := provider.lastReq.Messages[2]
	if toolMsg.Role != "tool" || toolMsg.ToolCallID != "call_keep" {
		t.Fatalf("tool message = %#v, want call_keep tool result", toolMsg)
	}

	sanitized := engine.ctxManager.GetNonSystemMessages()
	if len(sanitized) != 2 {
		t.Fatalf("sanitized message count = %d, want 2", len(sanitized))
	}
	if len(sanitized[0].ToolCalls) != 1 || sanitized[0].ToolCalls[0].ID != "call_keep" {
		t.Fatalf("sanitized assistant tool calls = %#v, want only call_keep", sanitized[0].ToolCalls)
	}
	if sanitized[1].ToolCallID != "call_keep" {
		t.Fatalf("sanitized tool result id = %q, want call_keep", sanitized[1].ToolCallID)
	}

	foundWarning := false
	for _, ev := range ex.events {
		if ev.Type == EventToolError && strings.Contains(ev.Message, "warning: removed") {
			foundWarning = true
			break
		}
	}
	if !foundWarning {
		t.Fatal("expected warning event for removed unpaired tool messages")
	}
}

func TestCallLLMSanitizesToolResultsSeparatedByAssistantReply(t *testing.T) {
	provider := &captureProvider{}
	engine := newEngineForContextTests(provider)

	cm := ctxmanager.NewManager(ctxmanager.ManagerConfig{
		ContextWindow: 8000,
		ReserveTokens: 4000,
	})
	engine.SetContextManager(cm)

	validArgs, err := json.Marshal(map[string]string{"path": "README.md"})
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}

	if err := cm.AddMessage(llm.Message{
		Role:    "assistant",
		Content: "need to inspect README",
		ToolCalls: []llm.ToolCall{{
			ID:   "call_gap",
			Type: "function",
			Function: llm.ToolCallFunc{
				Name:      "read",
				Arguments: validArgs,
			},
		}},
	}); err != nil {
		t.Fatalf("AddMessage assistant failed: %v", err)
	}
	if err := cm.AddMessage(llm.NewAssistantMessage("intervening assistant reply")); err != nil {
		t.Fatalf("AddMessage intervening assistant failed: %v", err)
	}
	if err := cm.AddMessage(llm.NewToolMessage("call_gap", "README contents")); err != nil {
		t.Fatalf("AddMessage late tool result failed: %v", err)
	}

	ex := &executor{engine: engine}
	if _, err := ex.callLLM(context.Background()); err != nil {
		t.Fatalf("callLLM failed: %v", err)
	}

	if provider.lastReq == nil {
		t.Fatal("expected provider to receive request")
	}
	if len(provider.lastReq.Messages) != 3 {
		t.Fatalf("provider message count = %d, want 3", len(provider.lastReq.Messages))
	}

	firstAssistant := provider.lastReq.Messages[1]
	if firstAssistant.Role != "assistant" {
		t.Fatalf("first assistant role = %q, want assistant", firstAssistant.Role)
	}
	if firstAssistant.Content != "need to inspect README" {
		t.Fatalf("first assistant content = %q, want preserved content", firstAssistant.Content)
	}
	if len(firstAssistant.ToolCalls) != 0 {
		t.Fatalf("first assistant tool calls = %#v, want none", firstAssistant.ToolCalls)
	}

	secondAssistant := provider.lastReq.Messages[2]
	if secondAssistant.Role != "assistant" || secondAssistant.Content != "intervening assistant reply" {
		t.Fatalf("second assistant message = %#v, want preserved intervening assistant", secondAssistant)
	}

	sanitized := engine.ctxManager.GetNonSystemMessages()
	if len(sanitized) != 2 {
		t.Fatalf("sanitized message count = %d, want 2", len(sanitized))
	}
	if len(sanitized[0].ToolCalls) != 0 {
		t.Fatalf("sanitized first assistant tool calls = %#v, want none", sanitized[0].ToolCalls)
	}
	if sanitized[1].Role != "assistant" || sanitized[1].Content != "intervening assistant reply" {
		t.Fatalf("sanitized second message = %#v, want intervening assistant", sanitized[1])
	}

	foundWarning := false
	for _, ev := range ex.events {
		if ev.Type == EventToolError && strings.Contains(ev.Message, "warning: removed") {
			foundWarning = true
			break
		}
	}
	if !foundWarning {
		t.Fatal("expected warning event for removed separated tool messages")
	}
}

func TestCallLLMSanitizesResponsesFollowupAgainstContextPairs(t *testing.T) {
	provider := &responsesCaptureProvider{}
	engine := newEngineForContextTests(provider)

	cm := ctxmanager.NewManager(ctxmanager.ManagerConfig{
		ContextWindow: 8000,
		ReserveTokens: 4000,
	})
	engine.SetContextManager(cm)

	validArgs, err := json.Marshal(map[string]string{"path": "README.md"})
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}

	if err := cm.AddMessage(llm.Message{
		Role: "assistant",
		ToolCalls: []llm.ToolCall{{
			ID:   "call_keep",
			Type: "function",
			Function: llm.ToolCallFunc{
				Name:      "read",
				Arguments: validArgs,
			},
		}},
	}); err != nil {
		t.Fatalf("AddMessage assistant failed: %v", err)
	}
	if err := cm.AddMessage(llm.NewToolMessage("call_keep", "README contents")); err != nil {
		t.Fatalf("AddMessage tool keep failed: %v", err)
	}
	if err := cm.AddMessage(llm.NewToolMessage("call_orphan", "orphan tool result")); err != nil {
		t.Fatalf("AddMessage tool orphan failed: %v", err)
	}

	ex := &executor{
		engine:              engine,
		responsesPreviousID: "resp_prev",
		responsesFollowup: []llm.Message{
			llm.NewToolMessage("call_keep", "README contents"),
			llm.NewToolMessage("call_orphan", "orphan tool result"),
		},
	}
	if _, err := ex.callLLM(context.Background()); err != nil {
		t.Fatalf("callLLM failed: %v", err)
	}

	if provider.lastReq == nil {
		t.Fatal("expected provider to receive request")
	}
	if len(provider.lastReq.Messages) != 2 {
		t.Fatalf("provider message count = %d, want 2", len(provider.lastReq.Messages))
	}
	if provider.lastReq.Messages[0].Role != "system" {
		t.Fatalf("first request message role = %q, want system", provider.lastReq.Messages[0].Role)
	}
	if provider.lastReq.Messages[1].Role != "tool" || provider.lastReq.Messages[1].ToolCallID != "call_keep" {
		t.Fatalf("followup request tool message = %#v, want call_keep tool result", provider.lastReq.Messages[1])
	}
	if len(ex.responsesFollowup) != 1 || ex.responsesFollowup[0].ToolCallID != "call_keep" {
		t.Fatalf("responsesFollowup = %#v, want only call_keep", ex.responsesFollowup)
	}
}

func TestCallLLMSanitizesResponsesFollowupSeparatedFromAssistantToolCall(t *testing.T) {
	provider := &responsesCaptureProvider{}
	engine := newEngineForContextTests(provider)

	cm := ctxmanager.NewManager(ctxmanager.ManagerConfig{
		ContextWindow: 8000,
		ReserveTokens: 4000,
	})
	engine.SetContextManager(cm)

	validArgs, err := json.Marshal(map[string]string{"path": "README.md"})
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}

	if err := cm.AddMessage(llm.Message{
		Role: "assistant",
		ToolCalls: []llm.ToolCall{{
			ID:   "call_gap",
			Type: "function",
			Function: llm.ToolCallFunc{
				Name:      "read",
				Arguments: validArgs,
			},
		}},
	}); err != nil {
		t.Fatalf("AddMessage assistant failed: %v", err)
	}
	if err := cm.AddMessage(llm.NewAssistantMessage("intervening assistant reply")); err != nil {
		t.Fatalf("AddMessage intervening assistant failed: %v", err)
	}
	if err := cm.AddMessage(llm.NewToolMessage("call_gap", "README contents")); err != nil {
		t.Fatalf("AddMessage late tool result failed: %v", err)
	}

	ex := &executor{
		engine:              engine,
		responsesPreviousID: "resp_prev",
		responsesFollowup: []llm.Message{
			llm.NewToolMessage("call_gap", "README contents"),
		},
	}
	if _, err := ex.callLLM(context.Background()); err != nil {
		t.Fatalf("callLLM failed: %v", err)
	}

	if provider.lastReq == nil {
		t.Fatal("expected provider to receive request")
	}
	if len(provider.lastReq.Messages) != 2 {
		t.Fatalf("provider message count = %d, want 2", len(provider.lastReq.Messages))
	}
	if provider.lastReq.Messages[0].Role != "system" {
		t.Fatalf("first request message role = %q, want system", provider.lastReq.Messages[0].Role)
	}
	if provider.lastReq.Messages[1].Role != "assistant" || provider.lastReq.Messages[1].Content != "intervening assistant reply" {
		t.Fatalf("second request message = %#v, want intervening assistant only", provider.lastReq.Messages[1])
	}
	if len(ex.responsesFollowup) != 0 {
		t.Fatalf("responsesFollowup = %#v, want empty after sanitization", ex.responsesFollowup)
	}
	if ex.responsesPreviousID != "" {
		t.Fatalf("responsesPreviousID = %q, want cleared when followup becomes empty", ex.responsesPreviousID)
	}
}
