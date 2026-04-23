package loop

import (
	"context"
	"strings"
	"testing"

	ctxmanager "github.com/mindspore-lab/mindspore-cli/agent/context"
	"github.com/mindspore-lab/mindspore-cli/integrations/llm"
)

func TestAddToolResultWithFallbackOnOversizedContent(t *testing.T) {
	cm := ctxmanager.NewManager(ctxmanager.ManagerConfig{
		ContextWindow:       120,
		ReserveTokens:       20,
		CompactionThreshold: 0.9,
	})

	engine := &Engine{ctxManager: cm}
	ex := &executor{engine: engine}

	oversized := strings.Repeat("x", 1000) // ~250 tokens, exceeds max usable 100
	if _, err := ex.addToolResultWithFallback("call_1", oversized); err != nil {
		t.Fatalf("addToolResultWithFallback returned error: %v", err)
	}

	msgs := cm.GetNonSystemMessages()
	if len(msgs) != 1 {
		t.Fatalf("expected exactly one tool message after fallback, got %d", len(msgs))
	}
	if msgs[0].Role != "tool" {
		t.Fatalf("expected tool role, got %q", msgs[0].Role)
	}
	if !strings.Contains(msgs[0].Content, "tool result replaced due to context limit") {
		t.Fatalf("expected fallback content, got %q", msgs[0].Content)
	}
}

func TestSyncContextTokenUsageFallsBackToEstimateWhenProviderUsageMissing(t *testing.T) {
	cm := ctxmanager.NewManager(ctxmanager.ManagerConfig{
		ContextWindow:       1000,
		ReserveTokens:       20,
		CompactionThreshold: 0.9,
	})
	if err := cm.AddMessage(llm.NewUserMessage("hello")); err != nil {
		t.Fatalf("AddMessage failed: %v", err)
	}

	estimated := cm.TokenUsage().Current
	engine := &Engine{ctxManager: cm}
	ex := &executor{engine: engine}

	ex.syncContextTokenUsage(llm.Usage{PromptTokens: 111})
	if got := cm.TokenUsage().Current; got != 111 {
		t.Fatalf("TokenUsage().Current with provider usage = %d, want 111", got)
	}

	ex.syncContextTokenUsage(llm.Usage{})
	if got := cm.TokenUsage().Current; got != estimated {
		t.Fatalf("TokenUsage().Current after fallback = %d, want %d", got, estimated)
	}
}

func TestHandleResponseSyncsProviderTotalUsageAfterAssistantMessage(t *testing.T) {
	cm := ctxmanager.NewManager(ctxmanager.ManagerConfig{
		ContextWindow:       1000,
		ReserveTokens:       20,
		CompactionThreshold: 0.9,
	})
	if err := cm.AddMessage(llm.NewUserMessage("hello")); err != nil {
		t.Fatalf("AddMessage failed: %v", err)
	}

	engine := &Engine{ctxManager: cm}
	ex := &executor{engine: engine}
	resp := &llm.CompletionResponse{
		Content: "ok",
		Usage: llm.Usage{
			PromptTokens:     1660,
			CompletionTokens: 149,
			TotalTokens:      1809,
		},
	}

	continueLoop, err := ex.handleResponse(context.Background(), resp)
	if err != nil {
		t.Fatalf("handleResponse() error = %v", err)
	}
	if continueLoop {
		t.Fatal("handleResponse() continueLoop = true, want false")
	}

	if got, want := cm.TokenUsage().Current, 1809; got != want {
		t.Fatalf("TokenUsage().Current = %d, want %d", got, want)
	}
	details := cm.TokenUsageDetails()
	if got, want := details.ProviderSnapshotTokens, 1809; got != want {
		t.Fatalf("TokenUsageDetails().ProviderSnapshotTokens = %d, want %d", got, want)
	}
	if got, want := details.ProviderTokenScope, ctxmanager.ProviderTokenScopeTotal; got != want {
		t.Fatalf("TokenUsageDetails().ProviderTokenScope = %q, want %q", got, want)
	}
}
