package app

import (
	"encoding/json"
	"strings"
	"testing"

	agentctx "github.com/mindspore-lab/mindspore-cli/agent/context"
	"github.com/mindspore-lab/mindspore-cli/integrations/llm"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

func TestCmdCtxOmitsSourceSectionForProviderBackedUsage(t *testing.T) {
	ctxManager := agentctx.NewManager(agentctx.ManagerConfig{
		ContextWindow: 1000,
		ReserveTokens: 100,
	})
	if err := ctxManager.AddMessage(llm.NewUserMessage("hello")); err != nil {
		t.Fatalf("AddMessage failed: %v", err)
	}
	ctxManager.SetPromptTokenUsage("openai-responses", 120)
	if err := ctxManager.AddMessage(llm.NewAssistantMessage("ok")); err != nil {
		t.Fatalf("AddMessage assistant failed: %v", err)
	}

	app := newModelCommandTestApp()
	app.ctxManager = ctxManager

	app.cmdCtx()

	ev := drainUntilEventType(t, app, model.TokenUpdate)
	reply := drainUntilEventType(t, app, model.AgentReply)

	if got, want := ev.CtxUsed, ctxManager.TokenUsage().Current; got != want {
		t.Fatalf("TokenUpdate.CtxUsed = %d, want %d", got, want)
	}
	if strings.Contains(reply.Message, "Source:") {
		t.Fatalf("reply message = %q, want no source section", reply.Message)
	}
	if !strings.Contains(reply.Message, "Current:") {
		t.Fatalf("reply message = %q, want base context usage", reply.Message)
	}
}

func TestCmdCtxOmitsSourceSectionForProviderBackedTotalSnapshot(t *testing.T) {
	ctxManager := agentctx.NewManager(agentctx.ManagerConfig{
		ContextWindow: 1000,
		ReserveTokens: 100,
	})
	if err := ctxManager.AddMessage(llm.NewUserMessage("hello")); err != nil {
		t.Fatalf("AddMessage failed: %v", err)
	}
	if err := ctxManager.AddMessage(llm.NewAssistantMessage("ok")); err != nil {
		t.Fatalf("AddMessage assistant failed: %v", err)
	}
	ctxManager.SetTotalTokenUsage("anthropic", 1809)

	app := newModelCommandTestApp()
	app.ctxManager = ctxManager

	app.cmdCtx()

	reply := drainUntilEventType(t, app, model.AgentReply)
	if strings.Contains(reply.Message, "Source:") {
		t.Fatalf("reply message = %q, want no source section", reply.Message)
	}
	if !strings.Contains(reply.Message, "Current:   1809") {
		t.Fatalf("reply message = %q, want provider-backed current usage", reply.Message)
	}
}

func TestCmdCtxShowsProviderUsageStatsFromRawPayload(t *testing.T) {
	ctxManager := agentctx.NewManager(agentctx.ManagerConfig{
		ContextWindow: 1000,
		ReserveTokens: 100,
	})
	if err := ctxManager.AddMessage(llm.NewUserMessage("hello")); err != nil {
		t.Fatalf("AddMessage failed: %v", err)
	}
	if err := ctxManager.AddMessage(llm.NewAssistantMessage("ok")); err != nil {
		t.Fatalf("AddMessage assistant failed: %v", err)
	}
	ctxManager.SetProviderTokenUsage("anthropic", llm.Usage{
		PromptTokens:     1660,
		CompletionTokens: 149,
		TotalTokens:      1809,
		Raw:              json.RawMessage(`{"prompt_tokens":1660,"completion_tokens":149,"total_tokens":1809,"cached_tokens":0,"service_tier":"standard","input_tokens_details":{"cached_tokens":32}}`),
	})

	app := newModelCommandTestApp()
	app.ctxManager = ctxManager

	app.cmdCtx()

	reply := drainUntilEventType(t, app, model.AgentReply)
	if strings.Contains(reply.Message, "Source:") {
		t.Fatalf("reply message = %q, want no source section", reply.Message)
	}
	if !strings.Contains(reply.Message, "Provider usage stats:") {
		t.Fatalf("reply message = %q, want provider usage stats section", reply.Message)
	}
	if !strings.Contains(reply.Message, "completion_tokens: 149") {
		t.Fatalf("reply message = %q, want completion_tokens line", reply.Message)
	}
	if !strings.Contains(reply.Message, "cached_tokens: 0") {
		t.Fatalf("reply message = %q, want cached_tokens line", reply.Message)
	}
	if !strings.Contains(reply.Message, "input_tokens_details.cached_tokens: 32") {
		t.Fatalf("reply message = %q, want nested cached_tokens line", reply.Message)
	}
	if !strings.Contains(reply.Message, "service_tier: standard") {
		t.Fatalf("reply message = %q, want service_tier line", reply.Message)
	}
}

func TestCmdCtxOmitsSourceSectionForLocalEstimateFallback(t *testing.T) {
	ctxManager := agentctx.NewManager(agentctx.ManagerConfig{
		ContextWindow: 1000,
		ReserveTokens: 100,
	})
	if err := ctxManager.AddMessage(llm.NewUserMessage("hello")); err != nil {
		t.Fatalf("AddMessage failed: %v", err)
	}

	app := newModelCommandTestApp()
	app.ctxManager = ctxManager

	app.cmdCtx()

	reply := drainUntilEventType(t, app, model.AgentReply)
	if strings.Contains(reply.Message, "Source:") {
		t.Fatalf("reply message = %q, want no source section", reply.Message)
	}
	if !strings.Contains(reply.Message, "Current:") {
		t.Fatalf("reply message = %q, want base context usage", reply.Message)
	}
}

func TestCmdCtxHidesBootstrapSystemPromptUsageBeforeFirstMessage(t *testing.T) {
	ctxManager := agentctx.NewManager(agentctx.ManagerConfig{
		ContextWindow: 1000,
		ReserveTokens: 100,
	})
	ctxManager.SetSystemPrompt("system prompt")
	if got := ctxManager.TokenUsage().Current; got == 0 {
		t.Fatal("raw context usage unexpectedly zero; test requires non-zero system prompt usage")
	}

	app := newModelCommandTestApp()
	app.ctxManager = ctxManager

	app.cmdCtx()

	ev := drainUntilEventType(t, app, model.TokenUpdate)
	reply := drainUntilEventType(t, app, model.AgentReply)
	if got := ev.CtxUsed; got != 0 {
		t.Fatalf("TokenUpdate.CtxUsed = %d, want 0", got)
	}
	if !strings.Contains(reply.Message, "Current:   0") {
		t.Fatalf("reply message = %q, want hidden bootstrap usage", reply.Message)
	}
	if strings.Contains(reply.Message, "Source:") {
		t.Fatalf("reply message = %q, want no source section", reply.Message)
	}
}
