package app

import (
	"strings"
	"testing"

	agentctx "github.com/mindspore-lab/mindspore-cli/agent/context"
	"github.com/mindspore-lab/mindspore-cli/integrations/llm"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

func TestCmdCompactCompactsContextAndEmitsTokenUpdate(t *testing.T) {
	ctxManager := agentctx.NewManager(agentctx.ManagerConfig{
		ContextWindow:       100,
		ReserveTokens:       10,
		CompactionThreshold: 0.9,
		EnableSmartCompact:  false,
	})
	for i := 0; i < 3; i++ {
		if err := ctxManager.AddMessage(llm.NewUserMessage(strings.Repeat("x", 80))); err != nil {
			t.Fatalf("AddMessage #%d failed: %v", i+1, err)
		}
	}

	before := ctxManager.TokenUsage().Current
	app := newModelCommandTestApp()
	app.ctxManager = ctxManager

	app.cmdCompact()

	drainUntilEventType(t, app, model.AgentThinking)
	ev := drainUntilEventType(t, app, model.TokenUpdate)
	reply := drainUntilEventType(t, app, model.AgentReply)

	if got := ctxManager.TokenUsage().Current; got > before/2 {
		t.Fatalf("context usage after cmdCompact = %d, want <= %d", got, before/2)
	}
	if got, want := ev.CtxUsed, ctxManager.TokenUsage().Current; got != want {
		t.Fatalf("TokenUpdate.CtxUsed = %d, want %d", got, want)
	}
	if got, want := ev.CtxMax, ctxManager.TokenUsage().ContextWindow; got != want {
		t.Fatalf("TokenUpdate.CtxMax = %d, want %d", got, want)
	}
	if !strings.Contains(reply.Message, "Context compacted:") {
		t.Fatalf("reply message = %q, want compaction summary", reply.Message)
	}
}
