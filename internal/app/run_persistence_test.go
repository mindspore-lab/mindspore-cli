package app

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	agentctx "github.com/mindspore-lab/mindspore-cli/agent/context"
	"github.com/mindspore-lab/mindspore-cli/agent/loop"
	"github.com/mindspore-lab/mindspore-cli/agent/session"
	"github.com/mindspore-lab/mindspore-cli/integrations/llm"
	"github.com/mindspore-lab/mindspore-cli/tools"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

type singleReplyProvider struct {
	content string
}

func (p *singleReplyProvider) Name() string {
	return "single-reply"
}

func (p *singleReplyProvider) Complete(context.Context, *llm.CompletionRequest) (*llm.CompletionResponse, error) {
	return &llm.CompletionResponse{Content: p.content, FinishReason: llm.FinishStop}, nil
}

func (p *singleReplyProvider) CompleteStream(context.Context, *llm.CompletionRequest) (llm.StreamIterator, error) {
	return &singleReplyIterator{
		chunks: []llm.StreamChunk{
			{Content: p.content, FinishReason: llm.FinishStop},
		},
	}, nil
}

func (p *singleReplyProvider) SupportsTools() bool {
	return true
}

func (p *singleReplyProvider) AvailableModels() []llm.ModelInfo {
	return nil
}

type singleReplyIterator struct {
	chunks []llm.StreamChunk
	index  int
}

func (it *singleReplyIterator) Next() (*llm.StreamChunk, error) {
	if it.index >= len(it.chunks) {
		return nil, io.EOF
	}
	chunk := it.chunks[it.index]
	it.index++
	return &chunk, nil
}

func (it *singleReplyIterator) Close() error {
	return nil
}

func TestRunTaskWithoutLLMDoesNotPersistSession(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	workDir := t.TempDir()
	runtimeSession, err := session.Create(workDir, "system prompt")
	if err != nil {
		t.Fatalf("create session: %v", err)
	}
	t.Cleanup(func() {
		_ = runtimeSession.Close()
	})

	ctxManager := agentctx.NewManager(agentctx.ManagerConfig{
		ContextWindow: 4096,
		ReserveTokens: 512,
	})
	ctxManager.SetSystemPrompt("system prompt")

	app := &Application{
		EventCh:    make(chan model.Event),
		llmReady:   false,
		session:    runtimeSession,
		ctxManager: ctxManager,
	}

	done := make(chan struct{})
	go func() {
		app.runTask("hello")
		close(done)
	}()

	ev := <-app.EventCh
	if ev.Type != model.AgentReply {
		t.Fatalf("event type = %q, want %q", ev.Type, model.AgentReply)
	}
	if ev.Message != provideAPIKeyFirstMsg {
		t.Fatalf("event message = %q, want %q", ev.Message, provideAPIKeyFirstMsg)
	}

	if _, err := os.Stat(runtimeSession.Path()); !os.IsNotExist(err) {
		t.Fatalf("expected no trajectory without live llm activity, got %v", err)
	}
	snapshotPath := filepath.Join(filepath.Dir(runtimeSession.Path()), "snapshot.json")
	if _, err := os.Stat(snapshotPath); !os.IsNotExist(err) {
		t.Fatalf("expected no snapshot without live llm activity, got %v", err)
	}
	if got := app.exitResumeHint(); got != "" {
		t.Fatalf("expected no resume hint without live llm activity, got %q", got)
	}

	<-done
}

func TestRunTaskPersistsSessionAfterLiveLLMReply(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	workDir := t.TempDir()
	runtimeSession, err := session.Create(workDir, "system prompt")
	if err != nil {
		t.Fatalf("create session: %v", err)
	}
	t.Cleanup(func() {
		_ = runtimeSession.Close()
	})

	ctxManager := agentctx.NewManager(agentctx.ManagerConfig{
		ContextWindow: 4096,
		ReserveTokens: 512,
	})
	ctxManager.SetSystemPrompt("system prompt")

	provider := &singleReplyProvider{content: "hi there"}
	engine := loop.NewEngine(loop.EngineConfig{
		MaxIterations: 1,
		ContextWindow: 4096,
	}, provider, tools.NewRegistry())
	engine.SetContextManager(ctxManager)

	app := &Application{
		Engine:     engine,
		EventCh:    make(chan model.Event, 32),
		llmReady:   true,
		session:    runtimeSession,
		ctxManager: ctxManager,
	}
	engine.SetTrajectoryRecorder(newTrajectoryRecorder(runtimeSession, ctxManager, app.noteLiveLLMActivity))

	app.runTask("hello")

	if _, err := os.Stat(runtimeSession.Path()); err != nil {
		t.Fatalf("expected trajectory after live llm reply, got %v", err)
	}
	snapshotPath := filepath.Join(filepath.Dir(runtimeSession.Path()), "snapshot.json")
	if _, err := os.Stat(snapshotPath); err != nil {
		t.Fatalf("expected snapshot after live llm reply, got %v", err)
	}

	trajectory, err := os.ReadFile(runtimeSession.Path())
	if err != nil {
		t.Fatalf("read trajectory: %v", err)
	}
	if !strings.Contains(string(trajectory), `"type":"user"`) {
		t.Fatalf("expected trajectory to contain user record, got %s", string(trajectory))
	}
	if !strings.Contains(string(trajectory), `"type":"assistant"`) {
		t.Fatalf("expected trajectory to contain assistant record, got %s", string(trajectory))
	}
	if got := app.exitResumeHint(); !strings.Contains(got, "mscli resume "+runtimeSession.ID()) {
		t.Fatalf("expected resume hint with session id after live llm reply, got %q", got)
	}
}
