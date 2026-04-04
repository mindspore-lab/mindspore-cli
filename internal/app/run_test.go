package app

import (
	"bytes"
	"context"
	"io"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/vigo999/mindspore-cli/agent/loop"
	"github.com/vigo999/mindspore-cli/integrations/llm"
	"github.com/vigo999/mindspore-cli/tools"
	"github.com/vigo999/mindspore-cli/ui/model"
)

type blockingStreamProvider struct {
	started chan struct{}
}

func (p *blockingStreamProvider) Name() string {
	return "blocking"
}

func (p *blockingStreamProvider) Complete(context.Context, *llm.CompletionRequest) (*llm.CompletionResponse, error) {
	return nil, io.EOF
}

func (p *blockingStreamProvider) CompleteStream(ctx context.Context, req *llm.CompletionRequest) (llm.StreamIterator, error) {
	select {
	case <-p.started:
	default:
		close(p.started)
	}
	return &blockingStreamIterator{ctx: ctx}, nil
}

func (p *blockingStreamProvider) SupportsTools() bool {
	return true
}

func (p *blockingStreamProvider) AvailableModels() []llm.ModelInfo {
	return nil
}

type blockingStreamIterator struct {
	ctx context.Context
}

func (it *blockingStreamIterator) Next() (*llm.StreamChunk, error) {
	<-it.ctx.Done()
	return nil, it.ctx.Err()
}

func (it *blockingStreamIterator) Close() error {
	return nil
}

func TestInterruptTokenCancelsActiveTask(t *testing.T) {
	provider := &blockingStreamProvider{started: make(chan struct{})}
	engine := loop.NewEngine(loop.EngineConfig{
		MaxIterations: 1,
		ContextWindow: 4096,
	}, provider, tools.NewRegistry())

	app := &Application{
		Engine:   engine,
		EventCh:  make(chan model.Event, 32),
		llmReady: true,
	}

	done := make(chan struct{})
	go func() {
		app.runTask("hello")
		close(done)
	}()

	select {
	case <-provider.started:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for task to start")
	}

	app.processInput(interruptActiveTaskToken)

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for task cancellation")
	}

	deadline := time.After(200 * time.Millisecond)
	for {
		select {
		case ev := <-app.EventCh:
			if ev.Type == model.ToolError && strings.Contains(strings.ToLower(ev.Message), "canceled") {
				t.Fatalf("expected interrupt cancellation to stay silent, got tool error %q", ev.Message)
			}
		case <-deadline:
			return
		}
	}
}

func TestRunTaskTimeoutEmitsWarningNotError(t *testing.T) {
	provider := &blockingStreamProvider{started: make(chan struct{})}
	engine := loop.NewEngine(loop.EngineConfig{
		MaxIterations:  1,
		ContextWindow:  4096,
		TimeoutPerTurn: 20 * time.Millisecond,
	}, provider, tools.NewRegistry())

	app := &Application{
		Engine:   engine,
		EventCh:  make(chan model.Event, 32),
		llmReady: true,
	}

	done := make(chan struct{})
	go func() {
		app.runTask("hello")
		close(done)
	}()

	select {
	case <-provider.started:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for task to start")
	}

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for timeout handling")
	}

	deadline := time.After(500 * time.Millisecond)
	var sawWarning bool
	for {
		select {
		case ev := <-app.EventCh:
			if ev.Type == model.ToolError && strings.Contains(strings.ToLower(ev.Message), "timeout") {
				t.Fatalf("expected timeout to emit warning, got tool error %q", ev.Message)
			}
			if ev.Type == model.ToolWarning {
				sawWarning = true
				if !strings.Contains(strings.ToLower(ev.Message), "timeout") && !strings.Contains(strings.ToLower(ev.Message), "deadline") {
					t.Fatalf("expected timeout warning message, got %q", ev.Message)
				}
			}
		case <-deadline:
			if !sawWarning {
				t.Fatal("expected timeout warning event")
			}
			return
		}
	}
}

type renderOnceModel struct {
	rendered chan struct{}
}

func (m *renderOnceModel) Init() tea.Cmd { return nil }

func (m *renderOnceModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	return m, nil
}

func (m *renderOnceModel) View() string {
	select {
	case <-m.rendered:
	default:
		close(m.rendered)
	}
	return "success\n"
}

func captureTUIStartupOutput(t *testing.T) string {
	t.Helper()

	var in bytes.Buffer
	var out bytes.Buffer

	m := &renderOnceModel{rendered: make(chan struct{})}
	p := tea.NewProgram(m, tuiProgramOptions(tea.WithInput(&in), tea.WithOutput(&out))...)

	go func() {
		<-m.rendered
		p.Quit()
	}()

	if _, err := p.Run(); err != nil {
		t.Fatal(err)
	}

	return out.String()
}

func TestTUIProgramOptionsInlineWithBracketedPaste(t *testing.T) {
	got := captureTUIStartupOutput(t)

	if !strings.Contains(got, "\x1b[?2004h") {
		t.Fatalf("expected startup output to include bracketed paste, got %q", got)
	}
	if strings.Contains(got, "\x1b[?1049h") {
		t.Fatal("alt screen should not be enabled in inline mode")
	}
	if strings.Contains(got, "\x1b[?1002h") {
		t.Fatal("mouse cell motion should not be enabled in inline mode")
	}
}

func TestConvertLoopEvent_TaskStartedIsNotRendered(t *testing.T) {
	ev := loop.Event{
		Type:    loop.EventTaskStarted,
		Message: "Task: repeated user input",
	}

	got := convertLoopEvent(ev)
	if got != nil {
		t.Fatalf("convertLoopEvent(TaskStarted) = %+v, want nil", got)
	}
}

func TestConvertLoopEvent_UnknownWithMessageFallsBackToAgentReply(t *testing.T) {
	ev := loop.Event{
		Type:    "UnknownEvent",
		Message: "some status",
	}

	got := convertLoopEvent(ev)
	if got == nil {
		t.Fatalf("convertLoopEvent(UnknownEvent) = nil, want non-nil")
	}
	if got.Type != model.AgentReply {
		t.Fatalf("convertLoopEvent type = %v, want %v", got.Type, model.AgentReply)
	}
	if got.Message != ev.Message {
		t.Fatalf("convertLoopEvent message = %q, want %q", got.Message, ev.Message)
	}
}

func TestConvertLoopEvent_ContextCompactedUsesContextNotice(t *testing.T) {
	ev := loop.Event{
		Type:    loop.EventContextCompacted,
		Message: "Context compacted automatically: 80 -> 40 tokens.",
	}

	got := convertLoopEvent(ev)
	if got == nil {
		t.Fatal("convertLoopEvent(ContextCompacted) = nil, want non-nil")
	}
	if got.Type != model.ContextNotice {
		t.Fatalf("convertLoopEvent type = %v, want %v", got.Type, model.ContextNotice)
	}
	if got.Message != ev.Message {
		t.Fatalf("convertLoopEvent message = %q, want %q", got.Message, ev.Message)
	}
}

func TestConvertLoopEvent_PreservesToolCallID(t *testing.T) {
	ev := loop.Event{
		Type:       loop.EventCmdOutput,
		ToolName:   "shell",
		ToolCallID: "call-shell-1",
		Message:    "PASS",
	}

	got := convertLoopEvent(ev)
	if got == nil {
		t.Fatal("convertLoopEvent(CmdOutput) = nil, want non-nil")
	}
	if got.Type != model.CmdOutput {
		t.Fatalf("convertLoopEvent type = %v, want %v", got.Type, model.CmdOutput)
	}
	if got.ToolCallID != ev.ToolCallID {
		t.Fatalf("convertLoopEvent ToolCallID = %q, want %q", got.ToolCallID, ev.ToolCallID)
	}
}
