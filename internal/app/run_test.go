package app

import (
	"bytes"
	"context"
	"io"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/vigo999/mindspore-code/agent/loop"
	"github.com/vigo999/mindspore-code/integrations/llm"
	"github.com/vigo999/mindspore-code/tools"
	"github.com/vigo999/mindspore-code/ui/model"
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

	deadline := time.NewTimer(300 * time.Millisecond)
	defer deadline.Stop()

	foundTaskDone := false
	for {
		select {
		case ev := <-app.EventCh:
			switch ev.Type {
			case model.TaskDone:
				foundTaskDone = true
			case model.ToolError:
				t.Fatalf("expected no ToolError after interrupt, got %q", ev.Message)
			}
		case <-deadline.C:
			if !foundTaskDone {
				t.Fatal("timed out waiting for TaskDone after interrupt")
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

func TestTUIProgramOptionsEnableAltScreenAndBracketedPaste(t *testing.T) {
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

	got := out.String()
	for _, want := range []string{
		"\x1b[?1049h", // alt screen
		"\x1b[?2004h", // bracketed paste
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("expected startup output to include %q, got %q", want, got)
		}
	}
	// Mouse cell motion must NOT be enabled (breaks terminal paste)
	if strings.Contains(got, "\x1b[?1002h") {
		t.Fatal("mouse cell motion should be disabled to allow terminal paste")
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
