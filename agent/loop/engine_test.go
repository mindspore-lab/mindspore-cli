package loop

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/vigo999/ms-cli/integrations/domain"
)

func TestShouldDirectReply(t *testing.T) {
	if !shouldDirectReply("hello") {
		t.Fatal("hello should trigger direct reply")
	}
	if !shouldDirectReply("你好") {
		t.Fatal("你好 should trigger direct reply")
	}
	if shouldDirectReply("fix bug in app/run.go") {
		t.Fatal("task-like input should not trigger direct reply")
	}
}

type testFactory struct{}

func (f testFactory) ClientFor(spec domain.ModelSpec) (domain.ModelClient, error) {
	return testClient{}, nil
}

func (f testFactory) Providers() []domain.ProviderInfo {
	return nil
}

type testClient struct{}

func (c testClient) Generate(ctx context.Context, req domain.GenerateRequest) (*domain.GenerateResponse, error) {
	return &domain.GenerateResponse{Text: "ok"}, nil
}

type shellLoopFactory struct{}

func (f shellLoopFactory) ClientFor(spec domain.ModelSpec) (domain.ModelClient, error) {
	return shellLoopClient{}, nil
}

func (f shellLoopFactory) Providers() []domain.ProviderInfo {
	return nil
}

type shellLoopClient struct{}

func (c shellLoopClient) Generate(ctx context.Context, req domain.GenerateRequest) (*domain.GenerateResponse, error) {
	return &domain.GenerateResponse{
		Text: `{"action":"shell","command":"ls -la"}`,
	}, nil
}

type nopFS struct{}

func (nopFS) Read(path string) (string, error) { return "", nil }
func (nopFS) Grep(path, pattern string, maxMatches int) ([]string, error) {
	return nil, nil
}
func (nopFS) Edit(path, oldText, newText string) (string, error) { return "", nil }
func (nopFS) Write(path, content string) (int, error)            { return len(content), nil }

type nopShell struct{}

func (nopShell) Run(ctx context.Context, command string) (string, int, error) {
	return "ok", 0, nil
}

func TestRunWithContext_Canceled(t *testing.T) {
	engine := NewEngine(Config{
		ModelFactory: testFactory{},
	})

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	events, err := engine.RunWithContext(ctx, Task{
		Description: "analyze code",
		Model: ModelSpec{
			Provider: "openrouter",
			Name:     "deepseek/deepseek-chat",
		},
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context canceled, got %v", err)
	}
	if len(events) == 0 {
		t.Fatalf("expected paused event")
	}
	if events[0].Type != EventReply {
		t.Fatalf("expected reply event, got %s", events[0].Type)
	}
}

func TestRunWithContextStream_EmitsEvents(t *testing.T) {
	engine := NewEngine(Config{
		ModelFactory:   testFactory{},
		DefaultMaxStep: 0, // unlimited
	})

	got := make([]EventType, 0, 4)
	err := engine.RunWithContextStream(context.Background(), Task{
		Description: "analyze code structure",
		Model: ModelSpec{
			Provider: "openrouter",
			Name:     "deepseek/deepseek-chat",
		},
	}, func(ev Event) {
		got = append(got, ev.Type)
	})
	if err != nil {
		t.Fatalf("RunWithContextStream failed: %v", err)
	}
	if len(got) < 2 {
		t.Fatalf("expected multiple streamed events, got %d", len(got))
	}
	if got[0] != EventThinking {
		t.Fatalf("first streamed event=%s want %s", got[0], EventThinking)
	}
	if got[len(got)-1] != EventReply {
		t.Fatalf("last streamed event=%s want %s", got[len(got)-1], EventReply)
	}
}

func TestRunWithContext_RepeatedShellLoopGuardStops(t *testing.T) {
	engine := NewEngine(Config{
		FS:               nopFS{},
		Shell:            nopShell{},
		ModelFactory:     shellLoopFactory{},
		DefaultMaxStep:   0, // unlimited
		MaxRepeatedShell: 2,
	})

	events, err := engine.RunWithContext(context.Background(), Task{
		Description: "分析代码结构",
		Model: ModelSpec{
			Provider: "openrouter",
			Name:     "deepseek/deepseek-chat",
		},
	})
	if err != nil {
		t.Fatalf("RunWithContext failed: %v", err)
	}
	if len(events) == 0 {
		t.Fatalf("expected events")
	}

	last := events[len(events)-1]
	if last.Type != EventReply {
		t.Fatalf("expected final reply, got %s", last.Type)
	}
	if !strings.Contains(last.Message, "重复命令循环") {
		t.Fatalf("expected loop-guard stop message, got %q", last.Message)
	}
}
