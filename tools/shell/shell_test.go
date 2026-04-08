package shell

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

	rshell "github.com/mindspore-lab/mindspore-cli/runtime/shell"
	"github.com/mindspore-lab/mindspore-cli/tools"
)

func TestShellToolExecute_DoesNotDuplicateCommandOrExit0InContent(t *testing.T) {
	runner := rshell.NewRunner(rshell.Config{
		WorkDir: ".",
		Timeout: 2 * time.Second,
	})
	tool := NewShellTool(runner)

	result, err := tool.Execute(context.Background(), []byte(`{"command":"printf 'hello\\n'"}`))
	if err != nil {
		t.Fatalf("execute shell tool: %v", err)
	}
	if result.Error != nil {
		t.Fatalf("unexpected result error: %v", result.Error)
	}

	if strings.Contains(result.Content, "$ printf") {
		t.Fatalf("expected content without command echo, got:\n%s", result.Content)
	}
	if strings.Contains(result.Content, "exit status 0") {
		t.Fatalf("expected content without exit status, got:\n%s", result.Content)
	}
	if strings.TrimSpace(result.Summary) == "exit 0" {
		t.Fatalf("expected summary not to be 'exit 0'")
	}
}

func TestShellToolExecuteStream_EmitsStartedAndOutput(t *testing.T) {
	runner := rshell.NewRunner(rshell.Config{
		WorkDir: ".",
		Timeout: 2 * time.Second,
	})
	tool := NewShellTool(runner)

	var (
		mu      sync.Mutex
		updates []tools.StreamEvent
	)
	result, err := tool.ExecuteStream(context.Background(), []byte(`{"command":"printf 'hello\\n'; printf 'warn\\n' >&2"}`), func(ev tools.StreamEvent) {
		mu.Lock()
		updates = append(updates, ev)
		mu.Unlock()
	})
	if err != nil {
		t.Fatalf("execute shell tool stream: %v", err)
	}
	if result.Error != nil {
		t.Fatalf("unexpected result error: %v", result.Error)
	}
	if len(updates) == 0 || updates[0].Type != tools.StreamEventStarted {
		t.Fatalf("expected first update to be started, got %#v", updates)
	}

	var sawStdout, sawStderr bool
	for _, update := range updates {
		if update.Type != tools.StreamEventOutput {
			continue
		}
		if strings.Contains(update.Message, "hello") {
			sawStdout = true
		}
		if strings.Contains(update.Message, "[stderr] warn") {
			sawStderr = true
		}
	}
	if !sawStdout {
		t.Fatalf("expected stdout update, got %#v", updates)
	}
	if !sawStderr {
		t.Fatalf("expected stderr update, got %#v", updates)
	}
}

func TestShellToolExecuteStream_ReturnsInterruptedSummaryWithPartialOutput(t *testing.T) {
	runner := rshell.NewRunner(rshell.Config{
		WorkDir: ".",
		Timeout: 5 * time.Second,
	})
	tool := NewShellTool(runner)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var once sync.Once
	result, err := tool.ExecuteStream(ctx, []byte(`{"command":"printf 'hello\\n'; sleep 5; printf 'done\\n'"}`), func(ev tools.StreamEvent) {
		if ev.Type == tools.StreamEventOutput && strings.Contains(ev.Message, "hello") {
			once.Do(cancel)
		}
	})
	if err != nil {
		t.Fatalf("execute shell tool stream: %v", err)
	}
	if result.Error != nil {
		t.Fatalf("unexpected result error: %v", result.Error)
	}
	if got, want := strings.TrimSpace(result.Summary), "interrupted"; got != want {
		t.Fatalf("summary = %q, want %q", got, want)
	}
	if !strings.Contains(result.Content, "hello") {
		t.Fatalf("expected partial stdout preserved, got:\n%s", result.Content)
	}
	if strings.Contains(result.Content, "done") {
		t.Fatalf("expected canceled command to omit trailing output, got:\n%s", result.Content)
	}
}
