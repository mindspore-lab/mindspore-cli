package shell

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	rshell "github.com/vigo999/ms-cli/runtime/shell"
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

func TestShellToolExecute_ReturnsContextCanceledOnInterrupt(t *testing.T) {
	runner := rshell.NewRunner(rshell.Config{
		WorkDir: ".",
		Timeout: 10 * time.Second,
	})
	tool := NewShellTool(runner)

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(100 * time.Millisecond)
		cancel()
	}()

	result, err := tool.Execute(ctx, []byte(`{"command":"sleep 100","timeout":120}`))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("execute shell tool error = %v, want context canceled", err)
	}
	if result != nil {
		t.Fatalf("expected nil result on cancel, got %#v", result)
	}
}
