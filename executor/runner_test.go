package executor

import (
	"context"
	"strings"
	"testing"
	"time"
)

func TestBashRunner_RunSuccess(t *testing.T) {
	r := NewBashRunner(".", 2*time.Second)
	res := r.Run(context.Background(), "echo hello")
	if res.Err != nil {
		t.Fatalf("unexpected error: %v", res.Err)
	}
	if res.ExitCode != 0 {
		t.Fatalf("unexpected exit code: %d", res.ExitCode)
	}
	if !strings.Contains(res.Output, "hello") {
		t.Fatalf("unexpected output: %q", res.Output)
	}
}

func TestBashRunner_Timeout(t *testing.T) {
	r := NewBashRunner(".", 100*time.Millisecond)
	res := r.Run(context.Background(), "sleep 1")
	if res.Err == nil {
		t.Fatalf("expected timeout error")
	}
}
